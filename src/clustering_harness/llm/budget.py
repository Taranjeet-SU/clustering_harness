"""Token accounting and hard spend caps.

Every non-cached call records its token usage in a :class:`Ledger` (SQLite, safe across threads and
processes). A :class:`Budget` refuses new calls once the recorded spend for its scope reaches the limit,
so a run fails loudly instead of overspending. Pointing several runs at the same ledger file enforces a
shared cap (e.g. "$20 in total for model X")."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PRICES_PER_MTOK", "Price", "Ledger", "Budget", "BudgetExceeded", "price_for"]


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, tin: int, tout: int) -> float:
        return tin / 1e6 * self.input_per_mtok + tout / 1e6 * self.output_per_mtok


# USD per million tokens (standard tier, prompts below any long-context surcharge). Verify current
# provider pricing before relying on these numbers; override per LLM with ``price=Price(...)``.
PRICES_PER_MTOK: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-opus-5-5": Price(4.00, 20.00),
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-4o": Price(2.50, 10.00),
}


def price_for(model: str) -> Price:
    """Known price for a hosted model name, or zero (local / unknown models).

    Provider-specific ids built from a known name match it, e.g. the Bedrock id
    ``us.anthropic.claude-haiku-5-5-v1:0`` is priced as ``claude-haiku-5-5`` and ``gpt-4o-mini-2024-07-18`` as
    ``gpt-4o-mini``. Near misses (``gpt-4o-mni``) and fine-tuned models (``ft:...``) are unknown.
    Prices differ by provider and region; pass an explicit :class:`Price` when it matters."""
    import re
    model = model or ""
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    if model.startswith("ft:") or ":ft:" in model:
        return Price(0.0, 0.0)            # fine-tuned models are priced differently: give an explicit price
    for name in sorted(PRICES_PER_MTOK, key=len, reverse=True):
        # the known name must stand alone: an optional provider prefix (``us.anthropic.``, ``openai/``) and an
        # optional version / date / region suffix (``-2024-07-18``, ``-v1:0``, ``@latest``), nothing else
        if re.search(r"(^|[./:])" + re.escape(name) + r"($|[-@:](v?\d|latest))", model):
            return PRICES_PER_MTOK[name]
    return Price(0.0, 0.0)


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Append-only usage ledger."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path) if str(path) == ":memory:" else self.path, check_same_thread=False,
                                     timeout=60, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS usage (t REAL, scope TEXT, model TEXT, stage TEXT, "
                           "input_tokens INTEGER, output_tokens INTEGER, usd REAL)")

    def record(self, scope: str, model: str, stage: str, tin: int, tout: int, usd: float) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO usage VALUES (?,?,?,?,?,?,?)", (time.time(), scope, model, stage, tin, tout, usd))

    def spent(self, scope: str) -> float:
        with self._lock:
            r = self._conn.execute("SELECT COALESCE(SUM(usd),0) FROM usage WHERE scope=?", (scope,)).fetchone()
        return float(r[0])

    def summary(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT scope, model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(usd) "
                                      "FROM usage GROUP BY scope, model").fetchall()
        return [dict(scope=s, model=m, calls=c, input_tokens=i, output_tokens=o, usd=round(u, 4)) for s, m, c, i, o, u in rows]


class Budget:
    """Hard cap in USD for one scope of a ledger."""

    def __init__(self, limit_usd: float, ledger: Ledger, scope: str):
        self.limit_usd = float(limit_usd)
        self.ledger = ledger
        self.scope = scope

    def check(self, reserved: float = 0.0, request_estimate: float = 0.0) -> None:
        """Raise :class:`BudgetExceeded` if the recorded spend, plus the estimated cost of requests already in
        flight (``reserved``) and of the next request, would pass the cap. Reserving in-flight costs keeps
        parallel requests from overshooting the cap."""
        spent = self.ledger.spent(self.scope)
        if spent + reserved + request_estimate > self.limit_usd or spent >= self.limit_usd:
            raise BudgetExceeded(f"budget for {self.scope!r} exhausted: ${spent:.4f} spent (+${reserved:.4f} in flight) "
                                 f"of ${self.limit_usd:.4g}. Raise budget_usd and rerun: finished work is cached, so "
                                 f"you only pay for the rest")

    def remaining(self) -> float:
        return self.limit_usd - self.ledger.spent(self.scope)
