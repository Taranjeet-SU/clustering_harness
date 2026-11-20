"""The LLM wrapper used by every algorithm stage.

Contract (identical for all providers):

* :meth:`LLM.chat` returns the response text, ``""`` if the backend failed twice.
* :meth:`LLM.chat_json` parses the first ``{...}`` object of the response. If parsing fails it asks once
  more, appending the bad answer and a correction request; if that fails too it returns
  ``{"_error": ..., "_raw": ...}`` instead of raising, so callers count contract failures explicitly.
* :meth:`LLM.call` is the cached entry point used by the stages. A *reasoning* model whose answer came
  back empty (its hidden reasoning used the whole budget) is retried once with a larger budget.

Concurrency is bounded per LLM by a semaphore; :func:`run_parallel` fans jobs out per model."""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

from .backends import OfflineCallError
from .base import BackendConfigError, ModelUnavailableError, Backend, BackendError, ChatRequest
import logging
from .budget import Budget, Ledger, Price, price_for
from .cache import ResponseCache, request_key

log = logging.getLogger("clustering_harness")

_SECRET = re.compile(r"\b(sk-[A-Za-z0-9*_\-]{6,}|sk-ant-[A-Za-z0-9*_\-]{6,}|(?:AKIA|ASIA)[A-Z0-9*]{8,}|AIza[0-9A-Za-z*_\-]{10,})")


def redact(text: str) -> str:
    """Hide API-key-like strings that providers echo back in error messages."""
    return _SECRET.sub("[redacted-key]", str(text))

__all__ = ["LLM", "run_parallel", "extract_json"]

_JSON = re.compile(r"\{.*\}", re.S)


def extract_json(raw: str) -> dict:
    """First-to-last brace span of ``raw`` parsed as JSON (raises on failure)."""
    return json.loads(_JSON.search(raw).group(0))


class LLM:
    """A configured model slot.

    Parameters
    ----------
    backend:
        Provider adapter (see :mod:`clustering_harness.llm.backends`).
    name:
        Short logical name used in logs (e.g. ``"a"``/``"b"`` or ``"qwen"``).
    cache:
        Shared :class:`ResponseCache`; ``None`` disables caching.
    max_concurrency:
        Maximum simultaneous requests to this model from this process.
    reasoning:
        Model spends hidden reasoning tokens from the generation budget (e.g. gpt-oss). Enables the
        empty-answer budget retry and text-mode judging.
    budget_usd, ledger, price:
        Optional hard spend cap. ``price`` defaults to the known price for the backend's model name.
    """

    def __init__(self, backend: Backend, name: str | None = None, cache: ResponseCache | None = None,
                 max_concurrency: int = 2, reasoning: bool = False, budget_usd: float | None = None,
                 ledger: Ledger | None = None, price: Price | None = None, transport_retries: int = 2,
                 max_consecutive_failures: int = 8):
        self.backend = backend
        self.name = name or backend.model_id
        self.cache = cache
        self.reasoning = reasoning
        self.max_concurrency = max_concurrency
        self._sem = threading.BoundedSemaphore(max_concurrency)
        self.price = price or price_for(getattr(backend, "model", ""))
        self.ledger = ledger
        self.budget = Budget(budget_usd, ledger, backend.model_id) if (budget_usd is not None and ledger is not None) else None
        if budget_usd is not None and self.price.input_per_mtok == 0 and self.price.output_per_mtok == 0 \
                and not backend.model_id.startswith(("ollama:", "offline", "callable")):
            log.warning("%s: budget_usd=%s cannot be enforced because the price of %s is unknown; pass price=Price(in, out) "
                        "(or price_in/price_out in a run spec)", self.name, budget_usd, backend.model_id)
        self.transport_retries = transport_retries
        self.max_consecutive_failures = max_consecutive_failures
        self.stats = {"requests": 0, "cache_hits": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0,
                      "uncached_failures": 0}
        self._stats_lock = threading.Lock()
        self._consecutive_failures = 0
        self._last_error: str | None = None
        self._inflight_usd = 0.0
        self._local = threading.local()

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    # ------------------------------------------------------------------ raw calls
    def chat(self, messages: list, max_tokens: int = 800, temperature: float = 0.0, seed: int = 0,
             json_mode: bool = False, stage: str = "") -> str:
        req = ChatRequest(messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed, json_mode=json_mode)
        estimate = 0.0
        if self.budget is not None:
            chars = sum(len(str(m.get("content", ""))) for m in messages)
            estimate = self.price.cost(chars // 3 + 1, max_tokens)          # conservative: ~3 chars per token
            with self._stats_lock:
                self.budget.check(self._inflight_usd, estimate)
                self._inflight_usd += estimate
        try:
            for attempt in range(self.transport_retries):
                try:
                    with self._sem:
                        res = self.backend.complete(req)
                except OfflineCallError:
                    raise
                except BackendConfigError as e:
                    msg = redact(e)
                    log.error("%s (%s): %s", self.name, self.model_id, msg)
                    raise ModelUnavailableError(f"{self.name} ({self.model_id}) cannot be used: {msg}. Fix the model settings "
                                                f"and rerun; failed requests were not cached.") from None
                except BackendError as e:
                    with self._stats_lock:
                        self.stats["failures"] += 1
                        self._last_error = redact(e)[:300]
                        n = self.stats["failures"]
                    if n <= 3 or n % 50 == 0:
                        log.warning("%s: request failed (attempt %d/%d, %d failures so far): %s", self.name, attempt + 1,
                                    self.transport_retries, n, redact(e)[:300])
                    continue
                with self._stats_lock:
                    self.stats["requests"] += 1
                    self.stats["input_tokens"] += res.input_tokens
                    self.stats["output_tokens"] += res.output_tokens
                    self._consecutive_failures = 0
                if self.ledger is not None:
                    self.ledger.record(self.model_id, self.model_id, stage, res.input_tokens, res.output_tokens,
                                       self.price.cost(res.input_tokens, res.output_tokens))
                return res.text
        finally:
            if estimate:
                with self._stats_lock:
                    self._inflight_usd -= estimate
        # every attempt failed: mark the call so that it is not cached, and give up if the model looks down
        self._local.failed = True
        with self._stats_lock:
            self._consecutive_failures += 1
            streak = self._consecutive_failures
        if streak >= self.max_consecutive_failures:
            raise ModelUnavailableError(f"{self.name} ({self.model_id}): {streak} requests in a row failed; last error: "
                                        f"{self._last_error}. Check the server / network / limits and rerun; failed "
                                        f"requests were not cached.")
        return ""

    def chat_json(self, messages: list, max_tokens: int = 800, temperature: float = 0.0, seed: int = 0, stage: str = "") -> dict:
        raw = self.chat(messages, max_tokens, temperature, seed, json_mode=True, stage=stage)
        for attempt in range(2):
            try:
                return extract_json(raw)
            except Exception as e:  # noqa: BLE001 - any parse failure is a contract failure
                if attempt == 1:
                    return {"_error": f"unparseable after retry: {str(e)[:80]}", "_raw": raw[:300]}
                raw = self.chat(messages + [{"role": "assistant", "content": raw[:2000]},
                                            {"role": "user", "content": f"That was not valid JSON ({str(e)[:80]}). Reply with the JSON object only."}],
                                max_tokens, temperature, seed, json_mode=True, stage=stage)
        return {"_error": "unreachable"}

    # ------------------------------------------------------------------ cached stage entry point
    def call(self, stage: str, messages: list, max_tokens: int = 600, seed: int = 0, as_json: bool = True,
             temperature: float = 0.0, tag: str = "") -> dict:
        """Cached call. Returns the parsed JSON dict (``as_json``) or ``{"text": ...}``."""
        key = request_key(self.model_id, messages, max_tokens, seed, as_json, temperature)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                with self._stats_lock:
                    self.stats["cache_hits"] += 1
                return hit
        t0 = time.time()
        n_calls = 1
        self._local.failed = False
        if as_json:
            out = self.chat_json(messages, max_tokens, temperature, seed, stage=stage)
            if "_error" in out and self.reasoning and not out.get("_raw") and max_tokens < 2400:
                n_calls += 1
                out = self.chat_json(messages, min(4000, max(2400, int(max_tokens * 2.5))), temperature, seed, stage=stage)
        else:
            out = {"text": self.chat(messages, max_tokens, temperature, seed, stage=stage)}
        if getattr(self._local, "failed", False):
            # a transport failure is not an answer: never cache it, so a rerun asks again
            with self._stats_lock:
                self.stats["uncached_failures"] += 1
            out = dict(out)
            out.setdefault("_error", "request failed")
            out["_transport_error"] = self._last_error or "request failed"
            return out
        if self.cache is not None:
            self.cache.put(key, out, model=self.model_id, stage=stage, tag=tag, n_calls=n_calls, secs=round(time.time() - t0, 2))
        return out

    def cache_key(self, messages: list, max_tokens: int, seed: int = 0, as_json: bool = True, temperature: float = 0.0) -> str:
        return request_key(self.model_id, messages, max_tokens, seed, as_json, temperature)

    def __repr__(self) -> str:
        return f"LLM({self.name!r}, {self.model_id!r}, concurrency={self.max_concurrency}, reasoning={self.reasoning})"


def run_parallel(jobs: Sequence[Any], fn: Callable[[Any], Any], llm_of: Callable[[Any], LLM],
                 workers_per_llm: int | None = None) -> list:
    """Run ``fn(job)`` for every job, with a thread pool per target LLM. Results keep job order."""
    pools: dict[int, ThreadPoolExecutor] = {}
    futs = []
    for jb in jobs:
        m = llm_of(jb)
        p = pools.get(id(m))
        if p is None:
            p = pools[id(m)] = ThreadPoolExecutor(workers_per_llm or max(1, m.max_concurrency * 2))
        futs.append(p.submit(fn, jb))
    try:
        return [f.result() for f in futs]
    finally:
        for p in pools.values():
            p.shutdown()
