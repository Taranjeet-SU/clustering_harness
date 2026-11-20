"""Model failures must be loud, must never be cached, and must stop a run when the model is unusable;
budgets must hold under parallel requests."""
from __future__ import annotations

import logging
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from clustering_harness.algorithms.gate import Gate
from clustering_harness.llm import (LLM, BackendConfigError, BackendError, BudgetExceeded, CallableBackend, ChatResult, Ledger,
                                    ModelUnavailableError, Price, ResponseCache)

MSG = [{"role": "user", "content": "Return JSON."}]


class Flaky:
    """Fails with ``exc`` until ``healthy`` is set, then answers a JSON object."""

    def __init__(self, exc=BackendError("connection reset")):
        self.exc, self.healthy, self.calls = exc, False, 0

    def __call__(self, req):
        self.calls += 1
        if not self.healthy:
            raise self.exc
        return ChatResult(text='{"score": 4}', input_tokens=10, output_tokens=5)


def test_transport_failure_is_logged_not_cached_and_retried_on_rerun(tmp_path, caplog):
    fl = Flaky()
    cache = ResponseCache(tmp_path / "c.sqlite")
    llm = LLM(CallableBackend(fl, "flaky:1"), name="a", cache=cache)
    with caplog.at_level(logging.WARNING, logger="clustering_harness"):
        out = llm.call("gate", MSG)
    assert out.get("_transport_error") and "_error" in out
    assert "request failed" in caplog.text and "connection reset" in caplog.text
    assert cache.stage_counts() == {}                              # nothing cached
    fl.healthy = True
    assert llm.call("gate", MSG) == {"score": 4}                   # the rerun asks the model again
    assert llm.call("gate", MSG) == {"score": 4} and llm.stats["cache_hits"] == 1


def test_config_errors_stop_at_once(tmp_path):
    fl = Flaky(BackendConfigError("401 invalid api key"))
    llm = LLM(CallableBackend(fl, "bad:key"), name="a", cache=ResponseCache(tmp_path / "c.sqlite"))
    with pytest.raises(ModelUnavailableError, match="invalid api key"):
        llm.call("gate", MSG)
    assert fl.calls == 1                                           # no retries for a config error


def test_consecutive_failures_stop_the_run(tmp_path):
    fl = Flaky()
    llm = LLM(CallableBackend(fl, "down:1"), name="a", cache=ResponseCache(tmp_path / "c.sqlite"), max_consecutive_failures=3)
    llm.call("s", MSG)
    with pytest.raises(ModelUnavailableError, match="in a row failed; last error: connection reset"):
        for _ in range(5):
            llm.call("s", [{"role": "user", "content": f"q{_}"}])


def test_one_success_resets_the_failure_streak(tmp_path):
    state = {"n": 0}

    def sometimes(req):
        state["n"] += 1
        if state["n"] % 2:
            raise BackendError("flaky")
        return ChatResult(text='{"ok": 1}')

    llm = LLM(CallableBackend(sometimes, "s:1"), name="a", transport_retries=1, max_consecutive_failures=2)
    for i in range(10):
        llm.chat_json([{"role": "user", "content": str(i)}])     # never two failures in a row


def test_budget_holds_under_parallel_requests(tmp_path):
    ledger = Ledger(tmp_path / "spend.sqlite")

    def answer(req):
        return ChatResult(text="{}", input_tokens=1000, output_tokens=100)

    price = Price(1000.0, 1000.0)       # $1 per 1k input tokens, $0.10 per 100 output: ~$1.1 per request
    llm = LLM(CallableBackend(answer, "priced:1"), name="a", max_concurrency=8, budget_usd=5.0, ledger=ledger, price=price)

    def one(i):
        try:
            llm.chat([{"role": "user", "content": "x" * 3000}], max_tokens=100)
            return 1
        except BudgetExceeded:
            return 0

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, range(40)))
    assert ledger.spent("priced:1") <= 5.0 + 1e-9


def _calls(scores):
    return [{"cid": f"c{i}", "rep": 0, "judge": j, "score": s, "ok": True, "name": "", "reason": "", "size": 10}
            for i, (sa, sb) in enumerate(scores) for j, s in (("a", sa), ("b", sb))]


def test_constant_judge_is_flagged(caplog):
    stub = types.SimpleNamespace(ctx=types.SimpleNamespace(load=lambda *a: None, save=lambda *a: None), calib_path=("x",))
    with caplog.at_level(logging.WARNING, logger="clustering_harness"):
        b = Gate.calibrate(stub, _calls([(4, 3), (4, 5), (4, 2), (4, 4), (4, 5), (4, 1)]))
    assert b["a"].get("constant") and not b["b"].get("constant")
    assert "judge 'a' gave every calibration group the same score" in caplog.text
    with caplog.at_level(logging.WARNING, logger="clustering_harness"):
        Gate.calibrate(stub, _calls([(4, 4)] * 6))
    assert "BOTH judges gave constant scores" in caplog.text


def test_keys_echoed_by_providers_are_redacted(tmp_path, caplog):
    fl = Flaky(BackendConfigError("401 Incorrect API key provided: sk-proj-abcDEF123456XYZ"))
    llm = LLM(CallableBackend(fl, "bad:key"), name="a")
    with caplog.at_level(logging.ERROR, logger="clustering_harness"), pytest.raises(ModelUnavailableError) as e:
        llm.chat(MSG)
    assert "sk-proj" not in str(e.value) and "sk-proj" not in caplog.text and "[redacted-key]" in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__        # the raw message is not chained either
