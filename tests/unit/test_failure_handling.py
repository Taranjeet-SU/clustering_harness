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


def _gate_stub(allow):
    llm = types.SimpleNamespace(model_id="stub:judge")
    ctx = types.SimpleNamespace(load=lambda *a: None, save=lambda *a: None, llm=lambda j: llm)
    return types.SimpleNamespace(ctx=ctx, calib_path=("x",), cfg=types.SimpleNamespace(allow_constant_judge=allow))


def test_constant_judge_stops_the_run():
    with pytest.raises(RuntimeError, match="judge 'a' .* gave all 6 calibration answers the same score"):
        Gate.calibrate(_gate_stub(False), _calls([(4, 3), (4, 5), (4, 2), (4, 4), (4, 5), (4, 1)]))


def test_constant_judge_allowed_when_configured(caplog):
    with caplog.at_level(logging.WARNING, logger="clustering_harness"):
        b = Gate.calibrate(_gate_stub(True), _calls([(4, 3), (4, 5), (4, 2), (4, 4), (4, 5), (4, 1)]))
    assert b["a"].get("constant") and not b["b"].get("constant")
    assert "allow_constant_judge is set" in caplog.text


def test_keys_echoed_by_providers_are_redacted(tmp_path, caplog):
    fl = Flaky(BackendConfigError("401 Incorrect API key provided: sk-proj-abcDEF123456XYZ"))
    llm = LLM(CallableBackend(fl, "bad:key"), name="a")
    with caplog.at_level(logging.ERROR, logger="clustering_harness"), pytest.raises(ModelUnavailableError) as e:
        llm.chat(MSG)
    assert "sk-proj" not in str(e.value) and "sk-proj" not in caplog.text and "[redacted-key]" in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__        # the raw message is not chained either


def test_openai_status_codes_are_classified_without_the_sdk():
    from types import SimpleNamespace

    from clustering_harness.llm import ChatRequest, OpenAICompatibleBackend

    class StatusError(Exception):
        def __init__(self, code):
            super().__init__(f"Error code: {code}")
            self.status_code = code

    def backend(code):
        be = OpenAICompatibleBackend("gpt-4o-mini", api_key="k")
        be._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: (_ for _ in ()).throw(StatusError(code)))))
        return be

    req = ChatRequest(messages=MSG)
    for code in (401, 403, 404):
        with pytest.raises(BackendConfigError):
            backend(code).complete(req)
    for code in (429, 500):
        with pytest.raises(BackendError) as e:
            backend(code).complete(req)
        assert not isinstance(e.value, BackendConfigError)


def test_unusable_model_refuses_later_requests_without_sending_them(tmp_path):
    fl = Flaky(BackendConfigError("401 invalid api key"))
    llm = LLM(CallableBackend(fl, "bad:key"), name="a")
    with pytest.raises(ModelUnavailableError):
        llm.call("s", MSG)
    for i in range(5):                                             # queued jobs stop at once
        with pytest.raises(ModelUnavailableError, match="invalid api key"):
            llm.call("s", [{"role": "user", "content": f"q{i}"}])
    assert fl.calls == 1


def test_run_parallel_drops_queued_jobs_after_a_fatal_error():
    from clustering_harness.llm.client import run_parallel
    fl = Flaky(BackendConfigError("403 forbidden"))
    llm = LLM(CallableBackend(fl, "bad:key"), name="a", max_concurrency=1)
    with pytest.raises(ModelUnavailableError):
        run_parallel(list(range(50)), lambda i: llm.call("s", [{"role": "user", "content": str(i)}]), lambda i: llm)
    assert fl.calls == 1


def test_zero_token_backend_is_still_charged(tmp_path, caplog):
    led = Ledger(tmp_path / "l.sqlite")
    llm = LLM(CallableBackend(lambda r: '{"ok": 1}', "custom:x"), name="a", ledger=led, budget_usd=0.01,
              price=Price(1000.0, 0.0))                          # input-only price keeps the estimates small
    with caplog.at_level(logging.WARNING, logger="clustering_harness"):
        llm.call("s", MSG)
    assert led.spent("custom:x") > 0 and "reports no token usage" in caplog.text
    with pytest.raises(BudgetExceeded):
        for i in range(200):
            llm.call("s", [{"role": "user", "content": f"q{i}"}])
    assert led.spent("custom:x") <= 0.01 + 1e-9


def test_settings_are_part_of_the_cache_key(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    calls = []
    be = CallableBackend(lambda r: calls.append(1) or '{"v": 1}', "m:1")
    LLM(be, cache=cache).call("s", MSG)
    LLM(be, cache=cache, reasoning=True).call("s", MSG)          # a different setting: asks again
    assert len(calls) == 2
    LLM(be, cache=cache, reasoning=True, legacy_cache_keys=True).call("s", MSG)   # legacy keys match the first entry
    assert len(calls) == 2


def test_backend_settings_change_the_key():
    from clustering_harness.llm import AnthropicBackend, OllamaBackend
    k = lambda be: LLM(be).cache_key(MSG, 100)                    # noqa: E731
    assert k(OllamaBackend("q")) == k(OllamaBackend("q", think=None))
    assert k(OllamaBackend("q")) != k(OllamaBackend("q", think="low"))
    assert k(AnthropicBackend("h")) != k(AnthropicBackend("h", effort="high"))


def test_budget_without_a_ledger_is_still_enforced():
    llm = LLM(CallableBackend(lambda r: ChatResult('{"ok": 1}', 1000, 1000), "custom:x"), name="a", budget_usd=0.001,
              price=Price(100.0, 100.0))
    with pytest.raises(BudgetExceeded):
        for i in range(50):
            llm.call("s", [{"role": "user", "content": f"q{i}"}])


def test_budget_with_unknown_price_is_refused():
    with pytest.raises(ValueError, match="price of custom:new-model is unknown"):
        LLM(CallableBackend(lambda r: "{}", "custom:new-model"), name="a", budget_usd=1.0)
    LLM(CallableBackend(lambda r: "{}", "custom:new-model"), name="a", budget_usd=1.0, price=Price(0, 0))   # explicit: ok


def test_budget_scope_shares_one_cap(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    mk = lambda mid: LLM(CallableBackend(lambda r: ChatResult('{"ok": 1}', 1000, 1000), mid), ledger=led,   # noqa: E731
                         budget_usd=0.01, price=Price(1000.0, 1000.0), budget_scope="gpt-4o-mini")
    a, b = mk("openai:gpt-4o-mini"), mk("azure-openai:x:gpt-4o-mini")
    with pytest.raises(BudgetExceeded):
        for i in range(10):
            (a if i % 2 else b).call("s", [{"role": "user", "content": f"q{i}"}])
    assert led.spent("gpt-4o-mini") <= 0.01 + 1e-9


def test_rejected_request_is_not_retried_cached_or_blocking(tmp_path):
    from clustering_harness.llm import RequestRejectedError
    calls = []

    def fn(req):
        calls.append(1)
        if "too long" in req.messages[-1]["content"]:
            raise RequestRejectedError("400 context length exceeded")
        return '{"ok": 1}'

    cache = ResponseCache(tmp_path / "c.sqlite")
    llm = LLM(CallableBackend(fn, "m:1"), cache=cache)
    out = llm.call("s", [{"role": "user", "content": "too long"}])
    assert out.get("_rejected") and "_transport_error" not in out
    assert len(calls) == 1                                         # no retry, no JSON repair request
    assert llm.stats["rejected"] == 1 and llm.stats["uncached_failures"] == 0
    assert cache.stage_counts() == {}
    assert llm.call("s", MSG) == {"ok": 1}                         # other requests carry on


def test_many_rejections_in_a_row_stop_the_model():
    from clustering_harness.llm import RequestRejectedError

    def fn(req):
        raise RequestRejectedError("400 unsupported parameter")

    llm = LLM(CallableBackend(fn, "m:1"), max_consecutive_failures=3)
    with pytest.raises(ModelUnavailableError):
        for i in range(5):
            llm.call("s", [{"role": "user", "content": f"q{i}"}])


def test_no_json_repair_request_after_a_transport_failure():
    fl = Flaky()
    llm = LLM(CallableBackend(fl, "down:1"), transport_retries=2)
    llm.call("s", MSG)
    assert fl.calls == 2                                           # two attempts, no third "fix your JSON" request


def test_backends_that_retry_internally_get_one_attempt():
    fl = Flaky()
    be = CallableBackend(fl, "sdk:1")
    be.retries_internally = True
    LLM(be).call("s", MSG)
    assert fl.calls == 1


def test_callable_backends_need_a_price_for_a_budget():
    with pytest.raises(ValueError, match="price of callable is unknown"):
        LLM(CallableBackend(lambda r: "{}"), budget_usd=1.0)


def test_unicode_text_is_not_junk():
    from clustering_harness.data import JunkRule
    j = JunkRule()
    assert not j.is_junk("Не могу войти в аккаунт", "")
    assert not j.is_junk("无法登录我的账户", "")
    assert j.is_junk("test", "") and j.is_junk("hi", "")


def test_custom_backend_exceptions_are_retried_not_fatal():
    calls = []

    def fn(req):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("HTTP 500")
        return '{"ok": 1}'

    llm = LLM(CallableBackend(fn, "custom:1"))
    assert llm.call("s", MSG) == {"ok": 1} and len(calls) == 2


def test_preflight_rejects_non_json_answers():
    llm = LLM(CallableBackend(lambda r: "Sure! Happy to help.", "chatty:1"), name="a")
    with pytest.raises(ModelUnavailableError, match="not JSON"):
        llm.preflight()


def test_prices_match_only_real_variants():
    from clustering_harness.llm import price_for
    assert price_for("gpt-4o-mini-2024-07-18") == price_for("gpt-4o-mini")
    assert price_for("us.anthropic.claude-haiku-5-5-v1:0") == price_for("claude-haiku-5-5")
    assert price_for("gpt-4o-mni") == Price(0, 0)                  # a typo is not gpt-4o
    assert price_for("ft:gpt-4o-mini:org::x") == Price(0, 0)       # fine-tunes need an explicit price


def test_indic_and_thai_text_is_not_junk():
    from clustering_harness.data import JunkRule
    j = JunkRule()
    assert not j.is_junk("मेरा कार्ड खो गया", "")
    assert not j.is_junk("บัตรของฉันหาย", "")
