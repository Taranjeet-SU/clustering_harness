"""Contract of the LLM layer: JSON repair, caching, reasoning retry, offline replay, budgets, parallelism and the
request formation of every provider backend (no network: transports and SDK clients are monkeypatched)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import KIND_MARKERS, FakeLLM, detect_kind  # noqa: E402

from clustering_harness.llm import (LLM, AnthropicBackend, BackendError, BudgetExceeded, CallableBackend, ChatRequest,  # noqa: E402
                                    ChatResult, Ledger, OfflineBackend, OfflineCallError, OllamaBackend,
                                    OpenAICompatibleBackend, Price, ResponseCache, price_for, request_key, run_parallel)
from clustering_harness.prompts import PromptSet  # noqa: E402

MSG = [{"role": "system", "content": "Reply with JSON only."}, {"role": "user", "content": "Classify the invoice ticket."}]


class Script:
    """Backend function answering from a list (or a function of the request); records requests."""

    def __init__(self, answers):
        self.answers = answers
        self.requests: list[ChatRequest] = []

    def __call__(self, req):
        self.requests.append(req)
        if callable(self.answers):
            return self.answers(req)
        return self.answers[min(len(self.requests) - 1, len(self.answers) - 1)]


def make(answers, **kw):
    s = Script(answers)
    return LLM(CallableBackend(s, kw.pop("model_id", "fake:test")), **kw), s


# ------------------------------------------------------------------------------------------- chat / chat_json
def test_chat_json_parses_first_brace_span():
    llm, s = make(['Sure! {"score": 4, "name": "Invoice totals wrong"} hope that helps'])
    assert llm.chat_json(MSG) == {"score": 4, "name": "Invoice totals wrong"}
    assert len(s.requests) == 1 and s.requests[0].json_mode is True


def test_chat_json_repairs_once_with_bad_answer_and_correction():
    llm, s = make(["not json at all", '{"ok": true}'])
    assert llm.chat_json(MSG, max_tokens=321, seed=5) == {"ok": True}
    assert len(s.requests) == 2
    rep = s.requests[1]
    assert rep.messages[:2] == MSG
    assert rep.messages[2] == {"role": "assistant", "content": "not json at all"}
    assert rep.messages[3]["role"] == "user" and "Reply with the JSON object only" in rep.messages[3]["content"]
    assert (rep.max_tokens, rep.seed, rep.json_mode) == (321, 5, True)


def test_chat_json_returns_error_dict_after_failed_repair():
    llm, s = make(["nope", "x" * 500])
    out = llm.chat_json(MSG)
    assert set(out) == {"_error", "_raw"}
    assert out["_error"].startswith("unparseable after retry")
    assert out["_raw"] == "x" * 300          # raw of the repair answer, cut to 300
    assert len(s.requests) == 2              # exactly one repair, never more


def test_chat_returns_empty_after_transport_failures_and_counts_them():
    def boom(req):
        raise BackendError("down")
    llm, s = make(boom)
    assert llm.chat(MSG) == ""
    assert llm.stats["failures"] == 2 and llm.stats["requests"] == 0
    out = llm.chat_json(MSG)
    assert "_error" in out and out["_raw"] == ""


def test_offline_backend_raises_on_miss_and_is_not_swallowed(tmp_path):
    off = OfflineBackend("fake:offline")
    llm = LLM(off, cache=ResponseCache(tmp_path / "c.sqlite"))
    with pytest.raises(OfflineCallError):
        llm.call("st", MSG)
    assert len(off.misses) == 1
    # pre-filled cache replays without touching the backend
    llm.cache.put(llm.cache_key(MSG, 600), {"x": 1})
    assert llm.call("st", MSG) == {"x": 1}
    assert len(off.misses) == 1


# ------------------------------------------------------------------------------------------- caching
def test_cache_hit_returns_identical_dict_and_skips_backend(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    llm, s = make(['{"a": [1, 2], "b": "export"}'], cache=cache)
    o1 = llm.call("stage1", MSG, max_tokens=100, seed=3)
    o2 = llm.call("stage1", MSG, max_tokens=100, seed=3)
    assert o1 == o2 == {"a": [1, 2], "b": "export"}
    assert len(s.requests) == 1 and llm.stats["cache_hits"] == 1
    # a new LLM over the same cache file (resumed run) also hits
    llm2, s2 = make(["{}"], cache=ResponseCache(tmp_path / "c.sqlite"))
    assert llm2.call("stage1", MSG, max_tokens=100, seed=3) == o1 and not s2.requests


def test_cache_key_covers_every_request_field():
    base = dict(model_id="m", messages=MSG, max_tokens=100, seed=0, as_json=True, temperature=0.0)
    k0 = request_key(**base)
    variants = [dict(model_id="m2"), dict(messages=MSG + [{"role": "user", "content": "more"}]), dict(max_tokens=101),
                dict(seed=1), dict(as_json=False), dict(temperature=0.5)]
    keys = {request_key(**{**base, **v}) for v in variants}
    assert k0 not in keys and len(keys) == len(variants)
    assert request_key(**base) == k0          # deterministic


def test_contract_failures_are_cached_too(tmp_path):
    llm, s = make(["garbage"], cache=ResponseCache(tmp_path / "c.sqlite"))
    o1 = llm.call("st", MSG)
    o2 = llm.call("st", MSG)
    assert "_error" in o1 and o1 == o2 and len(s.requests) == 2      # 1 call + 1 repair, then cached
    assert llm.cache.error_count() == 1


def test_text_mode_call_returns_text_dict():
    llm, s = make(["plain words"])
    assert llm.call("st", MSG, as_json=False) == {"text": "plain words"}
    assert s.requests[0].json_mode is False


# ------------------------------------------------------------------------------------------- reasoning retry
@pytest.mark.parametrize("max_tokens,expected", [(600, 2400), (900, 2400), (1200, 3000), (1500, 3750), (2000, 4000), (2399, 4000)])
def test_reasoning_empty_answer_retry_budget(max_tokens, expected):
    # rule: min(4000, max(2400, 2.5 * max_tokens)), only when the answer was empty and max_tokens < 2400
    llm, s = make(lambda r: '{"ok": 1}' if r.max_tokens >= 2400 else "", reasoning=True)
    assert llm.call("st", MSG, max_tokens=max_tokens) == {"ok": 1}
    assert [r.max_tokens for r in s.requests] == [max_tokens, max_tokens, expected]
    assert expected == min(4000, max(2400, int(max_tokens * 2.5)))


def test_reasoning_retry_recorded_as_two_calls(tmp_path):
    llm, s = make(lambda r: '{"ok": 1}' if r.max_tokens >= 2400 else "", reasoning=True, cache=ResponseCache(tmp_path / "c.sqlite"))
    llm.call("gate_x", MSG, max_tokens=900)
    assert llm.cache.stage_calls("gate_x") == 2


@pytest.mark.parametrize("case", ["not_reasoning", "nonempty_raw", "budget_already_large"])
def test_reasoning_retry_not_applied(case):
    kw = {"reasoning": case != "not_reasoning"}
    answer = "thinking out loud, no json" if case == "nonempty_raw" else ""
    mt = 2400 if case == "budget_already_large" else 900
    llm, s = make(lambda r: answer, **kw)
    out = llm.call("st", MSG, max_tokens=mt)
    assert "_error" in out
    assert [r.max_tokens for r in s.requests] == [mt, mt]       # call + repair only


# ------------------------------------------------------------------------------------------- budget
def test_budget_cost_from_price_table_and_cap(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    # 1M input tokens at $1 + 0.5M output at $2 = $2 per call
    s = Script(lambda r: ChatResult('{"ok": 1}', input_tokens=1_000_000, output_tokens=500_000))
    llm = LLM(CallableBackend(s, "fake:priced"), budget_usd=5.0, ledger=ledger, price=Price(1.0, 2.0))
    for _ in range(3):                        # spend 0 -> 2 -> 4 -> 6 (the cap is checked before each call)
        llm.chat_json(MSG)
    assert ledger.spent("fake:priced") == pytest.approx(6.0)
    with pytest.raises(BudgetExceeded):
        llm.chat_json(MSG)
    assert len(s.requests) == 3
    summ = ledger.summary()[0]
    assert summ["calls"] == 3 and summ["input_tokens"] == 3_000_000 and summ["output_tokens"] == 1_500_000


def test_budget_exact_limit_blocks(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite")
    s = Script(lambda r: ChatResult('{"ok": 1}', input_tokens=1_000_000, output_tokens=0))
    llm = LLM(CallableBackend(s, "fake:x"), budget_usd=1.0, ledger=ledger, price=Price(1.0, 0.0))
    llm.chat_json(MSG)                        # spend reaches exactly the limit
    with pytest.raises(BudgetExceeded):
        llm.chat_json(MSG)


def test_default_price_looked_up_from_model_name(tmp_path):
    class B:
        model_id = "openai:gpt-4o-mini"
        model = "gpt-4o-mini"

        def complete(self, req):
            return ChatResult('{"x": 1}', input_tokens=2_000_000, output_tokens=1_000_000)
    ledger = Ledger(tmp_path / "l.sqlite")
    llm = LLM(B(), ledger=ledger, budget_usd=100)
    llm.chat_json(MSG)
    p = price_for("gpt-4o-mini")
    assert ledger.spent("openai:gpt-4o-mini") == pytest.approx(2 * p.input_per_mtok + 1 * p.output_per_mtok)
    assert price_for("some-local-model") == Price(0.0, 0.0)


# ------------------------------------------------------------------------------------------- run_parallel
def test_run_parallel_preserves_job_order_across_models():
    a, _ = make(["{}"], max_concurrency=2)
    b, _ = make(["{}"], max_concurrency=1)
    seen = []
    lock = threading.Lock()

    def fn(j):
        time.sleep(0.002 * (20 - j))           # later jobs finish first
        with lock:
            seen.append(j)
        return j * 10
    out = run_parallel(list(range(20)), fn, lambda j: a if j % 2 else b)
    assert out == [j * 10 for j in range(20)]
    assert sorted(seen) == list(range(20))


# ------------------------------------------------------------------------------------------- backends
class FakeResponse:
    def __init__(self, j):
        self._j = j

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


class FakeSession:
    def __init__(self, reply):
        self.reply = reply
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json, timeout))
        if isinstance(self.reply, Exception):
            raise self.reply
        return FakeResponse(self.reply)


def test_ollama_request_body(monkeypatch):
    be = OllamaBackend("local-model", url="http://host:1/", num_ctx=4096, think="low", timeout=7, keep_alive="5m")
    sess = FakeSession({"message": {"content": '{"ok": 1}'}, "prompt_eval_count": 11, "eval_count": 3})
    monkeypatch.setattr(be, "_http", lambda: sess)
    res = be.complete(ChatRequest(messages=MSG, max_tokens=123, temperature=0.0, seed=9, json_mode=True))
    url, body, timeout = sess.posts[0]
    assert url == "http://host:1/api/chat" and timeout == 7
    assert body["model"] == "local-model" and body["messages"] == MSG and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_predict": 123, "num_ctx": 4096, "seed": 9}
    assert body["format"] == "json" and body["think"] == "low" and body["keep_alive"] == "5m"
    assert (res.text, res.input_tokens, res.output_tokens) == ('{"ok": 1}', 11, 3)
    assert be.model_id == "ollama:local-model"


def test_ollama_omits_optional_fields(monkeypatch):
    be = OllamaBackend("m")
    sess = FakeSession({"message": {"content": "hi"}})
    monkeypatch.setattr(be, "_http", lambda: sess)
    be.complete(ChatRequest(messages=MSG, json_mode=False))
    body = sess.posts[0][1]
    assert "format" not in body and "think" not in body and "keep_alive" not in body
    assert body["options"]["num_ctx"] == 8192


def test_ollama_think_false_is_sent(monkeypatch):
    be = OllamaBackend("m", think=False)
    sess = FakeSession({"message": {"content": "hi"}})
    monkeypatch.setattr(be, "_http", lambda: sess)
    be.complete(ChatRequest(messages=MSG))
    assert sess.posts[0][1]["think"] is False


def test_ollama_transport_error_becomes_backend_error(monkeypatch):
    be = OllamaBackend("m")
    monkeypatch.setattr(be, "_http", lambda: FakeSession(ConnectionError("refused")))
    with pytest.raises(BackendError):
        be.complete(ChatRequest(messages=MSG))


class FakeOpenAIClient:
    def __init__(self, content="{}"):
        self.kw = []
        outer = self

        class Completions:
            def create(self, **kw):
                outer.kw.append(kw)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                                       usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
        self.chat = SimpleNamespace(completions=Completions())


def test_openai_compatible_request():
    be = OpenAICompatibleBackend("gpt-test", api_key="k")
    be._client = FakeOpenAIClient('{"a": 1}')
    res = be.complete(ChatRequest(messages=MSG, max_tokens=77, temperature=0.0, seed=4, json_mode=True))
    kw = be._client.kw[0]
    assert kw == {"model": "gpt-test", "messages": MSG, "temperature": 0.0, "max_completion_tokens": 77, "seed": 4,
                  "response_format": {"type": "json_object"}}
    assert (res.text, res.input_tokens, res.output_tokens) == ('{"a": 1}', 5, 2)
    assert be.model_id == "openai:gpt-test"


def test_openai_compatible_options():
    be = OpenAICompatibleBackend("m", base_url="http://local/v1", api_key="k", supports_json_mode=False, supports_seed=False,
                                 token_param="max_tokens")
    be._client = FakeOpenAIClient()
    be.complete(ChatRequest(messages=MSG, max_tokens=50, json_mode=True))
    kw = be._client.kw[0]
    assert "response_format" not in kw and "seed" not in kw and kw["max_tokens"] == 50 and "max_completion_tokens" not in kw
    assert be.model_id == "openai-compatible:http://local/v1:m"


def test_openai_error_becomes_backend_error():
    be = OpenAICompatibleBackend("m", api_key="k")

    class Bad:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("429"))))
    be._client = Bad()
    with pytest.raises(BackendError):
        be.complete(ChatRequest(messages=MSG))


class FakeAnthropicClient:
    def __init__(self, text='{"ok": 1}', stop="end_turn"):
        self.kw = []
        outer = self

        class Messages:
            def create(self, **kw):
                outer.kw.append(kw)
                return SimpleNamespace(content=[SimpleNamespace(type="thinking", text="hidden"), SimpleNamespace(type="text", text=text)],
                                       stop_reason=stop, usage=SimpleNamespace(input_tokens=8, output_tokens=4))
        self.messages = Messages()


def test_anthropic_request_formation():
    pytest.importorskip("anthropic")
    be = AnthropicBackend("claude-test", api_key="k")
    be._client = FakeAnthropicClient()
    msgs = [{"role": "system", "content": "S1"}, {"role": "system", "content": "S2"}, {"role": "user", "content": "U"}]
    res = be.complete(ChatRequest(messages=msgs, max_tokens=900, temperature=0.0, seed=3, json_mode=True))
    kw = be._client.kw[0]
    assert kw["system"] == "S1\n\nS2"
    assert kw["messages"] == [{"role": "user", "content": "U"}]
    assert "temperature" not in kw and "seed" not in kw and "response_format" not in kw
    assert kw["thinking"] == {"type": "disabled"} and kw["output_config"] == {"effort": "low"}
    assert kw["max_tokens"] == 900 and kw["model"] == "claude-test"
    assert res.text == '{"ok": 1}' and (res.input_tokens, res.output_tokens) == (8, 4)


@pytest.mark.parametrize("asked,sent", [(10, 256), (300, 300), (99999, 4000)])
def test_anthropic_max_tokens_clamped(asked, sent):
    pytest.importorskip("anthropic")
    be = AnthropicBackend("claude-test", api_key="k")
    be._client = FakeAnthropicClient()
    be.complete(ChatRequest(messages=MSG, max_tokens=asked))
    assert be._client.kw[0]["max_tokens"] == sent


def test_anthropic_thinking_effort_and_refusal():
    pytest.importorskip("anthropic")
    be = AnthropicBackend("claude-test", api_key="k", thinking={"type": "adaptive"}, effort=None)
    be._client = FakeAnthropicClient(stop="refusal")
    res = be.complete(ChatRequest(messages=[{"role": "user", "content": "U"}]))
    kw = be._client.kw[0]
    assert kw["thinking"] == {"type": "adaptive"} and "output_config" not in kw and "system" not in kw
    assert res.text == "" and res.raw == {"stop_reason": "refusal"}


def test_anthropic_api_error_becomes_backend_error():
    anthropic = pytest.importorskip("anthropic")
    import httpx
    be = AnthropicBackend("claude-test", api_key="k")

    class Bad:
        class messages:
            @staticmethod
            def create(**kw):
                raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://invalid/v1/messages"))
    be._client = Bad()
    with pytest.raises(BackendError):
        be.complete(ChatRequest(messages=MSG))


# ------------------------------------------------------------------------------------------- the fake itself
def test_fake_prompt_markers_exist_in_default_preset():
    """The FakeLLM recognises prompts by template text; make sure the markers are really in the templates."""
    P = PromptSet.load("default")
    tpl = {"gate": "gate_user", "verify": "verify_user", "label": "label_user", "merge": "merge_user", "classify": "classify_user",
           "canonicalise": "canon_user", "judge": "judge_user", "rollup_decide": "rollup_decide_user", "rollup_name": "rollup_name_user"}
    for kind, marker in KIND_MARKERS:
        assert marker in P[tpl[kind]], kind
        others = [k for k, t in tpl.items() if k != kind and marker in P[t]]
        assert not others, (kind, others)
    assert detect_kind("", P.render("canon_user", labels="1. invoice")) == "canonicalise"


def test_fake_llm_rules_and_callback():
    f = FakeLLM("a", rules={"gate": {"score": 2, "name": "x"}}, callback=lambda i: '{"cb": 1}' if i.kind == "merge" else None)
    llm = LLM(CallableBackend(f, "fake:a"))
    P = PromptSet.load("default")
    gate_msgs = [{"role": "user", "content": P.render("gate_user", n=1, items="1. Title: invoice mismatch case 1")}]
    assert llm.chat_json(gate_msgs) == {"score": 2, "name": "x"}
    merge_msgs = [{"role": "user", "content": P.render("merge_user", name_a="A", examples_a="", name_b="B", examples_b="")}]
    assert llm.chat_json(merge_msgs) == {"cb": 1}
    assert f.kinds() == {"gate": 1, "merge": 1}
