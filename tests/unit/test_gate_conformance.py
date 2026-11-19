"""Doc step 5: the two-judge calibrated gate (calibration, decision bands, escalation, naming, frozen reuse)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import fake_llms, make_ctx, rid_of  # noqa: E402

from clustering_harness.algorithms.gate import Gate  # noqa: E402

IDENT = {"a": {"mu": 0.0, "sd": 1.0}, "b": {"mu": 0.0, "sd": 1.0}, "pooled_mu": 0.0, "pooled_sd": 1.0}   # adj == raw score


def call(judge, score, ok=True, name="", rep=0):
    return {"cid": "X", "rep": rep, "judge": judge, "score": score, "ok": ok, "name": name, "reason": "r", "size": 10}


@pytest.fixture
def ctx(tmp_path):
    return make_ctx(tmp_path, n_topics=3, per_topic=20)


# ------------------------------------------------------------------------------------------- calibration
def test_calibration_formula_population_sd_and_pooled(ctx):
    G = Gate(ctx)
    calls = [call("a", s) for s in (5, 4, 3)] + [call("b", s) for s in (2, 4, 4)] + [call("a", None, ok=False), call("b", None, ok=False)]
    b = G.calibrate(calls, force=True)
    assert b["a"] == {"mu": pytest.approx(4.0), "sd": pytest.approx(np.sqrt(2 / 3)), "n": 3}     # population sd (ddof=0)
    assert b["b"]["mu"] == pytest.approx(10 / 3) and b["b"]["sd"] == pytest.approx(np.std([2, 4, 4]))
    assert b["pooled_mu"] == pytest.approx(np.mean([5, 4, 3, 2, 4, 4]))
    assert b["pooled_sd"] == pytest.approx(np.std([5, 4, 3, 2, 4, 4]))
    # adj = pooled_mu + pooled_sd * (score - judge_mu) / judge_sd
    assert G.adjust(5, "a", b) == pytest.approx(b["pooled_mu"] + b["pooled_sd"] * (5 - 4.0) / np.sqrt(2 / 3))
    assert json.loads((ctx.workdir / "cache" / "gate_calibration.json").read_text())["a"]["n"] == 3


def test_calibration_zero_judge_sd_becomes_one(ctx):
    b = Gate(ctx).calibrate([call("a", 4)] * 3 + [call("b", s) for s in (2, 3, 4)], force=True)
    assert b["a"]["sd"] == 1.0
    assert Gate.adjust(5, "a", b) == pytest.approx(b["pooled_mu"] + b["pooled_sd"] * 1.0)


def test_calibration_requires_both_judges(ctx):
    with pytest.raises(RuntimeError):
        Gate(ctx).calibrate([call("a", 4), call("b", None, ok=False)], force=True)


def test_calibration_is_frozen(ctx):
    G = Gate(ctx)
    b1 = G.calibrate([call("a", 5), call("a", 3), call("b", 4), call("b", 2)], force=True)
    b2 = G.calibrate([call("a", 1), call("b", 1)])        # later batches never recalibrate
    assert b2 == json.loads(json.dumps(b1))


# ------------------------------------------------------------------------------------------- decision bands
@pytest.mark.parametrize("a,b,expected", [
    ([3.6], [3.6], "ACCEPT"),            # overall mean exactly 3.6, both judges >= 3.2
    ([3.2], [4.0], "ACCEPT"),            # judge mean exactly 3.2
    ([3.1], [4.1], "ESCALATE"),          # overall 3.6 but one judge below 3.2
    ([3.5], [3.6], "ESCALATE"),          # overall 3.55 < 3.6
    ([3.0], [3.0], "REJECT"),            # overall exactly 3.0
    ([2.0], [4.0], "REJECT"),            # mean 3.0
    ([3.0], [3.02], "ESCALATE"),         # just above 3.0
    ([5.0], [], "ESCALATE"),             # only one judge valid: cannot ACCEPT (both judge means required)
    ([2.0], [], "REJECT"),
])
def test_decision_bands(ctx, a, b, expected):
    d = Gate(ctx).decide([call("a", s) for s in a] + [call("b", s) for s in b], IDENT)
    assert d["decision"] == expected


def test_decision_uses_adjusted_scores(ctx):
    b = {"a": {"mu": 4.0, "sd": 0.5}, "b": {"mu": 3.0, "sd": 1.0}, "pooled_mu": 3.5, "pooled_sd": 1.0}
    # raw a=4 -> 3.5, raw b=4 -> 4.5; mean 4.0 ACCEPT though raw a == raw b
    d = Gate(ctx).decide([call("a", 4), call("b", 4)], b)
    assert d["per_judge"] == {"a": 3.5, "b": 4.5} and d["adj"] == 4.0 and d["decision"] == "ACCEPT"
    assert d["raw_mean"] == 4.0


def test_contract_fail_when_no_valid_call(ctx):
    d = Gate(ctx).decide([call("a", None, ok=False), call("b", None, ok=False)], IDENT)
    assert d["decision"] == "CONTRACT_FAIL" and d["adj"] is None and d["fails"] == 2


def test_name_is_first_nonempty_valid_name(ctx):
    calls = [call("a", 4, name=""), call("b", None, ok=False, name="Invalid call name"), call("b", 4, name="Invoice mismatch"),
             call("a", 4, name="Later name")]
    assert Gate(ctx).decide(calls, IDENT)["name"] == "Invoice mismatch"


# ------------------------------------------------------------------------------------------- gate_many with scripted judges
def scripted(scores):
    """Gate answers by (slot, replicate seed): scores[slot][rep] (None -> unparseable answer)."""
    def rule(slot):
        def f(info):
            s = scores[slot][info.seed]
            return "unparseable" if s is None else {"score": s, "name": f"{slot}{info.seed}", "reason": "r"}
        return f
    return fake_llms(rules_a={"gate": rule("a")}, rules_b={"gate": rule("b")})


def gate_ctx(tmp_path, scores, calib=IDENT):
    llms, fk = scripted(scores)
    ctx = make_ctx(tmp_path, llms=llms, n_topics=2, per_topic=20)
    if calib is not None:
        ctx.save(calib, "cache", "gate_calibration.json")
    return ctx, fk


def members(ctx, n=20):
    return ctx.eligible_ids[:n]


def test_escalation_fourth_sample_from_both_judges_then_3_4_rule_accept(tmp_path):
    ctx, fk = gate_ctx(tmp_path, {"a": [3, 3, 3, 3], "b": [4, 4, 4, 4]})       # mean 3.5 -> band -> 3.5 >= 3.4
    d = Gate(ctx).gate_many("t", [("X", members(ctx))])["X"]
    for s in "ab":
        assert sorted(i.seed for i in fk[s].of("gate")) == [0, 1, 2, 3]            # 4th sample (replicate 3) from both judges
    assert d["decision"] == "ACCEPT" and d.get("escalated") and d["calls"] == 8 and d["adj"] == 3.5


def test_escalation_3_4_rule_reject(tmp_path):
    ctx, _ = gate_ctx(tmp_path, {"a": [3, 3, 3, 3], "b": [4, 4, 4, 3]})       # 27/8 = 3.375 < 3.4
    d = Gate(ctx).gate_many("t", [("X", members(ctx))])["X"]
    assert d["decision"] == "REJECT" and d.get("escalated") and d["adj"] == 3.375


def test_escalation_reapplies_main_rule_first(tmp_path):
    ctx, _ = gate_ctx(tmp_path, {"a": [3, 3, 3, 5], "b": [4, 4, 4, 5]})       # 3.875, a=3.5, b=4.25 -> ACCEPT by main rule
    d = Gate(ctx).gate_many("t", [("X", members(ctx))])["X"]
    assert d["decision"] == "ACCEPT" and not d.get("escalated")
    ctx2, _ = gate_ctx(tmp_path / "2", {"a": [3, 3, 3, 1], "b": [4, 4, 4, 1]})  # 3.0 -> REJECT by main rule
    assert Gate(ctx2).gate_many("t", [("X", members(ctx2))])["X"]["decision"] == "REJECT"


def test_no_escalation_for_clear_decisions(tmp_path):
    ctx, fk = gate_ctx(tmp_path, {"a": [5, 5, 4, 1], "b": [4, 5, 5, 1]})
    d = Gate(ctx).gate_many("t", [("X", members(ctx))])["X"]
    assert d["decision"] == "ACCEPT" and d["calls"] == 6
    assert 3 not in {i.seed for i in fk["a"].of("gate")}


def test_contract_fail_escalates_then_stays_contract_fail(tmp_path):
    ctx, fk = gate_ctx(tmp_path, {"a": [None] * 4, "b": [None] * 4})
    d = Gate(ctx).gate_many("t", [("X", members(ctx))])["X"]
    assert d["decision"] == "CONTRACT_FAIL" and d["calls"] == 8 and d["fails"] == 8
    assert {i.seed for i in fk["b"].of("gate")} == {0, 1, 2, 3}


def test_contract_fail_rescued_by_fourth_sample(tmp_path):
    ctx, _ = gate_ctx(tmp_path, {"a": [None, None, None, 4], "b": [None, None, None, 4]})
    d = Gate(ctx).gate_many("t", [("X", members(ctx))])["X"]
    assert d["decision"] == "ACCEPT" and d["n"] == 2


def test_gate_name_in_replicate_then_judge_order(tmp_path):
    ctx, _ = gate_ctx(tmp_path, {"a": [5, 5, 5], "b": [5, 5, 5]})
    d = Gate(ctx).gate_many("t", [("X", members(ctx))])["X"]
    assert d["name"] == "a0"


def test_gate_many_requires_frozen_calibration(tmp_path):
    ctx, _ = gate_ctx(tmp_path, {"a": [5] * 4, "b": [5] * 4}, calib=None)
    with pytest.raises(RuntimeError):
        Gate(ctx).gate_many("later", [("X", members(ctx))])


def test_first_batch_freezes_calibration_from_replicates_0_to_2(tmp_path):
    # judge a: 3 on every replicate except 5 on replicate 3 (escalation only); judge b: 4.
    ctx, _ = gate_ctx(tmp_path, {"a": [3, 3, 4, 5], "b": [4, 4, 3, 5]}, calib=None)
    G = Gate(ctx)
    G.gate_many("s1", [("P0", members(ctx)), ("P1", ctx.eligible_ids[20:40])], calibrate_now=True)
    b = ctx.load("cache", "gate_calibration.json")
    assert b["a"]["n"] == 6 and b["b"]["n"] == 6                                    # 2 parents x reps 0..2, no 4th samples
    assert b["a"]["mu"] == pytest.approx(np.mean([3, 3, 4] * 2)) and b["pooled_mu"] == pytest.approx(np.mean([3, 3, 4, 4, 4, 3] * 2))
    frozen = dict(b)
    G.gate_many("later", [("Q", members(ctx))])                                      # reused, unchanged
    assert ctx.load("cache", "gate_calibration.json") == frozen


# ------------------------------------------------------------------------------------------- samples & prompt
def test_gate_sample_size_and_determinism(tmp_path):
    ctx = make_ctx(tmp_path, n_topics=2, per_topic=20)
    ms = ctx.eligible_ids[:30]
    s0 = ctx.gsample("P1", 0, ms)
    assert len(s0) == 8 and s0 == ctx.sort(s0) and set(s0) <= set(ms)
    assert ctx.gsample("P1", 0, list(reversed(ms))) == s0                            # member order does not matter
    assert len({tuple(ctx.gsample("P1", r, ms)) for r in range(4)}) > 1             # independent replicate samples
    assert ctx.gsample("P2", 0, ms) != s0 or ctx.gsample("P2", 1, ms) != ctx.gsample("P1", 1, ms)
    assert ctx.gsample("P1", 0, ms[:8]) == ctx.sort(ms[:8]) and ctx.gsample("P1", 0, ms[:3]) == ctx.sort(ms[:3])


def test_gate_prompt_shows_sample_with_description_head(tmp_path):
    llms, fk = fake_llms()
    ctx = make_ctx(tmp_path, llms=llms, n_topics=2, per_topic=20)
    ctx.save(IDENT, "cache", "gate_calibration.json")
    ms = ctx.eligible_ids[:20]
    Gate(ctx).gate_many("t", [("X", ms)])
    for info in fk["a"].of("gate"):
        shown = [rid_of(t) for t in info.titles()]
        assert shown == ctx.gsample("X", info.seed, ms)                               # replicate r uses seed r
        assert f"Below are {len(shown)} tickets" in info.user
    assert ctx.cfg.gate.desc_chars == 260 and ctx.cfg.gate.sample_size == 8 and ctx.cfg.gate.reps == 3


def test_reasoning_judge_retry_at_2400(tmp_path):
    # a reasoning judge that writes prose (non-empty, unparseable) under the normal budget is retried at 2400 tokens
    def rule(info):
        return {"score": 4, "name": "n"} if info.max_tokens >= 2400 else "Let me think about these tickets..."
    llms, fk = fake_llms(rules_a={"gate": rule}, rules_b={"gate": rule}, reasoning_b=True)
    ctx = make_ctx(tmp_path, llms=llms, n_topics=2, per_topic=20)
    ctx.save(IDENT, "cache", "gate_calibration.json")
    G = Gate(ctx)
    c = G.call("t", "X", ctx.eligible_ids[:20], 0, "b")
    assert c["ok"] and c["score"] == 4
    assert [i.max_tokens for i in fk["b"].of("gate")] == [900, 900, 2400]
    ca = G.call("t", "X", ctx.eligible_ids[:20], 0, "a")                              # non-reasoning judge: no budget retry
    assert not ca["ok"] and [i.max_tokens for i in fk["a"].of("gate")] == [300, 300]


@pytest.mark.parametrize("raw,score", [(0, 1), (7, 5), (3.4, 3), ("4", 4), (4.6, 5)])
def test_scores_rounded_and_clamped(tmp_path, raw, score):
    llms, _ = fake_llms(rules_a={"gate": {"score": raw, "name": "n"}})
    ctx = make_ctx(tmp_path, llms=llms, n_topics=2, per_topic=20)
    assert Gate(ctx).call("t", "X", ctx.eligible_ids[:10], 0, "a")["score"] == score


def test_reference_thresholds():
    from clustering_harness.config import PipelineConfig
    g = PipelineConfig.reference().gate
    assert (g.accept_mean, g.accept_min_judge, g.reject_mean, g.escalate_accept) == (3.6, 3.2, 3.0, 3.4)
    assert (g.sample_size, g.reps, g.desc_chars, g.retry_tokens_reasoning) == (8, 3, 260, 2400)


# regression test (bug fixed after conformance review)
def test_null_name_is_not_a_name(tmp_path):
    ctx, _ = gate_ctx(tmp_path, {"a": [5] * 4, "b": [5] * 4})
    ctx.llms["a"].backend.fn.rules["gate"] = {"score": 5, "name": None}
    ctx.llms["b"].backend.fn.rules["gate"] = {"score": 5, "name": None}
    assert Gate(ctx).gate_many("t", [("X", members(ctx))])["X"]["name"] == ""
