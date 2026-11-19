"""Doc step 19: guarded report-line rollup (identical-name round 0, guard rule, rounds while above target,
neighbourhood regrouping, final naming and LN ids by size)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import TOPICS, fake_llms, make_ctx, make_topic_dataset, topic_of  # noqa: E402

from clustering_harness.algorithms.rollup import Rollup, _norm  # noqa: E402
from clustering_harness.config import PipelineConfig  # noqa: E402


class FakeJudge:
    """Replaces LLMJudge: ``score(arm, cid, members) -> balanced score``; records every evaluated cluster."""

    def __init__(self, score):
        self.score = score
        self.seen = []

    def evaluate(self, arm, clusters, stage=None):
        rows = []
        for cid, ms in clusters.items():
            self.seen.append((arm, cid, tuple(ms)))
            rows.append({"arm": arm, "cluster": str(cid), "size": len(ms), "balanced": self.score(arm, cid, ms)})
        return pd.DataFrame(rows, columns=["arm", "cluster", "size", "balanced"])


def setup(tmp_path, sizes, names=None, target_max=150, score=None, rollup=None, callback_a=None, split=None):
    ds, X = make_topic_dataset(n_topics=len(sizes), per_topic=list(sizes), shuffle=False, seed=4)
    cfg = PipelineConfig.small()
    cfg.rollup = cfg.rollup.model_copy(update={"target_max": target_max, **(rollup or {})})
    llms, fk = fake_llms(callback_a=callback_a)
    ctx = make_ctx(tmp_path, ds=ds, X=X, cfg=cfg, llms=llms)
    cmem = {}
    for t, l in zip(ds.ids, ds.labels):
        k = [p for p, _ in TOPICS].index(l)
        cmem.setdefault(f"c{k:02d}", []).append(t)
    if split:                                       # split a topic cluster into several clusters
        for c, n in split.items():
            ms = cmem.pop(c)
            for i in range(n):
                cmem[f"{c}_{i}"] = ms[i::n]
    cname = names(cmem) if names else {c: f"{TOPICS[int(c[1:3])][0].title()} problems {c}" for c in cmem}
    ro = Rollup(ctx)
    ro.judge = FakeJudge(score or (lambda arm, cid, ms: 4.5))
    return ctx, ro, cmem, cname, fk


def test_norm():
    assert _norm("  Password   RESET-failure! ") == "password resetfailure"
    assert _norm(None) == ""


def test_round0_identical_names_and_guard(tmp_path):
    names = lambda cm: {c: ("Export failures" if c in ("c00", "c01") else "export   FAILURES!" if c == "c02" else f"Line {c}") for c in cm}  # noqa: E731
    ctx, ro, cmem, cname, fk = setup(tmp_path, [20, 30, 10, 15], names=names)
    out = ro.run(cmem, cname)
    guard = out["guard"]
    assert len(guard) == 1 and guard.iloc[0]["round"] == 0 and guard.iloc[0]["parts"] == 3 and guard.iloc[0]["accepted"]
    assert out["log"]["rounds"][0]["proposed"] == 1 and len(out["log"]["rounds"]) == 1      # 2 lines <= target: no rounds 1+
    L = out["lines"]
    assert L.line_id.tolist() == ["LN001", "LN002"] and L.tickets.tolist() == [60, 15]
    assert L.iloc[0]["round_formed"] == 0 and L.iloc[0]["cluster_ids"] == "c01;c00;c02"       # parts ordered by size
    assert out["cluster_to_line"] == {"c00": "LN001", "c01": "LN001", "c02": "LN001", "c03": "LN002"}
    assert not fk["a"].of("rollup_decide")


@pytest.mark.parametrize("parts,union,ok", [
    ((4.5, 4.0), 3.875, True),       # parts mean (30*4.5 + 10*4.0)/40 = 4.375; 4.375 - 0.5 = 3.875 (boundary kept)
    ((4.5, 4.0), 3.75, False),       # more than 0.5 below the parts mean
    ((4.0, 3.5), 3.5, True),         # guard_min boundary 3.5
    ((3.5, 3.5), 3.4375, False),     # below 3.5
    ((np.nan, np.nan), 3.5, True),   # no part scores: only the 3.5 floor applies
    ((4.0, 4.0), np.nan, False),     # union could not be judged
])
def test_guard_rule(tmp_path, parts, union, ok):
    names = lambda cm: {c: "Same line" for c in cm}   # noqa: E731
    pscore = {"c00": parts[0], "c01": parts[1]}
    score = lambda arm, cid, ms: union if arm == "rollup_lines" else pscore[cid]   # noqa: E731
    ctx, ro, cmem, cname, fk = setup(tmp_path, [30, 10], names=names, score=score)
    out = ro.run(cmem, cname)
    g = out["guard"].iloc[0]
    assert bool(g["accepted"]) is ok
    assert len(out["lines"]) == (1 if ok else 2)
    if ok:
        assert out["lines"].iloc[0]["judge_score"] == (round(union, 3) if union == union else None)


def test_census_and_guard_use_the_judge_arms(tmp_path):
    names = lambda cm: {c: "Same line" for c in cm}   # noqa: E731
    ctx, ro, cmem, cname, fk = setup(tmp_path, [12, 10])
    ro.run(cmem, {c: "Same line" for c in cmem})
    arms = [a for a, _, _ in ro.judge.seen]
    assert arms[:2] == ["clusters", "clusters"] and arms[2:] == ["rollup_lines"]
    assert sorted(len(m) for a, _, m in ro.judge.seen if a == "rollup_lines") == [22]


def test_rounds_only_above_target_and_neighbourhood_merges(tmp_path):
    # 4 topics, each split into 3 clusters with distinct names -> 12 units; target 5 forces rounds 1+
    ctx, ro, cmem, cname, fk = setup(tmp_path, [24, 24, 24, 24], target_max=5, split={f"c0{k}": 3 for k in range(4)})
    out = ro.run(cmem, cname)
    rounds = out["log"]["rounds"]
    assert rounds[0]["round"] == 0 and rounds[0]["proposed"] == 0
    assert rounds[1]["round"] == 1 and rounds[1]["proposed"] >= 1
    assert out["log"]["lines"] == 4 and len(out["lines"]) == 4
    for _, r in out["lines"].iterrows():
        topics = {c[:3] for c in r.cluster_ids.split(";")}
        assert len(topics) == 1 and r.n_clusters == 3
    # stops before the next round once lines <= target
    assert all(r["lines"] > 5 for r in rounds[:-1])
    decide = fk["a"].of("rollup_decide")
    for i in decide:
        rows = re.findall(r"^(c\d+)\. (.*?) \((\d+) tickets\)\. Examples: (.*)$", i.user, re.M)
        assert 2 <= len(rows) <= 30
        assert all(len(ex.split("; ")) <= 3 for *_, ex in rows)                       # three central title examples
    # accepted unions are named by the rollup model, as is every line without a criterion
    assert all(r.criterion for _, r in out["lines"].iterrows())
    assert all(n.endswith(" line") for n in out["line_names"].values())


def test_no_rounds_when_at_or_below_target(tmp_path):
    ctx, ro, cmem, cname, fk = setup(tmp_path, [20, 20, 20], target_max=3)
    out = ro.run(cmem, cname)
    assert len(out["log"]["rounds"]) == 1 and not fk["a"].of("rollup_decide")
    assert len(fk["a"].of("rollup_name")) == 3                                         # final naming of every line


def test_decide_parser_ignores_duplicates_unknown_and_singletons(tmp_path):
    def decide(info):
        if info.kind != "rollup_decide":
            return None
        ids = re.findall(r"^(c\d+)\. ", info.user, re.M)
        return {"lines": [{"ids": [ids[0], ids[1], "c999"], "why": "x"}, {"ids": [ids[1], ids[2]], "why": "dup"},
                          {"ids": [ids[3]], "why": "single"}, {"why": "no ids"}]}
    ctx, ro, cmem, cname, fk = setup(tmp_path, [10, 10, 10, 10, 10, 10], target_max=1, callback_a=decide,
                                     rollup=dict(thresholds=[None, 2.0]), names=lambda cm: {c: f"Distinct {i}" for i, c in enumerate(cm)})
    out = ro.run(cmem, cname)
    g = out["guard"]
    assert len(g) == 1 and g.iloc[0]["parts"] == 2                       # c999 dropped; ids[1] reused -> 1-unit line ignored


def test_max_group_regrouping_chunks_at_30(tmp_path):
    names = lambda cm: {c: "identical label" for c in cm}   # noqa: E731
    ctx, ro, cmem, cname, fk = setup(tmp_path, [70], target_max=10, names=names, split={"c00": 70},
                                     score=lambda arm, cid, ms: 1.0 if arm == "rollup_lines" else 4.0,
                                     rollup=dict(thresholds=[None, 0.22]))
    out = ro.run(cmem, cname)
    sizes = sorted(len(re.findall(r"^c\d+\. ", i.user, re.M)) for i in fk["a"].of("rollup_decide"))
    assert sizes == [10, 30, 30]                                          # > 30 units: tighter thresholds, then chunks of 30
    assert out["log"]["lines"] == 70


def test_line_ids_by_descending_size(tmp_path):
    ctx, ro, cmem, cname, fk = setup(tmp_path, [5, 40, 12, 25, 33])
    out = ro.run(cmem, cname)
    L = out["lines"]
    assert L.line_id.tolist() == [f"LN{i:03d}" for i in range(1, 6)]
    assert L.tickets.tolist() == [40, 33, 25, 12, 5]


def test_reference_rollup_settings():
    r = PipelineConfig.reference().rollup
    assert r.thresholds == [None, 0.22, 0.28, 0.34, 0.40, 0.46]
    assert (r.max_group, r.target_max, r.guard_min, r.guard_drop) == (30, 150, 3.5, 0.5)


def test_census_uses_two_judges_three_samples_of_eight(tmp_path):
    """The real LLMJudge: 3 seeded samples of <= 8 records x both judges, judge-balanced mean."""
    from clustering_harness.evaluation.llm_judge import LLMJudge
    ds, X = make_topic_dataset(n_topics=2, per_topic=[20, 6], shuffle=False)
    llms, fk = fake_llms(rules_b={"judge": {"coherence": 3, "name": "b"}})
    ctx = make_ctx(tmp_path, ds=ds, X=X, llms=llms)
    fr = LLMJudge(ctx).evaluate("clusters", {"x": ds.ids[:20], "y": ds.ids[20:]}).set_index("cluster")
    assert len(fk["a"].of("judge")) == 6 and len(fk["b"].of("judge")) == 6
    assert all(len(re.findall(r"Subject: ", i.user)) <= 8 for i in fk["a"].of("judge"))
    assert fr.loc["x", "judge_a"] == 5 and fr.loc["x", "judge_b"] == 3 and fr.loc["x", "balanced"] == 4.0
    assert fr.loc["x", "n"] == 6


# regression test (bug fixed after conformance review)
def test_malformed_decide_answer_is_ignored(tmp_path):
    def decide(info):
        return {"lines": ["c1", "c2"]} if info.kind == "rollup_decide" else None
    ctx, ro, cmem, cname, fk = setup(tmp_path, [10, 10, 10], target_max=1, callback_a=decide, rollup=dict(thresholds=[None, 2.0]),
                                     names=lambda cm: {c: f"Distinct {i}" for i, c in enumerate(cm)})
    out = ro.run(cmem, cname)
    assert len(out["lines"]) == 3
