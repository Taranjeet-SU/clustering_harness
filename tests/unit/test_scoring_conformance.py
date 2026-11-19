"""Doc step 17: per-cluster geometry (against brute-force definitions), evidence sampling, naming and the adaptive
two-model review (dual_first_fast_followup_v3)."""
from __future__ import annotations

import collections
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import TOPICS, fake_llms, make_ctx, make_topic_dataset  # noqa: E402

from clustering_harness.algorithms.scoring import ClusterScorer, cluster_geometry  # noqa: E402
from clustering_harness.config import PipelineConfig  # noqa: E402


# =========================================================================================== fixtures
def geo_setup(tmp_path, sizes=(25, 18, 12, 30), n_out=8, n_junk=3, dups=()):
    ds, X = make_topic_dataset(n_topics=len(sizes), per_topic=list(sizes), n_outliers=n_out, n_junk=n_junk, shuffle=False, seed=2)
    for src, dst in dups:                                     # exact duplicate texts (title + description)
        ds.titles[dst], ds.texts[dst] = ds.titles[src], ds.texts[src]
    ctx = make_ctx(tmp_path, ds=ds, X=X, cfg=PipelineConfig.small())
    groups = {}
    for t, l in zip(ds.ids, ds.labels):
        if l in [p for p, _ in TOPICS]:
            groups.setdefault(f"K{[p for p, _ in TOPICS].index(l)}", []).append(t)
    # leave the last 3 records of K3 unassigned
    unassigned = groups["K3"][-3:]
    groups["K3"] = groups["K3"][:-3]
    assignment = {t: "" for t in ds.ids}
    for c, ms in groups.items():
        for t in ms:
            assignment[t] = c
    return ctx, groups, assignment, unassigned


def hashes_of(ctx):
    from clustering_harness.data import TextTemplate
    return dict(zip(ctx.ds.ids, ctx.ds.text_hashes(TextTemplate(ctx.cfg.text.embed_with_title, ctx.cfg.text.embed_without_title))))


# =========================================================================================== geometry
def test_geometry_matches_brute_force(tmp_path):
    ctx, groups, assignment, _ = geo_setup(tmp_path)
    g = cluster_geometry(ctx, groups, assignment, hashes_of(ctx)).set_index("cluster_id")
    X = ctx.X.astype(np.float64)
    pos = ctx.pos
    cids = sorted(groups)
    C = np.stack([X[[pos[t] for t in groups[c]]].mean(0) for c in cids])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # silhouette: exact cosine distance among *all assigned* records, averaged per cluster
    act = [t for t in ctx.ds.ids if assignment[t]]
    A = X[[pos[t] for t in act]]
    D = 1 - A @ A.T
    lab = np.array([assignment[t] for t in act])
    sil = {}
    for i, t in enumerate(act):
        own = (lab == lab[i])
        own[i] = False
        a = D[i, own].mean()
        b = min(D[i, lab == o].mean() for o in cids if o != lab[i])
        sil[t] = (b - a) / max(a, b)
    nj = [ctx.ds.ids[i] for i in ctx.nonjunk]
    XN = X[ctx.nonjunk]
    SN = XN @ XN.T
    np.fill_diagonal(SN, -np.inf)
    nn = {t: [nj[j] for j in np.argsort(-SN[k])[:10]] for k, t in enumerate(nj)}
    for k, c in enumerate(cids):
        ms = groups[c]
        V = X[[pos[t] for t in ms]]
        n = len(ms)
        S = V @ V.T
        own = V @ C[k]
        other = np.max(np.delete(V @ C.T, k, axis=1), axis=1)
        r = g.loc[c]
        assert r.case_count == n
        assert r.cosine_silhouette_mean == pytest.approx(np.mean([sil[t] for t in ms]), abs=1e-4)
        assert r.mean_pairwise_cosine == pytest.approx((S.sum() - np.trace(S)) / (n * (n - 1)), abs=1e-5)   # self-pairs excluded
        assert r.mean_centroid_cosine == pytest.approx(own.mean(), abs=1e-5)
        assert r.p10_centroid_cosine == pytest.approx(np.percentile(own, 10), abs=1e-5)
        assert r.mean_centroid_margin == pytest.approx((own - other).mean(), abs=1e-5)
        assert r.negative_centroid_margin_fraction == pytest.approx(np.mean(own - other < 0))
        assert r.mean_10nn_same_cluster_fraction == pytest.approx(np.mean([np.mean([assignment[x] == c for x in nn[t]]) for t in ms]))
        cc = C @ C[k]
        cc[k] = -np.inf
        assert r.nearest_cluster_id == cids[int(np.argmax(cc))]
        assert r.nearest_centroid_cosine == pytest.approx(cc.max(), abs=1e-5)


def test_silhouette_is_sklearn_cosine_in_original_space(tmp_path):
    from sklearn.metrics import silhouette_samples
    ctx, groups, assignment, _ = geo_setup(tmp_path)
    g = cluster_geometry(ctx, groups, assignment, hashes_of(ctx)).set_index("cluster_id")
    act = [t for t in ctx.ds.ids if assignment[t]]
    s = silhouette_samples(ctx.X[[ctx.pos[t] for t in act]], [assignment[t] for t in act], metric="cosine")
    per = pd.Series(s, index=act)
    for c, ms in groups.items():
        assert g.loc[c, "cosine_silhouette_mean"] == pytest.approx(per[ms].mean(), abs=1e-6)


def test_duplicate_statistics(tmp_path):
    # rows 0..24 are K0; make 1 and 2 copies of 0, and 30 (K1) a copy of 3
    ctx, groups, assignment, _ = geo_setup(tmp_path, dups=[(0, 1), (0, 2), (3, 30)])
    g = cluster_geometry(ctx, groups, assignment, hashes_of(ctx)).set_index("cluster_id")
    assert g.loc["K0", "unique_exact_texts"] == 23 and g.loc["K0", "duplicate_text_fraction"] == pytest.approx(1 - 23 / 25)
    assert g.loc["K0", "cases_with_exact_text_also_in_other_clusters"] == 1 and g.loc["K1", "cases_with_exact_text_also_in_other_clusters"] == 1


# regression test (bug fixed after conformance review)
def test_duplicate_with_unassigned_copy_is_not_in_other_cluster(tmp_path):
    ctx, groups, assignment, unassigned = geo_setup(tmp_path)
    src = groups["K3"][0]
    dst = ctx.pos[unassigned[0]]
    ctx.ds.titles[dst], ctx.ds.texts[dst] = ctx.ds.title(src), ctx.ds.text(src)
    g = cluster_geometry(ctx, groups, assignment, hashes_of(ctx)).set_index("cluster_id")
    assert g.loc["K3", "cases_with_exact_text_also_in_other_clusters"] == 0


# =========================================================================================== evidence sampling
def make_scorer(tmp_path, sizes=(25, 18, 12, 30), dups=(), llms=None, **kw):
    ctx, groups, assignment, _ = geo_setup(tmp_path, sizes=sizes, dups=dups, **kw)
    if llms is not None:
        ctx.llms = llms
    names = {c: f"old name {c}" for c in groups}
    return ctx, ClusterScorer(ctx, groups, assignment, names), groups


def test_sample_roles_order_and_uniqueness(tmp_path):
    ctx, sc, groups = make_scorer(tmp_path, dups=[(0, 1), (0, 2)])
    picks = sc.sample("K0", 800, n=12)
    roles = [r for _, r in picks]
    assert len(picks) == 12 and roles[:2] == ["central"] * 2 and roles[2:4] == ["boundary"] * 2
    assert roles[4:6] == ["diverse"] * 2 and set(roles[6:]) == {"random"}
    ids = [t for t, _ in picks]
    h = hashes_of(ctx)
    assert len({h[t] for t in ids}) == 12                       # unique exact texts
    assert ctx.ds.ids[1] not in ids and ctx.ds.ids[2] not in ids  # duplicates collapse to the lowest numeric id
    # brute force: central / boundary / greedy-diverse over the de-duplicated pool
    pool = [t for t in ctx.sort(groups["K0"]) if t not in (ctx.ds.ids[1], ctx.ds.ids[2])]
    V = ctx.V(pool)
    own = V @ sc.C[sc.cpos["K0"]]
    order = np.argsort(-own)
    assert ids[:2] == [pool[i] for i in order[:2]]
    assert ids[2:4] == [pool[order[-1]], pool[order[-2]]]
    dist = np.ones(len(pool))
    for t in ids[:4]:
        dist = np.minimum(dist, 1 - V @ ctx.X[ctx.pos[t]])
    i1 = int(np.argmax(dist))
    dist = np.minimum(dist, 1 - V @ V[i1]); dist[i1] = -1
    i2 = int(np.argmax(dist))
    assert ids[4:6] == [pool[i1], pool[i2]]
    assert sc.sample("K0", 800, n=12) == picks                  # deterministic
    assert [t for t, _ in sc.sample("K0", 1, n=12)][6:] != ids[6:]   # seeded random part depends on the replicate


def test_sample_small_cluster_and_exclusion(tmp_path):
    ctx, sc, groups = make_scorer(tmp_path)
    small = sc.sample("K2", 0, n=12)
    assert sorted(t for t, _ in small) == ctx.sort(groups["K2"])  # 12 members, all shown once
    naming = {t for t, _ in sc.sample("K0", 800, n=12)}
    fresh = sc.sample("K0", 0, n=12, exclude=naming)              # 13 fresh >= min(8, 25) -> only fresh records
    assert not {t for t, _ in fresh} & naming and len(fresh) == 12
    many = set(ctx.sort(groups["K0"])[:20])                       # only 5 fresh < 8 -> whole pool is used
    mixed = sc.sample("K0", 0, n=12, exclude=many)
    assert len(mixed) == 12 and {t for t, _ in mixed} & many


def test_evidence_clipping(tmp_path):
    ctx, sc, groups = make_scorer(tmp_path)
    t = groups["K0"][0]
    ctx.ds.texts[ctx.pos[t]] = "A" * 700 + "Z" * 300
    e = sc.evidence(t, "central", 600)
    # 75% head + marker + 25% tail
    assert e["excerpt_clipped"] and e["text"].startswith(ctx.ds.title(t)) and e["text"].endswith("Z" * 150)
    assert len(e["text"]) == 450 + len(" […] ") + 150 and " […] " in e["text"]
    short = sc.evidence(groups["K0"][1], "central", 600)
    assert not short["excerpt_clipped"]


# =========================================================================================== naming & adaptive review
class Judges:
    """Scripted Name/Judgment answers. ``plan(slot, cid, panel) -> dict`` overrides a valid default judgment."""

    def __init__(self, plan=None, name_plan=None):
        self.plan = plan or (lambda s, c, p: {})
        self.name_plan = name_plan
        self.count = collections.Counter()
        self.lock = threading.Lock()

    def cb(self, slot):
        def f(info):
            if info.kind == "score_judgment":
                p = info.payload()
                key = (slot, p["cluster_id"])
                with self.lock:
                    if info.n_user == 1:
                        self.count[key] += 1
                    panel = self.count[key] - 1
                own = [c["id"] for c in p["own_cases"]]
                out = {"coherence": 5, "label_fit": 5, "distinction": 5, "decision": "ACCEPT", "reason": "Consistent synthetic panel.",
                       "supporting_ids": own[:2], "suspicious_ids": [], "subissues": [], "limitation": ""}
                out.update(self.plan(slot, p["cluster_id"], panel, own=own, attempt=info.seed) if _takes_kw(self.plan) else
                           self.plan(slot, p["cluster_id"], panel))
                return out
            if info.kind == "score_name" and self.name_plan:
                return self.name_plan(info)
            return None
        return f


def _takes_kw(f):
    import inspect
    return "own" in inspect.signature(f).parameters


def scored(tmp_path, plan=None, sizes=(25, 18, 12, 30), name_plan=None, clusters=None):
    j = Judges(plan, name_plan)
    llms, fk = fake_llms(callback_a=j.cb("a"), callback_b=j.cb("b"))
    ctx, sc, groups = make_scorer(tmp_path, sizes=sizes, llms=llms)
    rows = {c: sc.process(c) for c in (clusters or sorted(groups))}
    return ctx, sc, groups, rows, fk, j


def test_single_panel_when_no_rule_fires(tmp_path):
    ctx, sc, groups, rows, fk, j = scored(tmp_path)
    r = rows["K0"]
    assert r["panels_reviewed"] == 1 and r["judge_a_panels"] == 1 and r["judge_b_panels"] == 1
    assert r["quality_status"] == "MODEL_REVIEW_PASSED" and r["semantic_score_1_to_5"] == 5.0
    assert r["sampling_escalation_reason"] == "No additional panel required by rule"
    assert r["evaluation_protocol"] == "dual_first_fast_followup_v3"
    # naming by slot b only (name_with="b"); judgments excluded naming cases when possible
    assert len(fk["b"].of("score_name")) == 4 and not fk["a"].of("score_name")
    assert r["evaluation_cases_also_used_for_naming"] == 0


def test_large_cluster_gets_second_panel_from_second_model_only(tmp_path):
    ctx, sc, groups, rows, fk, j = scored(tmp_path, sizes=(110, 18, 12, 30), clusters=["K0"])
    r = rows["K0"]
    assert r["panels_reviewed"] == 2 and r["judge_a_panels"] == 1 and r["judge_b_panels"] == 2
    assert "size >=100" in r["sampling_escalation_reason"]
    assert r["judgment_sample_unique_cases"] == 24


@pytest.mark.parametrize("field,value", [("label_fit", 3), ("distinction", 3), ("coherence", 3), ("decision", "REPAIR"),
                                         ("suspicious_ids", "first")])
def test_first_panel_concern_triggers_second_panel(tmp_path, field, value):
    def plan(slot, cid, panel, own=None, attempt=0):
        if slot == "a" and panel == 0:
            return {field: [own[-1]] if value == "first" else value}
        return {}
    ctx, sc, groups, rows, fk, j = scored(tmp_path, plan, clusters=["K0"])
    r = rows["K0"]
    assert r["panels_reviewed"] in (2, 3) and r["judge_a_panels"] == 1
    assert "first panel flagged" in r["sampling_escalation_reason"]
    # quality status: suspicious ids, a model's coherence mean < 4 or any non-ACCEPT -> review; label_fit alone does not
    want = "MODEL_REVIEW_PASSED" if field in ("label_fit", "distinction") else "REVIEW_RECOMMENDED"
    assert r["quality_status"] == want


def test_persistent_disagreement_third_panel_and_equal_model_weight(tmp_path):
    plan = lambda slot, cid, panel: {"coherence": 5} if slot == "a" else {"coherence": 3, "decision": "ACCEPT"}   # noqa: E731
    ctx, sc, groups, rows, fk, j = scored(tmp_path, plan, sizes=(40, 18, 12, 30), clusters=["K0"])
    r = rows["K0"]
    assert r["panels_reviewed"] == 3 and r["judge_a_panels"] == 1 and r["judge_b_panels"] == 3
    assert "persistent judge disagreement" in r["sampling_escalation_reason"]
    # each model's panel mean first, then equal weight: (5 + 3) / 2, not the pooled mean (5+3+3+3)/4
    assert r["semantic_score_1_to_5"] == 4.0 and r["judge_disagreement"] == 2.0
    assert r["quality_status"] == "REVIEW_RECOMMENDED"


def test_disagreement_resolved_after_second_panel(tmp_path):
    plan = lambda slot, cid, panel: {"coherence": 4} if (slot == "b" and panel == 0) else {}   # noqa: E731
    ctx, sc, groups, rows, fk, j = scored(tmp_path, plan, sizes=(40, 18, 12, 30), clusters=["K0"])
    r = rows["K0"]
    assert r["panels_reviewed"] == 2 and r["judge_b_semantic_score"] == 4.5 and r["semantic_score_1_to_5"] == 4.75


def test_stop_when_all_distinct_texts_seen(tmp_path):
    plan = lambda slot, cid, panel: {"coherence": 2, "decision": "SPLIT"}   # noqa: E731
    ctx, sc, groups, rows, fk, j = scored(tmp_path, plan, sizes=(25, 18, 12, 30), clusters=["K2"])
    r = rows["K2"]                                                     # 12 distinct texts: the first panel sees them all
    assert r["panels_reviewed"] == 1 and r["judgment_sample_fraction"] == 1.0


def test_invalid_responses_retried_then_fail(tmp_path):
    def plan(slot, cid, panel, own=None, attempt=0):
        if cid == "K1" and slot == "a" and attempt == 0:
            return {"coherence": 9}                                    # schema violation -> retried with seed 1
        if cid == "K1" and slot == "b" and attempt < 2:
            return {"supporting_ids": ["not-an-own-id"]}               # ID validation failure
        if cid == "K2" and slot == "a":
            return {"decision": "MAYBE"}                               # never valid
        return {}
    j = Judges(plan)
    llms, fk = fake_llms(callback_a=j.cb("a"), callback_b=j.cb("b"))
    ctx, sc, groups = make_scorer(tmp_path, llms=llms)
    r = sc.process("K1")
    assert r["semantic_score_1_to_5"] == 5.0
    seeds_b = [i.seed for i in fk["b"].of("score_judgment") if i.payload()["cluster_id"] == "K1"]
    assert seeds_b[:3] == [0, 1, 2]
    with pytest.raises(RuntimeError):
        sc.process("K2")
    k2a = [i for i in fk["a"].of("score_judgment") if i.payload()["cluster_id"] == "K2"]
    assert len(k2a) == 3                                               # three attempts in total
    assert list((ctx.workdir / "cache" / "scoring").glob("K2.adaptive_rep0.*.a.attempt2.error.json"))
    with pytest.raises(RuntimeError, match="incomplete model reviews"):
        sc.run()


def test_naming_payload_and_competitors(tmp_path):
    ctx, sc, groups, rows, fk, j = scored(tmp_path, clusters=["K0"])
    p = fk["b"].of("score_name")[0].payload()
    assert p["cluster_id"] == "K0" and p["size"] == 25 and len(p["own_cases"]) == 12
    assert [c["role"] for c in p["own_cases"]][:4] == ["central", "central", "boundary", "boundary"]
    near = [sc.cids[i] for i in sc.nearest[sc.cpos["K0"]]]
    assert [c["cluster_id"] for c in p["competing_clusters"]] == near and len(near) == 2
    assert all(len(c["examples"]) == 2 and c["provisional_name"] == f"old name {c['cluster_id']}" for c in p["competing_clusters"])
    jp = fk["a"].of("score_judgment")[0].payload()
    assert jp["proposed_label"] == rows["K0"]["cluster_name"] and len(jp["own_cases"]) == 12
    assert not {c["id"] for c in jp["own_cases"]} & {c["id"] for c in p["own_cases"]}


def test_completed_reviews_and_names_are_reused(tmp_path):
    ctx, sc, groups, rows, fk, j = scored(tmp_path, clusters=["K0"])
    n = len(fk["a"].log) + len(fk["b"].log)
    sc2 = ClusterScorer(ctx, groups, {t: c for c, ms in groups.items() for t in ms}, {})
    assert sc2.process("K0") == json.loads(json.dumps(rows["K0"], default=float))
    assert len(fk["a"].log) + len(fk["b"].log) == n


def test_run_sorts_and_flags_name_collisions(tmp_path):
    name_plan = lambda info: {"name": "Same Name!", "inclusion_criterion": "Tickets that match this line.",   # noqa: E731
                              "exclusion_criterion": "Others.", "rationale": "Synthetic rationale.", "unsupported_cause_warning": ""} \
        if info.payload()["cluster_id"] in ("K1", "K2") else None
    j = Judges(None, name_plan)
    llms, fk = fake_llms(callback_a=j.cb("a"), callback_b=j.cb("b"))
    ctx, sc, groups = make_scorer(tmp_path, llms=llms)
    df = sc.run()
    assert df.case_count.tolist() == sorted(df.case_count, reverse=True)
    coll = dict(zip(df.cluster_id, df.name_collision_flag))
    assert coll == {"K0": False, "K1": True, "K2": True, "K3": False}
