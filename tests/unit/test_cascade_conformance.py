"""Doc steps 6-11: the cascade (S2 local splits and replacement, cascade A side output, S2b consensus cores and
depth-two children, S3B residual Leiden, P1-P2 pruning, P3 centroid reassignment, P4 merges).

The gate is replaced by :class:`fakes.FakeGate` (decisions scripted per stage and cluster id) and the models by
:class:`fakes.FakeLLM`; local splits are scripted where the rule under test is the *use* of the children."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import TOPICS, FakeGate, fake_llms, make_ctx, make_topic_dataset, rid_of  # noqa: E402

from clustering_harness.algorithms import cascade as cascade_mod  # noqa: E402
from clustering_harness.algorithms.cascade import Cascade, local_split  # noqa: E402
from clustering_harness.algorithms.partitions import knn_graph  # noqa: E402
from clustering_harness.config import PipelineConfig  # noqa: E402
from clustering_harness.data import Dataset  # noqa: E402


# =========================================================================================== helpers
def build(tmp_path, sizes, cascade=None, rules_a=None, rules_b=None, callback_a=None, callback_b=None, partitions=None, seed=0):
    ds, X = make_topic_dataset(n_topics=len(sizes), per_topic=list(sizes), shuffle=False, seed=seed)
    cfg = PipelineConfig.small()
    cfg.cascade = cfg.cascade.model_copy(update={"large_parent": 10_000, **(cascade or {})})
    llms, fk = fake_llms(rules_a=rules_a, rules_b=rules_b, callback_a=callback_a, callback_b=callback_b)
    ctx = make_ctx(tmp_path, ds=ds, X=X, cfg=cfg, llms=llms)
    write_grid(ctx, partitions or {})
    return ctx, fk


def block(ctx, k):
    """ids of topic k (dataset order)."""
    return [t for t, l in zip(ctx.ds.ids, ctx.ds.labels) if l == TOPICS[k][0]]


def write_grid(ctx, parts: dict):
    rows = []
    lo = ctx.cfg.bank.n_clusters_range[0]
    for rid, lab in parts.items():
        np.save(ctx.path("cache", "partitions", f"{rid}.npy"), np.asarray(lab, np.int32))
        rows.append(dict(run_id=rid, view="lg", seed=0, family="test", params="{}", n_clusters=lo, largest_frac=0.01, noise_frac=0.0))
    pd.DataFrame(rows, columns=["run_id", "view", "seed", "family", "params", "n_clusters", "largest_frac", "noise_frac"]).to_csv(
        ctx.path("results", "bank", "grid_runs.csv"), index=False)


def base_labels(ctx, groups: dict):
    """groups {int label: [ids]} -> label array over eligible records (base cluster c<l> becomes parent P<l>)."""
    pos = {t: i for i, t in enumerate(ctx.eligible_ids)}
    lab = np.full(len(pos), -1)
    for l, ids in groups.items():
        lab[[pos[t] for t in ids]] = l
    return lab


def table_rule(table: dict):
    """FakeGate rule from {stage: {cid: (decision, adj)} or callable(cid, ms)}."""
    def rule(stage, cid, ms):
        r = table.get(stage)
        if r is None:
            return None
        return r(cid, ms) if callable(r) else r.get(cid)
    return rule


def script_splits(monkeypatch, script: dict):
    calls = []

    def fake(self, key, ms, cap=None):
        calls.append((key, len(ms), cap))
        return [(k, list(g)) for k, g in script.get(key, [])]
    monkeypatch.setattr(Cascade, "_cached_split", fake)
    return calls


def no_reassign(monkeypatch):
    monkeypatch.setattr(Cascade, "reassign", lambda self, acc, stage, rg: (acc, {"added": 0}))


def ACC(adj=4.0):
    return ("ACCEPT", adj)


def REJ(adj=2.0):
    return ("REJECT", adj)


# =========================================================================================== local_split
class SplitStub:
    def __init__(self, monkeypatch, labels):
        import hdbscan
        import umap
        self.umap_kw, self.hdb_kw = [], []
        outer = self

        class U:
            def __init__(self, **kw):
                outer.umap_kw.append(kw)

            def fit_transform(self, X):
                return X[:, :5]

        class H:
            def __init__(self, **kw):
                outer.hdb_kw.append(kw)

            def fit_predict(self, U_):
                return np.asarray(labels[:len(U_)])
        monkeypatch.setattr(umap, "UMAP", U)
        monkeypatch.setattr(hdbscan, "HDBSCAN", H)


def test_local_split_size_floor(monkeypatch, tmp_path):
    ctx = make_ctx(tmp_path, n_topics=2, per_topic=40, cfg=PipelineConfig.reference())
    st = SplitStub(monkeypatch, [0] * 100)
    assert local_split(ctx, "P1", ctx.eligible_ids[:29]) == []          # doc step 6: no children below size 30
    assert st.umap_kw == []


def test_local_split_parameters_and_children(monkeypatch, tmp_path):
    ctx = make_ctx(tmp_path, n_topics=6, per_topic=40, cfg=PipelineConfig.reference())
    sizes = [40, 9, 12, 25, 11, 13, 14, 15, 30]
    labels = [i for i, s in enumerate(sizes) for _ in range(s)] + [-1] * 11
    ms = ctx.eligible_ids[:len(labels)]
    st = SplitStub(monkeypatch, labels)
    out = local_split(ctx, "P7", ms, seed=0)
    n = len(ms)
    assert st.umap_kw == [dict(n_components=5, metric="cosine", n_neighbors=15, min_dist=0.0, random_state=0)]
    assert st.hdb_kw == [dict(min_cluster_size=max(10, n // 8), cluster_selection_method="leaf")]
    # children >= 10 members, sorted by size, the six largest, ids <parent>.<i>
    assert [k for k, _ in out] == [f"P7.{i}" for i in range(6)]
    assert [len(g) for _, g in out] == [40, 30, 25, 15, 14, 13]
    assert local_split(ctx, "P7", ms, cap=4)[-1][0] == "P7.3"


def test_local_split_needs_two_children(monkeypatch, tmp_path):
    ctx = make_ctx(tmp_path, n_topics=2, per_topic=40, cfg=PipelineConfig.reference())
    SplitStub(monkeypatch, [0] * 40 + [1] * 9 + [-1] * 10)
    assert local_split(ctx, "P1", ctx.eligible_ids[:59]) == []


def test_local_split_small_parent_neighbours(monkeypatch, tmp_path):
    cfg = PipelineConfig.reference()
    cfg.cascade = cfg.cascade.model_copy(update=dict(split_min_size=5))
    ctx = make_ctx(tmp_path, n_topics=2, per_topic=40, cfg=cfg)
    st = SplitStub(monkeypatch, [0] * 6 + [1] * 6)
    local_split(ctx, "P1", ctx.eligible_ids[:12])
    assert st.umap_kw[0]["n_neighbors"] == 11                               # min(15, n - 1)


def test_local_split_real_two_topics(tmp_path):
    ds, X = make_topic_dataset(n_topics=2, per_topic=40, shuffle=False, seed=1)
    ctx = make_ctx(tmp_path, ds=ds, X=X, cfg=PipelineConfig.reference())
    out = local_split(ctx, "P3", ctx.eligible_ids)
    assert len(out) >= 2 and all(len(g) >= 10 for _, g in out)
    for _, g in out:
        assert len({ctx.ds.labels[ctx.pos[t]] for t in g}) == 1


def test_large_parent_threshold_reference():
    # doc step 6: large parent = size >= int(0.01 x eligible records), e.g. 200 for 20,000 eligible records
    cfg = PipelineConfig.reference()
    c = Cascade(SimpleNamespace(cfg=cfg, nonjunk=range(20000)), gate=None)
    assert c.large == 200
    cfg.cascade = cfg.cascade.model_copy(update=dict(large_parent=30))
    assert Cascade(SimpleNamespace(cfg=cfg, nonjunk=range(20000)), gate=None).large == 30


# =========================================================================================== S1 / S2 / cascade A
def test_s2_split_selection_and_replacement_rule(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [60, 60, 60, 60, 60, 60], cascade=dict(large_parent=50))
    b = [block(ctx, k) for k in range(6)]
    base = base_labels(ctx, {0: b[0], 1: b[1], 2: b[2], 3: b[3], 4: b[4][:40], 5: b[5]})
    splits = {
        "P0": [("P0.0", b[0][:40]), ("P0.1", b[0][40:])],      # full coverage, weighted 4.4 > 4.0 -> replaced
        "P1": [("P1.0", b[1][:30]), ("P1.1", b[1][30:])],      # coverage 0.5 < 0.6 -> parent retained
        "P2": [("P2.0", b[2][:36]), ("P2.1", b[2][36:])],      # weighted score == parent score (not >) -> retained
        "P3": [("P3.0", b[3][:36]), ("P3.1", b[3][36:])],      # coverage exactly 0.6 and 4.3 > 4.0 -> replaced by P3.0 only
        "P5": [("P5.0", b[5][:30]), ("P5.1", b[5][30:])],      # rejected parent keeps its accepted children
    }
    calls = script_splits(monkeypatch, splits)
    no_reassign(monkeypatch)
    gate = FakeGate(table_rule({
        "cas_s1": {"P0": ACC(4.0), "P1": ACC(4.0), "P2": ACC(4.0), "P3": ACC(4.0), "P4": ACC(4.0), "P5": REJ(2.5)},
        "cas_s2": {"P0.0": ACC(4.5), "P0.1": ACC(4.2), "P1.0": ACC(4.5), "P1.1": REJ(), "P2.0": ACC(4.0), "P2.1": ACC(4.0),
                   "P3.0": ACC(4.3), "P3.1": REJ(), "P5.0": ACC(3.9), "P5.1": REJ()},
    }))
    Cascade(ctx, gate).run(base)
    # split: every non-ACCEPT parent plus every parent of size >= 50 (P4 has 40 members)
    assert sorted(k for k, _, _ in calls if k.startswith("P") and "." not in k) == ["P0", "P1", "P2", "P3", "P5"]
    assert {c for c, _ in gate.items("cas_s2")} == {k for v in splits.values() for k, _ in v}
    s2 = ctx.load("cache", "cascade", "after_s2.json")
    assert set(s2) == {"P0.0", "P0.1", "P1", "P2", "P3.0", "P4", "P5.0"}
    assert s2["P1"]["stage"] == "S1" and s2["P0.0"]["stage"] == "S2" and s2["P3.0"]["members"] == ctx.sort(b[3][:36])


def test_s2_contract_fail_parent_is_split(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [40, 40])
    b = [block(ctx, k) for k in range(2)]
    calls = script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    gate = FakeGate(table_rule({"cas_s1": {"P0": ("CONTRACT_FAIL", None), "P1": ("ESCALATE", 3.5)}}))
    Cascade(ctx, gate).run(base_labels(ctx, {0: b[0], 1: b[1]}))
    assert sorted(k for k, _, _ in calls) == ["P0", "P1"]


def test_cascade_a_is_side_output_and_does_not_seed_b(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [50, 50, 50])
    b = [block(ctx, k) for k in range(3)]
    script_splits(monkeypatch, {})
    # base clusters hold only part of each topic; the rest is residual that reassignment can pick up
    base = base_labels(ctx, {0: b[0][:35], 1: b[1][:35], 2: b[2][:35]})
    gate = FakeGate(table_rule({"cas_s1": lambda c, m: ACC(), "cas_s3a_regate": lambda c, m: ACC()}))
    casc = Cascade(ctx, gate)
    casc.run(base)
    log = casc.log
    s2 = ctx.load("cache", "cascade", "after_s2.json")
    a = ctx.load("results", "cascade_A", "clusters.json")
    assert log["S3A"]["added"] > 0 and log["S3A"]["tickets"] == log["S2"]["tickets"] + log["S3A"]["added"]
    assert sum(len(v["members"]) for v in a.values()) > sum(len(v["members"]) for v in s2.values())
    # cascade B restarts from S2: no S2b candidates here, so B after S2b == S2, and its residual excludes A's additions
    assert log["S2b"]["tickets"] == log["S2"]["tickets"]
    assert log["S3B"]["residual"] == len(ctx.eligible_ids) - log["S2"]["tickets"]
    assert (ctx.workdir / "results" / "cascade_A" / "assignments.csv").exists()


def test_cascade_a_can_be_disabled(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [40, 40], cascade=dict(run_cascade_a=False))
    script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    casc = Cascade(ctx, FakeGate(table_rule({"cas_s1": lambda c, m: ACC()})))
    casc.run(base_labels(ctx, {0: block(ctx, 0), 1: block(ctx, 1)}))
    assert "S3A" not in casc.log and not (ctx.workdir / "results" / "cascade_A").exists()


# =========================================================================================== S2b consensus cores
REF_CORES = dict(cores_min_parent=20, core_min_size=10, candidate_min_size=10)


def coassoc_partitions(ctx, m, other_label=1000):
    """5 partitions over eligible records with designed co-association inside the member list ``m`` (>= 48 ids).

    X=m[0:12] always together (1.0); Y=Y1|Y2 (m[12:24]) together in 4/5 (0.8); chain A,B,C (m[24:36]) with
    co(A,B)=co(B,C)=0.8 but co(A,C)=0.6; Z1, Z2 (m[36:48]) together 3/5 (0.6, components of 6 < 10)."""
    pos = {t: i for i, t in enumerate(ctx.eligible_ids)}
    N = len(pos)
    P = np.tile(np.arange(N) + other_label, (5, 1))
    idx = lambda sl: [pos[t] for t in m[sl]]          # noqa: E731
    for p in range(5):
        P[p, idx(slice(0, 12))] = 1
        P[p, idx(slice(12, 18))] = 2
        P[p, idx(slice(18, 24))] = 2 if p < 4 else 3
        P[p, idx(slice(24, 28))] = 4 if p < 4 else 6       # A
        P[p, idx(slice(28, 32))] = 4                         # B
        P[p, idx(slice(32, 36))] = 5 if p == 0 else 4        # C
        P[p, idx(slice(36, 42))] = 7
        P[p, idx(slice(42, 48))] = 7 if p < 3 else 8
    return {f"part{p}": P[p] for p in range(5)}


def test_consensus_cores_connected_components(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [48, 60, 15, 40], cascade=REF_CORES)
    m = block(ctx, 0)
    write_grid(ctx, coassoc_partitions(ctx, m))
    b2 = block(ctx, 1)
    pos = {t: i for i, t in enumerate(ctx.eligible_ids)}
    parts = {k: np.load(ctx.workdir / "cache" / "partitions" / f"{k}.npy") for k in [f"part{p}" for p in range(5)]}
    for k, lab in parts.items():                      # parent P1: six blocks of 10 always together -> capped at 4 cores
        for j in range(6):
            lab[[pos[t] for t in b2[j * 10:(j + 1) * 10]]] = 100 + j
    write_grid(ctx, parts)
    script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    gate = FakeGate(table_rule({"cas_s1": {"P0": REJ(), "P1": REJ(), "P2": REJ(), "P3": ACC()}}))
    casc = Cascade(ctx, gate)
    casc.run(base_labels(ctx, {0: m, 1: b2, 2: block(ctx, 2), 3: block(ctx, 3)}))
    cores = dict(gate.items("cas_s2b"))
    p0 = {k: set(v) for k, v in cores.items() if k.startswith("P0.")}
    # X (clique), Y (co 0.8 counts: >= threshold) and the chain A-B-C (connected though co(A,C) = 0.6); Z pieces too small
    assert set(map(frozenset, p0.values())) == {frozenset(m[0:12]), frozenset(m[12:24]), frozenset(m[24:36])}
    assert all(k.startswith("P0.core") for k in p0)
    assert len([k for k in cores if k.startswith("P1.")]) == 4                 # cap: four cores per parent
    assert not [k for k in cores if k.startswith(("P2.", "P3."))]              # P2 < 20 members; P3 accepted at S1
    assert casc.log["S2b"]["bank_partitions_eligible"] == 5


def test_consensus_cores_jaccard_filter(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [48, 30], cascade=REF_CORES)
    m = block(ctx, 0)
    write_grid(ctx, coassoc_partitions(ctx, m))
    script_splits(monkeypatch, {"P0": [("P0.0", m[0:12] + m[36:38]),               # J(X) = 12/14 >= 0.6 -> X skipped
                                       ("P0.1", m[12:24] + m[38:47])]})             # J(Y) = 12/21 < 0.6 -> Y kept
    no_reassign(monkeypatch)
    gate = FakeGate(table_rule({"cas_s1": {"P0": REJ(), "P1": ACC()}, "cas_s2": {"P0.0": ACC(), "P0.1": ACC()}}))
    Cascade(ctx, gate).run(base_labels(ctx, {0: m, 1: block(ctx, 1)}))
    cores = {frozenset(v) for k, v in gate.items("cas_s2b")}
    assert cores == {frozenset(m[12:24]), frozenset(m[24:36])}


def test_consensus_cores_respect_min_size_and_threshold(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [48, 30], cascade={**REF_CORES, "core_min_size": 13})
    m = block(ctx, 0)
    write_grid(ctx, coassoc_partitions(ctx, m))
    script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    gate = FakeGate(table_rule({"cas_s1": {"P0": REJ(), "P1": ACC()}}))
    Cascade(ctx, gate).run(base_labels(ctx, {0: m, 1: block(ctx, 1)}))
    assert gate.items("cas_s2b") == []                                         # all components have 12 < 13 members


def test_depth_two_splits_and_overlap_removal(tmp_path, monkeypatch):
    yes = lambda i: {"answers": [{"n": k + 1, "belongs": True} for k in range(20)]} if i.kind == "verify" else None  # noqa: E731
    ctx, _ = build(tmp_path, [48, 30, 30], cascade={**REF_CORES, "deep_split_min": 12}, callback_a=yes, callback_b=yes)
    m, other = block(ctx, 0), block(ctx, 2)
    write_grid(ctx, coassoc_partitions(ctx, m))
    calls = script_splits(monkeypatch, {
        "P0": [("P0.0", m[36:48]), ("P0.1", m[24:35])],                             # both rejected; only P0.0 is >= 12
        "P0.0": [("P0.0.0", m[0:8] + other[:5]), ("P0.0.1", other[5:9])],
    })
    no_reassign(monkeypatch)
    gate = FakeGate(table_rule({
        "cas_s1": {"P0": REJ(), "P1": ACC()},
        "cas_s2": {"P0.0": REJ(), "P0.1": REJ()},
        # deep child first (4.5): takes 8 of X; X (4.0) keeps 4 < 10 -> dropped; Y (3.8) added; chain rejected
        "cas_s2b": lambda c, ms: ACC(4.5) if c == "P0.0.0" else ACC(4.0) if set(ms) == set(m[0:12]) else
        ACC(3.8) if set(ms) == set(m[12:24]) else ACC(4.2) if c == "P0.0.1" else REJ(),
    }))
    casc = Cascade(ctx, gate)
    casc.run(base_labels(ctx, {0: m, 1: block(ctx, 1)}))                         # topic 2 ("other") stays residual
    assert ("P0.0", 12, 4) in calls and not [c for c in calls if c[0] == "P0.1"]   # depth-two: rejected child >= min, cap 4
    final = ctx.load("results", "cascade_B", "clusters.json")
    assert set(final["P0.0.0"]["members"]) == set(m[0:8] + other[:5]) and final["P0.0.0"]["stage"] == "S2b"
    assert not any(set(v["members"]) >= set(m[8:12]) and k.startswith("P0.core") and len(v["members"]) < 10 for k, v in final.items())
    ycore = [k for k, v in final.items() if set(v["members"]) == set(m[12:24])]
    assert len(ycore) == 1
    assert "P0.0.1" not in final                                                 # 4 members < candidate_min_size
    assert casc.log["S2b"]["deep_children"] == 2 and casc.log["S2b"]["cores"] == 3
    # the X core lost 8 members to the higher-scored deep child and fell below ten
    assert not [k for k, v in final.items() if set(v["members"]) == set(m[0:12])]


# =========================================================================================== S3B
def doc_leiden(ctx, resid, scan=30, k=15, resolution=1.0, seed=0, min_size=10):
    """Independent implementation of doc step 9 for comparison."""
    import igraph as ig
    import leidenalg as la
    I, _ = knn_graph(ctx)
    elig = ctx.eligible_ids
    pos = {t: i for i, t in enumerate(elig)}
    rpos = {pos[t]: j for j, t in enumerate(resid)}
    nbrs = []
    for t in resid:
        nb = [int(x) for x in I[pos[t], 1:scan + 1] if int(x) in rpos][:k]
        nbrs.append({rpos[x] for x in nb})
    edges = [(i, j) for i in range(len(resid)) for j in sorted(nbrs[i]) if i < j and i in nbrs[j]]
    g = ig.Graph(n=len(resid), edges=edges)
    g.es["weight"] = [float(ctx.X[ctx.pos[resid[i]]] @ ctx.X[ctx.pos[resid[j]]]) for i, j in edges]
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution, seed=seed)
    return sorted([sorted(resid[i] for i in c) for c in part if len(c) >= min_size], key=len, reverse=True)


def test_s3b_leiden_on_residual(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [40, 30, 25, 20, 12], cascade=dict(leiden_min_size=10, leiden_max_gated=2))
    script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    b0 = block(ctx, 0)
    gate = FakeGate(table_rule({"cas_s1": {"P0": ACC()}, "cas_s3b": {"L0": ACC(4.1)}}))
    casc = Cascade(ctx, gate)
    casc.run(base_labels(ctx, {0: b0}))
    resid = [t for t in ctx.eligible_ids if t not in set(b0)]
    expected = doc_leiden(ctx, resid)
    got = gate.items("cas_s3b")
    assert [k for k, _ in got] == ["L0", "L1"]                                       # only the largest max_gated are gated
    assert [sorted(v) for _, v in got] == [sorted(c) for c in expected[:2]]
    assert all(len(v) >= 10 for _, v in got) and len(got[0][1]) >= len(got[1][1])
    final = ctx.load("results", "cascade_B", "clusters.json")
    assert final["L0"]["stage"] == "S3B" and "L1" not in final
    log = casc.log["S3B"]
    assert log["communities"] == len(expected) and log["gated"] == 2 and log["accepted"] == 1
    cov = log["tickets_added"]
    assert log["yield_rule_passed"] == (cov >= 0.03 * log["residual"] and 1 >= 0.1 * 2)
    assert "P" in casc.log                                                            # yield rule never halts pruning


def test_s3b_yield_rule_failure_does_not_branch(tmp_path, monkeypatch):
    ctx, _ = build(tmp_path, [40, 30, 30])
    script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    casc = Cascade(ctx, FakeGate(table_rule({"cas_s1": {"P0": ACC()}})))       # every community rejected
    casc.run(base_labels(ctx, {0: block(ctx, 0)}))
    assert casc.log["S3B"]["accepted"] == 0 and casc.log["S3B"]["yield_rule_passed"] is False
    assert casc.log["P"]["clusters_checked"] == 1 and "P4" in casc.log


# =========================================================================================== P1-P2 pruning
def test_pruning_double_no_and_dissolution(tmp_path, monkeypatch):
    ctx0, _ = build(tmp_path / "probe", [60, 60, 12])
    mem = {f"P{k}": ctx0.sort(block(ctx0, k)) for k in range(3)}
    samp = {c: ctx0.sort(random.Random(11).sample(ms, min(10, len(ms)))) for c, ms in mem.items()}   # doc: random.Random(11)
    v_no = {**{t: False for t in samp["P0"][:6]}, **{t: False for t in samp["P1"][:5]}, **{t: False for t in samp["P2"][:5]}}
    v_missing = set(samp["P1"][5:7])                       # verifier gives no answer -> never "no"
    r_no = set(samp["P0"][:6]) | set(samp["P1"][:4]) | set(samp["P2"][:5])
    r_missing = {samp["P1"][4]}                            # explicit verifier "no" but missing recheck -> kept

    def verifier(info):
        if info.kind != "verify" or info.stage != "cas_p1_check":
            return None
        ids = [rid_of(t) for t in info.titles(info.section("Candidate tickets:"))]
        return {"answers": [{"n": i + 1, "belongs": v_no.get(t, True)} for i, t in enumerate(ids) if t not in v_missing]}

    def rechecker(info):
        if info.kind != "verify" or info.stage != "cas_p2_recheck":
            return None
        ids = [rid_of(t) for t in info.titles(info.section("Candidate tickets:"))]
        return {"answers": [{"n": i + 1, "belongs": t not in r_no} for i, t in enumerate(ids) if t not in r_missing]}

    ctx, fk = build(tmp_path, [60, 60, 12], callback_a=verifier, callback_b=rechecker)
    script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    casc = Cascade(ctx, FakeGate(table_rule({"cas_s1": lambda c, m: ACC()})))
    casc.run(base_labels(ctx, {0: mem["P0"], 1: mem["P1"], 2: mem["P2"]}))
    prune = ctx.load("cache", "cascade", "prune.json")
    final = ctx.load("results", "cascade_B", "clusters.json")
    # P0: 6/10 double "no" -> 1 - 0.6 = 0.4 < 0.5 -> dissolved
    assert prune["P0"]["ejected"] == 6 and "P0" not in final and casc.log["P"]["dissolved"] == ["P0"]
    # P1: 5 verifier "no", 4 recheck "no", 1 recheck missing -> 4 ejected, nothing else touched
    assert prune["P1"]["ejected"] == 4 and set(final["P1"]["members"]) == set(mem["P1"]) - set(samp["P1"][:4])
    # P2: exactly 5/10 -> precision 0.5 is not < 0.5 -> kept, the 5 ejected
    assert prune["P2"]["precision_est"] == 0.5 and set(final["P2"]["members"]) == set(mem["P2"]) - set(samp["P2"][:5])
    # only explicit verifier "no" answers are rechecked
    rech = [i for i in fk["b"].log if i.stage == "cas_p2_recheck"]
    asked = {rid_of(t) for i in rech for t in i.titles(i.section("Candidate tickets:"))}
    assert asked == {t for t, v in v_no.items() if v is False}
    # sample used by the verifier == seeded sample of up to ten
    checks = [i for i in fk["a"].log if i.stage == "cas_p1_check"]
    assert sorted(tuple(rid_of(t) for t in i.titles(i.section("Candidate tickets:"))) for i in checks) == \
        sorted(tuple(s) for s in samp.values())


def test_pruning_prompt_material(tmp_path, monkeypatch):
    ctx, fk = build(tmp_path, [40, 12])
    script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    m0, m1 = ctx.sort(block(ctx, 0)), ctx.sort(block(ctx, 1))
    Cascade(ctx, FakeGate(table_rule({"cas_s1": lambda c, m: ACC()}))).run(base_labels(ctx, {0: m0, 1: m1}))
    # the labeler (slot b) writes line + criterion from the replicate-0 gate sample (<= 8 exemplars)
    labels = [i for i in fk["b"].log if i.stage == "cas_p1_label"]
    shown = sorted(tuple(rid_of(t) for t in i.titles()) for i in labels)
    assert shown == sorted([tuple(ctx.gsample("P0", 0, m0)), tuple(ctx.gsample("P1", 0, m1))])
    # verifier sees the line, the criterion and central exemplars that avoid sampled records when possible
    for i in [i for i in fk["a"].log if i.stage == "cas_p1_check"]:
        cands = [rid_of(t) for t in i.titles(i.section("Candidate tickets:"))]
        ex = [rid_of(t) for t in i.titles(i.section("Tickets already on this line:", "Candidate tickets:"))]
        ms = m0 if cands[0] in set(m0) else m1
        central = ctx.exemplars(ms, 3)
        assert "problems" in i.section("Report line: ", "\n") and "The ticket reports" in i.section("Inclusion criterion: ", "\n")
        assert 1 <= len(ex) <= 3 and set(ex) <= set(central)
        if set(central) - set(cands):
            assert not set(ex) & set(cands)


# =========================================================================================== P3 reassignment (direct)
def reassign_fixture(tmp_path, resid_specs, cos_ab=0.8, n_c=10, seed=0, rules_a=None, callback_a=None, gate_rule=None):
    """Two clusters A (around e0) and B (around b, cos(e0, b) = cos_ab) with own-centroid cosines c in
    linspace(0.80, 0.98, n_c), each twice (+/- an orthogonal axis so the centroids are exact); residual records
    with target cosines (s to A, t to B)."""
    d = 24
    phi = np.arccos(cos_ab)
    e = np.eye(d)
    a_dir, b_dir = e[0], np.cos(phi) * e[0] + np.sin(phi) * e[1]
    cs = np.linspace(0.80, 0.98, n_c)
    vecs, titles, ids = [], [], []

    def add(v, title):
        rid = str(1000 + len(ids))
        ids.append(rid); vecs.append(v); titles.append(f"{title} case {rid}")
    for k, c in enumerate(cs):
        for sgn in (1, -1):
            add(c * a_dir + sgn * np.sqrt(1 - c * c) * e[2 + k % 4], "password reset")
    for k, c in enumerate(cs):
        for sgn in (1, -1):
            add(c * b_dir + sgn * np.sqrt(1 - c * c) * e[6 + k % 4], "invoice mismatch")
    for j, (s, t) in enumerate(resid_specs):
        y = (t - s * np.cos(phi)) / np.sin(phi)
        z = 1 - s * s - y * y
        assert z >= 0, (s, t)
        add(s * e[0] + y * e[1] + np.sqrt(z) * e[10 + j % 14], "export failure")
    ds = Dataset(ids=ids, texts=[f"Customer reports something {i}." for i in ids], titles=titles)
    llms, fk = fake_llms(rules_a=rules_a, callback_a=callback_a)
    ctx = make_ctx(tmp_path, ds=ds, X=np.array(vecs), cfg=PipelineConfig.reference(), llms=llms)
    gate = FakeGate(gate_rule)
    acc = {"A": {"members": ids[:2 * n_c], "name": "Password reset problems", "adj": 4.0, "stage": "S1"},
           "B": {"members": ids[2 * n_c:4 * n_c], "name": "Invoice mismatch problems", "adj": 4.0, "stage": "S1"}}
    return ctx, fk, gate, acc, ids[4 * n_c:]


def test_reassign_thresholds_auto_and_band(tmp_path):
    specs = [(0.90, 0.75),    # 0 auto -> A (>= tau_hi, margin 0.15)
             (0.90, 0.895),   # 1 band: high similarity but margin 0.005 < 0.01
             (0.82, 0.60),    # 2 band: tau_lo <= best < tau_hi
             (0.70, 0.50),    # 3 below tau_lo: never considered
             (0.60, 0.90),    # 4 auto -> B
             (0.86, 0.851)]   # 5 band: margin 0.009
    ctx, fk, gate, acc, resid = reassign_fixture(tmp_path, specs, callback_a=lambda i: {"answers": []} if i.kind == "verify" else None)
    own = np.concatenate([ctx.V(acc[c]["members"]) @ ctx.centroid(acc[c]["members"]) for c in "AB"])
    acc2, st = Cascade(ctx, gate).reassign(acc, "t", "t_regate")
    assert st["tau_hi"] == pytest.approx(np.percentile(own, 25)) and st["tau_lo"] == pytest.approx(np.percentile(own, 5))
    assert st["tau_hi"] == pytest.approx(0.84, abs=1e-5) and st["tau_lo"] == pytest.approx(0.80, abs=1e-5)
    assert st["auto"] == 2 and st["band_tickets"] == 3
    asked = {rid_of(t) for i in fk["a"].of("verify") for t in i.titles(i.section("Candidate tickets:"))}
    assert asked == {resid[1], resid[2], resid[5]}
    assert resid[0] in acc2["A"]["members"] and resid[4] in acc2["B"]["members"]
    assert resid[3] not in acc2["A"]["members"] + acc2["B"]["members"]
    assert st["added"] == 2                                     # band verifier gave no answers -> nothing from the band


def band_specs(n_a, n_b):
    a = [(0.81 + 0.002 * i, 0.60) for i in range(n_a)]           # band of A (below tau_hi)
    b = [(0.55, 0.81 + 0.002 * i) for i in range(n_b)]           # band of B
    return a, b


def test_band_sampling_and_whole_band_admission(tmp_path):
    a_spec, b_spec = band_specs(10, 9)
    specs = a_spec + b_spec
    probe = reassign_fixture(tmp_path / "probe", specs)
    resid = probe[4]
    band_a, band_b = resid[:10], resid[10:19]
    # doc step 11: per destination, up to eight band candidates sampled with seed 7
    samp_a = sorted(random.Random(7).sample(band_a, 8), key=int)
    samp_b = sorted(random.Random(7).sample(band_b, 8), key=int)
    yes = set(samp_a[:6]) | set(samp_b[:5])                      # A: 6/8 = 75% -> whole band; B: 5/8 -> only the yes

    def verifier(info):
        if info.kind == "verify":
            ids = [rid_of(t) for t in info.titles(info.section("Candidate tickets:"))]
            return {"answers": [{"n": k + 1, "belongs": t in yes} for k, t in enumerate(ids)]}
    ctx, fk, gate, acc, _ = reassign_fixture(tmp_path, specs, callback_a=verifier, gate_rule=lambda s, c, m: ACC())
    acc2, st = Cascade(ctx, gate).reassign(acc, "t", "t_regate")
    reqs = fk["a"].of("verify")
    got = sorted(tuple(rid_of(t) for t in i.titles(i.section("Candidate tickets:"))) for i in reqs)
    assert got == sorted([tuple(samp_a), tuple(samp_b)])
    for i in reqs:                                                # exemplars: three central members of the destination
        ex = [rid_of(t) for t in i.titles(i.section("Tickets already on this line:", "Candidate tickets:"))]
        dest = "A" if "Password" in i.section("Report line: ", "\n") else "B"
        assert ex == ctx.exemplars(acc[dest]["members"], 3)
    assert set(band_a) <= set(acc2["A"]["members"])
    assert set(acc2["B"]["members"]) & set(band_b) == set(samp_b[:5])
    assert st["whole_band_admitted"] == 1 and st["asked"] == 16 and st["yes"] == 11


def test_band_admission_needs_four_sampled(tmp_path):
    a_spec, _ = band_specs(3, 0)
    ctx, fk, gate, acc, resid = reassign_fixture(tmp_path, a_spec, callback_a=lambda i: (
        {"answers": [{"n": 1, "belongs": True}, {"n": 2, "belongs": True}, {"n": 3, "belongs": False}]} if i.kind == "verify" else None))
    acc2, st = Cascade(ctx, gate).reassign(acc, "t", "t_regate")
    assert st["whole_band_admitted"] == 0 and len(set(acc2["A"]["members"]) & set(resid)) == 2


def test_growth_regate_and_revert(tmp_path):
    a_spec, b_spec = band_specs(6, 6)
    autos = [(0.90 + 0.001 * i, 0.70) for i in range(5)]          # 5 auto additions to A (old size 20: exactly 25%)
    specs = autos + a_spec + b_spec

    def verifier(info):
        if info.kind == "verify":
            ids = [rid_of(t) for t in info.titles(info.section("Candidate tickets:"))]
            return {"answers": [{"n": k + 1, "belongs": True} for k, _ in enumerate(ids)]}
    # A grows by 5 + 6 = 11 > 5 -> re-gated and ACCEPTed; B grows by 6 > 5 -> re-gated, REJECT -> all reverted
    rule = lambda s, c, m: ACC() if c.startswith("A+") else REJ()   # noqa: E731
    ctx, fk, gate, acc, resid = reassign_fixture(tmp_path, specs, callback_a=verifier, gate_rule=rule)
    acc2, st = Cascade(ctx, gate).reassign(acc, "p3", "p3_regate")
    stage, items = gate.calls[0]
    assert stage == "p3_regate" and sorted(k for k, _ in items) == ["A+p3", "B+p3"]
    assert dict(items)["A+p3"] == ctx.sort(acc["A"]["members"])         # the full expanded cluster is gated
    assert len(acc2["A"]["members"]) == 31 and len(acc2["B"]["members"]) == 20
    assert st["grown"] == 2 and st["reverted"] == 1 and st["added"] == 11


def test_growth_exactly_25_percent_is_not_regated(tmp_path):
    autos = [(0.90 + 0.001 * i, 0.70) for i in range(5)]
    ctx, fk, gate, acc, resid = reassign_fixture(tmp_path, autos)
    acc2, st = Cascade(ctx, gate).reassign(acc, "p3", "p3_regate")
    assert gate.calls == [("p3_regate", [])] and len(acc2["A"]["members"]) == 25 and st["regated"] == 0


def test_reassign_single_cluster_has_no_margin_test(tmp_path):
    ctx, fk, gate, acc, resid = reassign_fixture(tmp_path, [(0.90, 0.75)])
    acc.pop("B")
    acc2, st = Cascade(ctx, gate).reassign(acc, "t", "t_rg")
    assert st["auto"] == 1


# =========================================================================================== P4 merges
def merge_setup(tmp_path, monkeypatch, top_pairs=15, b_says_no=frozenset(), regate=None):
    def judge_b(info):
        if info.kind != "merge":
            return None
        a, b = info.section("Line A: ", "\n"), info.section("\nLine B: ", "\n")
        return {"same_line": not ({a, b} & set(b_says_no))}
    ctx, fk = build(tmp_path, [60, 60, 40, 30], cascade=dict(merge_top_pairs=top_pairs), callback_b=judge_b)
    b = [block(ctx, k) for k in range(4)]
    groups = {0: b[0][:30], 1: b[0][30:], 2: b[1][:20], 3: b[1][20:40], 4: b[1][40:], 5: b[2][:20], 6: b[2][20:], 7: b[3]}
    script_splits(monkeypatch, {})
    no_reassign(monkeypatch)
    gate = FakeGate(table_rule({"cas_s1": lambda c, m: ACC(), "cas_p4_regate": regate or (lambda c, m: ACC(4.4))}))
    casc = Cascade(ctx, gate)
    casc.run(base_labels(ctx, groups))
    return ctx, fk, gate, casc, groups


def test_merges_pairs_disjoint_and_both_judges(tmp_path, monkeypatch):
    ctx, fk, gate, casc, groups = merge_setup(tmp_path, monkeypatch, b_says_no={"name of P5"})
    acc = ctx.load("results", "cascade_B", "clusters.json")
    cids = list(groups)
    C = np.stack([ctx.centroid(groups[k]) for k in cids])
    S = C @ C.T
    pairs = sorted([(S[i, j], f"P{cids[i]}", f"P{cids[j]}") for i in range(len(cids)) for j in range(i + 1, len(cids)) if S[i, j] >= 0.85],
                   reverse=True)
    assert len(pairs) == 5                                        # same-topic pairs only
    assert len(fk["a"].of("merge")) == len(fk["b"].of("merge")) == 5
    # topic 1 has three clusters: only the most similar pair is proposed (disjoint); topic 2 vetoed by judge b
    t1 = [p for p in pairs if {p[1], p[2]} <= {"P2", "P3", "P4"}]
    first = t1[0]
    proposed = {k for k, _ in gate.items("cas_p4_regate")}
    assert proposed == {"P0|P1", f"{first[1]}|{first[2]}"}
    assert casc.log["P4"] == {"pairs_asked": 5, "both_same": 2, "merged": 2}
    assert set(acc[first[1]]["members"]) == set(groups[int(first[1][1:])]) | set(groups[int(first[2][1:])])
    assert first[2] not in acc and acc["P0"]["stage"] == "P4 merge" and "P5" in acc and "P6" in acc


def test_merges_top_pairs_cap_and_regate_veto(tmp_path, monkeypatch):
    ctx, fk, gate, casc, groups = merge_setup(tmp_path, monkeypatch, top_pairs=2,
                                              regate=lambda c, m: ACC() if c == "P0|P1" else REJ())
    assert len(fk["a"].of("merge")) == len(fk["b"].of("merge")) == 2 and casc.log["P4"]["pairs_asked"] == 2
    cids = list(groups)
    C = np.stack([ctx.centroid(groups[k]) for k in cids])
    S = C @ C.T
    top2 = sorted([(S[i, j], f"P{cids[i]}", f"P{cids[j]}") for i in range(len(cids)) for j in range(i + 1, len(cids))], reverse=True)[:2]
    asked = {(info.section("Line A: ", "\n"), info.section("\nLine B: ", "\n")) for info in fk["a"].of("merge")}
    assert asked == {(f"name of {x}", f"name of {y}") for _, x, y in top2}
    proposed = {k for k, _ in gate.items("cas_p4_regate")}
    assert proposed == {f"{x}|{y}" for _, x, y in top2}                  # top pairs are disjoint here
    acc = ctx.load("results", "cascade_B", "clusters.json")
    for _, x, y in top2:                                                  # merge only on re-gate ACCEPT
        assert (y not in acc) == (f"{x}|{y}" == "P0|P1")
    assert casc.log["P4"]["merged"] == int("P0|P1" in proposed)


def test_reference_cascade_thresholds():
    c = PipelineConfig.reference().cascade
    assert (c.split_min_size, c.split_umap_components, c.split_neighbors, c.split_mcs_divisor, c.split_min_child, c.split_cap) == (30, 5, 15, 8, 10, 6)
    assert (c.replace_min_coverage, c.cores_min_parent, c.coassociation, c.core_min_size, c.core_max_jaccard, c.cores_per_parent) == (0.6, 20, 0.8, 10, 0.6, 4)
    assert (c.deep_split_min, c.deep_split_cap, c.candidate_min_size) == (100, 4, 10)
    assert (c.leiden_scan, c.leiden_k, c.leiden_resolution, c.leiden_min_size, c.leiden_max_gated) == (30, 15, 1.0, 10, 40)
    assert (c.prune_sample, c.prune_seed, c.prune_min_precision) == (10, 11, 0.5)
    assert (c.reassign_hi_percentile, c.reassign_lo_percentile, c.reassign_margin, c.band_sample, c.band_seed) == (25, 5, 0.01, 8, 7)
    assert (c.band_admit_min, c.band_admit_fraction, c.growth_regate, c.merge_min_cosine, c.merge_top_pairs) == (4, 0.75, 0.25, 0.85, 15)


# regression test (bug fixed after conformance review)
def test_string_false_is_not_yes(tmp_path):
    from clustering_harness.algorithms.common import verify_membership
    llms, _ = fake_llms(rules_a={"verify": {"answers": [{"n": 1, "belongs": "false"}, {"n": 2, "belongs": "no"}]}})
    ctx = make_ctx(tmp_path, n_topics=2, per_topic=10, llms=llms)
    ans, err = verify_membership(ctx, "t", ctx.llms["a"], "Password reset problems", ctx.eligible_ids[:3], ctx.eligible_ids[3:5])
    assert not err and ans == {ctx.eligible_ids[3]: False, ctx.eligible_ids[4]: False}
