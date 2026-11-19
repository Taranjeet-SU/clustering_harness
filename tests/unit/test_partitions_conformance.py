"""Doc steps 3-4: consensus bank (151 partitions on the reference grid), label post-processing, base selection
and the hard-constraint filter used by the consensus cores."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import make_ctx, make_topic_dataset  # noqa: E402

from clustering_harness.algorithms import partitions as parts  # noqa: E402
from clustering_harness.algorithms.partitions import bank_eligible, build_bank, knn_graph, postprocess_labels  # noqa: E402
from clustering_harness.config import PipelineConfig  # noqa: E402


# ------------------------------------------------------------------------------------------- post-processing
def test_small_labels_become_noise_and_reindex_by_size():
    # doc step 3: labels with fewer than ten members -> -1; the rest re-indexed in descending size order
    lab = [7] * 9 + [3] * 12 + [5] * 30 + [-1] * 4 + [9] * 10
    out = postprocess_labels(lab, 10)
    assert (out[:9] == -1).all()
    assert (out[9:21] == 1).all() and (out[21:51] == 0).all() and (out[51:55] == -1).all() and (out[55:] == 2).all()


def test_postprocess_ties_and_empty():
    out = postprocess_labels([4] * 10 + [2] * 10, 10)
    assert out.tolist() == [1] * 10 + [0] * 10                 # ties broken by original label (stable)
    assert postprocess_labels([0, 1, 2], 10).tolist() == [-1, -1, -1]
    assert postprocess_labels(np.array([], int), 10).tolist() == []


def test_postprocess_is_idempotent():
    rng = np.random.default_rng(0)
    lab = rng.integers(-1, 30, 500)
    once = postprocess_labels(lab, 10)
    assert (postprocess_labels(once, 10) == once).all()
    sizes = np.bincount(once[once >= 0])
    assert (sizes >= 10).all() and (np.diff(sizes) <= 0).all()


# ------------------------------------------------------------------------------------------- knn
def test_knn_is_exact_cosine_with_self_first(tmp_path):
    ctx = make_ctx(tmp_path, n_topics=4, per_topic=30, n_junk=5)
    I, D = knn_graph(ctx)
    X = ctx.X[ctx.nonjunk]
    assert I.shape == (len(X), 51)                              # doc step 3: 51 entries per record
    S = X @ X.T
    assert (I[:, 0] == np.arange(len(X))).all()
    np.testing.assert_allclose(D, np.take_along_axis(S, I, 1), atol=1e-5)
    np.testing.assert_allclose(D[:, :10], -np.sort(-S, 1)[:, :10], atol=1e-5)
    I2, _ = knn_graph(ctx)                                       # cached
    assert (I2 == I).all()


# ------------------------------------------------------------------------------------------- reference grid (stubbed)
class Recorder:
    def __init__(self):
        self.calls = collections.defaultdict(list)


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the heavy algorithms by fast stubs that record their parameters."""
    import hdbscan
    import leidenalg
    import sklearn.cluster
    import sklearn.decomposition
    import umap
    from hdbscan import hdbscan_ as H
    rec = Recorder()

    def fake_labels(n, k, salt):
        return (np.arange(n) * 7 + salt) % k

    class UMAP:
        def __init__(self, **kw):
            rec.calls["umap"].append(kw)

        def fit_transform(self, X):
            return X[:, :10].copy()

    class HDBSCAN:
        def __init__(self, **kw):
            self.kw = kw
            rec.calls["hdbscan"].append(kw)

        def fit(self, Z):
            self.single_linkage_tree_ = SimpleNamespace(to_numpy=lambda: ("tree", len(Z), self.kw["min_samples"]))
            return self

        def fit_predict(self, Z):
            return fake_labels(len(Z), 12, self.kw["min_cluster_size"])

    def tree_to_labels(_, tree, mcs, sel):
        rec.calls["tree_to_labels"].append((tree[2], mcs, sel))
        return (fake_labels(tree[1], 15, mcs + (sel == "leaf")), None, None)

    class MBK:
        def __init__(self, k, **kw):
            rec.calls["kmeans"].append(dict(k=k, **kw))
            self.k = k

        def fit_predict(self, X):
            return fake_labels(len(X), min(self.k, 40), 1)

    class PCA:
        def __init__(self, n, **kw):
            rec.calls["pca"].append(dict(n=n, **kw))

        def fit_transform(self, X):
            return X[:, :20]

    def find_partition(graph, cls, weights=None, resolution_parameter=None, seed=None):
        rec.calls["leiden"].append(dict(cls=cls, resolution=resolution_parameter, seed=seed, weights=list(graph.es[weights]),
                                        n=graph.vcount()))
        return SimpleNamespace(membership=fake_labels(graph.vcount(), 20, seed).tolist())

    monkeypatch.setattr(umap, "UMAP", UMAP)
    monkeypatch.setattr(hdbscan, "HDBSCAN", HDBSCAN)
    monkeypatch.setattr(H, "_tree_to_labels", tree_to_labels)
    monkeypatch.setattr(sklearn.cluster, "MiniBatchKMeans", MBK)
    monkeypatch.setattr(sklearn.decomposition, "PCA", PCA)
    monkeypatch.setattr(leidenalg, "find_partition", find_partition)
    return rec


@pytest.fixture
def ref_ctx(tmp_path):
    # > 400 eligible records so that every k-means k of the reference grid is feasible
    ds, X = make_topic_dataset(n_topics=8, per_topic=60, dim=24, n_junk=6)
    return make_ctx(tmp_path, ds=ds, X=X, cfg=PipelineConfig.reference())


def test_reference_grid_has_151_partitions(stubbed, ref_ctx):
    grid, (rid, base) = build_bank(ref_ctx)
    fam = grid.family.value_counts().to_dict()
    # doc step 3: 151 partitions = 48 eom + 48 leaf + 15 k-means + 16 PCA-density + 24 Leiden
    assert fam == {"umap_hdbscan_eom": 48, "umap_hdbscan_leaf": 48, "kmeans_raw": 15, "hdbscan_pca": 16, "leiden_mknn": 24}
    assert len(grid) == 151 and grid.run_id.is_unique and (grid.view == "lg").all()
    saved = pd.read_csv(ref_ctx.workdir / "results" / "bank" / "grid_runs.csv")
    assert len(saved) == 151
    for r in grid.run_id:
        lab = np.load(ref_ctx.workdir / "cache" / "partitions" / f"{r}.npy")
        assert len(lab) == len(ref_ctx.nonjunk)                  # junk excluded from the bank
        assert (lab == postprocess_labels(lab, 10)).all()


def test_reference_grid_parameters(stubbed, ref_ctx):
    build_bank(ref_ctx)
    c = stubbed.calls
    # UMAP: 10 components, cosine, min_dist 0, neighbours {15, 30} x seeds {0, 1, 2}, precomputed neighbours
    assert len(c["umap"]) == 6
    assert {(k["n_neighbors"], k["random_state"]) for k in c["umap"]} == {(n, s) for n in (15, 30) for s in (0, 1, 2)}
    for k in c["umap"]:
        assert (k["n_components"], k["metric"], k["min_dist"]) == (10, "cosine", 0.0)
        knn_i, knn_d, _ = k["precomputed_knn"]
        assert knn_i.shape[1] == k["n_neighbors"] and (knn_d >= 0).all()
    # one HDBSCAN tree per (UMAP, min_samples), at min_cluster_size 25
    trees = [k for k in c["hdbscan"] if "cluster_selection_method" not in k]
    assert len(trees) == 12 and {k["min_cluster_size"] for k in trees} == {25} and {k["min_samples"] for k in trees} == {5, 15}
    assert collections.Counter((ms, mcs, sel) for ms, mcs, sel in c["tree_to_labels"]) == \
        {(ms, mcs, sel): 6 for ms in (5, 15) for mcs in (25, 50, 100, 200) for sel in ("eom", "leaf")}
    # MiniBatchKMeans on raw vectors
    assert sorted((k["k"], k["random_state"]) for k in c["kmeans"]) == [(k, s) for k in (100, 150, 200, 300, 400) for s in (0, 1, 2)]
    assert all(k["batch_size"] == 4096 and k["n_init"] == 3 for k in c["kmeans"])
    # PCA(20, random_state=0) + HDBSCAN
    assert c["pca"] == [dict(n=20, random_state=0)]
    pca_h = [k for k in c["hdbscan"] if "cluster_selection_method" in k]
    assert sorted((k["min_samples"], k["cluster_selection_method"], k["min_cluster_size"]) for k in pca_h) == \
        sorted((ms, s, m) for ms in (5, 15) for s in ("eom", "leaf") for m in (25, 50, 100, 200))
    # Leiden: RBConfiguration, resolutions x seeds per k, weights max(1e-8, cosine)
    import leidenalg
    assert len(c["leiden"]) == 24
    assert {(k["resolution"], k["seed"]) for k in c["leiden"]} == {(r, s) for r in (0.25, 0.5, 1.0, 2.0) for s in (0, 1, 2)}
    assert all(k["cls"] is leidenalg.RBConfigurationVertexPartition for k in c["leiden"])
    assert all(min(k["weights"]) >= 1e-8 for k in c["leiden"])


def test_leiden_graph_is_mutual_knn(stubbed, tmp_path):
    ds, X = make_topic_dataset(n_topics=3, per_topic=30, dim=16)
    cfg = PipelineConfig.reference()
    cfg.bank = cfg.bank.model_copy(update=dict(umap_neighbors=[30], seeds=[0], hdbscan_min_samples=[15], selections=["eom"],
                                               min_cluster_sizes=[50], kmeans_k=[], leiden_k=[15], leiden_resolutions=[1.0]))
    ctx = make_ctx(tmp_path, ds=ds, X=X, cfg=cfg)
    import igraph
    graphs = []
    orig = igraph.Graph

    class G(orig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            graphs.append(self)
    import unittest.mock as um
    with um.patch.object(igraph, "Graph", G):
        build_bank(ctx)
    I, _ = knn_graph(ctx)
    nb = [set(r[1:16]) for r in I]
    want = {(i, j) for i in range(len(nb)) for j in nb[i] if i < j and i in nb[j]}
    g = graphs[0]
    assert {tuple(sorted(e.tuple)) for e in g.es} == want
    Xn = ctx.X[ctx.nonjunk]
    for e in list(g.es)[:50]:
        i, j = e.tuple
        assert e["weight"] == pytest.approx(max(1e-8, float(Xn[i] @ Xn[j])), abs=1e-6)


def test_base_selection_matches_doc(stubbed, ref_ctx):
    grid, (rid, base) = build_bank(ref_ctx)
    # doc step 4: (nn, seed, min_samples, selection, min_cluster_size) = (30, 0, 15, eom, 50)
    assert rid == "umap_hdbscan_eom__lg__nn30_ms15_seleom_mcs50__s0"
    assert (base == np.load(ref_ctx.workdir / "cache" / "partitions" / f"{rid}.npy")).all()
    row = grid.set_index("run_id").loc[rid]
    assert json.loads(row.params) == {"nn": 30, "ms": 15, "sel": "eom", "mcs": 50} and row.seed == 0


def test_existing_partitions_are_reused(stubbed, ref_ctx):
    build_bank(ref_ctx)
    n = {k: len(v) for k, v in stubbed.calls.items()}
    grid2, _ = build_bank(ref_ctx)
    assert {k: len(v) for k, v in stubbed.calls.items()} == n      # nothing recomputed
    assert len(grid2) == 151


def test_base_selection_must_exist(stubbed, tmp_path):
    cfg = PipelineConfig.reference()
    cfg.base = cfg.base.model_copy(update=dict(min_cluster_size=77))
    ds, X = make_topic_dataset(n_topics=3, per_topic=30, dim=16)
    with pytest.raises(ValueError):
        build_bank(make_ctx(tmp_path, ds=ds, X=X, cfg=cfg))


# ------------------------------------------------------------------------------------------- hard constraints
def test_bank_eligible_hard_constraints(tmp_path):
    ctx = make_ctx(tmp_path, n_topics=2, per_topic=10, cfg=PipelineConfig.reference())
    rows = [("ok_lo", "lg", 60, 0.12, 0.40), ("ok_hi", "lg", 400, 0.05, 0.0), ("few", "lg", 59, 0.05, 0.1),
            ("many", "lg", 401, 0.05, 0.1), ("big", "lg", 100, 0.1201, 0.1), ("noisy", "lg", 100, 0.05, 0.4001),
            ("other_view", "sm", 100, 0.05, 0.1)]
    grid = pd.DataFrame(rows, columns=["run_id", "view", "n_clusters", "largest_frac", "noise_frac"])
    # doc step 8: 60-400 clusters, largest-cluster fraction <= 0.12, noise fraction <= 0.40, GTE-large view
    assert bank_eligible(ctx, grid).run_id.tolist() == ["ok_lo", "ok_hi"]


def test_grid_statistics_are_correct(stubbed, ref_ctx):
    grid, _ = build_bank(ref_ctx)
    N = len(ref_ctx.nonjunk)
    for _, r in grid.sample(20, random_state=0).iterrows():
        lab = np.load(ref_ctx.workdir / "cache" / "partitions" / f"{r.run_id}.npy")
        assert r.n_clusters == len(set(lab[lab >= 0]))
        assert r.noise_frac == pytest.approx(np.mean(lab < 0))
        assert r.largest_frac == pytest.approx(np.bincount(lab[lab >= 0]).max() / N)


# ------------------------------------------------------------------------------------------- real tiny build
def test_real_tiny_bank_build(tmp_path):
    ds, X = make_topic_dataset(n_topics=5, per_topic=40, dim=32, n_junk=4, seed=3)
    cfg = PipelineConfig.small()
    cfg.bank = cfg.bank.model_copy(update=dict(umap_neighbors=[10], seeds=[0], hdbscan_min_samples=[5], tree_min_cluster_size=6,
                                               min_cluster_sizes=[10, 20], kmeans_k=[4, 6], leiden_k=[10], leiden_resolutions=[1.0],
                                               min_cluster=5, workers=1))
    cfg.base = cfg.base.model_copy(update=dict(umap_neighbors=10, seed=0, min_samples=5, selection="eom", min_cluster_size=10))
    ctx = make_ctx(tmp_path, ds=ds, X=X, cfg=cfg)
    grid, (rid, base) = build_bank(ctx)
    assert grid.family.value_counts().to_dict() == {"umap_hdbscan_eom": 2, "umap_hdbscan_leaf": 2, "kmeans_raw": 2,
                                                    "hdbscan_pca": 4, "leiden_mknn": 1}
    from sklearn.metrics import adjusted_rand_score
    y = np.array(ds.labels)[ctx.nonjunk]
    m = base >= 0
    assert m.mean() > 0.8 and adjusted_rand_score(y[m], base[m]) > 0.9
    for r in grid.run_id:
        lab = np.load(ctx.workdir / "cache" / "partitions" / f"{r}.npy")
        sizes = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.array([])
        assert (sizes >= 5).all() and (np.diff(sizes) <= 0).all()


def test_kmeans_k_not_below_n_is_skipped_silently(stubbed, tmp_path):
    """Fragility: on data with fewer records than a grid k, the k-means member is silently dropped (the doc grid
    always has 15). Documented here so a change in that behaviour is noticed."""
    ds, X = make_topic_dataset(n_topics=3, per_topic=40, dim=24)
    grid, _ = build_bank(make_ctx(tmp_path, ds=ds, X=X, cfg=PipelineConfig.reference()))
    # 120 eligible records: only k=100 (x 3 seeds) survives; k in {150, 200, 300, 400} vanish without a log entry
    assert (grid.family == "kmeans_raw").sum() == 3 and len(grid) == 139
