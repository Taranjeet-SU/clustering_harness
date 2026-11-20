"""Consensus bank (pipeline doc step 3) and fixed base partition (step 4).

The bank holds alternative partitions of the eligible (non-junk) records:

* UMAP (cosine, ``min_dist=0``, exact neighbours) + HDBSCAN: for each (n_neighbors, seed) one UMAP; for each
  ``min_samples`` one HDBSCAN condensed tree; labels extracted for every (selection, min_cluster_size).
* MiniBatchKMeans on the raw vectors for each (k, seed).
* PCA + HDBSCAN for each (min_samples, selection, min_cluster_size).
* Leiden (RBConfiguration) on a mutual-kNN cosine graph for each (k, resolution, seed).

Every partition: clusters smaller than ``bank.min_cluster`` become noise (-1); labels are re-indexed by
descending size. The base partition is the UMAP+HDBSCAN member with the configured settings."""
from __future__ import annotations

import itertools
import json
import warnings

import numpy as np
import pandas as pd

import logging

from ..context import RunContext

log = logging.getLogger("clustering_harness")

__all__ = ["build_bank", "knn_graph", "bank_eligible", "postprocess_labels"]


def postprocess_labels(lab, min_cluster: int) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.int32).copy()
    u, ct = np.unique(lab[lab >= 0], return_counts=True)
    for c in u[ct < min_cluster]:
        lab[lab == c] = -1
    u, ct = np.unique(lab[lab >= 0], return_counts=True)
    mapping = {c: i for i, c in enumerate(u[np.argsort(-ct, kind="stable")])}
    return np.array([mapping.get(c, -1) for c in lab], np.int32)


def knn_graph(ctx: RunContext):
    """Exact cosine top-``bank.knn`` neighbours of every eligible record (faiss inner product). Cached."""
    p = ctx.path("cache", "geom", "knn.npz")
    if p.exists():
        z = np.load(p)
        return z["I"], z["D"]
    import faiss
    X = np.ascontiguousarray(ctx.X[ctx.nonjunk])
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    D, I = idx.search(X, min(ctx.cfg.bank.knn, len(X)))
    np.savez(p, I=I, D=D)
    return I, D


def build_bank(ctx: RunContext) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (or load) all bank partitions; returns (grid table, base labels over eligible records)."""
    import hdbscan
    import umap
    from hdbscan import hdbscan_ as H
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    B = ctx.cfg.bank
    sel = ctx.cfg.base
    X = np.ascontiguousarray(ctx.X[ctx.nonjunk])
    N = len(X)
    I, D = knn_graph(ctx)
    pdir = ctx.path("cache", "partitions", "x").parent
    records = []

    def record(rid, lab, params, seed, family):
        p = pdir / f"{rid}.npy"
        lab = postprocess_labels(lab, B.min_cluster)
        np.save(p, lab)
        u, ct = np.unique(lab[lab >= 0], return_counts=True)
        records.append(dict(run_id=rid, view="lg", seed=seed, family=family, params=json.dumps(params), n_clusters=len(u),
                            largest_frac=float(ct.max() / N) if len(ct) else 1.0, noise_frac=float(np.mean(lab < 0))))
        return lab

    def cached(rid):
        p = pdir / f"{rid}.npy"
        return np.load(p) if p.exists() else None

    primary = None
    for nn, seed in itertools.product(B.umap_neighbors, B.seeds):
        gp = ctx.path("cache", "geom", f"umap_nn{nn}_s{seed}.npy")
        Z = None
        for ms in B.hdbscan_min_samples:
            rids = [(s_, mcs, f"umap_hdbscan_{s_}__lg__nn{nn}_ms{ms}_sel{s_}_mcs{mcs}__s{seed}")
                    for s_, mcs in itertools.product(B.selections, B.min_cluster_sizes)]
            tree = None
            for s_, mcs, rid in rids:
                lab = cached(rid)
                if lab is None:
                    if tree is None:
                        if Z is None:
                            if gp.exists():
                                Z = np.load(gp)
                            else:
                                with warnings.catch_warnings():
                                    # expected: we pass exact neighbours (no NNDescent index) and a fixed seed
                                    warnings.filterwarnings("ignore", message=".*precomputed_knn.*")
                                    warnings.filterwarnings("ignore", message=".*overridden to 1 by setting random_state.*")
                                    Z = umap.UMAP(n_components=B.umap_components, n_neighbors=nn, min_dist=0.0, metric="cosine",
                                                  random_state=seed,
                                                  precomputed_knn=(I[:, :nn].copy(), np.maximum(0, 1 - D[:, :nn]).copy(), None)
                                                  ).fit_transform(X)
                                np.save(gp, Z)
                        tree = hdbscan.HDBSCAN(min_cluster_size=B.tree_min_cluster_size, min_samples=ms,
                                               core_dist_n_jobs=B.workers).fit(Z).single_linkage_tree_.to_numpy()
                    lab = H._tree_to_labels(None, tree, mcs, s_)[0]
                lab = record(rid, lab, dict(nn=nn, ms=ms, sel=s_, mcs=mcs), seed, "umap_hdbscan_" + s_)
                if (nn, seed, ms, s_, mcs) == (sel.umap_neighbors, sel.seed, sel.min_samples, sel.selection, sel.min_cluster_size):
                    primary = (rid, lab)
    for k, seed in itertools.product(B.kmeans_k, B.seeds):
        rid = f"kmeans_raw__lg__k{k}__s{seed}"
        if k >= N:
            log.warning("bank: skipping k-means k=%d (only %d eligible records)", k, N)
            continue
        lab = cached(rid)
        if lab is None:
            lab = MiniBatchKMeans(k, batch_size=B.kmeans_batch_size, n_init=B.kmeans_n_init, random_state=seed).fit_predict(X)
        record(rid, lab, {"k": k}, seed, "kmeans_raw")
    Zp = None
    for ms, s_, mcs in itertools.product(B.hdbscan_min_samples, B.selections, B.min_cluster_sizes):
        rid = f"hdbscan_pca__lg__ms{ms}_{s_}_mcs{mcs}__s0"
        lab = cached(rid)
        if lab is None:
            if Zp is None:
                Zp = PCA(min(B.pca_components, X.shape[1], N), random_state=0).fit_transform(X)
            lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=s_,
                                  core_dist_n_jobs=B.workers).fit_predict(Zp)
        record(rid, lab, dict(ms=ms, sel=s_, mcs=mcs), 0, "hdbscan_pca")
    import igraph as ig
    import leidenalg as la
    for k in B.leiden_k:
        graph = None
        for resolution, seed in itertools.product(B.leiden_resolutions, B.seeds):
            rid = f"leiden_mknn__lg__k{k}_res{resolution}__s{seed}"
            lab = cached(rid)
            if lab is None:
                if graph is None:
                    nb = [set(r[1:k + 1]) for r in I]
                    edges = [(i, int(j)) for i, ns in enumerate(nb) for j in ns if i < j and i in nb[j]]
                    graph = ig.Graph(n=N, edges=edges)
                    graph.es["weight"] = [max(1e-8, float(X[i] @ X[j])) for i, j in edges]
                lab = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                                        resolution_parameter=resolution, seed=seed).membership
            record(rid, lab, dict(k=k, res=resolution), seed, "leiden_mknn")
    if primary is None:
        raise ValueError("base selection does not match any bank member; check config.base against config.bank")
    grid = pd.DataFrame(records)
    grid.to_csv(ctx.path("results", "bank", "grid_runs.csv"), index=False)
    return grid, primary


def bank_eligible(ctx: RunContext, grid: pd.DataFrame) -> pd.DataFrame:
    """Bank members inside the hard constraints (used for consensus cores)."""
    B = ctx.cfg.bank
    lo, hi = B.n_clusters_range
    return grid[(grid.view == "lg") & grid.n_clusters.between(lo, hi) & (grid.largest_frac <= B.largest_frac_max)
                & (grid.noise_frac <= B.noise_frac_max)]
