"""Standard reference clusterers behind one uniform API.

Every baseline is a function ``f(X, random_state=0, **params) -> np.ndarray[int]`` returning one label per row
of ``X`` with noise ``-1``. Output labels are canonical: clusters are numbered ``0..k-1`` by decreasing size
(ties broken by first occurrence), so equal partitions give equal arrays. Rows of ``X`` need not be
normalised; methods that imply cosine geometry L2-normalise internally. Heavy optional libraries (umap,
hdbscan, faiss, igraph, leidenalg) are imported lazily inside the functions that need them.

Example::

    labels = run_baseline("kmeans", X, k=20, random_state=0)
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

__all__ = ["BASELINES", "run_baseline", "list_baselines", "canonicalize_labels", "l2_normalize"]


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows are left as zeros), as float32 C-contiguous array."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.ascontiguousarray(X / np.where(norms == 0, 1.0, norms), dtype=np.float32)


def canonicalize_labels(labels: np.ndarray, min_cluster_size: int = 1) -> np.ndarray:
    """Renumber clusters ``0..k-1`` by decreasing size (ties: first occurrence); clusters smaller than
    ``min_cluster_size`` and negative labels become noise ``-1``."""
    labels = np.asarray(labels).astype(np.int64).ravel()
    out = np.full(labels.size, -1, dtype=np.int64)
    keep = labels >= 0
    if not keep.any():
        return out
    uniq, first, counts = np.unique(labels[keep], return_index=True, return_counts=True)
    order = np.lexsort((first, -counts))
    mapping = np.full(uniq.size, -1, dtype=np.int64)
    nxt = 0
    for o in order:
        if counts[o] >= min_cluster_size:
            mapping[o] = nxt
            nxt += 1
    out[keep] = mapping[np.searchsorted(uniq, labels[keep])]
    return out


def _check_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array (n_samples, n_features)")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    return X


def _need_k(k: int | None, name: str) -> int:
    if k is None:
        raise ValueError(f"{name} requires k (number of clusters)")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    return int(k)


def kmeans(X: np.ndarray, random_state: int = 0, k: int | None = None, n_init: int = 10, max_iter: int = 300,
           normalize: bool = False) -> np.ndarray:
    """Lloyd k-means (scikit-learn ``KMeans``, k-means++ init). ``normalize=True`` L2-normalises rows first
    (approximate spherical k-means, the usual choice for text embeddings)."""
    from sklearn.cluster import KMeans

    k = _need_k(k, "kmeans")
    X = _check_X(X)
    X = l2_normalize(X) if normalize else X
    model = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state)
    return canonicalize_labels(model.fit_predict(X))


def minibatch_kmeans(X: np.ndarray, random_state: int = 0, k: int | None = None, batch_size: int = 1024,
                     n_init: int = 3, max_iter: int = 100, normalize: bool = False) -> np.ndarray:
    """Mini-batch k-means (Sculley 2010; scikit-learn ``MiniBatchKMeans``)."""
    from sklearn.cluster import MiniBatchKMeans

    k = _need_k(k, "minibatch_kmeans")
    X = _check_X(X)
    X = l2_normalize(X) if normalize else X
    model = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, n_init=n_init, max_iter=max_iter,
                            random_state=random_state)
    return canonicalize_labels(model.fit_predict(X))


def agglomerative_ward(X: np.ndarray, random_state: int = 0, k: int | None = None,
                       normalize: bool = False) -> np.ndarray:
    """Ward-linkage agglomerative clustering (Euclidean). Deterministic; ``random_state`` is unused.
    Memory is O(n^2): intended for up to a few tens of thousands of rows."""
    from sklearn.cluster import AgglomerativeClustering

    k = _need_k(k, "agglomerative_ward")
    X = _check_X(X)
    X = l2_normalize(X) if normalize else X
    return canonicalize_labels(AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X))


def agglomerative_cosine_average(X: np.ndarray, random_state: int = 0, k: int | None = None,
                                 distance_threshold: float | None = None) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine distance. Give exactly one of ``k`` or
    ``distance_threshold`` (cosine distance = 1 - cosine similarity, at which merging stops)."""
    from ._compat import agglomerative

    if (k is None) == (distance_threshold is None):
        raise ValueError("agglomerative_cosine_average requires exactly one of k or distance_threshold")
    X = l2_normalize(_check_X(X))
    model = agglomerative(None if k is None else int(k), metric="cosine", distance_threshold=distance_threshold,
                          linkage="average")
    return canonicalize_labels(model.fit_predict(X))


def _hdbscan_labels(Z: np.ndarray, min_cluster_size: int, min_samples: int | None, method: str,
                    epsilon: float) -> np.ndarray:
    import hdbscan as hdbscan_lib

    model = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
                                cluster_selection_method=method, cluster_selection_epsilon=epsilon)
    return model.fit_predict(np.asarray(Z, dtype=np.float64))


def hdbscan(X: np.ndarray, random_state: int = 0, min_cluster_size: int = 10, min_samples: int | None = None,
            cluster_selection_method: str = "eom", cluster_selection_epsilon: float = 0.0) -> np.ndarray:
    """HDBSCAN (Campello et al. 2013; McInnes et al. 2017) on the raw vectors with cosine geometry: rows are
    L2-normalised and Euclidean distance is used (monotone in cosine distance: ||a-b||^2 = 2(1-cos)).
    Deterministic; ``random_state`` is unused."""
    X = l2_normalize(_check_X(X))
    return canonicalize_labels(_hdbscan_labels(X, min_cluster_size, min_samples, cluster_selection_method,
                                               cluster_selection_epsilon))


def umap_hdbscan(X: np.ndarray, random_state: int = 0, n_components: int = 5, n_neighbors: int = 15,
                 min_dist: float = 0.0, min_cluster_size: int = 10, min_samples: int | None = None,
                 cluster_selection_method: str = "eom") -> np.ndarray:
    """UMAP (cosine metric, ``min_dist=0``) to ``n_components`` dims, then HDBSCAN (Euclidean, EOM).

    This is the BERTopic-style pipeline. A fixed ``random_state`` makes UMAP deterministic (and single
    threaded)."""
    import umap

    X = l2_normalize(_check_X(X))
    n_neighbors = max(2, min(n_neighbors, X.shape[0] - 1))
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
                        random_state=random_state, transform_seed=random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*overridden to 1 by setting random_state.*")
        Z = reducer.fit_transform(X)
    return canonicalize_labels(_hdbscan_labels(Z, min_cluster_size, min_samples, cluster_selection_method, 0.0))


def leiden_knn(X: np.ndarray, random_state: int = 0, n_neighbors: int = 15, resolution: float = 1.0,
               min_similarity: float = 0.0, min_cluster_size: int = 2, n_iterations: int = -1) -> np.ndarray:
    """Leiden community detection (Traag et al. 2019) on a mutual k-nearest-neighbour cosine graph.

    Rows are L2-normalised, exact neighbours come from ``faiss.IndexFlatIP``; an undirected edge (i, j) is
    kept when each is among the other's ``n_neighbors`` nearest neighbours and their cosine similarity is
    > ``min_similarity``; its weight is the similarity. Partitioning uses ``RBConfigurationVertexPartition``
    with ``resolution`` and ``seed=random_state``. Communities smaller than ``min_cluster_size`` (including
    vertices isolated by the mutual-kNN rule) are returned as noise.
    """
    import faiss
    import igraph as ig
    import leidenalg

    X = l2_normalize(_check_X(X))
    n = X.shape[0]
    if n == 1:
        return np.full(1, -1 if min_cluster_size > 1 else 0, dtype=np.int64)
    kk = min(n_neighbors, n - 1)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    sims, nbrs = index.search(X, kk + 1)
    rows = np.repeat(np.arange(n), kk + 1)
    cols, vals = nbrs.ravel(), sims.ravel()
    ok = (cols >= 0) & (cols != rows)
    rows, cols, vals = rows[ok], cols[ok], vals[ok]
    # Keep at most kk neighbours per row (self may be missing from the result when duplicates exist).
    order = np.lexsort((-vals, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    rank = np.arange(rows.size) - np.searchsorted(rows, rows)
    sel = rank < kk
    rows, cols, vals = rows[sel], cols[sel], vals[sel]
    directed = set(zip(rows.tolist(), cols.tolist()))
    edges, weights = [], []
    for i, j, w in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        if i < j and (j, i) in directed and w > min_similarity:
            edges.append((i, j))
            weights.append(float(w))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, n_iterations=n_iterations, seed=random_state)
    return canonicalize_labels(np.asarray(part.membership), min_cluster_size=min_cluster_size)


def gmm(X: np.ndarray, random_state: int = 0, k: int | None = None, covariance_type: str = "diag",
        n_init: int = 1, max_iter: int = 200, normalize: bool = False, reg_covar: float = 1e-6) -> np.ndarray:
    """Gaussian mixture model (EM, scikit-learn ``GaussianMixture``), hard assignment by max posterior.
    ``covariance_type`` defaults to ``"diag"`` because ``"full"`` is impractical for high-dimensional
    embeddings."""
    from sklearn.mixture import GaussianMixture

    k = _need_k(k, "gmm")
    X = _check_X(X)
    X = l2_normalize(X).astype(np.float64) if normalize else X
    model = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=n_init, max_iter=max_iter,
                            reg_covar=reg_covar, random_state=random_state)
    return canonicalize_labels(model.fit_predict(X))


BASELINES: dict[str, Callable[..., np.ndarray]] = {
    "kmeans": kmeans,
    "minibatch_kmeans": minibatch_kmeans,
    "agglomerative_ward": agglomerative_ward,
    "agglomerative_cosine_average": agglomerative_cosine_average,
    "hdbscan": hdbscan,
    "umap_hdbscan": umap_hdbscan,
    "leiden_knn": leiden_knn,
    "gmm": gmm,
}


def list_baselines() -> list[str]:
    """Names accepted by :func:`run_baseline`."""
    return sorted(BASELINES)


def run_baseline(name: str, X: np.ndarray, random_state: int = 0, **params: Any) -> np.ndarray:
    """Run baseline ``name`` on ``X`` (n_samples x n_features) and return int labels with noise ``-1``."""
    try:
        fn = BASELINES[name]
    except KeyError:
        raise ValueError(f"unknown baseline {name!r}; available: {list_baselines()}") from None
    return fn(X, random_state=random_state, **params)
