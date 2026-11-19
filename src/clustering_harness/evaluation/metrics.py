"""Clustering quality metrics with established, citable definitions.

Label convention (used across the package)
------------------------------------------
A *partition* is a 1-D array/list with one label per record, in dataset order. Any hashable value can be a
label. *Noise* (unassigned) is ``-1`` (or any negative integer) for integer labels and ``""`` / ``None`` /
``NaN`` for other labels. :func:`normalize_labels` maps a partition to non-negative ints with noise ``-1``.

External metrics compare a predicted partition against gold labels; internal metrics only use the vectors.
Formulas, ranges and references are documented in ``docs/evaluation.md``.
"""
from __future__ import annotations

from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import metrics as skm
from sklearn.metrics.cluster import contingency_matrix

__all__ = [
    "NOISE",
    "NoisePolicy",
    "normalize_labels",
    "apply_noise_policy",
    "purity_scores",
    "bcubed_scores",
    "external_metrics",
    "internal_metrics",
    "contingency_table",
    "cluster_label_report",
]

NOISE = -1
NoisePolicy = Literal["singleton", "cluster", "exclude"]
_POLICIES = ("singleton", "cluster", "exclude")
_NOISE_NAME = "<noise>"


def _is_noise(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, np.integer)):
        return int(x) < 0
    if isinstance(x, (float, np.floating)):
        return bool(np.isnan(x)) or (float(x).is_integer() and x < 0)
    return False


def normalize_labels(labels: Iterable[Any]) -> np.ndarray:
    """Map arbitrary hashable labels to ints ``0..k-1`` (order of first appearance); noise becomes ``-1``.

    Noise is a negative integer, ``None``, ``NaN`` or an empty / whitespace-only string.
    """
    if isinstance(labels, (np.ndarray, pd.Series, pd.Index)) and np.asarray(labels).dtype.kind in "iu":
        a = np.asarray(labels).ravel()
        out = np.full(a.size, NOISE, dtype=np.int64)
        keep = a >= 0
        if keep.any():
            out[keep] = pd.factorize(a[keep], sort=False)[0]
        return out
    arr = list(labels.tolist() if isinstance(labels, (np.ndarray, pd.Series, pd.Index)) else labels)
    out = np.full(len(arr), NOISE, dtype=np.int64)
    if not arr:
        return out
    noise = np.fromiter((_is_noise(x) for x in arr), dtype=bool, count=len(arr))
    keep = np.flatnonzero(~noise)
    if keep.size:
        codes, _ = pd.factorize(pd.Series([arr[i] for i in keep], dtype=object), sort=False)
        out[keep] = codes
    return out


def apply_noise_policy(labels: Iterable[Any], policy: NoisePolicy = "singleton") -> tuple[np.ndarray, np.ndarray]:
    """Resolve noise in a partition according to ``policy``.

    Returns ``(labels, keep)`` where ``labels`` are non-negative ints and ``keep`` is a boolean mask of the
    records that take part in the comparison. ``"singleton"``: every noise record becomes its own cluster;
    ``"cluster"``: all noise records form one extra cluster; ``"exclude"``: noise records are masked out
    (``keep`` is False for them, their label is left at ``-1``).
    """
    if policy not in _POLICIES:
        raise ValueError(f"noise policy must be one of {_POLICIES}, got {policy!r}")
    lab = normalize_labels(labels)
    noise = lab == NOISE
    keep = np.ones(lab.size, dtype=bool)
    if not noise.any():
        return lab, keep
    nxt = int(lab.max()) + 1 if (~noise).any() else 0
    lab = lab.copy()
    if policy == "singleton":
        lab[noise] = nxt + np.arange(int(noise.sum()))
    elif policy == "cluster":
        lab[noise] = nxt
    else:
        keep = ~noise
    return lab, keep


def _contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> sparse.csr_matrix:
    """Sparse classes x clusters count matrix (inputs are non-negative ints)."""
    return contingency_matrix(labels_true, labels_pred, sparse=True).tocsr()


def purity_scores(labels_true: Iterable[Any], labels_pred: Iterable[Any]) -> dict[str, float]:
    """Purity, inverse purity and their harmonic mean (no noise handling: every label is a class/cluster).

    ``purity = (1/n) * sum_clusters max_class n_ij`` (Zhao & Karypis 2001; Manning et al. 2008, sec. 16.3);
    ``inverse_purity`` swaps the roles of classes and clusters (Amigó et al. 2009).
    """
    t, p = normalize_labels(labels_true), normalize_labels(labels_pred)
    t[t < 0], p[p < 0] = t.max() + 1, p.max() + 1  # callers are expected to resolve noise first
    n = t.size
    if n == 0:
        return {"purity": np.nan, "inverse_purity": np.nan, "purity_f": np.nan}
    c = _contingency(t, p)
    pur = float(c.max(axis=0).sum()) / n
    inv = float(c.max(axis=1).sum()) / n
    f = 2 * pur * inv / (pur + inv) if pur + inv > 0 else 0.0
    return {"purity": pur, "inverse_purity": inv, "purity_f": f}


def bcubed_scores(labels_true: Iterable[Any], labels_pred: Iterable[Any]) -> dict[str, float]:
    """BCubed precision, recall and F1 (Bagga & Baldwin 1998; Amigó et al. 2009), computed in O(n + nnz).

    For record ``i`` with cluster ``C(i)`` and class ``L(i)``: precision(i) = |C(i) ∩ L(i)| / |C(i)|,
    recall(i) = |C(i) ∩ L(i)| / |L(i)| (the record itself included). Scores are averaged over records, so
    with contingency counts ``n_ij``: ``P = (1/n) sum_ij n_ij^2 / |C_j|`` and ``R = (1/n) sum_ij n_ij^2 / |L_i|``.
    F1 is the harmonic mean of the averaged P and R (Amigó et al.'s F with alpha = 0.5).
    """
    t, p = normalize_labels(labels_true), normalize_labels(labels_pred)
    t[t < 0], p[p < 0] = t.max() + 1, p.max() + 1
    n = t.size
    if n == 0:
        return {"bcubed_precision": np.nan, "bcubed_recall": np.nan, "bcubed_f1": np.nan}
    c = _contingency(t, p).tocoo()
    class_size = np.asarray(c.sum(axis=1)).ravel()
    cluster_size = np.asarray(c.sum(axis=0)).ravel()
    sq = c.data.astype(np.float64) ** 2
    prec = float((sq / cluster_size[c.col]).sum()) / n
    rec = float((sq / class_size[c.row]).sum()) / n
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"bcubed_precision": prec, "bcubed_recall": rec, "bcubed_f1": f1}


def external_metrics(labels_true: Iterable[Any], labels_pred: Iterable[Any],
                     noise: NoisePolicy = "singleton") -> dict[str, Any]:
    """Compare a predicted partition with gold labels.

    Records whose *gold* label is missing (noise convention) are always dropped (reported as
    ``n_gold_missing``). Predicted noise is resolved with ``noise``:

    - ``"singleton"`` (default, conservative): each noise record is its own cluster;
    - ``"cluster"``: all noise records form one cluster;
    - ``"exclude"``: noise records are dropped from both partitions (``n_evaluated`` reports how many remain).

    Returns ARI, AMI (arithmetic normalisation), NMI (arithmetic), homogeneity, completeness, V-measure,
    Fowlkes-Mallows, purity, inverse purity, purity F, BCubed P/R/F1, plus ``coverage`` (fraction of
    records with a non-noise prediction), ``n_clusters`` (non-noise predicted clusters), ``n_noise``, ``n``
    (records with a gold label), ``n_evaluated``, ``n_gold_missing``, ``n_classes`` and ``noise_policy``.
    """
    t_raw = normalize_labels(labels_true)
    p_raw = normalize_labels(labels_pred)
    if t_raw.size != p_raw.size:
        raise ValueError(f"labels_true and labels_pred differ in length ({t_raw.size} vs {p_raw.size})")
    gold = t_raw != NOISE
    n_gold_missing = int((~gold).sum())
    t_raw, p_raw = t_raw[gold], p_raw[gold]
    n = int(t_raw.size)
    n_noise = int((p_raw == NOISE).sum())
    n_clusters = int(np.unique(p_raw[p_raw != NOISE]).size)
    p, keep = apply_noise_policy(p_raw, noise)
    t, p = t_raw[keep], p[keep]
    out: dict[str, Any] = {}
    if t.size == 0:
        keys = ["ari", "ami", "nmi", "homogeneity", "completeness", "v_measure", "fowlkes_mallows",
                "purity", "inverse_purity", "purity_f", "bcubed_precision", "bcubed_recall", "bcubed_f1"]
        out.update({k: np.nan for k in keys})
    else:
        out["ari"] = float(skm.adjusted_rand_score(t, p))
        out["ami"] = float(skm.adjusted_mutual_info_score(t, p, average_method="arithmetic"))
        out["nmi"] = float(skm.normalized_mutual_info_score(t, p, average_method="arithmetic"))
        h, c, v = skm.homogeneity_completeness_v_measure(t, p)
        out.update(homogeneity=float(h), completeness=float(c), v_measure=float(v))
        out["fowlkes_mallows"] = float(skm.fowlkes_mallows_score(t, p))
        out.update(purity_scores(t, p))
        out.update(bcubed_scores(t, p))
    out.update(coverage=(n - n_noise) / n if n else np.nan, n_clusters=n_clusters, n_noise=n_noise, n=n,
               n_evaluated=int(t.size), n_gold_missing=n_gold_missing,
               n_classes=int(np.unique(t_raw).size), noise_policy=noise)
    return out


def internal_metrics(X: np.ndarray, labels: Iterable[Any], metric: str = "cosine", sample_size: int | None = None,
                     random_state: int | None = 0) -> dict[str, Any]:
    """Label-free quality metrics, computed on non-noise records only.

    - ``silhouette`` (Rousseeuw 1987) with ``metric`` (exact unless ``sample_size`` is given, in which case
      sklearn's random subsample of that size is used with ``random_state``);
    - ``davies_bouldin`` (Davies & Bouldin 1979; lower is better) and ``calinski_harabasz`` (Calinski &
      Harabasz 1974; higher is better). Both are Euclidean by definition; when ``metric="cosine"`` they are
      computed on L2-normalised vectors so that they are consistent with cosine geometry.

    Noise records are excluded before any computation. Metrics are NaN when undefined (fewer than 2 non-noise
    clusters, or as many clusters as non-noise points). Also returns ``coverage``, ``n_clusters``,
    ``n_evaluated`` and ``n``.
    """
    X = np.asarray(X, dtype=np.float64)
    lab = normalize_labels(labels)
    if X.ndim != 2 or X.shape[0] != lab.size:
        raise ValueError("X must be 2-D with one row per label")
    keep = lab != NOISE
    Xk, lk = X[keep], lab[keep]
    k = int(np.unique(lk).size)
    out: dict[str, Any] = {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if 2 <= k <= Xk.shape[0] - 1:
        out["silhouette"] = float(skm.silhouette_score(Xk, lk, metric=metric, sample_size=sample_size,
                                                       random_state=random_state))
        Xe = Xk
        if metric == "cosine":
            norms = np.linalg.norm(Xk, axis=1, keepdims=True)
            Xe = Xk / np.where(norms == 0, 1.0, norms)
        out["davies_bouldin"] = float(skm.davies_bouldin_score(Xe, lk))
        out["calinski_harabasz"] = float(skm.calinski_harabasz_score(Xe, lk))
    n = int(lab.size)
    out.update(coverage=float(keep.mean()) if n else np.nan, n_clusters=k, n_evaluated=int(keep.sum()), n=n,
               metric=metric)
    return out


def _display_labels(labels: Iterable[Any]) -> list:
    arr = list(labels.tolist() if isinstance(labels, (np.ndarray, pd.Series, pd.Index)) else labels)
    return [_NOISE_NAME if _is_noise(x) else x for x in arr]


def contingency_table(labels_true: Iterable[Any], labels_pred: Iterable[Any]) -> pd.DataFrame:
    """Counts of records per (gold label, predicted cluster); noise / missing shown as ``"<noise>"``.

    Rows are gold labels (index name ``"true"``), columns predicted clusters (``"pred"``).
    """
    t, p = _display_labels(labels_true), _display_labels(labels_pred)
    if len(t) != len(p):
        raise ValueError("labels_true and labels_pred differ in length")
    ts = pd.Series(t, name="true", dtype=object)
    ps = pd.Series(p, name="pred", dtype=object)
    return pd.crosstab(ts, ps)


def cluster_label_report(labels_true: Iterable[Any], labels_pred: Iterable[Any],
                         include_noise: bool = True) -> pd.DataFrame:
    """One row per predicted cluster: ``size``, ``majority_label``, ``majority_count``, ``purity``
    (majority_count / size), ``n_labels`` (distinct gold labels; missing gold counts as one) and ``is_noise``.

    Sorted by size (descending). The noise group (if any and ``include_noise``) is reported as a single row
    with ``cluster == "<noise>"``.
    """
    t, p = _display_labels(labels_true), _display_labels(labels_pred)
    if len(t) != len(p):
        raise ValueError("labels_true and labels_pred differ in length")
    df = pd.DataFrame({"cluster": pd.Series(p, dtype=object), "label": pd.Series(t, dtype=object)})
    if not include_noise:
        df = df[df["cluster"] != _NOISE_NAME]
    cols = ["cluster", "size", "majority_label", "majority_count", "purity", "n_labels", "is_noise"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    counts = df.groupby(["cluster", "label"], sort=False).size().rename("count").reset_index()
    counts = counts.sort_values(["cluster", "count"], ascending=[True, False], kind="stable",
                                key=lambda s: s.astype(str) if s.name == "cluster" else s)
    top = counts.drop_duplicates("cluster", keep="first").set_index("cluster")
    size = counts.groupby("cluster", sort=False)["count"].sum()
    nlab = counts.groupby("cluster", sort=False)["label"].nunique()
    rep = pd.DataFrame({"size": size, "majority_label": top["label"], "majority_count": top["count"],
                        "n_labels": nlab})
    rep["purity"] = rep["majority_count"] / rep["size"]
    rep["is_noise"] = rep.index.to_series().eq(_NOISE_NAME).to_numpy()
    rep = rep.rename_axis("cluster").reset_index()
    rep = rep.sort_values(["size"], ascending=False, kind="stable").reset_index(drop=True)
    return rep[cols]
