"""Agreement between several clusterings of the same records (run-to-run stability).

All partitions follow the package label convention (one label per record, noise ``-1`` / ``""`` / ``None``)
and must have the same length and record order. Definitions are documented in ``docs/evaluation.md``.
"""
from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import metrics as skm

from .metrics import NOISE, NoisePolicy, apply_noise_policy, normalize_labels

__all__ = ["pairwise_agreement", "agreement_summary", "coassignment_matrix", "consensus_stability"]

_AGREEMENT = {
    "ari": skm.adjusted_rand_score,
    "ami": lambda a, b: skm.adjusted_mutual_info_score(a, b, average_method="arithmetic"),
    "nmi": lambda a, b: skm.normalized_mutual_info_score(a, b, average_method="arithmetic"),
}


def _as_int_partitions(partitions: Sequence[Any]) -> list[np.ndarray]:
    parts = [normalize_labels(p) for p in partitions]
    if not parts:
        raise ValueError("at least one partition is required")
    n = parts[0].size
    if any(p.size != n for p in parts):
        raise ValueError("all partitions must have the same length")
    return parts


def _pair_score(a: np.ndarray, b: np.ndarray, metric: str, noise: NoisePolicy) -> float:
    la, ka = apply_noise_policy(a, noise)
    lb, kb = apply_noise_policy(b, noise)
    keep = ka & kb
    if not keep.any():
        return float("nan")
    return float(_AGREEMENT[metric](la[keep], lb[keep]))


def pairwise_agreement(partitions: Sequence[Any], names: Sequence[str] | None = None,
                       metric: Literal["ari", "ami", "nmi"] = "ari", noise: NoisePolicy = "singleton") -> pd.DataFrame:
    """Symmetric matrix of agreement scores between every pair of partitions.

    ``noise`` is applied to *both* partitions of a pair (see :func:`~.metrics.apply_noise_policy`); with
    ``"exclude"`` only records that are non-noise in both partitions are compared. The diagonal is the
    self-agreement (1.0 unless undefined).
    """
    if metric not in _AGREEMENT:
        raise ValueError(f"metric must be one of {sorted(_AGREEMENT)}, got {metric!r}")
    parts = _as_int_partitions(partitions)
    r = len(parts)
    names = list(names) if names is not None else [f"run{i}" for i in range(r)]
    if len(names) != r:
        raise ValueError("names must have one entry per partition")
    m = np.full((r, r), np.nan)
    for i in range(r):
        m[i, i] = _pair_score(parts[i], parts[i], metric, noise)
        for j in range(i + 1, r):
            m[i, j] = m[j, i] = _pair_score(parts[i], parts[j], metric, noise)
    return pd.DataFrame(m, index=names, columns=names)


def agreement_summary(matrix: pd.DataFrame | np.ndarray) -> dict[str, float]:
    """Mean, std, min and max of the off-diagonal entries of an agreement matrix (NaNs ignored)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    r = m.shape[0]
    off = m[~np.eye(r, dtype=bool)]
    off = off[~np.isnan(off)]
    if off.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n_runs": r, "n_pairs": 0}
    return {"mean": float(off.mean()), "std": float(off.std()), "min": float(off.min()), "max": float(off.max()),
            "n_runs": r, "n_pairs": int(off.size // 2)}


def _membership(labels: np.ndarray) -> sparse.csr_matrix:
    """N x k 0/1 indicator matrix of non-noise cluster membership (noise rows are empty)."""
    keep = np.flatnonzero(labels != NOISE)
    k = int(labels.max()) + 1 if keep.size else 0
    return sparse.csr_matrix((np.ones(keep.size), (keep, labels[keep])), shape=(labels.size, max(k, 1)))


def coassignment_matrix(partitions: Sequence[Any], dense: bool = False) -> sparse.csr_matrix | np.ndarray:
    """N x N matrix whose entry (i, j) is the fraction of runs in which records i and j share a cluster.

    Noise never co-clusters: a record that is noise in a run contributes nothing for that run, including on
    the diagonal, so ``C[i, i]`` is the fraction of runs in which record ``i`` was assigned to a cluster.
    This is the consensus matrix of Monti et al. (2003) with every record present in every run.
    Returns a ``scipy.sparse.csr_matrix`` (default) or a dense ``float64`` array if ``dense=True``.
    Memory is proportional to the number of co-clustered pairs (sum of squared cluster sizes).
    """
    parts = _as_int_partitions(partitions)
    n = parts[0].size
    acc = sparse.csr_matrix((n, n), dtype=np.float64)
    for p in parts:
        m = _membership(p)
        acc = acc + (m @ m.T)
    acc = (acc / len(parts)).tocsr()
    acc.sum_duplicates()
    acc.eliminate_zeros()
    return acc.toarray() if dense else acc


def consensus_stability(partitions: Sequence[Any]) -> np.ndarray:
    """Per-record stability: mean co-assignment of a record with its co-members, averaged over runs.

    Let ``C`` be :func:`coassignment_matrix` over the R runs. In run r, record i (if not noise) has the
    co-members ``S_r(i)`` = the other records in its cluster. Its agreement in run r is
    ``a_r(i) = mean_{j in S_r(i)} C[i, j]``, i.e. how often, across all runs, i is grouped with the records it
    is grouped with in run r (the *item consensus* of Monti et al. 2003, evaluated per run). The stability of
    record i is the mean of ``a_r(i)`` over the runs where i is non-noise and has at least one co-member.

    Returns a float array of length N in [0, 1]; NaN for records that are never in a cluster of size >= 2.
    1.0 means the record always lands with exactly the same co-members.
    """
    parts = _as_int_partitions(partitions)
    n = parts[0].size
    c = coassignment_matrix(parts)
    diag = c.diagonal()
    total = np.zeros(n)
    count = np.zeros(n)
    rows = np.arange(n)
    for p in parts:
        m = _membership(p)
        keep = p != NOISE
        if not keep.any():
            continue
        sizes = np.asarray(m.sum(axis=0)).ravel()
        s = (c @ m).tocsr()  # s[i, k] = sum_{j in cluster k} C[i, j]
        idx = rows[keep]
        own = np.asarray(s[idx, p[keep]]).ravel() - diag[idx]
        nco = sizes[p[keep]] - 1
        ok = nco > 0
        total[idx[ok]] += own[ok] / nco[ok]
        count[idx[ok]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)
