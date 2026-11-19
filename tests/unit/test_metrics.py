"""Tests for clustering_harness.evaluation.metrics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from clustering_harness.evaluation.metrics import (
    apply_noise_policy,
    bcubed_scores,
    cluster_label_report,
    contingency_table,
    external_metrics,
    internal_metrics,
    normalize_labels,
    purity_scores,
)

# Manning, Raghavan & Schütze, "Introduction to Information Retrieval" (2008), Figure 16.4 / section 16.3:
# 17 points in 3 clusters. Cluster 1 = 5 x + 1 o; cluster 2 = 1 x + 4 o + 1 d; cluster 3 = 2 x + 3 d.
# The book reports purity = 12/17 ~ 0.71, NMI ~ 0.36 and RI ~ 0.68.
IIR_PRED = [1] * 6 + [2] * 6 + [3] * 5
IIR_TRUE = ["x"] * 5 + ["o"] + ["x"] + ["o"] * 4 + ["d"] + ["x"] * 2 + ["d"] * 3


def _bcubed_bruteforce(t, p):
    t, p = np.asarray(t), np.asarray(p)
    same_c = p[:, None] == p[None, :]
    same_l = t[:, None] == t[None, :]
    both = (same_c & same_l).sum(1)
    return (both / same_c.sum(1)).mean(), (both / same_l.sum(1)).mean()


# ------------------------------------------------------------------ normalize_labels / noise policies
def test_normalize_labels_mixed_noise():
    out = normalize_labels(["a", "b", "", None, "a", float("nan"), "  "])
    assert out.tolist() == [0, 1, -1, -1, 0, -1, -1]
    assert normalize_labels(np.array([5, -1, 5, 7, -3])).tolist() == [0, -1, 0, 1, -1]
    assert normalize_labels([]).tolist() == []
    assert normalize_labels([-1, -1]).tolist() == [-1, -1]


def test_apply_noise_policy():
    lab = [0, -1, 1, -1]
    s, k = apply_noise_policy(lab, "singleton")
    assert k.all() and len(set(s.tolist())) == 4
    c, k = apply_noise_policy(lab, "cluster")
    assert k.all() and c[1] == c[3] and c[1] not in (c[0], c[2])
    e, k = apply_noise_policy(lab, "exclude")
    assert k.tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        apply_noise_policy(lab, "bogus")


# ------------------------------------------------------------------ textbook values
def test_iir_textbook_purity_nmi_ri():
    pur = purity_scores(IIR_TRUE, IIR_PRED)
    assert pur["purity"] == pytest.approx(12 / 17)
    assert round(pur["purity"], 2) == 0.71
    # inverse purity: x -> 5 (cluster 1), o -> 4 (cluster 2), d -> 3 (cluster 3)  => 12/17
    assert pur["inverse_purity"] == pytest.approx(12 / 17)
    m = external_metrics(IIR_TRUE, IIR_PRED)
    assert round(m["nmi"], 2) == 0.36
    assert round(skm.rand_score(normalize_labels(IIR_TRUE), IIR_PRED), 2) == 0.68


def test_iir_bcubed_hand_computed():
    # Precision = (1/17) * sum_cells n_ij^2 / |cluster_j|
    #   = (1/17) * [(5^2 + 1^2)/6 + (1^2 + 4^2 + 1^2)/6 + (2^2 + 3^2)/5] = (1/17) * (26/6 + 18/6 + 13/5)
    # Recall = (1/17) * sum_cells n_ij^2 / |class_i|, class sizes x=8, o=5, d=4
    #   = (1/17) * [(5^2 + 1^2 + 2^2)/8 + (1^2 + 4^2)/5 + (1^2 + 3^2)/4] = (1/17) * (30/8 + 17/5 + 10/4)
    p_exp = (26 / 6 + 18 / 6 + 13 / 5) / 17
    r_exp = (30 / 8 + 17 / 5 + 10 / 4) / 17
    b = bcubed_scores(IIR_TRUE, IIR_PRED)
    assert b["bcubed_precision"] == pytest.approx(p_exp)
    assert b["bcubed_recall"] == pytest.approx(r_exp)
    assert b["bcubed_f1"] == pytest.approx(2 * p_exp * r_exp / (p_exp + r_exp))


def test_bcubed_amigo_extremes():
    # Amigó et al. (2009): one cluster per item gives BCubed precision 1 and recall = mean(1/|class|);
    # a single cluster gives recall 1 and precision = sum_classes |class|^2 / n^2.
    t = ["a", "a", "b", "b", "b", "c"]
    sing = bcubed_scores(t, list(range(6)))
    assert sing["bcubed_precision"] == pytest.approx(1.0)
    assert sing["bcubed_recall"] == pytest.approx((2 * 1 / 2 + 3 * 1 / 3 + 1) / 6)
    one = bcubed_scores(t, [0] * 6)
    assert one["bcubed_recall"] == pytest.approx(1.0)
    assert one["bcubed_precision"] == pytest.approx((4 + 9 + 1) / 36)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bcubed_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 7, 300)
    p = rng.integers(0, 11, 300)
    b = bcubed_scores(t, p)
    pe, re = _bcubed_bruteforce(t, p)
    assert b["bcubed_precision"] == pytest.approx(pe)
    assert b["bcubed_recall"] == pytest.approx(re)


@pytest.mark.parametrize("seed", [0, 1])
def test_purity_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    t, p = rng.integers(0, 5, 200), rng.integers(0, 8, 200)
    pur = sum(np.bincount(t[p == c]).max() for c in np.unique(p)) / 200
    inv = sum(np.bincount(p[t == c]).max() for c in np.unique(t)) / 200
    s = purity_scores(t, p)
    assert s["purity"] == pytest.approx(pur)
    assert s["inverse_purity"] == pytest.approx(inv)
    assert s["purity_f"] == pytest.approx(2 * pur * inv / (pur + inv))


# ------------------------------------------------------------------ sklearn agreement / properties
def test_external_metrics_match_sklearn():
    rng = np.random.default_rng(3)
    t, p = rng.integers(0, 6, 500), rng.integers(0, 9, 500)
    m = external_metrics(t, p)
    assert m["ari"] == pytest.approx(skm.adjusted_rand_score(t, p))
    assert m["ami"] == pytest.approx(skm.adjusted_mutual_info_score(t, p))
    assert m["nmi"] == pytest.approx(skm.normalized_mutual_info_score(t, p))
    assert m["homogeneity"] == pytest.approx(skm.homogeneity_score(t, p))
    assert m["completeness"] == pytest.approx(skm.completeness_score(t, p))
    assert m["v_measure"] == pytest.approx(skm.v_measure_score(t, p))
    assert m["fowlkes_mallows"] == pytest.approx(skm.fowlkes_mallows_score(t, p))
    assert m["n"] == 500 and m["n_noise"] == 0 and m["coverage"] == 1.0 and m["n_clusters"] == 9


def test_perfect_partition_and_permutation_invariance():
    t = ["a"] * 5 + ["b"] * 3 + ["c"] * 4
    p = [7] * 5 + [2] * 3 + [9] * 4  # same partition, different label names
    m = external_metrics(t, p)
    for k in ("ari", "ami", "nmi", "homogeneity", "completeness", "v_measure", "fowlkes_mallows", "purity",
              "inverse_purity", "purity_f", "bcubed_precision", "bcubed_recall", "bcubed_f1"):
        assert m[k] == pytest.approx(1.0), k
    rng = np.random.default_rng(0)
    t2, p2 = rng.integers(0, 4, 100), rng.integers(0, 5, 100)
    base = external_metrics(t2, p2)
    perm = rng.permutation(100)
    relabel = np.array([3, 0, 4, 1, 2])
    other = external_metrics(t2[perm], relabel[p2[perm]])
    for k, v in base.items():
        if isinstance(v, float):
            assert other[k] == pytest.approx(v), k


def test_symmetric_metrics_are_symmetric():
    rng = np.random.default_rng(5)
    a, b = rng.integers(0, 4, 80), rng.integers(0, 6, 80)
    ab, ba = external_metrics(a, b), external_metrics(b, a)
    for k in ("ari", "ami", "nmi", "v_measure", "fowlkes_mallows", "purity_f", "bcubed_f1"):
        assert ab[k] == pytest.approx(ba[k]), k
    assert ab["purity"] == pytest.approx(ba["inverse_purity"])
    assert ab["bcubed_precision"] == pytest.approx(ba["bcubed_recall"])


# ------------------------------------------------------------------ noise policies
def test_noise_policies():
    t = ["a", "a", "a", "b", "b", "b"]
    p = [0, 0, -1, 1, 1, -1]
    s = external_metrics(t, p, noise="singleton")
    assert s["n_noise"] == 2 and s["coverage"] == pytest.approx(4 / 6) and s["n_clusters"] == 2
    assert s["n_evaluated"] == 6 and s["purity"] == pytest.approx(1.0) and s["ari"] < 1.0
    ex = external_metrics(t, p, noise="exclude")
    assert ex["n_evaluated"] == 4 and ex["ari"] == pytest.approx(1.0) and ex["coverage"] == pytest.approx(4 / 6)
    cl = external_metrics(t, p, noise="cluster")
    # noise cluster {a, b} has majority 1 of 2  -> purity (2 + 2 + 1) / 6
    assert cl["purity"] == pytest.approx(5 / 6)
    assert s["ari"] == pytest.approx(skm.adjusted_rand_score(t, [0, 0, 2, 1, 1, 3]))
    assert cl["ari"] == pytest.approx(skm.adjusted_rand_score(t, [0, 0, 2, 1, 1, 2]))
    assert s["noise_policy"] == "singleton"


def test_string_noise_and_missing_gold():
    t = ["a", "a", None, "b", "b"]
    p = ["x", "x", "y", "", None]
    m = external_metrics(t, p, noise="exclude")
    assert m["n_gold_missing"] == 1 and m["n"] == 4 and m["n_noise"] == 2 and m["n_evaluated"] == 2
    allnoise = external_metrics(["a", "b"], [-1, -1], noise="exclude")
    assert math.isnan(allnoise["ari"]) and allnoise["coverage"] == 0.0 and allnoise["n_clusters"] == 0
    with pytest.raises(ValueError):
        external_metrics([1, 2], [1])


# ------------------------------------------------------------------ internal metrics
def test_internal_metrics_match_sklearn_and_exclude_noise():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 3)) + [5, 0, 0], rng.normal(0, 0.1, (20, 3)) + [0, 5, 0]])
    lab = np.array([0] * 20 + [1] * 20)
    m = internal_metrics(X, lab, metric="euclidean")
    assert m["silhouette"] == pytest.approx(skm.silhouette_score(X, lab))
    assert m["davies_bouldin"] == pytest.approx(skm.davies_bouldin_score(X, lab))
    assert m["calinski_harabasz"] == pytest.approx(skm.calinski_harabasz_score(X, lab))
    noisy = lab.copy()
    noisy[:5] = -1
    mn = internal_metrics(X, noisy, metric="cosine")
    assert mn["silhouette"] == pytest.approx(skm.silhouette_score(X[5:], lab[5:], metric="cosine"))
    assert mn["coverage"] == pytest.approx(35 / 40) and mn["n_clusters"] == 2 and mn["n_evaluated"] == 35
    sampled = internal_metrics(X, lab, sample_size=30, random_state=1)
    assert -1 <= sampled["silhouette"] <= 1


def test_internal_metrics_undefined():
    X = np.random.default_rng(0).normal(size=(10, 4))
    m = internal_metrics(X, [0] * 10)
    assert math.isnan(m["silhouette"]) and math.isnan(m["davies_bouldin"]) and math.isnan(m["calinski_harabasz"])
    m = internal_metrics(X, [-1] * 9 + [0])
    assert m["n_clusters"] == 1 and math.isnan(m["silhouette"])


# ------------------------------------------------------------------ tables
def test_contingency_table_and_report():
    t = ["a", "a", "b", "b", "b", None]
    p = [0, 0, 0, 1, -1, 1]
    ct = contingency_table(t, p)
    assert isinstance(ct, pd.DataFrame) and int(ct.to_numpy().sum()) == 6
    assert ct.loc["a", 0] == 2 and ct.loc["b", 0] == 1 and ct.loc["b", "<noise>"] == 1
    rep = cluster_label_report(t, p)
    r0 = rep[rep["cluster"] == 0].iloc[0]
    assert r0["size"] == 3 and r0["majority_label"] == "a" and r0["purity"] == pytest.approx(2 / 3)
    assert r0["n_labels"] == 2 and not r0["is_noise"]
    assert rep["size"].is_monotonic_decreasing
    assert rep["is_noise"].sum() == 1
    assert cluster_label_report(t, p, include_noise=False)["is_noise"].sum() == 0
