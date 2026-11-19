"""Tests for clustering_harness.evaluation.stability."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse
from sklearn import metrics as skm

from clustering_harness.evaluation import (
    agreement_summary,
    coassignment_matrix,
    consensus_stability,
    pairwise_agreement,
)


def test_pairwise_agreement_identical_and_relabelled():
    a = [0, 0, 1, 1, 2, 2]
    b = ["x", "x", "y", "y", "z", "z"]
    m = pairwise_agreement([a, b, a], names=["a", "b", "c"])
    assert list(m.index) == ["a", "b", "c"]
    assert np.allclose(m.to_numpy(), 1.0)


def test_pairwise_agreement_matches_sklearn_and_symmetric():
    rng = np.random.default_rng(0)
    parts = [rng.integers(0, 4, 60) for _ in range(4)]
    for metric, fn in [("ari", skm.adjusted_rand_score), ("ami", skm.adjusted_mutual_info_score),
                       ("nmi", skm.normalized_mutual_info_score)]:
        m = pairwise_agreement(parts, metric=metric).to_numpy()
        assert np.allclose(m, m.T)
        assert m[0, 2] == pytest.approx(fn(parts[0], parts[2]))
    with pytest.raises(ValueError):
        pairwise_agreement(parts, metric="bogus")
    with pytest.raises(ValueError):
        pairwise_agreement([[0, 1], [0, 1, 2]])


def test_pairwise_agreement_noise_policies():
    a = [0, 0, -1, 1, 1, -1]
    b = [0, 0, 1, 1, 1, -1]
    ex = pairwise_agreement([a, b], noise="exclude").iloc[0, 1]
    assert ex == pytest.approx(1.0)  # only records non-noise in both are compared
    sg = pairwise_agreement([a, b], noise="singleton").iloc[0, 1]
    assert sg == pytest.approx(skm.adjusted_rand_score([0, 0, 2, 1, 1, 3], [0, 0, 1, 1, 1, 2]))


def test_agreement_summary():
    m = np.array([[1.0, 0.5, 0.7], [0.5, 1.0, 0.9], [0.7, 0.9, 1.0]])
    s = agreement_summary(m)
    assert s["mean"] == pytest.approx(0.7) and s["min"] == 0.5 and s["max"] == 0.9
    assert s["n_runs"] == 3 and s["n_pairs"] == 3
    assert math.isnan(agreement_summary(np.ones((1, 1)))["mean"])


def test_coassignment_matrix_values():
    p1 = [0, 0, 1, 1, -1]
    p2 = [0, 1, 1, 1, -1]
    c = coassignment_matrix([p1, p2])
    assert sparse.issparse(c)
    d = coassignment_matrix([p1, p2], dense=True)
    assert np.allclose(c.toarray(), d) and np.allclose(d, d.T)
    assert d[0, 1] == pytest.approx(0.5) and d[2, 3] == pytest.approx(1.0) and d[1, 2] == pytest.approx(0.5)
    assert d[0, 2] == 0.0
    # noise never co-clusters, not even with itself
    assert d[4].sum() == 0.0 and d[0, 0] == 1.0
    q = coassignment_matrix([[0, -1, 0], [0, 0, 0]], dense=True)
    assert q[1, 1] == pytest.approx(0.5) and q[0, 1] == pytest.approx(0.5)


def test_consensus_stability():
    same = [[0, 0, 1, 1, 2], [5, 5, 3, 3, 4]]
    s = consensus_stability(same)
    assert np.allclose(s[:4], 1.0) and math.isnan(s[4])  # record 4 is always a singleton
    p1 = [0, 0, 0, 1]
    p2 = [0, 0, 1, 1]
    s = consensus_stability([p1, p2])
    # C: C01=1, C02=.5, C12=.5, C23=.5, C03=0, C13=0
    # record 0: run1 co-members {1,2} -> (1+.5)/2 = .75; run2 {1} -> 1 ; mean .875
    # record 2: run1 {0,1} -> (.5+.5)/2 = .5; run2 {3} -> .5 ; mean .5
    # record 3: run1 singleton (skipped); run2 {2} -> .5
    assert s[0] == pytest.approx(0.875) and s[1] == pytest.approx(0.875)
    assert s[2] == pytest.approx(0.5) and s[3] == pytest.approx(0.5)


def test_consensus_stability_noise():
    s = consensus_stability([[0, 0, -1], [0, 0, 0]])
    # C01 = 1, C02 = C12 = .5. record 0: run1 {1}->1; run2 {1,2}->(1+.5)/2=.75 ; mean .875
    # record 2: run1 noise (skipped); run2 {0,1} -> .5
    assert s[0] == pytest.approx(0.875) and s[2] == pytest.approx(0.5)
