"""Tests for clustering_harness.baselines on synthetic blobs."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering_harness.baselines import BASELINES, canonicalize_labels, list_baselines, run_baseline


@pytest.fixture(scope="module")
def blobs():
    # Well-separated blobs in directions that also differ in angle (so cosine methods can separate them).
    centers = np.eye(8)[:4] * 10 + 1.0
    X, y = make_blobs(n_samples=240, centers=centers, cluster_std=0.4, random_state=0)
    return X, y


PARAMS = {
    "kmeans": {"k": 4},
    "minibatch_kmeans": {"k": 4, "batch_size": 64},
    "agglomerative_ward": {"k": 4},
    "agglomerative_cosine_average": {"k": 4},
    "hdbscan": {"min_cluster_size": 10},
    "umap_hdbscan": {"min_cluster_size": 10, "n_neighbors": 15},
    "leiden_knn": {"n_neighbors": 10, "resolution": 1.0},
    "gmm": {"k": 4},
}


def test_registry():
    assert set(list_baselines()) == set(PARAMS) == set(BASELINES)
    with pytest.raises(ValueError):
        run_baseline("nope", np.zeros((3, 2)))


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_recovers_blobs_and_is_deterministic(name, blobs):
    X, y = blobs
    a = run_baseline(name, X, random_state=0, **PARAMS[name])
    b = run_baseline(name, X, random_state=0, **PARAMS[name])
    assert a.dtype.kind == "i" and a.shape == (X.shape[0],)
    assert np.array_equal(a, b)
    assert adjusted_rand_score(y, a) > 0.9
    assert (a >= -1).all()


def test_k_required():
    X = np.random.default_rng(0).normal(size=(10, 3))
    for name in ("kmeans", "minibatch_kmeans", "agglomerative_ward", "gmm"):
        with pytest.raises(ValueError):
            run_baseline(name, X)
    with pytest.raises(ValueError):
        run_baseline("agglomerative_cosine_average", X)
    with pytest.raises(ValueError):
        run_baseline("agglomerative_cosine_average", X, k=2, distance_threshold=0.5)


def test_cosine_threshold_mode(blobs):
    X, y = blobs
    lab = run_baseline("agglomerative_cosine_average", X, distance_threshold=0.3)
    assert adjusted_rand_score(y, lab) > 0.9


def test_scale_invariance_of_cosine_methods(blobs):
    X, _ = blobs
    scale = np.random.default_rng(1).uniform(0.5, 3.0, size=(X.shape[0], 1))
    for name in ("hdbscan", "leiden_knn", "agglomerative_cosine_average"):
        a = run_baseline(name, X, **PARAMS[name])
        b = run_baseline(name, X * scale, **PARAMS[name])
        assert adjusted_rand_score(a, b) > 0.99, name


def test_canonicalize_labels():
    out = canonicalize_labels(np.array([5, 5, 2, 2, 2, -1, 9]), min_cluster_size=2)
    assert out.tolist() == [1, 1, 0, 0, 0, -1, -1]


def test_bad_input():
    with pytest.raises(ValueError):
        run_baseline("kmeans", np.array([[np.nan, 1.0]]), k=1)
