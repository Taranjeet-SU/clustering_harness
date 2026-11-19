"""Tests for clustering_harness.datasets.

Download tests are marked ``network`` and only run when ``CH_NETWORK_TESTS=1``.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from clustering_harness import datasets as chd
from clustering_harness.data import Dataset

network = pytest.mark.skipif(os.environ.get("CH_NETWORK_TESTS") != "1",
                             reason="set CH_NETWORK_TESTS=1 to run dataset download tests")


def test_registry_and_info():
    names = chd.available_datasets()
    assert {"banking77", "clinc150"} <= set(names)
    for n in names:
        info = chd.DATASET_INFO[n]
        for key in ("hf_repo", "revision", "license", "citation", "splits", "default_split"):
            assert info.get(key), (n, key)
        assert info["default_split"] in info["splits"]


def test_unknown_name_and_split():
    with pytest.raises(ValueError):
        chd.load_dataset("does-not-exist")
    with pytest.raises(ValueError):
        chd.load_dataset("banking77", split="dev")


def test_default_cache_dir_outside_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("CLUSTERING_HARNESS_CACHE", raising=False)
    assert chd.default_cache_dir() == Path.home() / ".cache" / "clustering_harness" / "datasets"
    monkeypatch.setenv("CLUSTERING_HARNESS_CACHE", str(tmp_path))
    assert chd.default_cache_dir() == tmp_path / "datasets"


def test_make_ids_stable():
    import numpy as np

    assert chd._make_ids("banking77", "test", np.array([0, 123])) == ["banking77-test-000000",
                                                                       "banking77-test-000123"]


@pytest.mark.network
@network
def test_banking77_download():
    ds = chd.load_dataset("banking77")
    assert isinstance(ds, Dataset) and len(ds) == 3080
    assert len(set(ds.labels)) == 77 and ds.ids[123] == "banking77-test-000123"
    assert all(t == "" for t in ds.titles) and all(ds.texts)
    small = chd.load_dataset("banking77", limit=200, seed=1)
    assert len(small) == 200 and small.ids == chd.load_dataset("banking77", limit=200, seed=1).ids


@pytest.mark.network
@network
def test_clinc150_download():
    ds = chd.load_dataset("clinc150")
    assert len(ds) == 4500 and len(set(ds.labels)) == 150 and "oos" not in set(ds.labels)
    with_oos = chd.load_dataset("clinc150", include_oos=True)
    assert len(with_oos) == 5500 and Counter(with_oos.labels)["oos"] == 1000


@pytest.mark.network
@network
def test_support_tickets_download():
    ds = chd.load_dataset("support_tickets")
    assert len(ds) > 10000 and len(set(ds.labels)) == 10
    assert any(ds.titles) and "\\n" not in ds.texts[0]
