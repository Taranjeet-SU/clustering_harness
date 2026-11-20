"""End-to-end ``Pipeline.run`` on a synthetic topic dataset with scripted models (offline, no downloads)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import HashEmbedder, fake_llms, make_topic_dataset  # noqa: E402

from clustering_harness import Pipeline, PipelineConfig  # noqa: E402
from clustering_harness.pipeline import STAGES  # noqa: E402

STATUSES = {"cluster", "unassigned", "junk"}


def config() -> PipelineConfig:
    cfg = PipelineConfig.small()
    # one UMAP per family member instead of six keeps the bank fast; everything else is the small preset
    cfg.bank = cfg.bank.model_copy(update=dict(seeds=[0], umap_neighbors=[30], workers=1))
    return cfg


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    ds, X = make_topic_dataset(n_topics=12, per_topic=[35] * 10 + [8, 7], seed=0, n_outliers=20, n_junk=8)
    wd = tmp_path_factory.mktemp("pipe") / "run"
    llms, fakes = fake_llms()
    progress = []
    pipe = Pipeline(config(), llms=llms, label_embedder=HashEmbedder(), vectors=X, workdir=wd,
                    progress=lambda k, d: progress.append(k))
    res = pipe.run(ds)
    return ds, X, wd, res, llms, fakes, progress


@pytest.mark.slow
def test_all_stages_complete_and_outputs_exist(run):
    ds, X, wd, res, llms, fakes, progress = run
    for f in ("assignments.csv", "clusters.csv", "report_lines.csv", "summary.json"):
        assert (wd / "output" / f).exists(), f
    for p in ("results/base/labels.npy", "results/bank/grid_runs.csv", "results/cascade_B/assignments.csv",
              "results/cascade_A/assignments.csv", "results/residual/assignments.csv", "results/refile/assignments.csv",
              "results/scoring/cluster_scores.csv", "results/rollup/lines.csv", "cache/gate_calibration.json"):
        assert (wd / p).exists(), p
    assert {"bank", "cascade", "residual", "refile", "scoring", "rollup", "export"} <= set(progress)
    assert STAGES[-1] == "export"
    summ = json.loads((wd / "output" / "summary.json").read_text())
    assert summ["records"] == len(ds) and summ["human_validated"] is False
    assert summ["clusters"] == len(res.clusters) and summ["report_lines"] == len(res.lines)


@pytest.mark.slow
def test_every_input_id_once_and_valid_statuses(run):
    ds, X, wd, res, *_ = run
    a = pd.read_csv(wd / "output" / "assignments.csv", dtype=str).fillna("")
    assert a.ticket_id.tolist() == ds.ids                                   # every id exactly once, dataset order
    assert set(a.status) <= STATUSES
    junk = np.array(ds.labels) == "junk"
    assert (a.status[junk] == "junk").all() and (a.status[~junk] != "junk").all()   # only rule junk is status=junk
    assert ((a.cluster_id != "") == (a.status == "cluster")).all()
    assert (a[a.status == "cluster"].line_id != "").all()
    sizes = a[a.status == "cluster"].groupby("cluster_id").size()
    cl = pd.read_csv(wd / "output" / "clusters.csv", dtype={"cluster_id": str}).set_index("cluster_id")
    assert dict(sizes) == dict(cl.case_count)                                # assignment/count agreement
    assert set(cl.quality_status) <= {"REVIEW_RECOMMENDED", "MODEL_REVIEW_PASSED"}
    lines = pd.read_csv(wd / "output" / "report_lines.csv")
    assert lines.tickets.sum() == (a.status == "cluster").sum()
    assert lines.tickets.is_monotonic_decreasing
    un = a[a.status == "unassigned"]
    assert all(r == "not placed" or r.startswith("LLM junk: ") for r in un.reason)
    placed = a[a.status == "cluster"].placement
    assert set(placed) <= {"cascade_B", "filed_existing_confirmed", "new_cluster", "reassigned", "refiled_filed_existing_confirmed",
                           "refiled_new_cluster", "refiled_reassigned"}


@pytest.mark.slow
def test_topic_recovery(run):
    from sklearn.metrics import adjusted_rand_score
    ds, X, wd, res, *_ = run
    a = res.assignments
    m = (a.status == "cluster").to_numpy()
    y = np.array(ds.labels)
    assert m[np.isin(y, ["junk"], invert=True)].mean() > 0.85                 # coverage of eligible records
    ari = adjusted_rand_score(y[m], a.cluster_id.to_numpy()[m])
    assert ari >= 0.7
    # report lines group clusters of the same topic
    assert adjusted_rand_score(y[m], a.line_id.to_numpy()[m]) >= 0.7


@pytest.mark.slow
def test_rerun_resumes_without_new_requests(run):
    ds, X, wd, res, llms, fakes, _ = run
    llms2, fakes2 = fake_llms()
    res2 = Pipeline(config(), llms=llms2, label_embedder=HashEmbedder(), vectors=X, workdir=wd).run(ds)
    assert sum(len(f.log) for f in fakes2.values()) == 0
    assert all(m.stats["requests"] == 0 for m in llms2.values())
    pd.testing.assert_frame_equal(res2.assignments.reset_index(drop=True), res.assignments.reset_index(drop=True), check_dtype=False)


@pytest.mark.slow
def test_workdir_refuses_other_config(run):
    ds, X, wd, *_ = run
    llms, _ = fake_llms()
    other = config()
    other.gate = other.gate.model_copy(update=dict(accept_mean=3.7))
    with pytest.raises(ValueError):
        Pipeline(other, llms=llms, label_embedder=HashEmbedder(), vectors=X, workdir=wd).run(ds)


@pytest.mark.slow
def test_cached_responses_replay_with_fresh_stage_outputs(run, tmp_path):
    """With every stage artefact removed except the LLM cache, a rerun is answered entirely from the cache and
    reproduces the same assignments (the cache keys cover every request field)."""
    import shutil
    ds, X, wd, res, *_ = run
    wd2 = tmp_path / "replay"
    (wd2 / "cache").mkdir(parents=True)
    # the cache runs in SQLite WAL mode and its connections are never checkpointed/closed, so the -wal sidecar
    # must travel with the database file (copying llm.sqlite alone loses the most recent answers)
    for f in (wd / "cache").glob("llm.sqlite*"):
        shutil.copy(f, wd2 / "cache" / f.name)
    from clustering_harness.llm import LLM, OfflineBackend, ResponseCache
    cache = ResponseCache(wd2 / "cache" / "llm.sqlite")
    off = {s: OfflineBackend(f"fake:{s}") for s in "ab"}
    llms = {s: LLM(off[s], name=s, cache=cache) for s in "ab"}
    res2 = Pipeline(config(), llms=llms, label_embedder=HashEmbedder(), vectors=X, workdir=wd2).run(ds)
    assert not off["a"].misses and not off["b"].misses
    assert res2.assignments.cluster_id.tolist() == res.assignments.cluster_id.tolist()


def test_workdir_refuses_other_records_of_the_same_size(run):
    ds, X, wd, *_ = run
    llms, _ = fake_llms()
    texts = list(ds.texts)
    texts[0] = texts[0] + " (edited)"
    other = type(ds)(ids=ds.ids, texts=texts, titles=ds.titles, labels=ds.labels, meta=ds.meta, name=ds.name)
    with pytest.raises(ValueError, match="data_fingerprint"):
        Pipeline(config(), llms=llms, label_embedder=HashEmbedder(), vectors=X, workdir=wd).run(other)


def test_workdir_refuses_other_models(run):
    ds, X, wd, *_ = run
    from clustering_harness.llm import LLM, CallableBackend
    llms, _ = fake_llms()
    llms["a"] = LLM(CallableBackend(lambda r: "{}", "another:model"), name="a")
    with pytest.raises(ValueError, match="models"):
        Pipeline(config(), llms=llms, label_embedder=HashEmbedder(), vectors=X, workdir=wd).run(ds)


def test_unknown_until_stage_is_rejected_before_any_work(run, tmp_path):
    ds, X, *_ = run
    llms, _ = fake_llms()
    with pytest.raises(ValueError, match="unknown stage 'bnak'"):
        Pipeline(config(), llms=llms, label_embedder=HashEmbedder(), vectors=X, workdir=tmp_path / "w").run(ds, until="bnak")
    assert not (tmp_path / "w" / "manifest.json").exists()      # nothing ran


def test_unusable_model_stops_the_run_and_caches_nothing(run, tmp_path):
    ds, X, *_ = run
    from clustering_harness.llm import LLM, BackendConfigError, CallableBackend, ModelUnavailableError

    def bad(req):
        raise BackendConfigError("401 Unauthorized: invalid api key")

    llms, _ = fake_llms()
    llms["a"] = LLM(CallableBackend(bad, "fake:a"), name="a")
    wd = tmp_path / "bad"
    with pytest.raises(ModelUnavailableError, match="invalid api key"):
        Pipeline(config(), llms=llms, label_embedder=HashEmbedder(), vectors=X, workdir=wd).run(ds)
    from clustering_harness.llm import ResponseCache
    assert ResponseCache(wd / "cache" / "llm.sqlite").error_count() == 0
