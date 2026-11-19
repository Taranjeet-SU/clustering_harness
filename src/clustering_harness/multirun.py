"""Multi-run experiments and side-by-side comparison of partitions.

* :func:`run_seeds` reruns the pipeline with different random seeds (base partition UMAP seed, local-split /
  Leiden seed, residual shuffling/sharding seed) and reports agreement between runs.
* :func:`compare_partitions` scores any named partitions — pipeline runs, baselines, gold labels — with
  external metrics (when gold labels exist), internal geometry and, optionally, the two-judge LLM evaluation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .context import RunContext
from .data import Dataset
from .evaluation.metrics import external_metrics, internal_metrics, normalize_labels
from .evaluation.stability import agreement_summary, pairwise_agreement

__all__ = ["run_seeds", "compare_partitions", "judge_partitions", "eval_context"]


def run_seeds(make_pipeline: Callable[[PipelineConfig, Path], "object"], ds: Dataset, config: PipelineConfig, seeds=(0, 1, 2),
              workdir: str | Path = "runs/multirun") -> dict:
    """Run the pipeline once per seed. ``make_pipeline(cfg, workdir)`` must return a configured :class:`Pipeline`.

    Seeds must be members of ``config.bank.seeds`` (the base partition is taken from the bank)."""
    workdir = Path(workdir)
    results, parts = {}, {}
    for s in seeds:
        if s not in config.bank.seeds:
            raise ValueError(f"seed {s} is not in config.bank.seeds {config.bank.seeds}")
        cfg = copy.deepcopy(config)
        cfg.base.seed, cfg.cascade.seed, cfg.residual.seed = s, s, s
        res = make_pipeline(cfg, workdir / f"seed_{s}").run(ds)
        results[s] = res
        parts[f"seed_{s}"] = res.labels()
    agree = {m: pairwise_agreement(list(parts.values()), names=list(parts), metric=m) for m in ("ari", "ami")}
    summary = {"seeds": list(seeds), "runs": {k: r.summary for k, r in zip(parts, results.values())},
               "agreement": {m: agreement_summary(v) for m, v in agree.items()}}
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "multirun_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    for m, v in agree.items():
        v.to_csv(workdir / f"pairwise_{m}.csv")
    return {"results": results, "agreement": agree, "summary": summary}


def eval_context(ds: Dataset, X: np.ndarray, llms: dict, workdir: str | Path, config: PipelineConfig | None = None, prompts=None) -> RunContext:
    """A :class:`RunContext` for evaluation only (LLM judging of arbitrary partitions)."""
    from .llm import ResponseCache
    workdir = Path(workdir)
    for m in llms.values():
        if m.cache is None:
            m.cache = ResponseCache(workdir / "cache" / "llm.sqlite")
    return RunContext(ds=ds, cfg=config or PipelineConfig(), llms=llms, workdir=workdir, X=np.asarray(X, np.float32), prompts=prompts)


def _clusters(ids, labels, min_size: int = 1) -> dict:
    lab = normalize_labels(labels)
    out: dict = {}
    for t, l in zip(ids, lab):
        if l >= 0:
            out.setdefault(str(int(l)), []).append(t)
    return {k: v for k, v in out.items() if len(v) >= min_size}


def judge_partitions(ctx: RunContext, partitions: dict, min_size: int = 2, max_clusters: int | None = None, seed: int = 0) -> pd.DataFrame:
    """LLM-judge every cluster (size >= ``min_size``) of each named partition. ``max_clusters`` samples clusters
    uniformly (seeded) to bound cost; the sample is reported."""
    from .evaluation.llm_judge import LLMJudge
    judge = LLMJudge(ctx)
    rows = []
    for name, labels in partitions.items():
        cl = _clusters(ctx.ds.ids, labels, min_size)
        keys = sorted(cl)
        if max_clusters and len(keys) > max_clusters:
            keys = sorted(np.random.default_rng(seed).choice(keys, size=max_clusters, replace=False).tolist())
        fr = judge.evaluate(f"eval_{name}", {k: cl[k] for k in keys})
        s = LLMJudge.summary(fr)
        s.update(partition=name, clusters_total=len(cl), clusters_judged_sample=len(keys))
        rows.append(s)
    return pd.DataFrame(rows).set_index("partition")


def compare_partitions(partitions: dict, X: np.ndarray | None = None, gold=None, noise: str = "singleton",
                       judge_ctx: RunContext | None = None, judge_max_clusters: int | None = None) -> pd.DataFrame:
    """One row per named partition with external metrics (vs ``gold``), internal metrics (on ``X``) and,
    if ``judge_ctx`` is given, LLM-judge scores."""
    rows = []
    for name, labels in partitions.items():
        r = {"partition": name}
        if gold is not None:
            r.update({f"ext_{k}": v for k, v in external_metrics(gold, labels, noise=noise).items()})
        if X is not None:
            r.update({f"int_{k}": v for k, v in internal_metrics(X, labels).items()})
        rows.append(r)
    out = pd.DataFrame(rows).set_index("partition")
    if judge_ctx is not None:
        j = judge_partitions(judge_ctx, partitions, max_clusters=judge_max_clusters)
        out = out.join(j.add_prefix("llm_"))
    return out
