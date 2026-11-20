"""Command line interface.

    clustering-harness init  [--preset small|reference] > run.yaml      # starter run spec
    clustering-harness run     run.yaml [--until STAGE]
    clustering-harness compare run.yaml --baselines kmeans:k=77 hdbscan:min_cluster_size=10 [--judge] [--judge-max-clusters 60]
    clustering-harness multirun run.yaml --seeds 0 1 2
    clustering-harness evaluate --assignments runs/x/output/assignments.csv --gold data.csv --id-col id --label-col label
    clustering-harness datasets [--export banking77 --split test --out gold.csv]

A run spec is YAML with sections ``data``, ``pipeline``, ``models`` (slots ``a`` and ``b``), ``embedder``,
``label_embedder`` and ``workdir``, plus two optional paths:

* ``ledger``: spend ledger (SQLite). Default ``<workdir>/spend.sqlite``, so ``budget_usd`` caps the spend of
  this run. Point several runs at one shared ledger to cap their *combined* spend instead.
* ``embedding_cache``: directory of cached embedding matrices, keyed by embedder name (model, pooling,
  length, precision) and the exact texts. Default ``~/.cache/clustering_harness/embeddings``.

See ``examples/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

__all__ = ["main", "load_spec", "build_from_spec"]

STARTER = """# clustering_harness run spec
workdir: runs/my_run
data:
  path: tickets.csv          # .csv / .tsv / .json / .jsonl / .parquet   (or:  dataset: banking77, split: test)
  # or MongoDB instead of path:   mongo: {{uri_env: MONGODB_URI, database: support, collection: [tickets],
  #                                       query: {{status: closed}}, text_field: [subject, description], title_field: subject}}
  id_col: id
  text_col: text
  title_col: title           # optional
  label_col: null            # optional gold labels for evaluation
pipeline:
  preset: {preset}          # small (~1k records) | reference (~15k) | auto
  prompts: default           # built-in preset name or path to a prompt YAML
  domain:
    product: Acme Cloud
    product_description: a SaaS platform with integrations, reporting and user management
models:                      # slot a: classifier/verifier/judge; slot b: canonicaliser/labeller/judge
  a: {{backend: ollama, model: "qwen3:8b", url: "http://localhost:11434", reasoning: false, concurrency: 2}}
  b: {{backend: openai, model: gpt-4o-mini, concurrency: 8, budget_usd: 5}}
embedder: {{backend: hf, model: Alibaba-NLP/gte-large-en-v1.5, pooling: cls, max_length: 8192}}
label_embedder: {{backend: st, model: BAAI/bge-large-en-v1.5, device: cpu}}
# embedder device defaults to cpu; set device: cuda (or auto) to use a GPU
# ledger: ~/.cache/clustering_harness/spend.sqlite   # optional shared ledger: budgets then cap spend across runs
"""


def load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    spec.setdefault("workdir", f"runs/{Path(path).stem}")
    return spec


def load_data(d: dict):
    from .data import Dataset
    if "mongo" in d:
        m = dict(d["mongo"])
        coll = m.pop("collection", None) or m.pop("collections", None)
        if not coll:
            raise ValueError("data.mongo needs `collection` (a name) or `collections` (a list of names)")
        ds = Dataset.from_mongo(coll, **m)
        if d.get("limit"):
            ds = ds.sample(int(d["limit"]), seed=d.get("seed", 0))
    elif "dataset" in d:
        from .datasets import load_dataset
        ds = load_dataset(d["dataset"], split=d.get("split"), limit=d.get("limit"), seed=d.get("seed", 0), **d.get("options", {}))
    else:
        kw = {k: d[k] for k in ("id_col", "text_col", "title_col", "label_col") if d.get(k)}
        ds = Dataset.from_file(d["path"], **kw)
        if d.get("limit"):
            ds = ds.sample(int(d["limit"]), seed=d.get("seed", 0))
    return ds


def build_config(p: dict, n: int):
    from .config import PipelineConfig, _deep_merge
    p = dict(p or {})
    preset = p.pop("preset", "auto")
    base = {"small": PipelineConfig.small, "reference": PipelineConfig.reference}.get(preset)
    cfg = base() if base else PipelineConfig.auto(n)
    return PipelineConfig.model_validate(_deep_merge(cfg.model_dump(), p))


def build_from_spec(spec: dict, ds=None, workdir: str | Path | None = None, config=None):
    from .factory import build_embedder, build_llm
    from .llm import Ledger, ResponseCache
    from .pipeline import Pipeline
    ds = ds if ds is not None else load_data(spec["data"])
    wd = Path(workdir or spec["workdir"]).expanduser()
    cfg = config or build_config(spec.get("pipeline"), len(ds))
    cache = ResponseCache(wd / "cache" / "llm.sqlite")
    ledger = Ledger(Path(spec.get("ledger", wd / "spend.sqlite")).expanduser())
    llms = {}
    for k, v in spec["models"].items():
        v = dict(v)
        name = v.pop("name", k)
        llms[k] = build_llm(v, cache=cache, ledger=ledger, name=name)
    emb_cache = Path(spec.get("embedding_cache", "~/.cache/clustering_harness/embeddings")).expanduser()
    pipe = Pipeline(cfg, llms=llms, embedder=build_embedder(spec.get("embedder"), emb_cache),
                    label_embedder=build_embedder(spec.get("label_embedder"), emb_cache), workdir=wd)
    return pipe, ds, ledger


def _parse_baseline(s: str):
    name, _, args = s.partition(":")
    kw = {}
    for part in filter(None, args.split(",")):
        k, v = part.split("=")
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    return name, kw


STAGES = "embed, bank, cascade, residual, refile, scoring, rollup, export"


def main(argv=None) -> int:
    from .baselines import list_baselines
    ap = argparse.ArgumentParser("clustering-harness", description="LLM-assisted clustering of short texts. "
                                 "A run is described by a YAML spec (see `clustering-harness init`).")
    ap.add_argument("-v", "--verbose", action="store_true", help="log stage-by-stage progress")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    s = sub.add_parser("init", help="print a starter run spec", description="Print a starter run spec to stdout.")
    s.add_argument("--preset", default="small", help="small (~1k records) | reference (~15k) | auto (default: small)")
    s = sub.add_parser("run", help="run the pipeline (resumable)", description="Run the pipeline. Re-running the same "
                       "spec resumes from the work directory's caches.")
    s.add_argument("spec", help="run spec YAML"); s.add_argument("--until", default=None, help=f"stop after this stage ({STAGES})")
    s = sub.add_parser("compare", help="run the pipeline, then compare it with baselines and gold labels",
                       description="Run (or resume) the pipeline, run the requested baselines on the same embeddings, and "
                       "write external/internal metrics (and optionally the LLM-judge evaluation) to output/comparison.csv.")
    s.add_argument("spec", help="run spec YAML")
    s.add_argument("--baselines", nargs="*", default=[], metavar="NAME[:k=v,...]",
                   help="e.g. kmeans:k=77 umap_hdbscan:min_cluster_size=10 leiden_knn:resolution=1.0; available: " + ", ".join(list_baselines()))
    s.add_argument("--judge", action="store_true", help="also run the two-judge LLM coherence evaluation (costs LLM calls)")
    s.add_argument("--judge-max-clusters", type=int, default=None, help="judge at most this many clusters per partition")
    s.add_argument("--out", default=None, help="output CSV (default: <workdir>/output/comparison.csv)")
    s = sub.add_parser("multirun", help="run several seeds and measure their agreement",
                       description="Run the pipeline once per seed (seeds must be in pipeline.bank.seeds). The seed equal to "
                       "the spec's own seed reuses the main run in <workdir>.")
    s.add_argument("spec", help="run spec YAML"); s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="default: 0 1 2")
    s = sub.add_parser("evaluate", help="score an assignments.csv against gold labels",
                       description="External metrics of output/assignments.csv against a gold CSV "
                       "(create one for a built-in dataset with `datasets --export`).")
    s.add_argument("--assignments", required=True, help="output/assignments.csv of a run")
    s.add_argument("--gold", required=True, help="CSV with record ids and gold labels")
    s.add_argument("--id-col", default="id", help="id column in the gold CSV (default: id)")
    s.add_argument("--label-col", default="label", help="label column in the gold CSV (default: label)")
    s.add_argument("--level", default="cluster_id", help="assignment column to score: cluster_id | line_id (default: cluster_id)")
    s = sub.add_parser("datasets", help="list built-in benchmark datasets, or export one with its gold labels")
    s.add_argument("--export", default=None, metavar="NAME", help="write NAME (id, text, label) to --out")
    s.add_argument("--split", default=None, help="dataset split (default: the dataset's default)")
    s.add_argument("--out", default=None, help="output CSV (default: NAME_SPLIT.csv)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "faiss", "faiss.loader", "datasets", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if a.cmd == "init":
        sys.stdout.write(STARTER.format(preset=a.preset))
        return 0
    if a.cmd == "datasets":
        from .datasets import DATASET_INFO, load_dataset
        if a.export:
            ds = load_dataset(a.export, split=a.split)
            out = a.out or f"{a.export}_{a.split or 'default'}.csv"
            pd.DataFrame({"id": ds.ids, "text": ds.texts, "label": ds.labels}).to_csv(out, index=False)
            print(f"wrote {len(ds)} records to {out}")
            return 0
        for k, v in DATASET_INFO.items():
            print(f"{k:18s} {v.get('license', '?'):14s} {v.get('hf_repo', '')}")
        return 0
    if a.cmd == "evaluate":
        from .evaluation.metrics import external_metrics
        asg = pd.read_csv(a.assignments, dtype=str).fillna("")
        gold = pd.read_csv(a.gold, dtype=str).fillna("")
        m = asg.merge(gold[[a.id_col, a.label_col]], left_on="ticket_id", right_on=a.id_col, how="inner")
        print(json.dumps(external_metrics(m[a.label_col].tolist(), m[a.level].tolist()), indent=1, default=float))
        return 0
    spec = load_spec(a.spec)
    if a.cmd == "run":
        pipe, ds, ledger = build_from_spec(spec)
        res = pipe.run(ds, until=a.until)
        if res is not None:
            print(json.dumps(res.summary, indent=1, default=str))
        print(json.dumps({"spend": ledger.summary()}, indent=1))
        return 0
    if a.cmd == "multirun":
        from .multirun import run_seeds
        ds = load_data(spec["data"])
        cfg = build_config(spec.get("pipeline"), len(ds))
        out = run_seeds(lambda c, w: build_from_spec(spec, ds=ds, workdir=w, config=c)[0], ds, cfg, a.seeds,
                        Path(spec["workdir"]).expanduser() / "multirun", base_workdir=Path(spec["workdir"]).expanduser())
        print(json.dumps(out["summary"]["agreement"], indent=1, default=float))
        return 0
    if a.cmd == "compare":
        from .baselines import run_baseline
        from .multirun import compare_partitions
        pipe, ds, _ = build_from_spec(spec)
        res = pipe.run(ds)
        X = np.load(Path(pipe.workdir) / "cache" / "embeddings.npy")
        parts = {"pipeline_clusters": res.labels("cluster")}
        if res.lines is not None:
            parts["pipeline_lines"] = res.labels("line")
        for b in a.baselines:
            name, kw = _parse_baseline(b)
            parts[b] = run_baseline(name, X, **kw)
        if ds.labels is not None:
            parts["gold"] = np.asarray(ds.labels)
        jctx = None
        if a.judge:
            from .context import RunContext
            jctx = RunContext(ds=ds, cfg=pipe.cfg, llms=pipe.llms, workdir=pipe.workdir, X=X, prompts=pipe.prompts)
        table = compare_partitions(parts, X=X, gold=ds.labels, judge_ctx=jctx, judge_max_clusters=a.judge_max_clusters)
        out = Path(a.out or Path(pipe.workdir) / "output" / "comparison.csv")
        table.to_csv(out)
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(table.T)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
