"""Command line interface.

    clustering-harness init  [--preset small|reference] > run.yaml      # starter run spec
    clustering-harness run     run.yaml [--until STAGE]
    clustering-harness compare run.yaml --baselines kmeans:k=77 hdbscan:min_cluster_size=10 [--judge] [--judge-max-clusters 60]
    clustering-harness multirun run.yaml --seeds 0 1 2
    clustering-harness reset   run.yaml [--from-stage STAGE]         # start over, keeping the paid model answers
    clustering-harness evaluate --assignments runs/x/output/assignments.csv --gold data.csv --id-col id --label-col label
    clustering-harness datasets [--export banking77 --split test --out gold.csv]

A run spec is YAML with sections ``data``, ``pipeline``, ``models`` (slots ``a`` and ``b``), ``embedder``,
``label_embedder`` and ``workdir``, plus two optional paths:

* ``ledger``: spend ledger (SQLite). Default ``<workdir>/spend.sqlite``, so ``budget_usd`` caps the spend of
  this run. Point several runs at one shared ledger to cap their *combined* spend instead.
* ``embedding_cache``: directory of cached embedding matrices, keyed by embedder name (model, pooling,
  length, precision) and the exact texts. Default ``$CLUSTERING_HARNESS_CACHE/embeddings``, else
  ``~/.cache/clustering_harness/embeddings``.

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

from .pipeline import STAGES

__all__ = ["main", "load_spec", "build_from_spec"]

STARTER = """# clustering_harness run spec
workdir: runs/my_run
data:
  path: tickets.csv          # .csv / .tsv / .json / .jsonl / .parquet   (or:  dataset: banking77, split: test)
  # or MongoDB instead of path:   mongo: {{uri_env: MONGODB_URI, database: support, collection: [tickets],
  #                                       query: {{status: closed}}, text_field: description, title_field: subject}}
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


SPEC_KEYS = ("workdir", "data", "pipeline", "models", "embedder", "label_embedder", "ledger", "embedding_cache")
DATA_KEYS = {"path": ("path", "id_col", "text_col", "title_col", "label_col", "limit", "seed"),
             "dataset": ("dataset", "split", "limit", "seed", "options"),
             "mongo": ("mongo", "limit", "seed")}
PRESETS = ("small", "reference", "auto")


def _unknown(keys, allowed, where: str) -> None:
    import difflib
    bad = [k for k in keys if k not in allowed]
    if bad:
        hint = "; ".join(f"{k!r}" + (f" (did you mean {m[0]!r}?)" if (m := difflib.get_close_matches(str(k), allowed, 1)) else "")
                         for k in bad)
        raise ValueError(f"unknown {where} key(s): {hint}. Allowed: {', '.join(allowed)}")


def _resolve_input(p, base: Path | None) -> str:
    """An input file path as written; if it does not exist relative to the current folder, relative to the spec."""
    q = Path(str(p)).expanduser()
    if q.is_absolute() or q.exists() or base is None or not (base / q).exists():
        return str(q)
    return str(base / q)


def load_spec(path: str | Path) -> dict:
    """Read and check a run spec. Input files (``data.path``, a prompt YAML) are looked up in the current folder
    first, then next to the spec; outputs (``workdir``, ``ledger``) are relative to the current folder."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"run spec not found: {path}")
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: a run spec must be a YAML mapping (see `clustering-harness init`)")
    _unknown(spec, SPEC_KEYS, "run spec")
    for k in ("data", "models"):
        if k not in spec:
            raise ValueError(f"{path}: the run spec needs a `{k}` section (see `clustering-harness init`)")
    base = path.resolve().parent
    d = spec["data"]
    src = [k for k in DATA_KEYS if k in d]
    if len(src) != 1:
        raise ValueError(f"data: give exactly one of `path`, `dataset` or `mongo` (found: {src or 'none'})")
    _unknown(d, DATA_KEYS[src[0]], "data")
    if "path" in d:
        d["path"] = _resolve_input(d["path"], base)
    p = spec.get("pipeline") or {}
    if p.get("preset", "auto") not in PRESETS:
        import difflib
        near = difflib.get_close_matches(str(p["preset"]), PRESETS, 1)
        raise ValueError(f"pipeline.preset {p['preset']!r} is not one of {', '.join(PRESETS)}"
                         + (f" (did you mean {near[0]!r}?)" if near else ""))
    pr = str(p.get("prompts", ""))
    if pr.endswith((".yaml", ".yml")):
        p["prompts"] = _resolve_input(pr, base)
    from .prompts import PromptSet
    PromptSet.load(p.get("prompts", "default"), domain=p.get("domain") or None)   # unknown preset / domain keys fail now
    missing = {"a", "b"} - set(spec["models"] or {})
    if missing:
        raise ValueError(f"models: slots 'a' and 'b' are both required (missing {sorted(missing)})")
    spec.setdefault("workdir", f"runs/{path.stem}")
    return spec


def load_data(d: dict):
    from .data import Dataset
    if "mongo" in d:
        m = dict(d["mongo"])
        coll = m.pop("collection", None) or m.pop("collections", None)
        if not coll:
            raise ValueError("data.mongo needs `collection` (a name) or `collections` (a list of names)")
        import inspect
        from .mongo import load_mongo_frame
        allowed = [k for k in inspect.signature(load_mongo_frame).parameters if k not in ("collection", "client")] + ["name"]
        _unknown(m, allowed, "data.mongo")
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
    from pydantic import ValidationError
    try:
        return PipelineConfig.model_validate(_deep_merge(cfg.model_dump(), p))
    except ValidationError as e:
        raise ValueError("invalid pipeline settings: " + "; ".join(_pydantic_hint(err, cfg) for err in e.errors())) from None


def _pydantic_hint(err: dict, cfg) -> str:
    """One readable line per pydantic error, with a "did you mean" for unknown keys."""
    import difflib
    loc = [str(x) for x in err.get("loc", ())]
    where = "pipeline." + ".".join(loc)
    if err.get("type") == "extra_forbidden":
        node = cfg.model_dump()
        for part in loc[:-1]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        near = difflib.get_close_matches(loc[-1], list(node) if isinstance(node, dict) else [], 1)
        return f"unknown setting {where!r}" + (f" (did you mean {'pipeline.' + '.'.join(loc[:-1] + [near[0]])!r}?)" if near else "")
    return f"{where}: {err.get('msg')} (got {err.get('input')!r})"


def build_from_spec(spec: dict, ds=None, workdir: str | Path | None = None, config=None, ledger_path: str | Path | None = None):
    from .factory import build_embedder, build_llm
    from .llm import Ledger, ResponseCache
    from .pipeline import Pipeline
    ds = ds if ds is not None else load_data(spec["data"])
    wd = Path(workdir or spec["workdir"]).expanduser()
    cfg = config or build_config(spec.get("pipeline"), len(ds))
    cache = ResponseCache(wd / "cache" / "llm.sqlite")
    ledger = Ledger(Path(ledger_path or spec.get("ledger") or wd / "spend.sqlite").expanduser())
    llms = {}
    for k, v in spec["models"].items():
        v = dict(v)
        name = v.pop("name", k)
        llms[k] = build_llm(v, cache=cache, ledger=ledger, name=name)
    import os
    default_cache = Path(os.environ.get("CLUSTERING_HARNESS_CACHE", "~/.cache/clustering_harness")) / "embeddings"
    emb_cache = Path(spec.get("embedding_cache", default_cache)).expanduser()
    pipe = Pipeline(cfg, llms=llms, embedder=build_embedder(spec.get("embedder"), emb_cache),
                    label_embedder=build_embedder(spec.get("label_embedder"), emb_cache), workdir=wd)
    return pipe, ds, ledger


def _parse_baseline(s: str):
    name, _, args = s.partition(":")
    kw = {}
    for part in filter(None, args.split(",")):
        if part.count("=") != 1:
            raise ValueError(f"baseline option {part!r} in {s!r} must look like key=value")
        k, v = part.split("=")
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    return name, kw




def main(argv=None) -> int:
    verbose = "-v" in (argv if argv is not None else sys.argv[1:]) or "--verbose" in (argv if argv is not None else sys.argv[1:])
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume from the caches", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError, ImportError, OSError, yaml.YAMLError) as e:
        if verbose:
            raise
        msg = str(e) if not isinstance(e, KeyError) else f"missing key {e}"
        print(f"error: {msg}\n(rerun with -v for the full traceback)", file=sys.stderr)
        return 2


def _main(argv=None) -> int:
    from .baselines import list_baselines
    ap = argparse.ArgumentParser("clustering-harness", description="LLM-assisted clustering of short texts. "
                                 "A run is described by a YAML spec (see `clustering-harness init`).")
    ap.add_argument("-v", "--verbose", action="store_true", help="log stage-by-stage progress")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    s = sub.add_parser("init", help="print a starter run spec", description="Print a starter run spec to stdout.")
    s.add_argument("--preset", default="small", help="small (~1k records) | reference (~15k) | auto (default: small)")
    s = sub.add_parser("run", help="run the pipeline (resumable)", description="Run the pipeline. Re-running the same "
                       "spec resumes from the work directory's caches.")
    s.add_argument("spec", help="run spec YAML")
    s.add_argument("--until", default=None, choices=STAGES, metavar="STAGE", help=f"stop after this stage: {', '.join(STAGES)}")
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
    s = sub.add_parser("reset", help="delete a run's stage outputs but keep the paid model answers",
                       description="Delete the stage outputs and manifest of the spec's workdir, keeping the model-answer "
                       "cache, label vectors and spend ledger. The next `run` may then use changed data, settings, prompts "
                       "or models; requests whose prompt did not change are answered from the cache for free. With "
                       "--from-stage only that stage and later ones are deleted; use it only when your change cannot "
                       "affect the earlier stages (e.g. a scoring or rollup setting).")
    s.add_argument("spec", help="run spec YAML")
    s.add_argument("--from-stage", default=None, choices=STAGES, metavar="STAGE", help=f"one of: {', '.join(STAGES)}")
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
    # -v shows this package's progress; other libraries only speak up for warnings and errors
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logging.getLogger("clustering_harness").setLevel(logging.INFO if a.verbose else logging.WARNING)
    for noisy in ("faiss", "faiss.loader"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if a.cmd in ("run", "compare", "multirun"):
        # plugins referenced as "module:attr" in the spec may live next to the spec or in the current folder
        for d in (Path.cwd(), Path(a.spec).resolve().parent):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))

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
        for col, frame, name in ((a.id_col, gold, "--gold"), (a.label_col, gold, "--gold"), ("ticket_id", asg, "--assignments"),
                                 (a.level, asg, "--assignments")):
            if col not in frame.columns:
                raise ValueError(f"column {col!r} not in {name} file (columns: {', '.join(map(str, frame.columns))})")
        m = asg.merge(gold[[a.id_col, a.label_col]], left_on="ticket_id", right_on=a.id_col, how="inner")
        if m.empty:
            raise ValueError(f"no record ids match between {a.assignments} (ticket_id, e.g. {asg.ticket_id.head(2).tolist()}) "
                             f"and {a.gold} ({a.id_col}, e.g. {gold[a.id_col].head(2).tolist()})")
        if len(m) < len(asg):
            print(f"warning: only {len(m)} of {len(asg)} assignment rows have a gold label", file=sys.stderr)
        print(json.dumps(external_metrics(m[a.label_col].tolist(), m[a.level].tolist()), indent=1, default=float))
        return 0
    if a.cmd == "reset":
        from .pipeline import reset_workdir
        spec = yaml.safe_load(Path(a.spec).read_text()) or {}
        wd = Path(spec.get("workdir") or f"runs/{Path(a.spec).stem}").expanduser()
        if not wd.exists():
            raise FileNotFoundError(f"workdir {wd} does not exist (nothing to reset)")
        removed = reset_workdir(wd, a.from_stage)
        print(f"removed {len(removed)} item(s) from {wd}; kept the model-answer cache and spend ledger")
        if a.from_stage:
            print(f"the stages before {a.from_stage!r} were kept as they are; the next run trusts them")
        return 0
    spec = load_spec(a.spec)
    if a.cmd == "run":
        pipe, ds, ledger = build_from_spec(spec)
        res = pipe.run(ds, until=a.until)
        if res is not None:
            print(json.dumps(res.summary, indent=1, default=str))
            for slot, p in (res.summary.get("model_problems") or {}).items():
                print(f"warning: model {slot}: {p['unparseable_answers']} unparseable answer(s), {p['rejected_requests']} "
                      f"rejected request(s) in stage(s) {', '.join(p['stages'])}; those items were treated as 'no answer'. "
                      f"To ask them again: clustering-harness reset {a.spec} --from-stage {p['stages'][0]}", file=sys.stderr)
        print(json.dumps({"spend": ledger.summary()}, indent=1))
        return 0
    if a.cmd == "multirun":
        from .multirun import run_seeds
        ds = load_data(spec["data"])
        cfg = build_config(spec.get("pipeline"), len(ds))
        main_wd = Path(spec["workdir"]).expanduser()
        shared = spec.get("ledger") or main_wd / "spend.sqlite"      # one ledger: budgets cap all seeds together
        out = run_seeds(lambda c, w: build_from_spec(spec, ds=ds, workdir=w, config=c, ledger_path=shared)[0], ds, cfg, a.seeds,
                        main_wd / "multirun", base_workdir=main_wd)
        print(json.dumps(out["summary"]["agreement"], indent=1, default=float))
        return 0
    if a.cmd == "compare":
        from .baselines import BASELINES, run_baseline
        from .multirun import compare_partitions
        import inspect
        for b in a.baselines:     # check names and options before the (long) pipeline run
            name, kw = _parse_baseline(b)
            if name not in BASELINES:
                raise ValueError(f"unknown baseline {name!r}; available: {', '.join(list_baselines())}")
            sig = inspect.signature(BASELINES[name]).parameters
            if not any(p.kind is p.VAR_KEYWORD for p in sig.values()):
                _unknown(kw, [k for k in sig if k not in ("X", "random_state")], f"baseline {name}")
        pipe, ds, _ = build_from_spec(spec)
        res = pipe.run(ds)
        X = np.load(Path(pipe.workdir) / "cache" / "embeddings.npy")
        parts = {"pipeline_clusters": res.labels("cluster")}
        if res.lines is not None:
            lines_lab, clus_lab = res.labels("line"), res.labels("cluster")
            from sklearn.metrics import adjusted_rand_score
            if adjusted_rand_score(clus_lab, lines_lab) < 1.0:     # identical partitions are compared (and judged) once
                parts["pipeline_lines"] = lines_lab
            else:
                print("note: report lines equal the clusters (no merges), so pipeline_lines is not listed separately",
                      file=sys.stderr)
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
