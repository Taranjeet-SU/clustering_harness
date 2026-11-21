"""End-to-end pipeline orchestration.

>>> from clustering_harness import Pipeline, PipelineConfig, Dataset
>>> from clustering_harness.llm import LLM, OllamaBackend
>>> pipe = Pipeline(PipelineConfig.small(), llms={"a": LLM(OllamaBackend("qwen3:8b")), "b": LLM(OllamaBackend("gpt-oss:20b"), reasoning=True)},
...                 embedder=..., label_embedder=..., workdir="runs/demo")
>>> result = pipe.run(Dataset.from_csv("tickets.csv", id_col="id", text_col="body", title_col="subject"))

Stages (each persists its output under ``workdir`` and is skipped on rerun):

``embed`` → ``bank`` (consensus bank + base partition) → ``cascade`` → ``residual`` (C_full) →
``refile`` (C_full_v2) → ``scoring`` → ``rollup`` → ``export``.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .context import RunContext, write_json
from .data import Dataset, TextTemplate
from .embeddings import Embedder, l2_normalize
from .llm import ResponseCache
from .prompts import PromptSet

__all__ = ["Pipeline", "ClusteringResult", "STAGES", "IncompleteStageError", "reset_workdir"]

log = logging.getLogger("clustering_harness")

#: files and folders (relative to the workdir) written by each stage; ``*`` globs
STAGE_FILES = {
    "embed": ("cache/embeddings.npy", "cache/embeddings.ids.json"),
    "bank": ("results/bank", "results/base", "cache/partitions", "cache/geom"),
    "cascade": ("results/cascade_*", "results/cascade_log.json", "cache/cascade", "cache/gate_calibration.json", "results/model_problems/cascade.json"),
    "residual": ("results/residual", "cache/residual", "results/model_problems/residual.json"),
    "refile": ("results/refile", "cache/refile", "results/model_problems/refile.json"),
    "scoring": ("results/scoring", "cache/scoring", "cache/scoring_evidence", "results/model_problems/scoring.json"),
    "rollup": ("results/rollup", "results/model_problems/rollup.json"),
    "export": ("output",),
}


def reset_workdir(workdir: str | Path, from_stage: str | None = None) -> list[str]:
    """Delete stage outputs but keep the paid model answers (``cache/llm.sqlite``), label vectors and spend ledger.

    Without ``from_stage`` every stage is removed together with the manifest, so the next run may use new
    data, settings, prompts or models; requests whose prompt did not change are still answered from the cache.
    With ``from_stage`` that stage and all later ones are removed and the earlier ones are kept. The manifest is
    kept, so the next run still refuses changes that a kept stage depends on (see :data:`STAGE_DEPENDS`).
    Returns removed paths."""
    import shutil
    wd = Path(workdir)
    if from_stage is not None and from_stage not in STAGES:
        raise ValueError(f"unknown stage {from_stage!r}; stages are: {', '.join(STAGES)}")
    stages = STAGES[STAGES.index(from_stage):] if from_stage else STAGES
    pats = [p for st in stages for p in STAGE_FILES[st]] + ["progress.json"]
    if from_stage is None:
        pats += ["manifest.json", "config.yaml", "cache/judge.jsonl"]
    removed = []
    for pat in pats:
        for p in sorted(wd.glob(pat)):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed.append(str(p.relative_to(wd)))
    return removed


#: what each stage's saved output depends on (manifest keys; ``config.<section>`` for pipeline settings). A change is
#: accepted once every saved stage from the earliest dependent stage on has been removed (``reset --from-stage``).
STAGE_DEPENDS = {
    "embed": ("records", "data_fingerprint", "embedder", "embedder_options", "vectors_fingerprint", "config.text"),
    "bank": ("config.bank", "config.base", "config.junk"),
    "cascade": ("models", "model_settings", "prompts_fingerprint", "label_embedder", "label_embedder_options",
                "config.gate", "config.cascade", "config.masking", "config.roles", "config.prompts", "config.domain"),
    "residual": ("config.residual",),
    "refile": (),
    "scoring": ("config.scoring",),
    "rollup": ("config.rollup",),
    "export": (),
}
_LABELS = {"data_fingerprint": "data (record ids, titles or texts)", "prompts_fingerprint": "prompts (file contents or domain)",
           "vectors_fingerprint": "precomputed vectors"}


def _stage_of(key: str) -> int:
    for i, st in enumerate(STAGES):
        if key in STAGE_DEPENDS[st]:
            return i
    return 0          # unknown keys are treated as affecting everything


def _short(v, n: int = 80) -> str:
    t = json.dumps(v, default=str, sort_keys=True)
    return t if len(t) <= n else t[:n - 3] + "..."


def _manifest_changes(prev: dict, new: dict) -> list:
    """[(key, human-readable description)] of guarded differences between two manifests."""
    out = []
    for k in ("records", "data_fingerprint", "models", "model_settings", "prompts_fingerprint", "embedder",
              "embedder_options", "label_embedder", "label_embedder_options", "vectors_fingerprint"):
        if k in prev and prev[k] != new[k]:
            a, b = prev[k], new[k]
            if k in _LABELS:
                out.append((k, _LABELS[k]))
            elif isinstance(a, dict) and isinstance(b, dict):     # e.g. models: name the slot and show the full change
                out.append((k, "; ".join(f"{k}.{s}: {_short(a.get(s), 200)} -> {_short(b.get(s), 200)}"
                                         for s in sorted(set(a) | set(b)) if a.get(s) != b.get(s))))
            else:
                out.append((k, f"{k}: {_short(a, 200)} -> {_short(b, 200)}"))
    if "config" in prev:
        for sect in sorted(set(prev["config"]) | set(new["config"])):
            a, b = prev["config"].get(sect), new["config"].get(sect)
            if a == b:
                continue
            if isinstance(a, dict) and isinstance(b, dict):
                fields = [f"pipeline.{sect}.{f}: {_short(a.get(f), 40)} -> {_short(b.get(f), 40)}"
                          for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
            else:
                fields = [f"pipeline.{sect}: {_short(a, 40)} -> {_short(b, 40)}"]
            out.append((f"config.{sect}", "; ".join(fields)))
    elif prev.get("config_fingerprint") not in (None, new["config_fingerprint"]):
        out.append(("config", "pipeline settings (this workdir predates per-setting records)"))
    return out


class IncompleteStageError(RuntimeError):
    """Model requests failed during a stage, so the stage was not saved; rerun to retry only those requests."""


STAGES = ("embed", "bank", "cascade", "residual", "refile", "scoring", "rollup", "export")


@dataclass
class ClusteringResult:
    """Final outputs. ``assignments`` has one row per input record (dataset order)."""
    assignments: pd.DataFrame
    clusters: pd.DataFrame
    lines: pd.DataFrame | None
    summary: dict
    workdir: Path
    logs: dict = field(default_factory=dict)

    def labels(self, level: str = "cluster") -> np.ndarray:
        """Cluster (or report-line) id per record; ``""`` for unassigned / junk records."""
        col = "cluster_id" if level == "cluster" else "line_id"
        return self.assignments[col].fillna("").astype(str).to_numpy()

    def clusters_dict(self, level: str = "cluster") -> dict:
        col = "cluster_id" if level == "cluster" else "line_id"
        a = self.assignments[self.assignments[col].fillna("") != ""]
        return {c: g.ticket_id.astype(str).tolist() for c, g in a.groupby(col)}





def _data_fingerprint(ds: Dataset) -> str:
    """Hash of the records the run sees (ids, titles, texts in order)."""
    import hashlib
    h = hashlib.sha256()
    for rid, t, x in zip(ds.ids, ds.titles, ds.texts):
        h.update(json.dumps([rid, t, x]).encode())
    return h.hexdigest()[:20]


def _vectors_fingerprint(vectors) -> str | None:
    """Hash of precomputed vectors (shape, dtype and values), so a rerun with changed vectors is noticed."""
    if vectors is None:
        return None
    import hashlib
    v = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    h = hashlib.sha256(json.dumps(list(v.shape)).encode())
    h.update(v.tobytes())
    return h.hexdigest()[:20]


def _environment() -> dict:
    """Versions that can change embeddings or clusterings numerically."""
    import platform
    from importlib import metadata
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("clustering-harness", "numpy", "scikit-learn", "umap-learn", "hdbscan", "faiss-cpu", "leidenalg",
                "torch", "transformers", "sentence-transformers"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return env


def _embedder_settings(e) -> dict | None:
    e = getattr(e, "inner", e)
    if e is None:
        return None
    # only options that change the vectors (the name already records fp16 on GPU); device/batch size do not
    keys = ("model", "store_float16", "pooling", "max_length", "prefix")
    return {"class": type(e).__name__, **{k: getattr(e, k) for k in keys if hasattr(e, k)}}

class Pipeline:
    """The full clustering pipeline. Models and embedders are injected, so any provider can be used."""

    def __init__(self, config: PipelineConfig | None = None, llms: dict | None = None, embedder: Embedder | None = None,
                 label_embedder: Embedder | None = None, workdir: str | Path = "runs/default", prompts: PromptSet | None = None,
                 vectors: np.ndarray | None = None, progress: Callable[[str, dict], None] | None = None,
                 preflight: bool = True):
        self.cfg = config or PipelineConfig()
        self.llms = llms or {}
        self.embedder = embedder
        self.label_embedder = label_embedder or embedder   # labels/titles fall back to the main embedder
        self.workdir = Path(workdir)
        self.prompts = prompts
        self.vectors = vectors
        self.progress = progress
        self.preflight = preflight
        missing = {"a", "b"} - set(self.llms)
        if missing:
            raise ValueError(f"llms must define slots 'a' and 'b' (missing {sorted(missing)})")
        a, b = self.llms["a"], self.llms["b"]
        if a.model_id == b.model_id and a.settings == b.settings:
            log.warning("slots a and b use the same model (%s): the two judges send identical requests and share cached "
                        "answers, so the two-judge checks act as one judge and quality scores run high. Use two "
                        "different models for an independent second opinion.", a.model_id)
        for m in self.llms.values():
            if m.cache is None:
                m.cache = ResponseCache(self.workdir / "cache" / "llm.sqlite")

    # ------------------------------------------------------------------ helpers
    def _prompts(self) -> PromptSet:
        if self.prompts is None:
            self.prompts = PromptSet.load(self.cfg.prompts, domain=self.cfg.domain or None)
        return self.prompts

    def _preflight(self) -> None:
        """One tiny request per model before the slow stages, so a wrong key, model name or URL fails in seconds."""
        from .llm.backends import OfflineBackend
        seen = set()
        for m in self.llms.values():
            if id(m) in seen or isinstance(m.backend, OfflineBackend):
                continue
            seen.add(id(m))
            m.preflight()

    def _embed(self, ds: Dataset) -> np.ndarray:
        p = self.workdir / "cache" / "embeddings.npy"
        ids_p = self.workdir / "cache" / "embeddings.ids.json"
        if self.vectors is None and p.exists() and ids_p.exists() and json.loads(ids_p.read_text()) == ds.ids:
            return np.load(p)
        if self.vectors is not None:   # always taken from the caller (cheap), never from a possibly stale cache
            X = l2_normalize(self.vectors)
        else:
            if self.embedder is None:
                raise ValueError("an embedder (or precomputed vectors) is required")
            t = self.cfg.text
            name = getattr(self.embedder, "name", type(self.embedder).__name__)
            log.info("embed: %d records with %s (the first run downloads the model; later runs use the cache)", len(ds), name)
            t0 = time.time()
            X = l2_normalize(self.embedder.encode(ds.embedding_texts(TextTemplate(t.embed_with_title, t.embed_without_title))))
            log.info("embed: done in %.1f min", (time.time() - t0) / 60)
        if len(X) != len(ds) or not np.isfinite(X).all():
            raise ValueError("embeddings must be finite, one row per record")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, X)
        ids_p.write_text(json.dumps(ds.ids))
        return X

    def _assign_frame(self, ctx: RunContext, lab: dict, names: dict, how: dict | None = None, reasons: dict | None = None) -> pd.DataFrame:
        rows = []
        for i, t in enumerate(ctx.ds.ids):
            c = lab.get(t)
            j = bool(ctx.junk[i])
            rows.append({"ticket_id": t, "cluster_id": c or "", "cluster_name": ctx.mask(names.get(c, "")) if c else "",
                         "status": "junk" if j else ("cluster" if c else "unassigned"), "placement": (how or {}).get(t, ""),
                         "reason": "junk" if j else ("" if c else (reasons or {}).get(t, "not placed"))})
        return pd.DataFrame(rows)

    @staticmethod
    def _lab(frame: pd.DataFrame) -> tuple[dict, dict]:
        f = frame[frame.status == "cluster"]
        lab = dict(zip(f.ticket_id.astype(str), f.cluster_id.astype(str)))
        names = f.groupby("cluster_id").cluster_name.first().to_dict()
        return lab, names

    # ------------------------------------------------------------------ run
    def run(self, ds: Dataset, until: str | None = None) -> ClusteringResult | None:
        """Run (or resume) the pipeline. ``until`` stops after one of :data:`STAGES`.

        A stage during which any model request failed (after retries) is not saved: its files are removed and
        :class:`IncompleteStageError` is raised. The answers that did arrive are cached, so rerunning asks only
        the failed requests again."""
        self._snap = None
        try:
            return self._run(ds, until)
        except BaseException:
            self._discard_failed_stage()
            raise

    def reset(self, from_stage: str | None = None) -> list[str]:
        """See :func:`reset_workdir`."""
        return reset_workdir(self.workdir, from_stage)

    def _model_problems(self) -> dict:
        """Per slot: answers that stayed unparseable and requests the provider rejected in the *saved* stages (kept
        across reruns; removed with their stage by ``reset``), with the stages they belong to."""
        out: dict = {}
        files = (self.workdir / "results" / "model_problems").glob("*.json")
        for f in sorted(files, key=lambda f: STAGES.index(f.stem) if f.stem in STAGES else 99):
            for slot, v in json.loads(f.read_text()).items():
                o = out.setdefault(slot, {"unparseable_answers": 0, "rejected_requests": 0, "stages": []})
                o["unparseable_answers"] += v["unparseable_answers"]
                o["rejected_requests"] += v["rejected_requests"]
                o["stages"].append(f.stem)
        return out

    # ------------------------------------------------------------------ failed-request guard
    def _unique_llms(self) -> list:
        out, seen = [], set()
        for m in self.llms.values():
            if id(m) not in seen:
                seen.add(id(m))
                out.append(m)
        return out

    def _failed_requests(self) -> int:
        return sum(m.stats.get("uncached_failures", 0) for m in self._unique_llms())

    _KEEP = ("manifest.json", "config.yaml", "spend.sqlite", "cache/llm.sqlite", "cache/embeddings")

    def _files(self) -> set:
        wd = self.workdir
        return {p for p in wd.rglob("*") if p.is_file()
                and not str(p.relative_to(wd)).replace("\\", "/").startswith(self._KEEP)}

    def _problem_counts(self) -> dict:
        return {k: (m.stats.get("contract_failures", 0), m.stats.get("rejected", 0)) for k, m in self.llms.items()}

    def _stage_start(self, stage: str) -> None:
        self._snap = (stage, self._failed_requests(), self._files())
        self._problems0 = self._problem_counts()

    def _discard_failed_stage(self) -> int:
        """Remove the files written by the current stage if model requests failed during it; returns the count."""
        if not getattr(self, "_snap", None):
            return 0
        stage, f0, files0 = self._snap
        n = self._failed_requests() - f0
        if n > 0:
            for p in self._files() - files0:
                p.unlink(missing_ok=True)
                for d in p.parents:          # and the folders that are now empty
                    if d == self.workdir or any(d.iterdir()):
                        break
                    d.rmdir()
            log.error("%s: %d model request(s) failed; the stage was not saved", stage, n)
        self._snap = None
        return n

    def _stage_end(self) -> None:
        stage = self._snap[0] if self._snap else "?"
        n = self._discard_failed_stage()
        if not n and stage != "?":
            # answers that stayed unparseable / were rejected are part of this saved stage: record them with it
            now, before = self._problem_counts(), getattr(self, "_problems0", {})
            delta = {k: {"unparseable_answers": now[k][0] - before.get(k, (0, 0))[0],
                         "rejected_requests": now[k][1] - before.get(k, (0, 0))[1]} for k in now}
            delta = {k: v for k, v in delta.items() if v["unparseable_answers"] or v["rejected_requests"]}
            if delta:
                write_json(delta, self.workdir / "results" / "model_problems" / f"{stage}.json")
        if n:
            raise IncompleteStageError(
                f"stage {stage!r}: {n} model request(s) failed after retries (see the warnings above), so its results "
                f"were not saved. Rerun the same command when the model server is healthy: answers that arrived are "
                f"cached, so only the failed requests are sent again.")

    def _run(self, ds: Dataset, until: str | None = None) -> ClusteringResult | None:
        T0 = time.time()
        if until is not None and until not in STAGES:
            raise ValueError(f"unknown stage {until!r} for until; stages are: {', '.join(STAGES)}")
        if len(ds) == 0:
            raise ValueError(f"dataset {ds.name!r} has no records (check the file, query or collection names)")
        wd = self.workdir
        wd.mkdir(parents=True, exist_ok=True)
        manifest = {"dataset": ds.name, "records": len(ds), "data_fingerprint": _data_fingerprint(ds),
                    "config_fingerprint": self.cfg.fingerprint(),
                    "config": self.cfg.result_settings(),
                    "models": {k: m.model_id for k, m in self.llms.items()},
                    "embedder": getattr(self.embedder, "name", "precomputed" if self.vectors is not None else None),
                    "label_embedder": getattr(self.label_embedder, "name", None),
                    "embedder_options": _embedder_settings(self.embedder),
                    "label_embedder_options": _embedder_settings(self.label_embedder),
                    "model_settings": {k: m.settings for k, m in self.llms.items()},
                    "prompts_fingerprint": self._prompts().fingerprint(),
                    "vectors_fingerprint": _vectors_fingerprint(self.vectors),
                    "environment": _environment()}
        old = wd / "manifest.json"
        if old.exists():
            prev = json.loads(old.read_text())
            manifest["first_run_environment"] = prev.get("first_run_environment") or prev.get("environment")
            changes = _manifest_changes(prev, manifest)
            if changes:
                first = min(_stage_of(k) for k, _ in changes)
                saved = [st for st in STAGES[first:] if any(any(wd.glob(p)) for p in STAGE_FILES[st])]
                if saved:
                    what = "\n  ".join(d for _, d in changes)
                    raise ValueError(
                        f"{wd} holds a run made with different settings:\n  {what}\nThe saved stage(s) {', '.join(saved)} "
                        f"depend on them and would be mixed with the new settings. Rerun from the first affected stage with "
                        f"`clustering-harness reset <spec> --from-stage {STAGES[first]}` (keeps earlier stages and the paid "
                        f"model answers), start over with `clustering-harness reset <spec>`, or use a new workdir.")
                log.info("settings changed (%s); no saved stage depends on them", ", ".join(k for k, _ in changes))
        manifest["first_run_environment"] = manifest.get("first_run_environment") or manifest["environment"]
        if self.preflight and until not in ("embed", "bank") and not (wd / "output" / "assignments.csv").exists():
            self._preflight()          # before the manifest is written, so a corrected model name is not refused
        write_json(manifest, old)
        self.cfg.to_file(wd / "config.yaml")
        X = self._embed(ds)
        ctx = RunContext(ds=ds, cfg=self.cfg, llms=self.llms, workdir=wd, X=X, label_embedder=self.label_embedder,
                         prompts=self._prompts(), progress_cb=self.progress)
        if until == "embed":
            return None
        from .algorithms.cascade import Cascade
        from .algorithms.gate import Gate
        from .algorithms.partitions import build_bank
        from .algorithms.residual import ResidualPass
        gate = Gate(ctx)

        # bank + base -----------------------------------------------------------
        base_p = wd / "results" / "base" / "labels.npy"
        if base_p.exists():
            base_labels = np.load(base_p)
        else:
            ctx.progress("bank", stage="building consensus bank")
            grid, (rid, base_labels) = build_bank(ctx)
            np.save(ctx.path("results", "base", "labels.npy"), base_labels)
            full = {t: f"c{l}" for t, l in zip(ctx.eligible_ids, base_labels) if l >= 0}
            self._assign_frame(ctx, full, {}, reasons={t: "noise" for t in ctx.eligible_ids if t not in full}).to_csv(
                ctx.path("results", "base", "assignments.csv"), index=False)
            ctx.save({"source_run": rid, "partitions": len(grid), "clusters": int(base_labels.max() + 1) if len(base_labels) else 0,
                      "assigned": int((base_labels >= 0).sum())}, "results", "base", "source.json")
        ctx.progress("bank", stage="done", clusters=int(base_labels.max() + 1) if len(base_labels) else 0,
                     assigned=int((base_labels >= 0).sum()))
        if until == "bank":
            return None

        # cascade ---------------------------------------------------------------------
        cb = wd / "results" / "cascade_B" / "assignments.csv"
        self._stage_start("cascade")
        if not cb.exists():
            Cascade(ctx, gate).run(base_labels)
        self._stage_end()
        if until == "cascade":
            return None

        # residual (C_full) -------------------------------------------------------------
        cf_p = wd / "results" / "residual" / "assignments.csv"
        self._stage_start("residual")
        if not cf_p.exists():
            b = pd.read_csv(cb, dtype=str).fillna("")
            lab, names = self._lab(b)
            resid = b[b.status == "unassigned"].ticket_id.tolist()
            r = ResidualPass(ctx, gate, "v1").run(lab, names, resid, "residual")
            reasons = {t: f"LLM junk: {v}" for t, v in r["junk"].items()}
            f = self._assign_frame(ctx, r["labels"], r["names"], {**{t: "cascade_B" for t in lab}, **r["how"]}, reasons)
            cf_p.parent.mkdir(parents=True, exist_ok=True)
            f.to_csv(cf_p, index=False)
            ctx.save(r["log"], "results", "residual", "log.json")
        self._stage_end()
        if until == "residual":
            return None

        # refile (C_full_v2) --------------------------------------------------------------
        v2_p = wd / "results" / "refile" / "assignments.csv"
        self._stage_start("refile")
        if not v2_p.exists():
            a = pd.read_csv(cf_p, dtype=str).fillna("")
            if self.cfg.residual.refile:
                lab, names = self._lab(a)
                pool = a[a.status == "unassigned"].ticket_id.tolist()
                r = ResidualPass(ctx, gate, "v2").run(lab, names, pool, "refile")
                old_how = dict(zip(a.ticket_id, a.placement))
                how = {t: (old_how.get(t, "") if t in lab else r["how"].get(t, "")) for t in r["labels"]}
                reasons = {t: f"LLM junk: {v}" for t, v in r["junk"].items()}
                f = self._assign_frame(ctx, r["labels"], r["names"], how, reasons)
                f["previous_reason"] = a.reason.values
                v2_p.parent.mkdir(parents=True, exist_ok=True)
                f.to_csv(v2_p, index=False)
                ctx.save(r["log"], "results", "refile", "log.json")
            else:
                v2_p.parent.mkdir(parents=True, exist_ok=True)
                a.to_csv(v2_p, index=False)
        self._stage_end()
        if until == "refile":
            return None

        # scoring ------------------------------------------------------------------------
        sc_p = wd / "results" / "scoring" / "cluster_scores.csv"
        final = pd.read_csv(v2_p, dtype=str).fillna("")
        self._stage_start("scoring")
        if not sc_p.exists():
            from .algorithms.scoring import ClusterScorer
            groups = {c: g.ticket_id.tolist() for c, g in final[final.status == "cluster"].groupby("cluster_id")}
            prov = final[final.status == "cluster"].groupby("cluster_id").cluster_name.first().to_dict()
            assignment = dict(zip(final.ticket_id, final.cluster_id))
            if groups:
                score = ClusterScorer(ctx, groups, assignment, prov).run()
            else:   # nothing was clustered: empty, well-formed outputs
                score = pd.DataFrame(columns=["cluster_id", "cluster_name", "case_count", "semantic_score_1_to_5", "cosine_silhouette_mean"])
            sc_p.parent.mkdir(parents=True, exist_ok=True)
            score.to_csv(sc_p, index=False)
        score = pd.read_csv(sc_p, dtype={"cluster_id": str})
        newnames = dict(zip(score.cluster_id, score.cluster_name))
        final["cluster_name"] = [newnames.get(c, "") for c in final.cluster_id]
        self._stage_end()
        if until == "scoring":
            return None

        # rollup --------------------------------------------------------------------------
        lines = None
        ro_p = wd / "results" / "rollup" / "lines.csv"
        self._stage_start("rollup")
        if self.cfg.rollup.enabled and len(score):
            if not ro_p.exists():
                from .algorithms.rollup import Rollup
                cl = final[final.status == "cluster"]   # names as rewritten by scoring
                cmem = {c: g.ticket_id.tolist() for c, g in cl.groupby("cluster_id")}
                cname = cl.groupby("cluster_id").cluster_name.first().to_dict()
                out = Rollup(ctx).run(cmem, cname)
                ro_p.parent.mkdir(parents=True, exist_ok=True)
                out["lines"].to_csv(ro_p, index=False)
                ctx.save(out["cluster_to_line"], "results", "rollup", "cluster_to_line.json")
                ctx.save(out["line_names"], "results", "rollup", "line_names.json")
                ctx.save(out["log"], "results", "rollup", "log.json")
                out["guard"].to_csv(ctx.path("results", "rollup", "guard.csv"), index=False)
                out["census"].to_csv(ctx.path("results", "rollup", "census.csv"), index=False)
            lines = pd.read_csv(ro_p)
            c2l = ctx.load("results", "rollup", "cluster_to_line.json")
            lnames = ctx.load("results", "rollup", "line_names.json")
            final["line_id"] = [c2l.get(c, "") for c in final.cluster_id]
            final["line_name"] = [lnames.get(l, "") for l in final.line_id]
        self._stage_end()
        if until == "rollup":
            return None

        # export ---------------------------------------------------------------------------
        out_dir = wd / "output"
        out_dir.mkdir(exist_ok=True)
        final.insert(1, "title", [ctx.mask(t) for t in ds.titles])
        final.to_csv(out_dir / "assignments.csv", index=False)
        score.to_csv(out_dir / "clusters.csv", index=False)
        if lines is not None:
            lines.to_csv(out_dir / "report_lines.csv", index=False)
        st = final.status.value_counts().to_dict()
        summary = {"records": len(ds), "clusters": int(score.shape[0]), "report_lines": int(len(lines)) if lines is not None else None,
                   "statuses": st, "coverage_all": round(st.get("cluster", 0) / len(ds), 4),
                   "coverage_eligible": round(st.get("cluster", 0) / max(1, len(ds) - st.get("junk", 0)), 4),
                   "semantic_mean": float(score.semantic_score_1_to_5.mean()) if len(score) else None,
                   "semantic_size_weighted": float((score.semantic_score_1_to_5 * score.case_count).sum() / score.case_count.sum()) if len(score) else None,
                   "silhouette_size_weighted": float((score.cosine_silhouette_mean * score.case_count).sum() / score.case_count.sum()) if len(score) else None,
                   "llm_usage": {k: m.stats for k, m in self.llms.items()},
                   "model_problems": self._model_problems(),
                   "wall_minutes_this_call": round((time.time() - T0) / 60, 1), "human_validated": False}
        prev_p = out_dir / "summary.json"
        if prev_p.exists():
            # a resume or `compare` reaches export again: keep the usage and time of the calls that did the work
            try:
                prev = json.loads(prev_p.read_text())
            except Exception:  # noqa: BLE001
                prev = {}
            summary["llm_usage_this_call"] = summary["llm_usage"]
            summary["llm_usage"] = {k: {s: (v or 0) + (prev.get("llm_usage", {}).get(k, {}).get(s) or 0)
                                        for s, v in u.items()} for k, u in summary["llm_usage"].items()}
            summary["wall_minutes_total"] = round((prev.get("wall_minutes_total") or prev.get("wall_minutes_this_call") or 0)
                                                  + summary["wall_minutes_this_call"], 1)
        else:
            summary["wall_minutes_total"] = summary["wall_minutes_this_call"]
        write_json(summary, out_dir / "summary.json")
        for slot, p in summary["model_problems"].items():
            log.warning("model %s (%s): %d unparseable answer(s) and %d rejected request(s) in stage(s) %s; those items were "
                        "handled as 'no answer'. To ask them again after fixing the cause: clustering-harness reset <spec> "
                        "--from-stage %s", slot, self.llms[slot].model_id, p["unparseable_answers"], p["rejected_requests"],
                        ", ".join(p["stages"]), p["stages"][0])
        for c in {id(m.cache): m.cache for m in self.llms.values() if m.cache is not None}.values():
            c.checkpoint()
        logs = {k: ctx.load("results", *k.split("/"), default=None) for k in ("cascade_log.json", "residual/log.json", "refile/log.json", "rollup/log.json")}
        ctx.progress("export", stage="done", **{k: v for k, v in summary.items() if k not in ("llm_usage", "statuses")})
        return ClusteringResult(final, score, lines, summary, wd, logs)
