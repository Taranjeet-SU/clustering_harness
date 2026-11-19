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

__all__ = ["Pipeline", "ClusteringResult", "STAGES"]

log = logging.getLogger("clustering_harness")

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


class Pipeline:
    """The full clustering pipeline. Models and embedders are injected, so any provider can be used."""

    def __init__(self, config: PipelineConfig | None = None, llms: dict | None = None, embedder: Embedder | None = None,
                 label_embedder: Embedder | None = None, workdir: str | Path = "runs/default", prompts: PromptSet | None = None,
                 vectors: np.ndarray | None = None, progress: Callable[[str, dict], None] | None = None):
        self.cfg = config or PipelineConfig()
        self.llms = llms or {}
        self.embedder = embedder
        self.label_embedder = label_embedder or embedder   # labels/titles fall back to the main embedder
        self.workdir = Path(workdir)
        self.prompts = prompts
        self.vectors = vectors
        self.progress = progress
        missing = {"a", "b"} - set(self.llms)
        if missing:
            raise ValueError(f"llms must define slots 'a' and 'b' (missing {sorted(missing)})")
        for m in self.llms.values():
            if m.cache is None:
                m.cache = ResponseCache(self.workdir / "cache" / "llm.sqlite")

    # ------------------------------------------------------------------ helpers
    def _embed(self, ds: Dataset) -> np.ndarray:
        p = self.workdir / "cache" / "embeddings.npy"
        ids_p = self.workdir / "cache" / "embeddings.ids.json"
        if p.exists() and ids_p.exists() and json.loads(ids_p.read_text()) == ds.ids:
            return np.load(p)
        if self.vectors is not None:
            X = l2_normalize(self.vectors)
        else:
            if self.embedder is None:
                raise ValueError("an embedder (or precomputed vectors) is required")
            t = self.cfg.text
            X = l2_normalize(self.embedder.encode(ds.embedding_texts(TextTemplate(t.embed_with_title, t.embed_without_title))))
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
        T0 = time.time()
        wd = self.workdir
        wd.mkdir(parents=True, exist_ok=True)
        manifest = {"dataset": ds.name, "records": len(ds), "config_fingerprint": self.cfg.fingerprint(),
                    "models": {k: m.model_id for k, m in self.llms.items()},
                    "embedder": getattr(self.embedder, "name", "precomputed" if self.vectors is not None else None),
                    "label_embedder": getattr(self.label_embedder, "name", None)}
        old = wd / "manifest.json"
        if old.exists():
            prev = json.loads(old.read_text())
            if prev.get("config_fingerprint") != manifest["config_fingerprint"] or prev.get("records") != len(ds):
                raise ValueError(f"{wd} holds a run with a different config or dataset; use a new workdir")
        write_json(manifest, old)
        self.cfg.to_file(wd / "config.yaml")
        X = self._embed(ds)
        ctx = RunContext(ds=ds, cfg=self.cfg, llms=self.llms, workdir=wd, X=X, label_embedder=self.label_embedder,
                         prompts=self.prompts, progress_cb=self.progress)
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
        if not cb.exists():
            Cascade(ctx, gate).run(base_labels)
        if until == "cascade":
            return None

        # residual (C_full) -------------------------------------------------------------
        cf_p = wd / "results" / "residual" / "assignments.csv"
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
        if until == "residual":
            return None

        # refile (C_full_v2) --------------------------------------------------------------
        v2_p = wd / "results" / "refile" / "assignments.csv"
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
        if until == "refile":
            return None

        # scoring ------------------------------------------------------------------------
        sc_p = wd / "results" / "scoring" / "cluster_scores.csv"
        final = pd.read_csv(v2_p, dtype=str).fillna("")
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
        if until == "scoring":
            return None

        # rollup --------------------------------------------------------------------------
        lines = None
        ro_p = wd / "results" / "rollup" / "lines.csv"
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
                   "wall_minutes_this_call": round((time.time() - T0) / 60, 1), "human_validated": False}
        write_json(summary, out_dir / "summary.json")
        for c in {id(m.cache): m.cache for m in self.llms.values() if m.cache is not None}.values():
            c.checkpoint()
        logs = {k: ctx.load("results", *k.split("/"), default=None) for k in ("cascade_log.json", "residual/log.json", "refile/log.json", "rollup/log.json")}
        ctx.progress("export", stage="done", **{k: v for k, v in summary.items() if k not in ("llm_usage", "statuses")})
        return ClusteringResult(final, score, lines, summary, wd, logs)
