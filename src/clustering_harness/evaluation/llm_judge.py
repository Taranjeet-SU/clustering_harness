"""LLM-as-judge evaluation of any partition.

Protocol (the reference project's unified judging protocol): for every cluster, ``reps`` seeded samples of
up to ``sample_size`` members are shown to each of two judge models with a fixed 1–5 rubric ("would a
support lead file all of these under one recurring issue type?"). A cluster's score is the mean of all
valid scores; the *judge-balanced* score averages the two judges' means so neither model dominates.
Contract failures are counted, never imputed.

Because the protocol only needs ``{cluster: [ids]}``, the same evaluator scores this pipeline, baselines
(k-means, HDBSCAN, ...) and gold partitions on equal terms.

Usage::

    from clustering_harness.evaluation.llm_judge import LLMJudge
    judge = LLMJudge(ctx)                      # a RunContext (models, prompts, dataset)
    frame = judge.evaluate("kmeans_k77", clusters)   # DataFrame, one row per cluster
    LLMJudge.summary(frame)
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time

import numpy as np
import pandas as pd

__all__ = ["LLMJudge", "parse_coherence"]

_JSONRE = re.compile(r"\{.*\}", re.S)


def parse_coherence(content: str) -> dict:
    s = (content or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).rstrip("`").strip()
    try:
        j = json.loads(s)
    except Exception:  # noqa: BLE001
        m = _JSONRE.search(s)
        if not m:
            raise ValueError("no json")
        j = json.loads(m.group(0))
    c = min(5, max(1, int(j["coherence"])))
    return {"coherence": c, "name": str(j.get("name", ""))[:80], "reason": str(j.get("reason", ""))[:200]}


def mhash(members, ctx) -> str:
    return hashlib.md5(",".join(ctx.sort(members)).encode()).hexdigest()[:12]


class LLMJudge:
    def __init__(self, ctx, reps: int = 3, sample_size: int = 8, snippet_chars: int = 300, judges=("a", "b")):
        self.ctx = ctx
        self.reps = reps
        self.k = sample_size
        self.snip = snippet_chars
        self.judges = judges
        self.path = ctx.workdir / "cache" / "judge.jsonl"
        self._lock = threading.Lock()
        self._mem: dict | None = None

    # ------------------------------------------------------------------ record store
    def records(self) -> dict:
        with self._lock:
            if self._mem is None:
                self._mem = {}
                if self.path.exists():
                    for line in self.path.read_text().splitlines():
                        if line.strip():
                            r = json.loads(line)
                            self._mem[r["key"]] = r
            return self._mem

    def _append(self, rec):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self._mem[rec["key"]] = rec

    # ------------------------------------------------------------------ prompts
    def items(self, tids) -> str:
        P, ctx = self.ctx.prompts, self.ctx
        return "\n".join(P.render("judge_item", title=ctx.title(t), desc=ctx.desc_head(t, self.snip) or P["judge_empty_desc"]) for t in tids)

    def messages(self, tids) -> list:
        P = self.ctx.prompts
        prompt = P.render("judge_user", n=len(tids), items=self.items(tids), rubric=P["judge_rubric"])
        return [{"role": "system", "content": P["judge_system"]}, {"role": "user", "content": prompt}]

    def ask(self, slot: str, msgs: list, stage: str) -> dict:
        llm = self.ctx.llm(slot)
        for att in range(3):
            if not llm.reasoning:
                out = llm.call(stage, msgs, max_tokens=300, seed=att, as_json=True)
                if "_error" not in out:
                    try:
                        return {"coherence": min(5, max(1, int(out["coherence"]))), "name": str(out.get("name", ""))[:80],
                                "reason": str(out.get("reason", ""))[:200], "attempts": att + 1}
                    except Exception:  # noqa: BLE001
                        pass
            else:
                txt = llm.call(stage, msgs, max_tokens=1200 + 800 * att, seed=0, as_json=False)["text"]
                try:
                    r = parse_coherence(txt)
                    r["attempts"] = att + 1
                    return r
                except Exception:  # noqa: BLE001
                    pass
        return {"coherence": None, "name": "", "reason": "contract_failure", "attempts": 3}

    def job(self, arm, cid, members, rep, judge, stage) -> dict:
        h = mhash(members, self.ctx)
        key = f"{arm}|{cid}|{rep}|R|{judge}|{h}"
        hit = self.records().get(key)
        if hit is not None:
            return hit
        sids = self.ctx.jsample(arm, cid, rep, members, self.k)
        t0 = time.time()
        r = self.ask(judge, self.messages(sids), stage)
        rec = {"key": key, "arm": arm, "cluster": str(cid), "replicate": rep, "judge": judge, "size": len(members), "mhash": h,
               "sample_ids": sids, "stage": stage, "secs": round(time.time() - t0, 2), **r}
        self._append(rec)
        return rec

    # ------------------------------------------------------------------ API
    def evaluate(self, arm: str, clusters: dict, stage: str | None = None) -> pd.DataFrame:
        """Judge every cluster of ``clusters`` {cluster id: [record ids]} and return a per-cluster frame."""
        stage = stage or f"eval_{arm}"
        jobs = [(arm, cid, ms, r, j) for r in range(self.reps) for cid, ms in clusters.items() for j in self.judges]
        self.ctx.parallel(jobs, lambda jb: self.job(*jb, stage), lambda jb: jb[4])
        return self.frame(arm, clusters)

    def frame(self, arm: str, clusters: dict) -> pd.DataFrame:
        recs = self.records()
        rows = []
        for cid, ms in clusters.items():
            h = mhash(ms, self.ctx)
            keys = {j: [f"{arm}|{cid}|{r}|R|{j}|{h}" for r in range(self.reps)] for j in self.judges}
            vals = {j: [recs[k]["coherence"] for k in ks if k in recs and recs[k].get("coherence") is not None] for j, ks in keys.items()}
            allv = [v for j in self.judges for v in vals[j]]
            names = [recs[k]["name"] for r in range(self.reps) for j in self.judges
                     if (k := f"{arm}|{cid}|{r}|R|{j}|{h}") in recs and recs[k].get("name")]
            per = {j: float(np.mean(v)) if v else np.nan for j, v in vals.items()}
            rows.append({"arm": arm, "cluster": str(cid), "size": len(ms), "n": len(allv), "mean": float(np.mean(allv)) if allv else np.nan,
                         **{f"judge_{j}": per[j] for j in self.judges},
                         "balanced": float(np.mean(list(per.values()))) if all(v == v for v in per.values()) else np.nan,
                         "fails": sum(1 for ks in keys.values() for k in ks if k in recs and recs[k].get("coherence") is None),
                         "judge_name": names[0] if names else ""})
        return pd.DataFrame(rows)

    @staticmethod
    def summary(frame: pd.DataFrame, n_records: int | None = None) -> dict:
        """Unweighted and size-weighted mean scores; ``coverage`` if the population size is given."""
        f = frame.dropna(subset=["balanced"])
        out = {"clusters": int(len(frame)), "judged": int(len(f)),
               "mean_balanced": float(f.balanced.mean()) if len(f) else np.nan,
               "size_weighted_balanced": float((f.balanced * f["size"]).sum() / f["size"].sum()) if len(f) else np.nan,
               "share_clusters_ge_4": float((f.balanced >= 4).mean()) if len(f) else np.nan,
               "contract_failures": int(frame.fails.sum())}
        if n_records:
            out["coverage"] = float(frame["size"].sum() / n_records)
        return out
