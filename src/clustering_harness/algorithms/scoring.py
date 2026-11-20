"""Per-cluster scoring (pipeline doc step 17): exact geometry, evidence-grounded naming and adaptive
two-model semantic review.

Geometry (in the embedding space, assigned records only):
  * cosine silhouette per record (exact, :func:`sklearn.metrics.silhouette_samples`), averaged per cluster;
  * mean pairwise cosine (analytic, self-pairs excluded), mean / 10th-percentile cosine to the centroid;
  * mean share of each member's 10 nearest eligible neighbours in the same cluster;
  * centroid margin = own-centroid cosine minus best other-centroid cosine (and its negative share);
  * exact-duplicate text counts.

Naming: the naming model sees 12 member excerpts (central, boundary, diverse, seeded random; unique texts)
and 2 excerpts from each of the two nearest clusters, and returns a name with inclusion/exclusion criteria.

Review: both models score a 12-case panel (excluding naming cases where possible) for coherence, label fit
and distinction. A second panel (second model only) is drawn for large clusters, any flagged concern or a
judge disagreement >= 1; a third only when the disagreement persists. Each model's mean has equal weight."""
from __future__ import annotations

import collections
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..context import RunContext
from ..llm.base import ModelUnavailableError
from .partitions import knn_graph

__all__ = ["ClusterScorer", "Name", "Judgment", "cluster_geometry"]


class Name(BaseModel):
    name: str = Field(min_length=5, max_length=120)
    inclusion_criterion: str = Field(min_length=15, max_length=700)
    exclusion_criterion: str = Field(min_length=5, max_length=700)
    rationale: str = Field(min_length=10, max_length=1000)
    unsupported_cause_warning: str


class Judgment(BaseModel):
    coherence: int = Field(ge=1, le=5)
    label_fit: int = Field(ge=1, le=5)
    distinction: int = Field(ge=1, le=5)
    decision: Literal['ACCEPT', 'REPAIR', 'SPLIT', 'REJECT', 'UNCERTAIN']
    reason: str = Field(min_length=15, max_length=1800)
    supporting_ids: list[str]
    suspicious_ids: list[str]
    subissues: list[str]
    limitation: str


def cluster_geometry(ctx: RunContext, groups: dict, assignment: dict, hashes: dict) -> pd.DataFrame:
    """Geometry table for ``groups`` {cluster: [ids]}; ``assignment`` {id: cluster or ''} for all records."""
    from sklearn import config_context
    from sklearn.metrics import silhouette_samples
    X, pos = ctx.X, ctx.pos
    cids = sorted(groups)
    cpos = {c: i for i, c in enumerate(cids)}
    C = np.stack([X[[pos[t] for t in groups[c]]].mean(0) for c in cids])
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    cs = C @ C.T
    np.fill_diagonal(cs, -np.inf)
    nearest = np.argsort(-cs, axis=1)[:, :min(2, max(0, len(cids) - 1))]
    active = [(t, assignment[t]) for t in ctx.ds.ids if assignment.get(t)]
    aidx = np.array([pos[t] for t, _ in active])
    labs = np.array([c for _, c in active])
    with config_context(working_memory=256):
        sil = silhouette_samples(X[aidx], labs, metric="cosine") if len(cids) > 1 else np.full(len(active), np.nan)
    silmap = dict(zip([t for t, _ in active], sil))
    I, _ = knn_graph(ctx)
    nj = ctx.nonjunk
    npidx = {int(i): k for k, i in enumerate(nj)}
    hashclusters = collections.defaultdict(set)
    for t, c in assignment.items():
        if c:
            hashclusters[hashes[t]].add(c)
    rows = []
    for c in cids:
        tids = groups[c]
        ii = np.array([pos[t] for t in tids])
        xx = X[ii]
        n = len(ii)
        s = xx @ C.T
        own = s[:, cpos[c]].copy()
        s[:, cpos[c]] = -np.inf
        alt = s.max(1) if len(cids) > 1 else np.full(n, np.nan)
        margin = own - alt
        consistency = []
        for t, i in zip(tids, ii):
            j = npidx[int(i)]
            neighbors = [int(v) for v in I[j] if v != j][:10]
            consistency.append(np.mean([assignment.get(ctx.ds.ids[nj[v]], "") == c for v in neighbors]))
        hs = [hashes[t] for t in tids]
        pairwise = (float(np.square(xx.sum(0, dtype=np.float64)).sum()) - n) / (n * (n - 1)) if n > 1 else np.nan
        rows.append({"cluster_id": c, "case_count": n, "unique_exact_texts": len(set(hs)), "duplicate_text_fraction": 1 - len(set(hs)) / n,
                     "cases_with_exact_text_also_in_other_clusters": sum(len(hashclusters[h]) > 1 for h in hs),
                     "cosine_silhouette_mean": float(np.nanmean([silmap[t] for t in tids])), "mean_pairwise_cosine": pairwise,
                     "mean_centroid_cosine": float(own.mean()), "p10_centroid_cosine": float(np.percentile(own, 10)),
                     "mean_10nn_same_cluster_fraction": float(np.mean(consistency)), "mean_centroid_margin": float(np.nanmean(margin)),
                     "negative_centroid_margin_fraction": float(np.mean(margin < 0)),
                     "nearest_cluster_id": cids[nearest[cpos[c], 0]] if len(cids) > 1 else "",
                     "nearest_centroid_cosine": float(cs[cpos[c], nearest[cpos[c], 0]]) if len(cids) > 1 else np.nan})
    return pd.DataFrame(rows)


class ClusterScorer:
    def __init__(self, ctx: RunContext, groups: dict, assignment: dict, provisional_names: dict):
        self.ctx = ctx
        self.S = ctx.cfg.scoring
        self.groups = {c: list(v) for c, v in groups.items()}
        self.assignment = assignment
        self.oldnames = provisional_names
        self.cids = sorted(self.groups)
        self.cpos = {c: i for i, c in enumerate(self.cids)}
        X, pos = ctx.X, ctx.pos
        C = np.stack([X[[pos[t] for t in self.groups[c]]].mean(0) for c in self.cids])
        self.C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        cs = self.C @ self.C.T
        np.fill_diagonal(cs, -np.inf)
        self.nearest = np.argsort(-cs, axis=1)[:, :min(2, max(0, len(self.cids) - 1))]
        tpl_texts = ctx.ds.text_hashes(_template(ctx))
        self.hashes = dict(zip(ctx.ds.ids, tpl_texts))
        gp = ctx.workdir / "results" / "scoring" / "cluster_geometry.csv"
        if gp.exists():
            self.geom = pd.read_csv(gp, dtype={"cluster_id": str}).set_index("cluster_id")
        else:
            g = cluster_geometry(ctx, self.groups, assignment, self.hashes)
            g.to_csv(ctx.path("results", "scoring", "cluster_geometry.csv"), index=False)
            self.geom = pd.read_csv(gp, dtype={"cluster_id": str}).set_index("cluster_id")
        self.rev = ctx.workdir / "cache" / "scoring"
        self.rev.mkdir(parents=True, exist_ok=True)
        self._file_lock = threading.Lock()

    # ------------------------------------------------------------------ evidence
    def sample(self, c, rep=0, n=12, exclude=()):
        ctx, X, pos = self.ctx, self.ctx.X, self.ctx.pos
        ids = ctx.sort(self.groups[c])
        seen, unique = set(), []
        for t in ids:
            h = self.hashes[t]
            if h not in seen:
                unique.append(t); seen.add(h)
        preferred = [t for t in unique if t not in set(exclude)]
        pool = preferred if len(preferred) >= min(8, len(unique)) else unique
        rng = np.random.default_rng(int(hashlib.sha256(f"{c}|{rep}".encode()).hexdigest()[:8], 16))
        xx = X[[pos[t] for t in pool]]
        own = xx @ self.C[self.cpos[c]]
        order = np.argsort(-own)
        picks, used = [], set()

        def add(i, role):
            t = pool[int(i)]
            if t not in used and len(picks) < n:
                picks.append((t, role)); used.add(t)
        for i in order[:2]:
            add(i, "central")
        for i in order[-2:][::-1]:
            add(i, "boundary")
        if len(pool):
            dist = np.ones(len(pool))
            for t, _ in picks:
                dist = np.minimum(dist, 1 - xx @ X[pos[t]])
            for _ in range(min(2, len(pool))):
                i = int(np.argmax(dist)); add(i, "diverse")
                dist = np.minimum(dist, 1 - xx @ xx[i]); dist[i] = -1
        for i in rng.permutation(len(pool)):
            add(i, "random")
        return picks

    def evidence(self, t, role, n=700):
        ctx = self.ctx
        text = ctx.mask(str(ctx.ds.title(t)) + "\n" + str(ctx.ds.text(t)))
        clip = len(text) > n
        excerpt = text if not clip else text[:n * 3 // 4] + " […] " + text[-n // 4:]
        return {"id": t, "role": role, "text": excerpt, "excerpt_clipped": clip}

    # ------------------------------------------------------------------ cached structured calls
    def _write(self, obj, path):
        from ..context import write_json
        write_json(obj, path)

    def structured(self, stage, slot, payload, Schema, path, ids=None):
        if path.exists():
            return Schema.model_validate_json(path.read_text()).model_dump()
        P = self.ctx.prompts
        llm = self.ctx.llm(slot)
        msgs = [{"role": "system", "content": P["judge_system"] + P["score_system_suffix"]},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False) + "\nJSON schema: " + json.dumps(Schema.model_json_schema())}]
        id_only_failure = None
        for attempt in range(3):
            out = llm.call(stage, msgs, max_tokens=2400, seed=attempt)
            try:
                result = Schema.model_validate(out).model_dump()
                if ids is not None:
                    unknown = sorted((set(result["supporting_ids"]) | set(result["suspicious_ids"])) - ids)
                    both = sorted(set(result["supporting_ids"]) & set(result["suspicious_ids"]))
                    if unknown or both:
                        id_only_failure = result
                        raise ValueError(f"ids not among own_cases: {unknown[:10]}; ids listed as both supporting and suspicious: {both[:10]}")
                self._write(result, path)
                return result
            except ModelUnavailableError:
                raise
            except Exception as e:  # noqa: BLE001
                self._write({"error": str(e)[:800], "response": out}, path.with_suffix(f".attempt{attempt}.error.json"))
                msgs = msgs[:2] + [{"role": "user", "content": "Previous response failed schema/ID validation. Return a complete valid object. "
                                                              "IDs must be supplied own-case IDs. Error: " + str(e)[:400]}]
        if id_only_failure is not None:
            # schema-valid answer whose id lists never validated: keep it with the invalid ids removed, flagged
            r = dict(id_only_failure)
            both = set(r["supporting_ids"]) & set(r["suspicious_ids"])
            r["supporting_ids"] = [t for t in r["supporting_ids"] if t in ids and t not in both]
            r["suspicious_ids"] = [t for t in r["suspicious_ids"] if t in ids]
            r["limitation"] = (r.get("limitation") or "") + " [ids sanitised after 3 invalid attempts]"
            self._write(r, path)
            return r
        raise RuntimeError(f"{stage} {slot}: invalid responses for {path.name}")

    # ------------------------------------------------------------------ naming & review
    def name_cluster(self, c):
        ctx, S = self.ctx, self.S
        ms = self.groups[c]
        near = [self.cids[i] for i in self.nearest[self.cpos[c]]]
        competitors = [{"cluster_id": k, "provisional_name": self.oldnames.get(k, ""),
                        "examples": [self.evidence(t, "competing_cluster", S.competitor_chars) for t, _ in self.sample(k, 99, 2)]} for k in near]
        naming = self.sample(c, 800, n=S.panel_size)
        payload = {"task": ctx.prompts["naming_task"], "cluster_id": c, "size": len(ms), "unique_texts": int(self.geom.loc[c, "unique_exact_texts"]),
                   "own_cases": [self.evidence(t, r, S.naming_chars) for t, r in naming], "competing_clusters": competitors}
        ctx.save(payload, "cache", "scoring_evidence", f"{c}.naming.json")
        final = self.structured("score_name", S.name_with, payload, Name, self.rev / f"{c}.name_final.json")
        final = {k: ctx.mask(v) for k, v in final.items()}
        return ms, competitors, naming, final

    def judge_payload(self, c, ms, final, picked, competitors):
        P = self.ctx.prompts
        return {"task": P["judgment_task"], "rubric": P["judge_rubric"], "label_fit_rubric": P["label_fit_rubric"],
                "distinction_rubric": P["distinction_rubric"], "cluster_id": c, "cluster_size": len(ms), "proposed_label": final["name"],
                "inclusion": final["inclusion_criterion"], "own_cases": [self.evidence(t, r, self.S.evidence_chars) for t, r in picked],
                "competing_clusters": competitors}

    def process(self, c):
        ctx, S = self.ctx, self.S
        completed = self.rev / f"{c}.complete.json"
        if completed.exists():
            return json.loads(completed.read_text())["row"]
        ms, competitors, naming, final = self.name_cluster(c)
        judgments, evaluated, overlap, reasons, escalation = [], set(), set(), [], []
        panel_count = 0
        namehash = hashlib.sha256(final["name"].encode()).hexdigest()[:10]
        follow = "b" if S.name_with == "b" else "a"
        for rep in range(S.max_panels):
            picked = self.sample(c, rep, n=S.panel_size, exclude={t for t, _ in naming} | evaluated)
            ownids = {t for t, _ in picked}
            evaluated |= ownids
            overlap |= ownids & {t for t, _ in naming}
            ep = self.judge_payload(c, ms, final, picked, competitors)
            ctx.save(ep, "cache", "scoring_evidence", f"{c}.adaptive_rep{rep}.{namehash}.json")
            judges = ["a", "b"] if rep == 0 else [follow]
            with ThreadPoolExecutor(2) as ex:
                fs = {j: ex.submit(self.structured, "score_judgment", j, ep, Judgment, self.rev / f"{c}.adaptive_rep{rep}.{namehash}.{j}.json", ownids)
                      for j in judges}
                for j, f in fs.items():
                    z = f.result()
                    judgments.append({"judge": j, "rep": rep, **z}); reasons.append(ctx.llm(j).name + ": " + z["reason"])
            panel_count += 1
            current = [z for z in judgments if z["rep"] == rep]
            concerns = any(z["coherence"] < 4 or z["label_fit"] < 4 or z["distinction"] < 4 or z["decision"] != "ACCEPT" or z["suspicious_ids"] for z in current)
            qscore = np.mean([z["coherence"] for z in judgments if z["judge"] == "a"])
            gscore = np.mean([z["coherence"] for z in judgments if z["judge"] == "b"])
            if len(evaluated) >= int(self.geom.loc[c, "unique_exact_texts"]):
                break
            if rep == 0:
                if len(ms) >= S.escalate_size:
                    escalation.append(f"size >={S.escalate_size}")
                if concerns:
                    escalation.append("first panel flagged quality, name, distinction or suspicious members")
                if abs(qscore - gscore) >= 1:
                    escalation.append("first-panel judge disagreement")
                if not escalation:
                    break
            elif rep == 1:
                pooled = {j: np.mean([z["coherence"] for z in judgments if z["judge"] == j]) for j in ("a", "b")}
                if abs(pooled["a"] - pooled["b"]) < 1:
                    break
                escalation.append("persistent judge disagreement after two panels")
        per = {j: np.mean([z["coherence"] for z in judgments if z["judge"] == j]) for j in ("a", "b")}
        decisions = collections.Counter(z["decision"] for z in judgments)
        suspicious = ctx.sort({t for z in judgments for t in z["suspicious_ids"]})
        g = self.geom.loc[c].to_dict()
        row = {"cluster_id": c, "cluster_name": final["name"], "inclusion_criterion": final["inclusion_criterion"],
               "exclusion_criterion": final["exclusion_criterion"], "naming_rationale": final["rationale"],
               "unsupported_cause_warning": final["unsupported_cause_warning"], **g,
               "semantic_score_1_to_5": float(np.mean(list(per.values()))), "judge_a_semantic_score": float(per["a"]),
               "judge_b_semantic_score": float(per["b"]), "judge_disagreement": float(abs(per["a"] - per["b"])),
               "label_fit_score_1_to_5": float(np.mean([np.mean([z["label_fit"] for z in judgments if z["judge"] == j]) for j in ("a", "b")])),
               "distinction_score_1_to_5": float(np.mean([np.mean([z["distinction"] for z in judgments if z["judge"] == j]) for j in ("a", "b")])),
               "judgment_decisions": json.dumps(decisions), "judgment_sample_unique_cases": len(evaluated),
               "judgment_sample_fraction": len(evaluated) / len(ms), "evaluation_cases_also_used_for_naming": len(overlap),
               "suspicious_sample_ids": "; ".join(suspicious), "example_case_ids": "; ".join(t for t, _ in naming[:5]),
               "semantic_reasons": " | ".join(reasons),
               "quality_status": "REVIEW_RECOMMENDED" if suspicious or min(per.values()) < 4 or any(z["decision"] != "ACCEPT" for z in judgments) else "MODEL_REVIEW_PASSED",
               "judge_a_panels": sum(z["judge"] == "a" for z in judgments), "judge_b_panels": sum(z["judge"] == "b" for z in judgments),
               "review_limitation": "Sampled, model-assisted text-only judgment; not human-audited purity. Same model families used during clustering. Exact duplicates excluded from semantic sampling where possible.",
               "evaluation_protocol": "dual_first_fast_followup_v3",
               "panels_reviewed": panel_count, "sampling_escalation_reason": "; ".join(escalation) or "No additional panel required by rule"}
        ctx.save({"row": row, "judgments": judgments}, "cache", "scoring", f"{c}.complete.json")
        return row

    def run(self) -> pd.DataFrame:
        rows, failures = [], []
        with ThreadPoolExecutor(self.S.concurrency) as ex:
            jobs = {ex.submit(self.process, c): c for c in sorted(self.cids, key=lambda c: -len(self.groups[c]))}
            for f in as_completed(jobs):
                c = jobs[f]
                try:
                    rows.append(f.result())
                except ModelUnavailableError:
                    for other in jobs:
                        other.cancel()
                    raise
                except Exception as e:  # noqa: BLE001
                    failures.append({"cluster_id": c, "error": str(e)})
                self.ctx.progress("scoring", clusters_total=len(self.cids), clusters_completed=len(rows), clusters_failed=len(failures))
        if failures:
            raise RuntimeError(f"incomplete model reviews: {failures[:3]}")
        score = pd.DataFrame(rows)
        score = score.sort_values(["case_count", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
        norm = score.cluster_name.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
        score["same_name_cluster_count"] = norm.map(norm.value_counts())
        score["name_collision_flag"] = score.same_name_cluster_count > 1
        return score


def _template(ctx):
    from ..data import TextTemplate
    return TextTemplate(ctx.cfg.text.embed_with_title, ctx.cfg.text.embed_without_title)
