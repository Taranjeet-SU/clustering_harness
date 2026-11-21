"""Guarded report-line rollup (pipeline doc step 19).

Units start as the final clusters, each judged with the unified two-judge protocol (census score).
Round 0 proposes merging clusters with identical normalised names. Rounds 1+ (only while there are more than
``target_max`` lines) group units by average linkage on ``1 - (0.5 * name cosine + 0.5 * centroid cosine)``
at increasing thresholds (groups larger than ``max_group`` are split tighter) and ask the rollup model to
partition each neighbourhood into report lines. Every proposed merge is judged with the same protocol and
kept only if its judge-balanced score is >= ``guard_min`` and not more than ``guard_drop`` below the
size-weighted mean of its parts. Finally every line gets a name and an inclusion criterion."""
from __future__ import annotations

import hashlib
import re
import time

import numpy as np
import pandas as pd

from ..context import RunContext
from ..evaluation.llm_judge import LLMJudge

__all__ = ["Rollup"]

def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (s or "").lower())).strip()


def _uid(clusters):
    return "U" + hashlib.md5(",".join(sorted(clusters)).encode()).hexdigest()[:10]


class Rollup:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.R = ctx.cfg.rollup
        self.judge = LLMJudge(ctx)
        self.parts_arm, self.lines_arm = self.R.census_arm, self.R.guard_arm

    def run(self, cmem: dict, cname: dict) -> dict:
        """``cmem`` {cluster: [ids]}, ``cname`` {cluster: name}. Returns lines, mapping and a log."""
        ctx, R, P = self.ctx, self.R, self.ctx.prompts
        T0 = time.time()
        cmem = {c: ctx.sort(cmem[c]) for c in sorted(cmem)}
        units = {_uid([c]): {"clusters": [c], "members": ms, "name": cname[c], "criterion": "", "round": -1} for c, ms in cmem.items()}
        ctx.progress("rollup", stage="census judging of clusters", clusters=len(cmem))
        fr = self.judge.evaluate(self.parts_arm, cmem, stage="rollup_parts")
        cscore = dict(zip(fr.cluster, fr.balanced))
        for u in units.values():
            u["score"] = float(cscore.get(u["clusters"][0], np.nan))
        rlog = []
        rollup_llm = ctx.role("rollup")
        decisions = []

        def exemplars(ms, k=3):
            c = ctx.centroid(ms)
            return sorted(ms, key=lambda t: -float(ctx.X[ctx.pos[t]] @ c))[:k]

        def name_line(u):
            parts = sorted(u["clusters"], key=lambda c: -len(cmem[c]))
            pl = "\n".join(f"- {ctx.mask(cname[c])} ({len(cmem[c])} tickets)" for c in parts[:25])
            samp = ctx.jsample("rollup_name", ",".join(sorted(u["clusters"])), 0, u["members"], 10)
            it = "\n".join(f"- {ctx.title(t)} | {ctx.desc_head(t, 120) or P['item_empty_desc']}" for t in samp)
            o = rollup_llm.call("rollup_name", [{"role": "system", "content": P["rollup_system"]},
                                                {"role": "user", "content": P.render("rollup_name_user", parts=pl, samples=it)}], max_tokens=600)
            if "_error" not in o and o.get("name"):
                u["name"] = ctx.mask(str(o["name"]).strip())[:100]
                u["criterion"] = ctx.mask(str(o.get("criterion", "")).strip())[:400]

        def decide(group, r):
            ids = {f"c{i+1}": g for i, g in enumerate(group)}
            rows = []
            for k, g in ids.items():
                u = units[g]
                ex = "; ".join(ctx.title(t)[:90] for t in exemplars(u["members"]))
                rows.append(P.render("rollup_row", k=k, name=ctx.mask(u["name"]), n=len(u["members"]), examples=ex))
            user = P.render("rollup_decide_user", rubric=P["rollup_rubric"], rows="\n".join(rows))
            o = rollup_llm.call("rollup_decide", [{"role": "system", "content": P["rollup_system"]}, {"role": "user", "content": user}], max_tokens=2500)
            seen, out = set(), []
            for ln in (o.get("lines") or []) if "_error" not in o else []:
                if not isinstance(ln, dict) or not isinstance(ln.get("ids"), list):
                    continue
                gs = [ids[x] for x in ln["ids"] if isinstance(x, str) and x in ids and ids[x] not in seen]
                seen |= set(gs)
                if len(gs) >= 2:
                    out.append((gs, str(ln.get("why", ""))[:200]))
            decisions.append({"round": r, "group": [units[g]["name"] for g in group], "merged": [[units[x]["name"] for x in gs] for gs, _ in out],
                              "error": o.get("_error", "")})
            return out

        def guard(proposals, r):
            cand = {}
            for gs, why in proposals:
                cls = sorted(c for g in gs for c in units[g]["clusters"])
                cand[_uid(cls)] = (gs, cls, why)
            if not cand:
                return []
            lm = {k: ctx.sort({t for g in v[0] for t in units[g]["members"]}) for k, v in cand.items()}
            fr2 = self.judge.evaluate(self.lines_arm, lm, stage="rollup_guard")
            sc = dict(zip(fr2.cluster, fr2.balanced))
            acc = []
            for k, (gs, cls, why) in cand.items():
                sz = np.array([len(units[g]["members"]) for g in gs], float)
                ps = np.array([units[g]["score"] for g in gs], float)
                ok = ~np.isnan(ps)
                pm = float((sz[ok] * ps[ok]).sum() / sz[ok].sum()) if ok.any() else np.nan
                s = float(sc.get(k, np.nan))
                keep = (not np.isnan(s)) and s >= R.guard_min and (np.isnan(pm) or s >= pm - R.guard_drop)
                rlog.append({"round": r, "line": k, "parts": len(gs), "clusters": len(cls), "tickets": int(sz.sum()), "score": round(s, 3),
                             "parts_mean": round(pm, 3) if pm == pm else None, "accepted": bool(keep), "why": why,
                             "part_names": [units[g]["name"] for g in gs][:30]})
                if keep:
                    acc.append((k, gs, cls, s))
            return acc

        def apply(acc, r):
            for k, gs, cls, s in acc:
                ms = ctx.sort({t for g in gs for t in units[g]["members"]})
                big = max(gs, key=lambda g: len(units[g]["members"]))
                nu = {"clusters": cls, "members": ms, "name": units[big]["name"], "criterion": "", "round": r, "score": s}
                for g in gs:
                    units.pop(g)
                units[k] = nu
            newu = [units[k] for k, *_ in acc]
            ctx.parallel(newu, name_line, lambda u: ctx.cfg.roles.rollup)

        def rstat(r, proposed, accepted):
            sizes = sorted((len(u["members"]) for u in units.values()), reverse=True)
            cum = np.cumsum(sizes) / sum(sizes)
            return {"round": r, "proposed": proposed, "accepted": accepted, "lines": len(units),
                    "lines_for_80pct": int(np.searchsorted(cum, 0.8) + 1), "lines_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
                    "minutes": round((time.time() - T0) / 60, 1)}

        stats = []
        byn: dict = {}
        for k, u in units.items():
            byn.setdefault(_norm(u["name"]), []).append(k)
        props = [(ks, "identical name") for ks in byn.values() if len(ks) >= 2]
        acc = guard(props, 0); apply(acc, 0); stats.append(rstat(0, len(props), len(acc)))
        from .._compat import agglomerative
        for r in range(1, len(R.thresholds)):
            if len(units) <= R.target_max:
                break
            keys = sorted(units)
            NV = ctx.label_vectors([_norm(units[k]["name"]) for k in keys], f"rollup_names_r{r}")
            CV = np.stack([ctx.centroid(units[k]["members"]) for k in keys])
            S = 0.5 * (NV @ NV.T) + 0.5 * (CV @ CV.T)
            D = np.clip(1 - S, 0, 2)
            np.fill_diagonal(D, 0)

            def groups(idx, th):
                if len(idx) < 2:
                    return []
                lab = agglomerative(None, metric="precomputed", linkage="average", distance_threshold=th).fit_predict(D[np.ix_(idx, idx)])
                out = []
                for l_ in set(lab):
                    sub = [idx[i] for i in np.where(lab == l_)[0]]
                    if len(sub) < 2:
                        continue
                    if len(sub) > R.max_group:
                        deeper = groups(sub, th - 0.03) if th > 0.08 else []
                        out += deeper if deeper else [sub[i:i + R.max_group] for i in range(0, len(sub), R.max_group)]
                    else:
                        out.append(sub)
                return out
            G = [[keys[i] for i in g] for g in groups(list(range(len(keys))), R.thresholds[r])]
            ctx.progress("rollup", stage=f"round {r}", neighbourhoods=len(G), units=len(keys))
            dec = ctx.parallel(G, lambda g: decide(g, r), lambda g: ctx.cfg.roles.rollup)
            props = [p for d in dec for p in d]
            acc = guard(props, r); apply(acc, r); stats.append(rstat(r, len(props), len(acc)))
        ctx.progress("rollup", stage="naming lines", lines=len(units))
        ctx.parallel([u for u in units.values() if not u["criterion"]], name_line, lambda u: ctx.cfg.roles.rollup)
        order = sorted(units, key=lambda k: -len(units[k]["members"]))
        lid = {k: f"LN{i+1:03d}" for i, k in enumerate(order)}
        lrows, c2l = [], {}
        for k in order:
            u = units[k]
            for c in u["clusters"]:
                c2l[c] = lid[k]
            lrows.append({"line_id": lid[k], "line_name": u["name"], "criterion": u["criterion"], "tickets": len(u["members"]),
                          "n_clusters": len(u["clusters"]), "judge_score": round(u["score"], 3) if u["score"] == u["score"] else None,
                          "round_formed": u["round"], "cluster_ids": ";".join(sorted(u["clusters"], key=lambda c: -len(cmem[c]))),
                          "cluster_names": " || ".join(ctx.mask(cname[c]) for c in sorted(u["clusters"], key=lambda c: -len(cmem[c])))[:2000]})
        L = pd.DataFrame(lrows)
        rl = pd.DataFrame(rlog)
        csc = pd.Series(cscore).dropna()
        log = {"clusters": len(cmem), "lines": len(units), "rounds": stats, "proposals": int(len(rl)),
               "proposals_accepted": int(rl.accepted.sum()) if len(rl) else 0,
               "mean_judge_cluster_level": round(float(csc.mean()), 3) if len(csc) else None,
               "coverage_weighted_judge_cluster_level": round(float(sum(len(cmem[c]) * v for c, v in csc.items()) / sum(len(cmem[c]) for c in csc.index)), 3) if len(csc) else None,
               "wall_minutes": round((time.time() - T0) / 60, 1)}
        return {"lines": L, "cluster_to_line": c2l, "line_names": {lid[k]: units[k]["name"] for k in units}, "guard": rl,
                "decisions": decisions, "census": fr, "log": log}
