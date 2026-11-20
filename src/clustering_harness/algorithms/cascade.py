"""Cascade: gate, split, recover and prune the base partition (pipeline doc steps 5–11).

S1   gate every base parent; the gate calibration is frozen from these calls.
S2   split rejected or large parents locally (UMAP 5-d + HDBSCAN leaf) and gate the children.
A    (side output) S2 + centroid reassignment = "cascade A"; cascade B restarts from S2.
S2b  consensus cores inside rejected parents (co-association over eligible bank partitions) and
     depth-two splits of large rejected children; gated.
S3B  Leiden communities in the residual (mutual-kNN graph); the largest are gated.
P    sampled label-then-verify pruning; P3 centroid reassignment; P4 two-judge merges + re-gate.
"""
from __future__ import annotations

import copy
import json
import warnings
import random

import numpy as np

from ..context import RunContext
from .common import as_bool, verify_membership, write_line
from .gate import Gate
from .partitions import bank_eligible, knn_graph

__all__ = ["Cascade", "local_split"]


def local_split(ctx: RunContext, cid: str, ms: list, seed: int = 0, cap: int | None = None) -> list:
    """Split one cluster locally: UMAP(5-d, cosine) + HDBSCAN leaf. Returns up to ``cap`` (child id, members)
    children of at least ``split_min_child`` members, or [] if fewer than two such children exist."""
    import hdbscan
    import umap
    C = ctx.cfg.cascade
    cap = C.split_cap if cap is None else cap
    n = len(ms)
    if n < C.split_min_size:
        return []
    X = ctx.V(ms)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*overridden to 1 by setting random_state.*")   # expected with a fixed seed
        U = umap.UMAP(n_components=C.split_umap_components, metric="cosine", n_neighbors=min(C.split_neighbors, n - 1),
                      min_dist=0.0, random_state=seed).fit_transform(X)
    lab = hdbscan.HDBSCAN(min_cluster_size=max(C.split_min_child, n // C.split_mcs_divisor), cluster_selection_method="leaf").fit_predict(U)
    groups = [[ms[i] for i in np.where(lab == c)[0]] for c in set(lab) if c >= 0]
    groups = sorted([g for g in groups if len(g) >= C.split_min_child], key=len, reverse=True)
    if len(groups) < 2:
        return []
    return [(f"{cid}.{i}", g) for i, g in enumerate(groups[:cap])]


class Cascade:
    def __init__(self, ctx: RunContext, gate: Gate):
        self.ctx = ctx
        self.gate = gate
        self.C = ctx.cfg.cascade
        self.log: dict = {}
        n_elig = len(ctx.nonjunk)
        self.large = int(self.C.large_parent * n_elig) if self.C.large_parent < 1 else int(self.C.large_parent)

    # ------------------------------------------------------------------ helpers
    def _cached_split(self, key: str, ms: list, cap: int | None = None) -> list:
        p = self.ctx.workdir / "cache" / "cascade" / f"split_{key}.json"
        if p.exists():
            return json.loads(p.read_text())
        out = local_split(self.ctx, key, ms, seed=self.C.seed, cap=cap)
        self.ctx.save(out, "cache", "cascade", f"split_{key}.json")
        return out

    def reassign(self, acc: dict, stage: str, regate_stage: str) -> tuple[dict, dict]:
        """Centroid reassignment of the residual with sampled verification of the uncertain band (doc step 11)."""
        ctx, C = self.ctx, self.C
        assigned = {t for v in acc.values() for t in v["members"]}
        resid = [ctx.ds.ids[i] for i in ctx.nonjunk if ctx.ds.ids[i] not in assigned]
        cids = list(acc)
        stats = {"band_tickets": 0, "auto": 0, "clusters_verified": 0, "whole_band_admitted": 0, "yes": 0, "asked": 0}
        if not cids or not resid:
            stats.update(added=0, grown=0, regated=0, reverted=0)
            return acc, stats
        Cm = np.stack([ctx.centroid(acc[c]["members"]) for c in cids])
        own = np.concatenate([ctx.V(acc[c]["members"]) @ Cm[i] for i, c in enumerate(cids)])
        tau_hi, tau_lo = np.percentile(own, C.reassign_hi_percentile), np.percentile(own, C.reassign_lo_percentile)
        S = ctx.V(resid) @ Cm.T
        o = np.argsort(-S, 1)
        s1 = S[np.arange(len(resid)), o[:, 0]]
        s2 = S[np.arange(len(resid)), o[:, 1]] if len(cids) > 1 else np.full(len(resid), -np.inf)
        auto = (s1 >= tau_hi) & (s1 - s2 >= C.reassign_margin)
        band = (~auto) & (s1 >= tau_lo)
        add = {c: [] for c in cids}
        for i in np.where(auto)[0]:
            add[cids[o[i, 0]]].append(resid[i])
        bandc = {c: [] for c in cids}
        for i in np.where(band)[0]:
            bandc[cids[o[i, 0]]].append(resid[i])
        jobs = [(c, ctx.sort(random.Random(C.band_seed).sample(v, min(C.band_sample, len(v))))) for c, v in bandc.items() if v]
        verifier = ctx.role("verifier")
        res = ctx.parallel(jobs, lambda j: verify_membership(ctx, stage, verifier, acc[j[0]]["name"], ctx.exemplars(acc[j[0]]["members"]), j[1]),
                           lambda j: ctx.cfg.roles.verifier)
        stats.update(band_tickets=int(band.sum()), auto=int(auto.sum()), clusters_verified=len(jobs))
        for (c, cands), (ans, err) in zip(jobs, res):
            yes = [t for t in cands if ans.get(t)]
            stats["yes"] += len(yes); stats["asked"] += len(cands)
            if len(cands) >= C.band_admit_min and len(yes) / len(cands) >= C.band_admit_fraction:
                add[c] += bandc[c]; stats["whole_band_admitted"] += 1
            else:
                add[c] += yes
        grown = [c for c in cids if len(add[c]) > C.growth_regate * len(acc[c]["members"])]
        items = [(f"{c}+{stage}", ctx.sort(acc[c]["members"] + add[c])) for c in grown]
        rg = self.gate.gate_many(regate_stage, items)
        reverted = 0
        for c in cids:
            k = f"{c}+{stage}"
            if c in grown and (k not in rg or rg[k]["decision"] != "ACCEPT"):
                add[c] = []; reverted += 1
            acc[c]["members"] = ctx.sort(acc[c]["members"] + add[c])
        stats.update(tau_hi=float(tau_hi), tau_lo=float(tau_lo), grown=len(grown), regated=len(items), reverted=reverted,
                     added=sum(len(v) for v in add.values()))
        return acc, stats

    def write_arm(self, arm: str, acc: dict) -> None:
        ctx = self.ctx
        lab = {t: c for c, v in acc.items() for t in v["members"]}
        rows = []
        for i, t in enumerate(ctx.ds.ids):
            c = lab.get(t)
            j = bool(ctx.junk[i])
            rows.append({"ticket_id": t, "cluster_id": c or "", "cluster_name": ctx.mask(acc[c]["name"]) if c else "",
                         "status": "junk" if j else ("cluster" if c else "unassigned"), "stage": acc[c]["stage"] if c else "",
                         "reason": "junk" if j else ("" if c else "not in an accepted cluster")})
        import pandas as pd
        pd.DataFrame(rows).to_csv(ctx.path("results", arm, "assignments.csv"), index=False)
        ctx.save(acc, "results", arm, "clusters.json")

    # ------------------------------------------------------------------ run
    def run(self, base_labels: np.ndarray) -> tuple[dict, dict]:
        """``base_labels``: base partition over eligible records (dataset order of ``ctx.nonjunk``)."""
        ctx, C, G = self.ctx, self.C, self.gate
        elig = ctx.eligible_ids
        groups: dict = {}
        for t, l in zip(elig, base_labels):
            if l >= 0:
                groups.setdefault(f"c{l}", []).append(t)
        base = {f"P{k[1:]}": ctx.sort(groups[k]) for k in sorted(groups)}
        ctx.progress("cascade", step="S1", parents=len(base), large=self.large)

        # S1 ----------------------------------------------------------------
        s1 = G.gate_many("cas_s1", list(base.items()), calibrate_now=True)
        acc = {c: {"members": base[c], "name": s1[c]["name"], "adj": s1[c]["adj"], "stage": "S1"} for c in base if s1[c]["decision"] == "ACCEPT"}
        self.log["S1"] = {"parents": len(base), "accepted": len(acc), "tickets": sum(len(v["members"]) for v in acc.values())}
        ctx.progress("cascade", step="S1 done", **self.log["S1"])

        # S2 ----------------------------------------------------------------
        to_split = [c for c in base if s1[c]["decision"] != "ACCEPT" or len(base[c]) >= self.large]
        children = {c: self._cached_split(c, base[c]) for c in to_split}
        items = [(k, g) for c in to_split for k, g in children[c]]
        s2 = G.gate_many("cas_s2", items)
        for c in to_split:
            ch = [(k, g) for k, g in children[c] if k in s2 and s2[k]["decision"] == "ACCEPT"]
            if s1[c]["decision"] == "ACCEPT":
                cov = sum(len(g) for _, g in ch) / len(base[c])
                if ch and cov >= C.replace_min_coverage and \
                        np.average([s2[k]["adj"] for k, _ in ch], weights=[len(g) for _, g in ch]) > s1[c]["adj"]:
                    acc.pop(c)
                    for k, g in ch:
                        acc[k] = {"members": g, "name": s2[k]["name"], "adj": s2[k]["adj"], "stage": "S2"}
            else:
                for k, g in ch:
                    acc[k] = {"members": g, "name": s2[k]["name"], "adj": s2[k]["adj"], "stage": "S2"}
        self.log["S2"] = {"parents_split": len(to_split), "with_children": sum(1 for c in to_split if children[c]), "children_gated": len(items),
                          "children_accepted": sum(1 for k in s2 if s2[k]["decision"] == "ACCEPT"), "clusters": len(acc),
                          "tickets": sum(len(v["members"]) for v in acc.values())}
        ctx.progress("cascade", step="S2 done", **self.log["S2"])
        ctx.save(acc, "cache", "cascade", "after_s2.json")

        # Cascade A (side output) ---------------------------------------------
        if C.run_cascade_a:
            accA, self.log["S3A"] = self.reassign(copy.deepcopy(acc), "cas_s3a", "cas_s3a_regate")
            self.log["S3A"].update(clusters=len(accA), tickets=sum(len(v["members"]) for v in accA.values()))
            self.write_arm("cascade_A", accA)
            ctx.progress("cascade", step="A done", **{k: v for k, v in self.log["S3A"].items() if not isinstance(v, float)})

        # S2b ---------------------------------------------------------------
        import pandas as pd
        grid = pd.read_csv(ctx.path("results", "bank", "grid_runs.csv"))
        ok = bank_eligible(ctx, grid)
        nj_pos = {t: k for k, t in enumerate(elig)}
        rejected = [c for c in base if s1[c]["decision"] != "ACCEPT"]
        cores = []
        if len(ok):
            from scipy.sparse.csgraph import connected_components
            import scipy.sparse as sp
            Pm = np.stack([np.load(ctx.workdir / "cache" / "partitions" / f"{r}.npy") for r in ok.run_id])
            for c in rejected:
                ms = base[c]
                if len(ms) < C.cores_min_parent:
                    continue
                L = Pm[:, [nj_pos[t] for t in ms]]
                co = np.zeros((len(ms), len(ms)), np.float32)
                for r in L:
                    co += ((r[:, None] == r[None, :]) & (r[:, None] >= 0))
                co /= len(L)
                n, comp = connected_components(sp.csr_matrix(co >= C.coassociation), directed=False)
                got = sorted([[ms[i] for i in np.where(comp == k)[0]] for k in range(n)], key=len, reverse=True)
                kids = [set(g) for k, g in children.get(c, []) if k in s2 and s2[k]["decision"] == "ACCEPT"]
                j = 0
                for g in got:
                    if len(g) < C.core_min_size or j >= C.cores_per_parent:
                        continue
                    if any(len(set(g) & s) / len(set(g) | s) >= C.core_max_jaccard for s in kids):
                        continue
                    cores.append((f"{c}.core{j}", g)); j += 1
        deep = []
        for k, g in items:
            if k in s2 and s2[k]["decision"] != "ACCEPT" and len(g) >= C.deep_split_min:
                deep += [(kk, gg) for kk, gg in self._cached_split(k, g, cap=C.deep_split_cap)]
        cand = sorted(cores + deep, key=lambda x: -len(x[1]))
        s2b = G.gate_many("cas_s2b", cand)
        accB = copy.deepcopy(acc)
        taken = {t for v in accB.values() for t in v["members"]}
        for k, g in sorted(cand, key=lambda x: -(s2b[x[0]]["adj"] or 0) if x[0] in s2b else 0):
            if k in s2b and s2b[k]["decision"] == "ACCEPT":
                g2 = [t for t in g if t not in taken]
                if len(g2) >= C.candidate_min_size:
                    accB[k] = {"members": g2, "name": s2b[k]["name"], "adj": s2b[k]["adj"], "stage": "S2b"}; taken |= set(g2)
        self.log["S2b"] = {"bank_partitions_eligible": int(len(ok)), "cores": len(cores), "deep_children": len(deep), "gated": len(cand),
                           "accepted": sum(1 for k in s2b if s2b[k]["decision"] == "ACCEPT"),
                           "clusters": len(accB), "tickets": sum(len(v["members"]) for v in accB.values())}
        ctx.progress("cascade", step="S2b done", **self.log["S2b"])

        # S3B ---------------------------------------------------------------
        import igraph as ig
        import leidenalg as la
        resid = [t for t in elig if t not in taken]
        I, _ = knn_graph(ctx)
        I = I[:, 1:C.leiden_scan + 1]
        rset = np.array([nj_pos[t] for t in resid], dtype=int)
        inr = np.full(len(elig), -1)
        inr[rset] = np.arange(len(rset))
        edges = {}
        for a_, gi in enumerate(rset):
            nb = [x for x in I[gi] if inr[x] >= 0][:C.leiden_k]
            for b_ in nb:
                edges[(a_, int(inr[b_]))] = 1
        mut = [(i, j) for (i, j) in edges if i < j and (j, i) in edges]
        g = ig.Graph(n=len(rset), edges=mut)
        g.es["weight"] = [float(ctx.X[ctx.pos[resid[i]]] @ ctx.X[ctx.pos[resid[j]]]) for i, j in mut]
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=C.leiden_resolution, seed=C.seed)
        comms = sorted([[resid[i] for i in c] for c in part if len(c) >= C.leiden_min_size], key=len, reverse=True)
        lcand = [(f"L{i}", ctx.sort(c)) for i, c in enumerate(comms[:C.leiden_max_gated])]
        s3b = G.gate_many("cas_s3b", lcand)
        accL = [(k, g_) for k, g_ in lcand if s3b[k]["decision"] == "ACCEPT"]
        for k, g_ in accL:
            accB[k] = {"members": g_, "name": s3b[k]["name"], "adj": s3b[k]["adj"], "stage": "S3B"}
        cov_add = sum(len(g_) for _, g_ in accL)
        self.log["S3B"] = {"residual": len(resid), "communities": len(comms), "gated": len(lcand), "accepted": len(accL), "tickets_added": cov_add,
                           "yield_rule_passed": bool(cov_add >= 0.03 * len(resid) and len(accL) >= 0.1 * max(len(lcand), 1))}
        ctx.progress("cascade", step="S3B done", **self.log["S3B"])

        # P1-P2 pruning -------------------------------------------------------
        cids = list(accB)
        labeler, verifier, rechecker = ctx.role("labeler"), ctx.role("verifier"), ctx.role("rechecker")
        labels = dict(zip(cids, ctx.parallel(cids, lambda c: write_line(ctx, "cas_p1_label", labeler, c, accB[c]["members"]),
                                              lambda c: ctx.cfg.roles.labeler)))

        def check(c):
            ms = accB[c]["members"]
            samp = ctx.sort(random.Random(C.prune_seed).sample(ms, min(C.prune_sample, len(ms))))
            line, crit = labels[c]
            ex = [t for t in ctx.exemplars(ms, 3) if t not in samp][:3] or ctx.exemplars(ms, 3)
            a1, _ = verify_membership(ctx, "cas_p1_check", verifier, line or accB[c]["name"], ex, samp, crit=crit)
            no = [t for t in samp if a1.get(t) is False]
            a2 = {}
            if no:
                a2, _ = verify_membership(ctx, "cas_p2_recheck", rechecker, line or accB[c]["name"], ex, no, crit=crit)
            return c, samp, [t for t in no if a2.get(t) is False], line
        res = ctx.parallel(cids, check, lambda c: ctx.cfg.roles.verifier)
        prune, dissolved = {}, []
        for c, samp, eject, line in res:
            prec = 1 - len(eject) / max(len(samp), 1)
            prune[c] = {"sampled": len(samp), "ejected": len(eject), "precision_est": prec, "line": line}
            if prec < C.prune_min_precision:
                dissolved.append(c); accB.pop(c)
            else:
                accB[c]["members"] = [t for t in accB[c]["members"] if t not in set(eject)]
                if line:
                    accB[c]["line"] = line
        self.log["P"] = {"clusters_checked": len(res), "members_sampled": sum(p["sampled"] for p in prune.values()),
                         "ejected": sum(p["ejected"] for p in prune.values()), "dissolved": dissolved,
                         "mean_precision_est": float(np.mean([p["precision_est"] for p in prune.values()])) if prune else None}
        ctx.save(prune, "cache", "cascade", "prune.json")
        ctx.progress("cascade", step="P done", **{k: v for k, v in self.log["P"].items() if k != "dissolved"})

        # P3 reassignment + P4 merge ------------------------------------------
        accB, self.log["P3"] = self.reassign(accB, "cas_p3", "cas_p3_regate")
        ctx.progress("cascade", step="P3 done", **{k: v for k, v in self.log["P3"].items() if not isinstance(v, float)})
        cids = list(accB)
        pairs = []
        if len(cids) > 1:
            Cm = np.stack([ctx.centroid(accB[c]["members"]) for c in cids])
            Sm = Cm @ Cm.T
            np.fill_diagonal(Sm, -1)
            iu = np.triu_indices(len(cids), 1)
            pairs = sorted([(float(Sm[i, j]), cids[i], cids[j]) for i, j in zip(*iu) if Sm[i, j] >= C.merge_min_cosine], reverse=True)[:C.merge_top_pairs]
        P = ctx.prompts

        def same(p):
            _, x, y = p
            ex = lambda c: "\n".join(f"- {ctx.item(t, 160)}" for t in ctx.gsample(c, 0, accB[c]["members"])[:5])
            user = P.render("merge_user", name_a=accB[x]["name"], examples_a=ex(x), name_b=accB[y]["name"], examples_b=ex(y))
            v = {}
            for j in ("a", "b"):
                llm = ctx.llm(j)
                o = llm.call("cas_p4_merge", [{"role": "system", "content": P["merge_system"]}, {"role": "user", "content": user}],
                             max_tokens=1200 if llm.reasoning else 200)
                v[j] = as_bool(o.get("same_line")) is True if "_error" not in o else False
            return v
        votes = [same(p) for p in pairs]
        mcands, used = [], set()
        for p, v in zip(pairs, votes):
            if all(v.values()) and p[1] not in used and p[2] not in used:
                mcands.append((f"{p[1]}|{p[2]}", ctx.sort(set(accB[p[1]]["members"]) | set(accB[p[2]]["members"]))))
                used |= {p[1], p[2]}
        mg = G.gate_many("cas_p4_regate", mcands)
        merged = 0
        for k, g_ in mcands:
            if mg[k]["decision"] == "ACCEPT":
                x, y = k.split("|")
                accB[x] = {"members": g_, "name": mg[k]["name"] or accB[x]["name"], "adj": mg[k]["adj"], "stage": "P4 merge"}
                accB.pop(y); merged += 1
        self.log["P4"] = {"pairs_asked": len(pairs), "both_same": len(mcands), "merged": merged}
        self.log["final_B"] = {"clusters": len(accB), "tickets": sum(len(v["members"]) for v in accB.values())}
        ctx.progress("cascade", step="done", **self.log["final_B"])
        self.write_arm("cascade_B", accB)
        ctx.save(self.log, "results", "cascade_log.json")
        return accB, self.log
