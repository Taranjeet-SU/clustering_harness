"""LLM residual classification (pipeline doc steps 12–16).

Records left unassigned are sharded into parallel *streams* (k-means on their vectors). Each stream is read
sequentially in batches by the classifier model, which files every record under an accepted report line,
reuses a provisional label proposed earlier in the stream, proposes a NEW label, or marks it JUNK. Then:

1. provisional labels are pre-grouped by label-embedding similarity and canonicalised by the second model;
2. canonical labels with at least ``min_label_size`` records are gated (two-judge gate);
3. filings under existing lines are confirmed by the verifier (batches of ``confirm_batch``);
4. accepted new clusters are pruned (label-then-verify on a sample, double "no" ejects);
5. what is still left is reassigned by margin-gated nearest centroid with sampled verification, and
   clusters that grew by more than ``growth_regate`` are re-gated (additions reverted unless ACCEPT).

``variant="v1"`` is the first residual pass (C_full; its JUNK reasons include ``service_request``).
``variant="v2"`` is the refile pass (C_full_v2): a corrected classifier prompt that keeps service requests,
a repair pass for failed batches, and fully id-sorted orderings."""
from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..context import RunContext
from .common import verify_membership, write_line
from .gate import Gate

__all__ = ["ResidualPass", "ResidualResult"]


class ResidualResult(dict):
    """``labels`` {id: cluster}, ``how`` {id: placement}, ``junk`` {id: reason}, ``names`` {cluster: name}, ``log``."""


class ResidualPass:
    def __init__(self, ctx: RunContext, gate: Gate, variant: str = "v1"):
        assert variant in ("v1", "v2")
        self.ctx, self.gate, self.v2 = ctx, gate, variant == "v2"
        self.R = ctx.cfg.residual
        self.P_ = "V2_" if self.v2 else "Cf_"
        self.tag = "R" if self.v2 else ""
        self.generic = re.compile(self.R.generic_label, re.I)

    def run(self, lab: dict, names: dict, resid: list, cache_key: str) -> ResidualResult:
        """``lab``: {id: cluster} of accepted clusters (dataset order); ``names``: {cluster: name};
        ``resid``: ids to classify (dataset order)."""
        ctx, R, v2, P_ = self.ctx, self.R, self.v2, self.P_
        P = ctx.prompts
        T0 = time.time()
        if not resid:
            return ResidualResult(labels=dict(lab), how={}, junk={}, names=dict(names), log={"pool": 0, "placed": 0})
        NS = min(R.refile_streams if v2 else R.streams, len(resid))
        members: dict = {}
        for t, c in lab.items():
            members.setdefault(c, []).append(t)
        lines = sorted(members)
        lidx = {c: i for i, c in enumerate(lines)}
        cent = np.stack([ctx.V(members[c]).mean(0) for c in lines]) if lines else np.zeros((0, ctx.X.shape[1]), np.float32)
        if len(cent):
            cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        resid = list(resid)
        if v2:
            resid = ctx.sort(resid)
        random.Random(R.seed).shuffle(resid)
        from sklearn.cluster import KMeans
        stream = KMeans(NS, n_init=3, random_state=R.seed).fit_predict(ctx.V(resid)) if NS > 1 else np.zeros(len(resid), int)
        TV = ctx.label_vectors([ctx.title(t) for t in resid], f"{cache_key}_resid_titles")
        tpos = {t: i for i, t in enumerate(resid)}
        ctx.progress(cache_key, stage="classifying", residual=len(resid), streams=np.bincount(stream).tolist())
        rules = P["classify_rules_v2" if v2 else "classify_rules_v1"]
        classifier = ctx.role("classifier")

        def classify_batch(batch, prov, pvec, prov_members, filed, junk):
            if len(cent):
                bv = ctx.V(batch).mean(0)
                near = np.argsort(-(cent @ bv))[:R.near_lines]
            else:
                near = []
            lid = {f"L{j+1}": lines[k] for j, k in enumerate(near)}
            acc_txt = "\n".join(f"{k}. {ctx.mask(names[c])}" for k, c in lid.items())
            if prov:
                qv = TV[[tpos[t] for t in batch]].mean(0)
                pn = np.argsort(-(np.array(pvec) @ qv))[:R.near_provisional]
            else:
                pn = []
            pid = {f"N{j+1}": int(k) for j, k in enumerate(pn)}
            pv = "\n".join(f"{k}. {prov[v]}" for k, v in pid.items()) or P["classify_no_provisional"]
            items = "\n".join(P.render("classify_item", n=j + 1, title=ctx.title(t), desc=ctx.desc_head(t, R.desc_chars) or P["item_empty_desc"])
                              for j, t in enumerate(batch))
            user = P.render("classify_user", accepted=acc_txt, provisional=pv, rules=rules, items=items)
            o = classifier.call(P_ + "cls", [{"role": "system", "content": P["classify_system"]}, {"role": "user", "content": user}], max_tokens=1200)
            answered = set()
            for it in (o.get("tickets") or []) if "_error" not in o else []:
                try:
                    t = batch[int(it["n"]) - 1]
                    ch = str(it["choice"]).strip()
                except Exception:  # noqa: BLE001
                    continue
                if t in answered:          # a record answered twice keeps its first answer
                    continue
                answered.add(t)
                if ch in lid:
                    filed[t] = lid[ch]
                elif ch in pid:
                    prov_members.setdefault(pid[ch], []).append(t)
                elif re.match(r"(?i)NEW\s*:", ch):
                    l_ = ch.split(":", 1)[-1].strip()[:80]
                    if l_ and not self.generic.search(l_):
                        prov.append(l_); pvec.append(TV[tpos[t]]); prov_members.setdefault(len(prov) - 1, []).append(t)
                elif re.match(r"(?i)JUNK\s*:", ch):
                    junk[t] = ch.split(":", 1)[-1].strip()[:40]
            return "_error" not in o

        def run_stream(s):
            saved = ctx.load("cache", cache_key, f"stream_{s}.json")
            if saved is not None:
                return saved
            ts = [t for t, k in zip(resid, stream) if k == s]
            prov, pvec, prov_members, filed, junk = [], [], {}, {}, {}
            for i in range(0, len(ts), R.batch_size):
                classify_batch(ts[i:i + R.batch_size], prov, pvec, prov_members, filed, junk)
            res = {"prov": prov, "prov_members": {str(k): v for k, v in prov_members.items()}, "filed": filed, "junk": junk, "n": len(ts)}
            ctx.save(res, "cache", cache_key, f"stream_{s}.json")
            return res

        def repair(s, res):
            saved = ctx.load("cache", cache_key, f"stream_{s}_repair.json")
            if saved is not None:
                return saved
            ts = [t for t, k in zip(resid, stream) if k == s]
            pm = {int(k): list(v) for k, v in res["prov_members"].items()}
            done = set(res["filed"]) | set(res["junk"]) | {t for v in pm.values() for t in v}
            miss = [t for t in ts if t not in done]
            prov = list(res["prov"])
            pvec = [TV[tpos[pm[k][0]]] for k in range(len(prov))]
            filed, junk = dict(res["filed"]), dict(res["junk"])
            for i in range(0, len(miss), R.repair_batch_size):
                classify_batch(miss[i:i + R.repair_batch_size], prov, pvec, pm, filed, junk)
            done2 = set(filed) | set(junk) | {t for v in pm.values() for t in v}
            out = {"prov": prov, "prov_members": {str(k): v for k, v in pm.items()}, "filed": filed, "junk": junk, "n": len(ts),
                   "missing_before_repair": len(miss), "missing_after_repair": len([t for t in ts if t not in done2])}
            ctx.save(out, "cache", cache_key, f"stream_{s}_repair.json")
            return out

        with ThreadPoolExecutor(NS) as ex:
            S = list(ex.map(run_stream, range(NS)))
        unclassified = None
        if v2:
            with ThreadPoolExecutor(NS) as ex:
                S = list(ex.map(lambda s: repair(s, S[s]), range(NS)))
            unclassified = {"before_repair": sum(x["missing_before_repair"] for x in S), "after_repair": sum(x["missing_after_repair"] for x in S)}
        filed = {k: v for s in S for k, v in s["filed"].items()}
        junk = {k: v for s in S for k, v in s["junk"].items()}
        labels, lmembers = [], []
        for s in S:
            for k, l_ in enumerate(s["prov"]):
                labels.append(l_); lmembers.append(s["prov_members"].get(str(k), []))
        ctx.progress(cache_key, stage="canonicalising", provisional=len(labels), filed=len(filed), junk=len(junk))

        # 1. canonicalise ------------------------------------------------------
        from .._compat import agglomerative
        LV = ctx.label_vectors(labels, f"{cache_key}_prov_labels") if labels else np.zeros((0, 1024))
        grp = (agglomerative(None, metric="cosine", linkage="average", distance_threshold=R.canon_distance).fit_predict(LV)
               if len(labels) > 1 else np.zeros(len(labels), int))
        order = np.argsort(grp, kind="stable")
        chunks = [order[i:i + R.canon_chunk] for i in range(0, len(order), R.canon_chunk)]
        canonicalizer = ctx.role("canonicalizer")

        def canon_chunk(ch):
            user = P.render("canon_user", labels="\n".join(f"{j+1}. {labels[k]}" for j, k in enumerate(ch)))
            o = canonicalizer.call(P_ + "merge", [{"role": "system", "content": P["classify_system"]}, {"role": "user", "content": user}], max_tokens=3000)
            out = {}
            for it in (o.get("map") or []) if "_error" not in o else []:
                try:
                    out[int(ch[int(it["n"]) - 1])] = str(it["canonical"]).strip().lower()
                except Exception:  # noqa: BLE001
                    pass
            return out
        canon = {}
        for r in ctx.parallel(chunks, canon_chunk, lambda c: ctx.cfg.roles.canonicalizer):
            canon.update(r)
        groups: dict = {}
        for k, ms in enumerate(lmembers):
            groups.setdefault(canon.get(k, labels[k].lower()), []).extend(ms)
        gl = sorted(groups.items(), key=(lambda x: (-len(set(x[1])), x[0])) if v2 else (lambda x: -len(x[1])))
        pre = f"{self.tag}N"
        items = [(f"{pre}{i}", ctx.sort(set(ms))) for i, (c, ms) in enumerate(gl) if len(set(ms)) >= R.min_label_size]
        gname = {f"{pre}{i}": c for i, (c, ms) in enumerate(gl)}

        # 2. gate --------------------------------------------------------------
        ctx.progress(cache_key, stage="gating canonical labels", canonical=len(groups), gated=len(items))
        g = self.gate.gate_many(P_ + "gate", items)

        # 3. confirm filings -----------------------------------------------------
        ctx.progress(cache_key, stage="confirming filings", filings=len(filed))
        byline: dict = {}
        for t, c in filed.items():
            byline.setdefault(c, []).append(t)
        jobs = []
        for c, ts in (sorted(byline.items()) if v2 else byline.items()):
            ex_ = sorted(members[c], key=lambda t: -float(ctx.X[ctx.pos[t]] @ cent[lidx[c]]))[:3]
            jobs += [(c, ex_, ts[k:k + R.confirm_batch]) for k in range(0, len(ts), R.confirm_batch)]
        verifier = ctx.role("verifier")
        res = ctx.parallel(jobs, lambda j: verify_membership(ctx, P_ + "confirm", verifier, names[j[0]], j[1], j[2])[0],
                           lambda j: ctx.cfg.roles.verifier)
        confirmed = 0
        newlab = dict(lab)
        for (c, _, ts), ans in zip(jobs, res):
            for t in ts:
                if ans.get(t):
                    newlab[t] = c; confirmed += 1
        acc_new = {}
        for k, ms in items:
            if g[k]["decision"] == "ACCEPT":
                ms2 = [t for t in ms if t not in newlab]
                if len(ms2) >= R.min_label_size:
                    acc_new[f"C{k}"] = {"members": ms2, "name": g[k]["name"] or gname[k]}

        # 4. prune new clusters ----------------------------------------------------
        ctx.progress(cache_key, stage="pruning new clusters", new_clusters=len(acc_new))
        labeler, rechecker = ctx.role("labeler"), ctx.role("rechecker")
        C = ctx.cfg.cascade

        def prune(cid):
            ms = acc_new[cid]["members"]
            line, crit = write_line(ctx, P_ + "p1_label", labeler, cid, ms)
            samp = ctx.sort(random.Random(C.prune_seed).sample(ms, min(C.prune_sample, len(ms))))
            exs = [t for t in ms if t not in samp][:3] or ms[:3]
            a1, _ = verify_membership(ctx, P_ + "p1_check", verifier, line or acc_new[cid]["name"], exs, samp, crit=crit)
            no = [t for t in samp if a1.get(t) is False]
            a2 = verify_membership(ctx, P_ + "p2_recheck", rechecker, line or acc_new[cid]["name"], exs, no, crit=crit)[0] if no else {}
            return cid, samp, [t for t in no if a2.get(t) is False]
        pr = ctx.parallel(sorted(acc_new) if v2 else list(acc_new), prune, lambda c: ctx.cfg.roles.verifier)
        dissolved, ejected = [], 0
        for cid, samp, ej in pr:
            if 1 - len(ej) / max(len(samp), 1) < C.prune_min_precision:
                dissolved.append(cid); acc_new.pop(cid)
            else:
                acc_new[cid]["members"] = [t for t in acc_new[cid]["members"] if t not in set(ej)]
                ejected += len(ej)
        names = dict(names)
        for cid, v in acc_new.items():
            for t in v["members"]:
                newlab[t] = cid
            names[cid] = v["name"]

        # 5. reassignment of what is left -------------------------------------------
        ctx.progress(cache_key, stage="reassignment")
        allc: dict = {}
        for t, c in newlab.items():
            allc.setdefault(c, []).append(t)
        cids = sorted(allc) if v2 else list(allc)
        if v2:
            for c in cids:
                allc[c] = ctx.sort(allc[c])
        add = {c: [] for c in cids}
        bandc = {c: [] for c in cids}
        n_auto = n_band = 0
        tau_hi = tau_lo = float("nan")
        left = [t for t in (ctx.sort(resid) if v2 else resid) if t not in newlab and t not in junk]
        if cids:
            Cm = np.stack([ctx.V(allc[c]).mean(0) for c in cids])
            Cm /= np.linalg.norm(Cm, axis=1, keepdims=True)
            own = np.concatenate([ctx.V(allc[c]) @ Cm[i] for i, c in enumerate(cids)])
            tau_hi, tau_lo = np.percentile(own, C.reassign_hi_percentile), np.percentile(own, C.reassign_lo_percentile)
            if left:
                Sx = ctx.V(left) @ Cm.T
                o_ = np.argsort(-Sx, 1)
                s1 = Sx[np.arange(len(left)), o_[:, 0]]
                s2 = Sx[np.arange(len(left)), o_[:, 1]] if len(cids) > 1 else np.full(len(left), -np.inf)
                auto = (s1 >= tau_hi) & (s1 - s2 >= C.reassign_margin)
                band = (~auto) & (s1 >= tau_lo)
                n_auto, n_band = int(auto.sum()), int(band.sum())
                for i in np.where(auto)[0]:
                    add[cids[o_[i, 0]]].append(left[i])
                for i in np.where(band)[0]:
                    bandc[cids[o_[i, 0]]].append(left[i])
        vj = [(c, ctx.sort(random.Random(C.band_seed).sample(v, min(C.band_sample, len(v))))) for c, v in bandc.items() if v]
        vr = ctx.parallel(vj, lambda j: verify_membership(ctx, P_ + "s3", verifier, names[j[0]], allc[j[0]][:3], j[1])[0],
                          lambda j: ctx.cfg.roles.verifier)
        for (c, cands), ans in zip(vj, vr):
            yes = [t for t in cands if ans.get(t)]
            add[c] += bandc[c] if (len(cands) >= C.band_admit_min and len(yes) / len(cands) >= C.band_admit_fraction) else yes
        grown = [c for c in cids if len(add[c]) > C.growth_regate * len(allc[c])]
        suffix = self.tag if v2 else "cf"
        ctx.progress(cache_key, stage="re-gating grown clusters", grown=len(grown))
        rg = self.gate.gate_many(P_ + "regate", [(f"{c}+{suffix}", ctx.sort(allc[c] + add[c])) for c in grown])
        reverted = 0
        for c in cids:
            if c in grown and rg[f"{c}+{suffix}"]["decision"] != "ACCEPT":
                add[c] = []; reverted += 1
            for t in add[c]:
                newlab[t] = c

        # placements -------------------------------------------------------------------
        addset = {t for v in add.values() for t in v}
        how = {}
        newc = set(acc_new)
        prefix = "refiled_" if v2 else ""
        for t, c in newlab.items():
            if t in lab:
                continue
            if v2:
                if t in addset and not (t in filed and filed[t] == c):
                    how[t] = prefix + "reassigned"
                elif t in filed and c == filed[t]:
                    how[t] = prefix + "filed_existing_confirmed"
                elif c in newc:
                    how[t] = prefix + "new_cluster"
                else:
                    how[t] = prefix + "reassigned"
            else:
                how[t] = ("reassigned" if t in addset
                          else "filed_existing_confirmed" if (t in filed and c == filed[t])
                          else "new_cluster" if c in newc else "reassigned")
        log = {"pool": len(resid), "streams": np.bincount(stream).tolist(), "classified_filed_existing": len(filed),
               "filed_confirmed": confirmed, "provisional_labels": len(labels), "canonical_labels": len(groups), "gated": len(items),
               "accepted_new": sum(g[k]["decision"] == "ACCEPT" for k, _ in items), "new_clusters_after_prune": len(acc_new),
               "new_cluster_tickets": sum(len(v["members"]) for v in acc_new.values()), "junk_flagged": len(junk),
               "junk_reasons": pd.Series(list(junk.values())).value_counts().to_dict() if junk else {},
               "pruned_dissolved": dissolved, "pruned_ejected": ejected,
               "reassign": {"tau_hi": round(float(tau_hi), 4), "tau_lo": round(float(tau_lo), 4), "auto": n_auto, "band": n_band,
                            "verify_calls": len(vj), "added": len(addset - set(lab)), "grown": len(grown), "reverted": reverted},
               "placed": len([t for t in newlab if t not in lab]), "clusters": len(set(newlab.values())), "tickets_in_clusters": len(newlab),
               "wall_minutes": round((time.time() - T0) / 60, 1)}
        if unclassified is not None:
            log["classifier_contract_failures"] = unclassified
        return ResidualResult(labels=newlab, how=how, junk=junk, names=names, new=acc_new, log=log)
