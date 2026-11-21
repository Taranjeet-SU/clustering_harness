"""Two-judge calibrated report-line gate.

For a candidate cluster, each of the two judge models scores ``reps`` independently seeded samples of up
to ``sample_size`` members on a 1–5 "one report line?" scale. Scores are z-normalised per judge onto the
pooled scale, with (mean, sd) frozen from the first gated batch (the base parents):

    adj = pooled_mu + pooled_sd * (score - judge_mu) / judge_sd      (judge_sd == 0 -> 1)

Decision (pipeline doc, step 5):
    ACCEPT   mean(adj) >= accept_mean and both judges' mean(adj) >= accept_min_judge
    REJECT   mean(adj) <= reject_mean
    ESCALATE otherwise (or all calls failed): one more sample from both judges, re-decide; if still in the
             band, ACCEPT iff mean(adj) >= escalate_accept. CONTRACT_FAIL if no valid score at all.
The first valid non-empty name, in call order (replicate 0 judge a, replicate 0 judge b, replicate 1 ...), names
the cluster."""
from __future__ import annotations

from typing import Sequence

import logging

import numpy as np

from ..context import RunContext

__all__ = ["Gate"]

log = logging.getLogger("clustering_harness")
JUDGES = ("a", "b")


class Gate:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.cfg = ctx.cfg.gate
        self.calib_path = ("cache", "gate_calibration.json")

    # ------------------------------------------------------------------ one call
    def messages(self, sids: Sequence[str]) -> list:
        P = self.ctx.prompts
        items = "\n".join(f"{i+1}. {self.ctx.item(t, self.cfg.desc_chars)}" for i, t in enumerate(sids))
        return [{"role": "system", "content": P["gate_system"].strip()},
                {"role": "user", "content": P.render("gate_user", n=len(sids), items=items).strip()}]

    def call(self, stage: str, cid: str, members, rep: int, judge: str, seed: int = 0) -> dict:
        llm = self.ctx.llm(judge)
        sids = self.ctx.gsample(cid, rep, members, seed)
        msgs = self.messages(sids)
        out = llm.call(stage, msgs, max_tokens=self.cfg.max_tokens_reasoning if llm.reasoning else self.cfg.max_tokens,
                       seed=rep, tag=f"gate|{cid}|{rep}")
        if "_error" in out and llm.reasoning:
            out = llm.call(stage, msgs, max_tokens=self.cfg.retry_tokens_reasoning, seed=rep, tag=f"gate|{cid}|{rep}|budget")
        ok = "_error" not in out
        score = None
        if ok:
            try:
                score = min(5, max(1, int(round(float(out["score"])))))
            except Exception:  # noqa: BLE001
                ok = False
        return {"cid": cid, "rep": rep, "judge": judge, "score": score, "ok": ok, "name": str(out.get("name") or "")[:90],
                "reason": str(out.get("reason", ""))[:200], "size": len(members)}

    # ------------------------------------------------------------------ calibration & decision
    def calibrate(self, calls: list, force: bool = False) -> dict:
        b = self.ctx.load(*self.calib_path)
        if b is not None and not force:
            return b
        b = {}
        for j in JUDGES:
            v = [c["score"] for c in calls if c["ok"] and c["judge"] == j]
            if not v:
                n = sum(1 for c in calls if c["judge"] == j)
                raise RuntimeError(f"gate calibration: judge {j!r} returned no valid score in {n} calls (every answer "
                                   f"failed the JSON contract). Look at the model's raw answers under <workdir>/results/; "
                                   f"a model that cannot follow the JSON instructions needs reasoning: true, more tokens, or "
                                   f"a different model")
            b[j] = {"mu": float(np.mean(v)), "sd": float(np.std(v)) or 1.0, "n": len(v)}
            if len(v) >= 5 and float(np.std(v)) == 0.0:
                b[j]["constant"] = True
                msg = (f"gate calibration: judge {j!r} ({self.ctx.llm(j).model_id}) gave all {len(v)} calibration answers "
                       f"the same score ({v[0]}), so it cannot tell good groups from bad ones. Check the model and its "
                       f"answers (a stub, a broken prompt, or a model too weak to judge) and use another model")
                if not self.cfg.allow_constant_judge:
                    raise RuntimeError(msg + " (or set pipeline.gate.allow_constant_judge: true to continue anyway).")
                log.warning("%s; continuing because allow_constant_judge is set.", msg)
        allv = [c["score"] for c in calls if c["ok"]]
        b["pooled_mu"] = float(np.mean(allv))
        b["pooled_sd"] = float(np.std(allv)) or 1.0   # a uniform calibration batch would otherwise collapse every adjusted score
        b["frozen_from"] = "first gated batch (base parents), replicates 0..reps-1"
        self.ctx.save(b, *self.calib_path)
        return b

    @staticmethod
    def adjust(score: float, judge: str, b: dict) -> float:
        return b["pooled_mu"] + b["pooled_sd"] * (score - b[judge]["mu"]) / b[judge]["sd"]

    def decide(self, calls: list, b: dict) -> dict:
        good = [c for c in calls if c["ok"]]
        if not good:
            return {"decision": "CONTRACT_FAIL", "adj": None, "n": 0, "fails": len(calls)}
        per = {j: [self.adjust(c["score"], j, b) for c in good if c["judge"] == j] for j in JUDGES}
        pj = {j: float(np.mean(v)) for j, v in per.items() if v}
        m = float(np.mean([x for v in per.values() for x in v]))
        if m >= self.cfg.accept_mean and len(pj) == 2 and min(pj.values()) >= self.cfg.accept_min_judge:
            d = "ACCEPT"
        elif m <= self.cfg.reject_mean:
            d = "REJECT"
        else:
            d = "ESCALATE"
        names = [c["name"] for c in good if c["name"]]
        return {"decision": d, "adj": round(m, 3), "per_judge": {k: round(v, 3) for k, v in pj.items()},
                "raw_mean": round(float(np.mean([c["score"] for c in good])), 3), "n": len(good), "fails": len(calls) - len(good),
                "name": names[0] if names else ""}

    # ------------------------------------------------------------------ batch
    def gate_many(self, stage: str, items: list, seed: int = 0, calibrate_now: bool = False) -> dict:
        """``items``: list of (cluster id, members). Returns {cluster id: decision dict}."""
        if not items:
            return {}
        if not calibrate_now and self.ctx.load(*self.calib_path) is None:
            raise RuntimeError("gate calibration not frozen yet: gate the base parents first")
        reps = self.cfg.reps
        base = [(cid, ms, r, j) for cid, ms in items for r in range(reps) for j in JUDGES]
        res = self.ctx.parallel(base, lambda x: self.call(stage, x[0], x[1], x[2], x[3], seed), lambda x: x[3])
        b = self.calibrate(res)
        by: dict = {}
        for c in res:
            by.setdefault(c["cid"], []).append(c)
        esc = [(cid, ms, reps, j) for cid, ms in items for j in JUDGES
               if self.decide(by.get(cid, []), b)["decision"] in ("ESCALATE", "CONTRACT_FAIL")]
        res2 = self.ctx.parallel(esc, lambda x: self.call(stage, x[0], x[1], x[2], x[3], seed), lambda x: x[3])
        for c in res2:
            by.setdefault(c["cid"], []).append(c)
        out = {}
        for cid, ms in items:
            calls = by.get(cid, [])
            d = self.decide(calls, b)
            if len(calls) > reps * 2 and d["decision"] in ("ESCALATE", "CONTRACT_FAIL"):
                d["escalated"] = True
                if d["adj"] is not None:
                    d["decision"] = "ACCEPT" if d["adj"] >= self.cfg.escalate_accept else "REJECT"
            d.update(cid=cid, size=len(ms), calls=len(calls), reasons=[c["reason"] for c in calls if c["ok"]][:3])
            out[cid] = d
        return out
