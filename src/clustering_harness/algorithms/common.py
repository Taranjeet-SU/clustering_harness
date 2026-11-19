"""LLM helpers shared by several stages: membership verification and report-line labelling."""
from __future__ import annotations

from typing import Sequence

from ..context import RunContext
from ..llm import LLM

__all__ = ["verify_membership", "write_line", "as_bool"]


def as_bool(v) -> bool | None:
    """Strict truth value of a model answer: JSON booleans, numbers, and yes/no/true/false strings; None if unclear."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    return None


def verify_membership(ctx: RunContext, stage: str, llm: LLM, line: str, exemplars: Sequence[str], cands: Sequence[str],
                      crit: str = "", seed: int = 0) -> tuple[dict, bool]:
    """Ask ``llm`` whether each candidate belongs on report line ``line`` (shown with exemplar members).

    Returns ({candidate id: bool}, contract_failed). Missing answers are absent from the dict (never "no")."""
    P = ctx.prompts
    ex = "\n".join(f"- {ctx.item(t, 160)}" for t in exemplars)
    it = "\n".join(f"{i+1}. {ctx.item(t, 220)}" for i, t in enumerate(cands))
    user = P.render("verify_user", line=line, criterion=crit or P["verify_default_criterion"], exemplars=ex, candidates=it)
    out = llm.call(stage, [{"role": "system", "content": P["verify_system"]}, {"role": "user", "content": user}],
                   max_tokens=1500 if llm.reasoning else 400, seed=seed)
    ans = {}
    for x in (out.get("answers") or []) if "_error" not in out else []:
        try:
            b = as_bool(x["belongs"])
            if b is not None:
                ans[cands[int(x["n"]) - 1]] = b
        except Exception:  # noqa: BLE001
            pass
    return ans, "_error" in out


def write_line(ctx: RunContext, stage: str, llm: LLM, cid: str, members: Sequence[str]) -> tuple[str, str]:
    """Report line + one-sentence inclusion criterion from the gate sample (replicate 0) of a cluster."""
    P = ctx.prompts
    ex = "\n".join(f"- {ctx.item(t, 200)}" for t in ctx.gsample(cid, 0, members))
    out = llm.call(stage, [{"role": "system", "content": P["label_system"]}, {"role": "user", "content": P.render("label_user", exemplars=ex)}],
                   max_tokens=1500)
    return out.get("line", "") or "", out.get("criterion", "") or ""
