"""Reusable, deterministic test doubles for the clustering_harness conformance tests.

* :class:`FakeLLM` — a scripted model behind :class:`~clustering_harness.llm.CallableBackend`. It recognises every
  prompt type of the pipeline by its template text and answers either from per-kind rules, from a callback or
  from built-in *topic-aware* defaults (records carry a topic phrase in their title, so the fake can judge
  purity, file tickets, verify membership, ... exactly like an ideal model would).
* :func:`make_topic_dataset` — synthetic records with recognisable topic phrases plus matching vectors
  (well-separated topic centroids + noise, L2-normalised).
* :class:`HashEmbedder` — deterministic bag-of-words label embedder (no model download).
* :func:`fake_llms` / :func:`make_ctx` — model slots and a :class:`RunContext` in a temporary directory.

Every text here is invented and generic.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from clustering_harness.config import PipelineConfig
from clustering_harness.context import RunContext
from clustering_harness.data import Dataset
from clustering_harness.llm import LLM, CallableBackend, ChatRequest, ChatResult

__all__ = ["TOPICS", "FakeLLM", "PromptInfo", "StageLLM", "fake_llms", "make_topic_dataset", "make_ctx", "HashEmbedder",
           "topic_of", "rid_of", "KIND_MARKERS", "FakeGate"]

# ---------------------------------------------------------------------------------------------- topics
#: (topic phrase, extra vocabulary). Phrases are mutually exclusive substrings and never generic words.
TOPICS: list[tuple[str, list[str]]] = [
    ("password reset", ["link", "mail", "expired", "token", "forgot"]),
    ("invoice mismatch", ["amount", "billing", "tax", "currency", "credit"]),
    ("export failure", ["csv", "stuck", "timeout", "download", "file"]),
    ("sso login", ["saml", "redirect", "certificate", "idp", "assertion"]),
    ("license renewal", ["seats", "quote", "expiry", "contract", "term"]),
    ("upload error", ["attachment", "size", "rejected", "format", "limit"]),
    ("notification delay", ["alert", "digest", "late", "queue", "schedule"]),
    ("api quota", ["throttled", "requests", "limit", "key", "burst"]),
    ("dashboard slowness", ["widgets", "loading", "spinner", "charts", "latency"]),
    ("user provisioning", ["onboarding", "roles", "accounts", "group", "access"]),
    ("calendar sync", ["events", "duplicate", "timezone", "invite", "meeting"]),
    ("printer setup", ["driver", "queue", "spooler", "tray", "paper"]),
    ("search indexing", ["crawler", "stale", "results", "reindex", "facets"]),
    ("webhook retries", ["payload", "endpoint", "signature", "backoff", "delivery"]),
]
PHRASES = [p for p, _ in TOPICS]
_FILLER = ["customer", "reports", "since", "yesterday", "please", "advise", "urgent", "team", "noticed", "again", "after",
           "update", "tenant", "environment", "screen", "steps", "attached", "logs", "issue", "seen"]


def topic_of(text: str | None) -> str | None:
    """The topic phrase contained in ``text`` (case-insensitive), or ``None``."""
    s = (text or "").lower()
    for p in PHRASES:
        if p in s:
            return p
    return None


_RID = re.compile(r"\bcase (\d+)\b")


def rid_of(text: str | None) -> str | None:
    m = _RID.search(text or "")
    return m.group(1) if m else None


def make_topic_dataset(n_topics: int = 6, per_topic: int | list[int] = 40, seed: int = 0, dim: int = 32, noise: float = 0.3,
                       n_outliers: int = 0, n_junk: int = 0, id_start: int = 1000, shuffle: bool = True,
                       name: str = "synthetic") -> tuple[Dataset, np.ndarray]:
    """Synthetic support-ticket-like records.

    Topic records: title ``"<topic phrase> case <id>"`` and a description repeating the phrase with topic
    vocabulary. Outliers have no topic phrase and random vectors; junk records have a placeholder title and
    an empty description (rule junk). ``ds.labels`` holds the topic phrase, ``"outlier"`` or ``"junk"``.
    Vectors: unit topic centroid + ``noise`` * N(0, I/dim), L2-normalised; float32."""
    assert n_topics <= len(TOPICS)
    rng = np.random.default_rng(seed)
    sizes = [per_topic] * n_topics if isinstance(per_topic, int) else list(per_topic)
    cents = rng.normal(size=(n_topics, dim))
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    kinds = [("topic", k) for k in range(n_topics) for _ in range(sizes[k])] + [("outlier", -1)] * n_outliers + [("junk", -2)] * n_junk
    order = rng.permutation(len(kinds)) if shuffle else np.arange(len(kinds))
    ids, titles, texts, labels, vecs = [], [], [], [], []
    for row, o in enumerate(order):
        kind, k = kinds[o]
        rid = str(id_start + row)
        ids.append(rid)
        if kind == "topic":
            phrase, vocab = TOPICS[k]
            words = list(rng.choice(vocab, 3)) + list(rng.choice(_FILLER, 4))
            titles.append(f"{phrase} case {rid}")
            texts.append(f"Customer reports {phrase} problem. " + " ".join(words) + f". The {phrase} keeps happening.")
            labels.append(phrase)
            vecs.append(cents[k] + noise * rng.normal(size=dim) / np.sqrt(dim))
        elif kind == "outlier":
            titles.append(f"question case {rid}")
            texts.append("General question about " + " ".join(rng.choice(_FILLER, 6)) + ".")
            labels.append("outlier")
            vecs.append(rng.normal(size=dim))
        else:
            titles.append("test")
            texts.append("")
            labels.append("junk")
            vecs.append(rng.normal(size=dim))
    X = np.asarray(vecs, dtype=np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return Dataset(ids=ids, texts=texts, titles=titles, labels=labels, name=name), X.astype(np.float32)


# ---------------------------------------------------------------------------------------------- embedder
class HashEmbedder:
    """Deterministic bag-of-words embedder for labels and titles (hashing trick, L2-normalised)."""

    def __init__(self, dim: int = 64, name: str = "hash-bow-64"):
        self.dim = dim
        self.name = name
        self.calls = 0

    def encode(self, texts) -> np.ndarray:
        self.calls += 1
        out = np.full((len(texts), self.dim), 1e-3, np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", (t or "").lower()):
                if w in ("case", "the"):
                    continue
                out[i, int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------- prompt kinds
#: Fixed template fragments that identify each prompt kind (checked against the default preset by the tests).
KIND_MARKERS: list[tuple[str, str]] = [
    ("gate", "GRANULARITY CONTRACT"),
    ("verify", "Candidate tickets:"),
    ("label", "a one-sentence inclusion criterion that decides whether another ticket belongs on it"),
    ("merge", "Should A and B be ONE line of the report"),
    ("classify", 'Return JSON: {"tickets"'),
    ("canonicalise", "Canonicalise these provisional report-line labels"),
    ("judge", "Then name the cluster in under 10 words"),
    ("rollup_decide", "Partition ALL the clusters above into report lines"),
    ("rollup_name", "This level-1 report line groups these clusters"),
]

_tls = threading.local()


def current_stage() -> str:
    return getattr(_tls, "stage", "")


class StageLLM(LLM):
    """``LLM`` that exposes the calling stage to the backend (thread-local), so fakes can script per stage."""

    def call(self, stage, messages, *a, **kw):
        prev = getattr(_tls, "stage", "")
        _tls.stage = stage
        try:
            return super().call(stage, messages, *a, **kw)
        finally:
            _tls.stage = prev


@dataclass
class PromptInfo:
    kind: str
    slot: str
    stage: str
    request: ChatRequest
    system: str
    user: str                     # first user message (the prompt proper)
    n_user: int                   # number of user messages (> 1 for repair / schema retries)
    extra: dict = field(default_factory=dict)

    # ---- parsed views
    @property
    def seed(self) -> int:
        return self.request.seed

    @property
    def max_tokens(self) -> int:
        return self.request.max_tokens

    def titles(self, text: str | None = None) -> list[str]:
        return re.findall(r"Title: (.*)", self.user if text is None else text)

    def payload(self) -> dict:
        return json.loads(self.user.split("\nJSON schema: ")[0])

    def schema_title(self) -> str:
        return json.loads(self.user.split("\nJSON schema: ", 1)[1]).get("title", "")

    def section(self, start: str, end: str | None = None) -> str:
        s = self.user.split(start, 1)[1] if start in self.user else ""
        return s.split(end, 1)[0] if end and end in s else s


def detect_kind(system: str, user: str) -> str:
    if "\nJSON schema: " in user:
        t = json.loads(user.split("\nJSON schema: ", 1)[1]).get("title", "")
        return {"Name": "score_name", "Judgment": "score_judgment"}.get(t, "score_other")
    for kind, marker in KIND_MARKERS:
        if marker in user:
            return kind
    return "unknown"


def _majority(xs):
    xs = [x for x in xs if x]
    if not xs:
        return None, 0.0
    t, n = collections.Counter(xs).most_common(1)[0]
    return t, n / len(xs)


def _purity_score(topics) -> tuple[int, str | None]:
    if not topics:
        return 1, None
    t, frac = _majority(topics)
    frac = frac * sum(1 for x in topics if x) / len(topics)
    if t is None:
        return 1, None
    return (5 if frac == 1 else 4 if frac >= 0.85 else 3 if frac >= 0.6 else 2), t


class FakeLLM:
    """Callable backend function. ``rules[kind]`` may be a dict / str (fixed answer) or a callable
    ``f(info) -> dict | str | None`` (``None`` falls through to the default). ``callback(info)`` is tried first
    for every kind. Every request is recorded in :attr:`log`."""

    def __init__(self, slot: str, rules: dict | None = None, callback: Callable[[PromptInfo], Any] | None = None):
        self.slot = slot
        self.rules = dict(rules or {})
        self.callback = callback
        self.log: list[PromptInfo] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ dispatch
    def __call__(self, req: ChatRequest):
        system = "\n".join(m["content"] for m in req.messages if m["role"] == "system")
        users = [m["content"] for m in req.messages if m["role"] == "user"]
        info = PromptInfo(detect_kind(system, users[0] if users else ""), self.slot, current_stage(), req, system,
                          users[0] if users else "", len(users))
        with self._lock:
            self.log.append(info)
        out = None
        if self.callback is not None:
            out = self.callback(info)
        if out is None and info.kind in self.rules:
            r = self.rules[info.kind]
            out = r(info) if callable(r) else r
        if out is None:
            out = getattr(self, "default_" + info.kind, self.default_unknown)(info)
        if isinstance(out, ChatResult):
            return out
        return out if isinstance(out, str) else json.dumps(out)

    def kinds(self) -> collections.Counter:
        return collections.Counter(i.kind for i in self.log)

    def of(self, kind: str) -> list[PromptInfo]:
        return [i for i in self.log if i.kind == kind]

    # ------------------------------------------------------------------ topic-aware defaults
    def default_unknown(self, info):
        return {}

    def default_gate(self, info):
        score, t = _purity_score([topic_of(x) for x in info.titles()])
        if self.slot == "b" and score == 5 and info.seed % 2 == 1:
            score = 4          # gives the second judge some variance for the calibration
        return {"score": score, "verdict": "pass" if score >= 4 else "too_broad", "name": f"{t.title()} problems" if t else "",
                "outliers": [], "reason": "synthetic judgement"}

    def default_verify(self, info):
        line = info.section("Report line: ", "\n")
        ex_t = [topic_of(x) for x in info.titles(info.section("Tickets already on this line:", "Candidate tickets:"))]
        ref = topic_of(line) or _majority(ex_t)[0]
        cands = info.titles(info.section("Candidate tickets:"))
        return {"answers": [{"n": i + 1, "belongs": bool(ref) and topic_of(c) == ref} for i, c in enumerate(cands)]}

    def default_label(self, info):
        t = _majority([topic_of(x) for x in info.titles()])[0]
        return {"line": f"{t.title()} problems" if t else "Unclear tickets", "criterion": f"The ticket reports {t or 'this'}."}

    def default_merge(self, info):
        a = info.section("Line A: ", "\nLine B: ")
        b = info.section("\nLine B: ")
        ta, tb = _majority([topic_of(x) for x in info.titles(a)])[0], _majority([topic_of(x) for x in info.titles(b)])[0]
        return {"same_line": bool(ta) and ta == tb, "reason": "synthetic"}

    def default_classify(self, info):
        acc = dict(re.findall(r"^(L\d+)\. (.*)$", info.section("Report lines already accepted", "Provisional labels"), re.M))
        prov = dict(re.findall(r"^(N\d+)\. (.*)$", info.section("Provisional labels"), re.M))
        out = []
        for n, title in re.findall(r"^\[(\d+)\] Title: (.*)$", info.user, re.M):
            t = topic_of(title)
            if t is None:
                ch = "JUNK: insufficient_information"
            else:
                ch = next((k for k, v in acc.items() if topic_of(v) == t), None) or \
                     next((k for k, v in prov.items() if topic_of(v) == t), None) or f"NEW: {t} requests"
            out.append({"n": int(n), "choice": ch})
        return {"tickets": out}

    def default_canonicalise(self, info):
        rows = re.findall(r"^(\d+)\. (.*)$", info.user, re.M)
        return {"map": [{"n": int(n), "canonical": (topic_of(l) or l) + " requests"} for n, l in rows]}

    def default_judge(self, info):
        score, t = _purity_score([topic_of(x) for x in re.findall(r"Subject: (.*)", info.user)])
        return {"coherence": score, "name": f"{t.title()} problems" if t else "Mixed", "reason": "synthetic"}

    def default_score_name(self, info):
        p = info.payload()
        t = _majority([topic_of(c["text"]) for c in p["own_cases"]])[0] or "unclear"
        return {"name": f"{t.title()} issue reports", "inclusion_criterion": f"Tickets that describe a {t} problem.",
                "exclusion_criterion": "Anything else.", "rationale": "All sampled cases share the topic.",
                "unsupported_cause_warning": ""}

    def default_score_judgment(self, info):
        p = info.payload()
        ts = [topic_of(c["text"]) for c in p["own_cases"]]
        score, t = _purity_score(ts)
        bad = [c["id"] for c, x in zip(p["own_cases"], ts) if x != t]
        good = [c["id"] for c, x in zip(p["own_cases"], ts) if x == t][:2]
        return {"coherence": score, "label_fit": 5, "distinction": 5, "decision": "ACCEPT" if score >= 4 else "SPLIT",
                "reason": "Synthetic judgement of the panel.", "supporting_ids": good, "suspicious_ids": bad,
                "subissues": [], "limitation": ""}

    def default_rollup_decide(self, info):
        rows = re.findall(r"^(c\d+)\. (.*?) \(\d+ tickets\)\. Examples: (.*)$", info.user, re.M)
        by = collections.defaultdict(list)
        for k, name, ex in rows:
            by[topic_of(ex) or topic_of(name) or k].append(k)
        return {"lines": [{"ids": v, "why": "same topic"} for v in by.values()]}

    def default_rollup_name(self, info):
        t = _majority([topic_of(x) for x in re.findall(r"^- (.*)$", info.user, re.M)])[0]
        return {"name": f"{t.title()} line" if t else "Unnamed line", "criterion": f"Tickets about {t}."}


def fake_llms(rules_a: dict | None = None, rules_b: dict | None = None, callback_a=None, callback_b=None,
              reasoning_a: bool = False, reasoning_b: bool = False, cache=None, max_concurrency: int = 2):
    """Two scripted model slots. Returns ``({"a": LLM, "b": LLM}, {"a": FakeLLM, "b": FakeLLM})``."""
    fa, fb = FakeLLM("a", rules_a, callback_a), FakeLLM("b", rules_b, callback_b)
    llms = {"a": StageLLM(CallableBackend(fa, "fake:a"), name="a", reasoning=reasoning_a, cache=cache, max_concurrency=max_concurrency),
            "b": StageLLM(CallableBackend(fb, "fake:b"), name="b", reasoning=reasoning_b, cache=cache, max_concurrency=max_concurrency)}
    return llms, {"a": fa, "b": fb}


def make_ctx(tmp_path: Path, ds: Dataset | None = None, X: np.ndarray | None = None, cfg: PipelineConfig | None = None,
             llms: dict | None = None, label_embedder=None, **dataset_kw) -> RunContext:
    """A :class:`RunContext` over a synthetic dataset in ``tmp_path / "run"``."""
    if ds is None:
        ds, X = make_topic_dataset(**dataset_kw)
    if llms is None:
        llms, _ = fake_llms()
    X = np.asarray(X, np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return RunContext(ds=ds, cfg=cfg or PipelineConfig.small(), llms=llms, workdir=Path(tmp_path) / "run", X=X,
                      label_embedder=label_embedder or HashEmbedder())


# ---------------------------------------------------------------------------------------------- gate double
class FakeGate:
    """Stand-in for :class:`~clustering_harness.algorithms.gate.Gate` that decides from a rule.

    ``rule(stage, cid, members) -> (decision, adj) | None``; ``None`` -> ``default``. Every call is recorded."""

    def __init__(self, rule: Callable[[str, str, list], tuple | None] | None = None, default=("REJECT", 2.0)):
        self.rule = rule
        self.default = default
        self.calls: list[tuple[str, list]] = []

    def items(self, stage_prefix: str) -> list:
        return [it for s, its in self.calls if s.startswith(stage_prefix) for it in its]

    def gate_many(self, stage, items, seed=0, calibrate_now=False):
        items = [(c, list(m)) for c, m in items]
        self.calls.append((stage, items))
        out = {}
        for cid, ms in items:
            r = self.rule(stage, cid, ms) if self.rule else None
            d, adj = r if r is not None else self.default
            out[cid] = {"decision": d, "adj": adj, "name": f"name of {cid}", "cid": cid, "size": len(ms)}
        return out
