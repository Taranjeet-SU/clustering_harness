"""Run context shared by all stages: data, vectors, models, prompts, artifact store and text views."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import PipelineConfig
from .data import Dataset, JunkRule, one_line
from .embeddings import Embedder
from .llm import LLM, run_parallel
from .masking import NoMasker, RegexMasker
from .prompts import PromptSet

__all__ = ["RunContext", "write_json"]

log = logging.getLogger("clustering_harness")


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


@dataclass
class RunContext:
    ds: Dataset
    cfg: PipelineConfig
    llms: dict                      # {"a": LLM, "b": LLM}
    workdir: Path
    X: np.ndarray                   # L2-normalised float32 vectors, dataset row order
    label_embedder: Embedder | None = None
    prompts: PromptSet | None = None
    progress_cb: Callable[[str, dict], None] | None = None
    junk: np.ndarray | None = None
    _label_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.workdir = Path(self.workdir)
        self.prompts = self.prompts or PromptSet.load(self.cfg.prompts, domain=self.cfg.domain or None)
        m = self.cfg.masking
        self.mask = RegexMasker(names=m.names, requester_prefix=m.requester_prefix) if m.enabled else NoMasker()
        if self.junk is None:
            j = self.cfg.junk
            self.junk = self.ds.junk_mask(JunkRule(j.placeholder_title, j.min_alnum, j.min_desc_alnum))
        self.nonjunk = np.where(~self.junk)[0]
        self.pos = self.ds.positions
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paths & persistence
    def path(self, *parts) -> Path:
        p = self.workdir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, obj, *parts) -> None:
        write_json(obj, self.path(*parts))

    def load(self, *parts, default=None):
        p = self.workdir.joinpath(*parts)
        return json.loads(p.read_text()) if p.exists() else default

    def progress(self, _key: str, /, **kw) -> None:
        kw["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            d = self.load("progress.json", default={})
            d[_key] = kw
            self.save(d, "progress.json")
        log.info("%s %s", _key, {k: v for k, v in kw.items() if k != "updated"})
        if self.progress_cb:
            self.progress_cb(_key, kw)

    # ------------------------------------------------------------------ ids & vectors
    def sort(self, ids) -> list:
        return self.ds.sort(ids)

    def V(self, ms: Sequence[str]) -> np.ndarray:
        return self.X[[self.pos[t] for t in ms]]

    def centroid(self, ms: Sequence[str]) -> np.ndarray:
        v = self.V(ms).mean(0)
        return v / np.linalg.norm(v)

    def exemplars(self, ms: Sequence[str], k: int = 3) -> list:
        c = self.centroid(ms)
        s = self.V(ms) @ c
        return [ms[i] for i in np.argsort(-s)[:k]]

    @property
    def eligible_ids(self) -> list:
        return [self.ds.ids[i] for i in self.nonjunk]

    # ------------------------------------------------------------------ text views (masked, one line)
    def title(self, rid: str) -> str:
        return one_line(self.mask(self.ds.titles[self.pos[rid]]), self.cfg.text.title_chars)

    def desc_head(self, rid: str, n: int = 300) -> str:
        return one_line(self.mask(self.ds.texts[self.pos[rid]]), n)

    def item(self, rid: str, n: int | None = None) -> str:
        n = self.cfg.gate.desc_chars if n is None else n
        return self.prompts.render("item", title=self.title(rid), desc=self.desc_head(rid, n) or self.prompts["item_empty_desc"])

    # ------------------------------------------------------------------ samples
    def gsample(self, cid: str, rep: int, members: Sequence[str], seed: int = 0) -> list:
        """Gate sample: up to ``gate.sample_size`` members, seeded by (cluster id, replicate)."""
        k = self.cfg.gate.sample_size
        ms = self.sort(members)
        if len(ms) <= k:
            return ms
        s = int(hashlib.md5(f"G|{cid}|{rep}|{seed}".encode()).hexdigest()[:8], 16)
        return self.sort(random.Random(s).sample(ms, k))

    def jsample(self, arm: str, cid: str, rep: int, members: Sequence[str], k: int = 8) -> list:
        """Judge sample: up to ``k`` members, seeded by (arm, cluster id, replicate)."""
        seed = int(hashlib.md5(f"{arm}|{cid}|{rep}".encode()).hexdigest()[:8], 16)
        ms = self.sort(members)
        if len(ms) <= k:
            return ms
        return self.sort(random.Random(seed).sample(ms, k))

    # ------------------------------------------------------------------ models
    def llm(self, slot: str) -> LLM:
        return self.llms[slot]

    def role(self, name: str) -> LLM:
        return self.llms[getattr(self.cfg.roles, name)]

    def parallel(self, jobs, fn, slot_of: Callable[[Any], str] | None = None, workers: int | None = None) -> list:
        return run_parallel(jobs, fn, lambda jb: self.llms[slot_of(jb)] if slot_of else self.llms["a"], workers)

    def label_vectors(self, texts: Sequence[str], cache_name: str | None = None) -> np.ndarray:
        """Embed short labels with the label embedder (cached on disk per text list)."""
        if self.label_embedder is None:
            raise RuntimeError("this stage needs a label_embedder")
        texts = list(texts)
        h = hashlib.sha256(json.dumps([self.label_embedder.name, texts]).encode()).hexdigest()[:16]
        p = self.workdir / "cache" / "labels" / f"{cache_name or 'labels'}_{h}.npy"
        if p.exists():
            return np.load(p)
        V = np.asarray(self.label_embedder.encode(texts), dtype=np.float32)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, V)
        return V
