"""Pluggable text embedders.

Every embedder implements ``encode(texts) -> np.ndarray`` returning one L2-normalised float32 row per
text (same order) and exposes a stable ``name`` used for caching. Implementations:

* :class:`HFTransformerEmbedder` — any Hugging Face encoder with CLS or mean pooling, long-context aware,
  length-sorted token-budget batching (reference default: ``Alibaba-NLP/gte-large-en-v1.5``, CLS, 8,192).
* :class:`SentenceTransformerEmbedder` — any ``sentence-transformers`` model (reference label
  embedder: ``BAAI/bge-large-en-v1.5``).
* :class:`OllamaEmbedder`, :class:`OpenAIEmbedder` — remote embedding APIs.
* :class:`PrecomputedEmbedder` — vectors you already have.
* :class:`CachedEmbedder` — wraps any embedder with an on-disk cache keyed by model name + text hash.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import logging
import time

import numpy as np

log = logging.getLogger("clustering_harness")
__all__ = ["Embedder", "HFTransformerEmbedder", "SentenceTransformerEmbedder", "OllamaEmbedder", "OpenAIEmbedder",
           "PrecomputedEmbedder", "CachedEmbedder", "l2_normalize"]


def l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


@runtime_checkable
class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


def _token_budget_batches(lengths: np.ndarray, batch_tokens: int):
    """Length-sorted batches whose padded size (rows x longest row) stays within ``batch_tokens``."""
    order = np.argsort(lengths, kind="stable")
    cur, cur_max = [], 0
    for i in order:
        L = int(lengths[i]); new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > batch_tokens:
            yield cur
            cur, cur_max = [], 0
            new_max = L
        cur.append(int(i)); cur_max = new_max
    if cur:
        yield cur


def _extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
    """The ``PreTrainedModel.get_extended_attention_mask`` of transformers 4.x (encoder case), which remote
    model code still calls but recent transformers releases no longer provide."""
    import torch
    dtype = dtype or self.dtype
    if attention_mask.dim() == 3:
        ext = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        ext = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"wrong shape for attention_mask: {tuple(attention_mask.shape)}")
    ext = ext.to(dtype=dtype)
    return (1.0 - ext) * torch.finfo(dtype).min


def _ensure_extended_attention_mask(model) -> None:
    import types
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m, "dtype") and not hasattr(m, "get_extended_attention_mask"):
            m.get_extended_attention_mask = types.MethodType(_extended_attention_mask, m)


def _fix_gte_buffers(model) -> None:
    """``Alibaba-NLP/gte-*-en-v1.5`` remote code can leave position ids and rotary caches uninitialised under
    recent ``transformers`` versions; recompute them exactly as the model's ``__init__`` does. Also restores
    ``get_extended_attention_mask``, which the remote code calls and recent transformers removed."""
    import torch
    _ensure_extended_attention_mask(model)
    emb = getattr(model, "embeddings", None)
    rot = getattr(emb, "rotary_emb", None) if emb is not None else None
    if rot is None or not hasattr(rot, "_set_cos_sin_cache"):
        return
    n = model.config.max_position_embeddings
    dev = emb.word_embeddings.weight.device
    emb.register_buffer("position_ids", torch.arange(n, device=dev), persistent=False)
    rot.inv_freq = 1.0 / (rot.base ** (torch.arange(0, rot.dim, 2).float().to(dev) / rot.dim))
    rot._set_cos_sin_cache(seq_len=n, device=dev, dtype=torch.get_default_dtype())
    factor = getattr(rot, "scaling_factor", None)
    if factor:
        rot._set_cos_sin_cache(seq_len=int(n * factor), device=dev, dtype=torch.get_default_dtype())


class HFTransformerEmbedder:
    """Hugging Face ``AutoModel`` encoder.

    Parameters mirror the reference implementation: inputs are truncated to ``max_length`` tokens, batched
    by length under a padded-token budget, pooled (``"cls"`` = first token, ``"mean"`` = attention-masked
    mean), L2-normalised in float32, and optionally stored as float16 (``store_float16``) before being
    returned as float32 (re-normalised).

    ``device`` defaults to ``"cpu"``; pass ``"cuda"``, ``"cuda:1"`` or ``"auto"`` (CUDA when available) to use a
    GPU. ``half=True`` runs the model in float16 on CUDA; the precision is part of :attr:`name` (``:fp16``) so
    caches never mix float16-GPU and float32 vectors.
    """

    def __init__(self, model: str = "Alibaba-NLP/gte-large-en-v1.5", pooling: str = "cls", max_length: int = 8192,
                 device: str | None = "cpu", batch_tokens: int = 32768, half: bool = True, store_float16: bool = True,
                 trust_remote_code: bool = True, local_files_only: bool = False, query_prefix: str = "",
                 revision: str | None = None, code_revision: str | None = None):
        self.model = model
        # pin the model weights (revision) and, for models that run their own code, that code (code_revision)
        self.revision = revision
        self.code_revision = code_revision
        self.pooling = pooling
        self.max_length = max_length
        if device in (None, "auto"):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_tokens = batch_tokens
        self.half = half
        self.store_float16 = store_float16
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.query_prefix = query_prefix
        self.name = (f"hf:{model}" + (f"@{revision}" if revision else "") + f":{pooling}:{max_length}"
                     + (":fp16" if half and device.startswith("cuda") else ""))
        self.last_stats: dict = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer
        texts = [self.query_prefix + (t or "") for t in texts]
        dev = self.device
        pins = {k: v for k, v in (("revision", self.revision), ("code_revision", self.code_revision)) if v}
        tok = AutoTokenizer.from_pretrained(self.model, local_files_only=self.local_files_only, trust_remote_code=self.trust_remote_code,
                                            **{k: v for k, v in pins.items() if k == "revision"})
        enc = tok(texts, truncation=True, max_length=self.max_length, add_special_tokens=True)["input_ids"]
        raw_len = np.array([len(x) for x in tok(texts, truncation=False, add_special_tokens=True)["input_ids"]])
        lengths = np.array([len(x) for x in enc])
        log.info("embed: loading %s on %s", self.model, dev)
        am = AutoModel.from_pretrained(self.model, trust_remote_code=self.trust_remote_code, local_files_only=self.local_files_only,
                                       dtype=torch.float32, **pins).to(dev).eval()
        _fix_gte_buffers(am)
        if self.half and dev.startswith("cuda"):
            am = am.half()
        dim = am.config.hidden_size
        out = np.zeros((len(texts), dim), dtype=np.float16 if self.store_float16 else np.float32)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        done, next_report, t0 = 0, 0.1, time.time()
        for rows in _token_budget_batches(lengths, self.batch_tokens):
            L = int(lengths[rows].max())
            ids_t = torch.full((len(rows), L), pad, dtype=torch.long)
            mask = torch.zeros((len(rows), L), dtype=torch.long)
            for j, r in enumerate(rows):
                x = enc[r]; ids_t[j, : len(x)] = torch.tensor(x); mask[j, : len(x)] = 1
            with torch.no_grad():
                def run(a, m):
                    hs = am(input_ids=a.to(dev), attention_mask=m.to(dev)).last_hidden_state
                    if self.pooling == "cls":
                        return hs[:, 0]
                    mm = m.to(dev).unsqueeze(-1).to(hs.dtype)
                    return (hs * mm).sum(1) / mm.sum(1).clamp(min=1)
                try:
                    h = run(ids_t, mask)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    h = torch.cat([run(ids_t[j:j + 1], mask[j:j + 1]) for j in range(len(rows))])
                h = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
            out[rows] = h.astype(out.dtype)
            done += len(rows)
            if len(texts) and done / len(texts) >= next_report:
                log.info("embed: %d/%d records (%.0f%%, %.1f min)", done, len(texts), 100 * done / len(texts), (time.time() - t0) / 60)
                next_report = min(1.0, done / len(texts) + 0.1)
        am = None
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        self.last_stats = {"n": len(texts), "dim": dim, "tokens_median": int(np.median(raw_len)) if len(raw_len) else 0,
                           "tokens_max": int(raw_len.max()) if len(raw_len) else 0,
                           "truncated": int((raw_len > self.max_length).sum())}
        return l2_normalize(out.astype(np.float32))


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-large-en-v1.5", device: str = "cpu", batch_size: int = 32, prefix: str = ""):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.prefix = prefix
        self.name = f"st:{model}"
        self._m = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if self._m is None:
            self._m = SentenceTransformer(self.model, device=self.device)
        V = self._m.encode([self.prefix + t for t in texts], normalize_embeddings=True, batch_size=self.batch_size,
                           show_progress_bar=False)
        return np.asarray(V, dtype=np.float32)


class OllamaEmbedder:
    def __init__(self, model: str = "bge-m3", url: str = "http://localhost:11434", batch_size: int = 64, timeout: float = 600):
        self.model, self.url, self.batch_size, self.timeout = model, url.rstrip("/"), batch_size, timeout
        self.name = f"ollama-embed:{model}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import requests
        rows = []
        for i in range(0, len(texts), self.batch_size):
            r = requests.post(self.url + "/api/embed", json={"model": self.model, "input": list(texts[i:i + self.batch_size])},
                              timeout=self.timeout)
            r.raise_for_status()
            rows += r.json()["embeddings"]
        return l2_normalize(np.array(rows, dtype=np.float32))


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", base_url: str | None = None, api_key_env: str | None = "auto",
                 batch_size: int = 256, dimensions: int | None = None):
        from .llm.backends import resolve_api_key
        self.model, self.base_url, self.batch_size, self.dimensions = model, base_url, batch_size, dimensions
        self.api_key = resolve_api_key(api_key_env, base_url)   # never OPENAI_API_KEY for another host unless named
        self.name = f"openai-embed:{model}" + (f":{dimensions}" if dimensions else "")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from openai import OpenAI
        from .llm.backends import OPENAI_URL
        client = OpenAI(api_key=self.api_key, base_url=self.base_url or OPENAI_URL)   # not OPENAI_BASE_URL
        rows = []
        for i in range(0, len(texts), self.batch_size):
            kw = {"model": self.model, "input": [t or " " for t in texts[i:i + self.batch_size]]}
            if self.dimensions:
                kw["dimensions"] = self.dimensions
            rows += [d.embedding for d in client.embeddings.create(**kw).data]
        return l2_normalize(np.array(rows, dtype=np.float32))


class PrecomputedEmbedder:
    """Vectors supplied by the caller, in dataset row order (``encode`` ignores the texts but checks the count)."""

    def __init__(self, vectors: np.ndarray, name: str = "precomputed"):
        self.vectors = l2_normalize(vectors)
        self.name = name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) != len(self.vectors):
            raise ValueError(f"precomputed vectors have {len(self.vectors)} rows, got {len(texts)} texts")
        return self.vectors


class CachedEmbedder:
    """On-disk cache in front of any embedder (one ``.npy`` per (model, ordered texts) combination)."""

    def __init__(self, inner: Embedder, cache_dir: str | os.PathLike):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.name = inner.name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        h = hashlib.sha256(json.dumps([self.inner.name, list(texts)]).encode()).hexdigest()[:20]
        p = self.cache_dir / f"emb_{h}.npy"
        if p.exists():
            return np.load(p)
        V = self.inner.encode(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(p, V)
        return V
