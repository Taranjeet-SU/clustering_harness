"""Build models and embedders from plain dictionaries (YAML run specs, CLI)."""
from __future__ import annotations

from pathlib import Path

from .embeddings import (CachedEmbedder, HFTransformerEmbedder, OllamaEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder)
from .llm import (LLM, AnthropicBackend, Ledger, OfflineBackend, OllamaBackend, OpenAICompatibleBackend, Price, ResponseCache)

__all__ = ["build_llm", "build_embedder"]


def build_llm(spec: dict, cache: ResponseCache | None = None, ledger: Ledger | None = None, name: str | None = None) -> LLM:
    """``spec`` keys: ``backend`` (ollama | openai | openai_compatible | anthropic | offline), ``model``, backend options
    (``url``, ``num_ctx``, ``think``, ``base_url``, ``api_key_env``, ``effort``), and wrapper options
    (``reasoning``, ``concurrency``, ``budget_usd``, ``price_in``, ``price_out``)."""
    s = dict(spec)
    kind = s.pop("backend")
    reasoning = bool(s.pop("reasoning", False))
    concurrency = int(s.pop("concurrency", 2))
    budget = s.pop("budget_usd", None)
    pin, pout = s.pop("price_in", None), s.pop("price_out", None)
    if kind == "ollama":
        be = OllamaBackend(s.pop("model"), url=s.pop("url", "http://localhost:11434"), num_ctx=s.pop("num_ctx", 8192),
                           think=s.pop("think", None), timeout=s.pop("timeout", 600))
    elif kind in ("openai", "openai_compatible"):
        be = OpenAICompatibleBackend(s.pop("model"), base_url=s.pop("base_url", None), api_key_env=s.pop("api_key_env", "OPENAI_API_KEY"),
                                     supports_json_mode=s.pop("json_mode", True), supports_seed=s.pop("seed", True),
                                     token_param=s.pop("token_param", "max_completion_tokens" if kind == "openai" else "max_tokens"))
    elif kind == "anthropic":
        be = AnthropicBackend(s.pop("model", "claude-haiku-5-5"), effort=s.pop("effort", "low"),
                              thinking=s.pop("thinking", None), max_tokens_cap=s.pop("max_tokens_cap", 4000))
    elif kind == "offline":
        be = OfflineBackend(s.pop("model_id"))
    else:
        raise ValueError(f"unknown backend {kind!r}")
    if s:
        raise ValueError(f"unknown options for {kind}: {sorted(s)}")
    price = Price(pin, pout) if pin is not None and pout is not None else None
    return LLM(be, name=name, cache=cache, max_concurrency=concurrency, reasoning=reasoning, budget_usd=budget, ledger=ledger, price=price)


def build_embedder(spec: dict | None, cache_dir: str | Path | None = None):
    if not spec:
        return None
    s = dict(spec)
    kind = s.pop("backend")
    if kind in ("hf", "transformers"):
        e = HFTransformerEmbedder(**s)
    elif kind in ("sentence_transformers", "st"):
        e = SentenceTransformerEmbedder(**s)
    elif kind == "ollama":
        e = OllamaEmbedder(**s)
    elif kind == "openai":
        e = OpenAIEmbedder(**s)
    else:
        raise ValueError(f"unknown embedder backend {kind!r}")
    return CachedEmbedder(e, cache_dir) if cache_dir else e
