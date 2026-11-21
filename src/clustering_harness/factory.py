"""Build models and embedders from plain dictionaries (YAML run specs, CLI)."""
from __future__ import annotations

from pathlib import Path

from .embeddings import (CachedEmbedder, HFTransformerEmbedder, OllamaEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder)
from .llm import (LLM, AnthropicBackend, Ledger, OfflineBackend, OllamaBackend, OpenAICompatibleBackend, Price, ResponseCache)

__all__ = ["build_llm", "build_embedder", "build_backend", "register_backend", "backend_kinds"]

_GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _ollama(s):
    return OllamaBackend(s.pop("model"), url=s.pop("url", "http://localhost:11434"), num_ctx=s.pop("num_ctx", 8192),
                         think=s.pop("think", None), timeout=s.pop("timeout", 600), keep_alive=s.pop("keep_alive", None))


def _openai_like(kind):
    def make(s):
        if kind == "gemini":
            s.setdefault("base_url", _GEMINI_OPENAI_URL)
            s.setdefault("api_key_env", "GEMINI_API_KEY")
            s.setdefault("seed", False)
        if kind == "azure_openai":
            s.setdefault("api_key_env", "AZURE_OPENAI_API_KEY")
            if "deployment" in s:
                s["model"] = s.pop("deployment")
        return OpenAICompatibleBackend(
            s.pop("model"), base_url=s.pop("base_url", None), api_key_env=s.pop("api_key_env", "OPENAI_API_KEY"),
            api_key_provider=s.pop("api_key_provider", None), azure_endpoint=s.pop("azure_endpoint", None),
            api_version=s.pop("api_version", None), default_headers=s.pop("default_headers", None),
            supports_json_mode=s.pop("json_mode", True), supports_seed=s.pop("seed", True), timeout=s.pop("timeout", 120),
            max_retries=s.pop("max_retries", 5),
            token_param=s.pop("token_param", "max_completion_tokens" if kind in ("openai", "azure_openai") else "max_tokens"))
    return make


def _anthropic(s):
    return AnthropicBackend(s.pop("model", "claude-haiku-5-5"), effort=s.pop("effort", "low"), thinking=s.pop("thinking", None),
                            max_tokens_cap=s.pop("max_tokens_cap", 4000), api_key_provider=s.pop("api_key_provider", None),
                            api_key_env=s.pop("api_key_env", "ANTHROPIC_API_KEY"), base_url=s.pop("base_url", None),
                            timeout=s.pop("timeout", 120), max_retries=s.pop("max_retries", 8))


def _bedrock(s):
    from .llm.bedrock import BedrockBackend
    keys = ("region", "credentials", "credentials_provider", "aws_access_key_id", "aws_secret_access_key", "aws_session_token",
            "profile_name", "role_arn", "external_id", "role_session_name", "role_duration_seconds", "bearer_token",
            "bearer_token_env", "max_tokens_cap", "send_temperature", "additional_model_request_fields", "timeout",
            "max_retries", "endpoint_url")
    return BedrockBackend(s.pop("model"), **{k: s.pop(k) for k in keys if k in s})


def _offline(s):
    return OfflineBackend(s.pop("model_id"))


_BACKENDS: dict = {"ollama": _ollama, "openai": _openai_like("openai"), "openai_compatible": _openai_like("openai_compatible"),
                   "azure_openai": _openai_like("azure_openai"), "gemini": _openai_like("gemini"), "anthropic": _anthropic,
                   "bedrock": _bedrock, "offline": _offline}


def register_backend(kind: str, factory) -> None:
    """Make ``backend: <kind>`` available in run specs. ``factory(options: dict) -> Backend`` must pop every option
    it uses from the dict (leftover options are reported as errors)."""
    _BACKENDS[kind] = factory


def backend_kinds() -> list[str]:
    _load_entry_points()
    return sorted(_BACKENDS)


_EP_LOADED = False


def _load_entry_points():
    """Backends published by installed packages under the ``clustering_harness.backends`` entry-point group."""
    global _EP_LOADED
    if _EP_LOADED:
        return
    _EP_LOADED = True
    from importlib.metadata import entry_points
    eps = entry_points()
    group = (eps.select(group="clustering_harness.backends") if hasattr(eps, "select")      # Python 3.10+
             else eps.get("clustering_harness.backends", []))                              # Python 3.9
    for ep in group:
        _BACKENDS.setdefault(ep.name, ep.load())


def build_backend(spec: dict):
    """A backend from a spec dict. ``backend`` is a built-in kind (see :func:`backend_kinds`), a kind added with
    :func:`register_backend` or an entry point, or an import path ``"package.module:attr"`` naming a Backend class
    or factory function that is called with the remaining options as keyword arguments."""
    s = dict(spec)
    kind = s.pop("backend")
    _load_entry_points()
    if kind in _BACKENDS:
        be = _BACKENDS[kind](s)
        if s:
            raise ValueError(f"unknown options for backend {kind!r}: {sorted(s)}")
    elif ":" in str(kind):
        from .llm.bedrock import resolve_callable
        be = resolve_callable(kind)(**s)
    else:
        raise ValueError(f"unknown backend {kind!r}; built-in: {sorted(_BACKENDS)}; or use 'package.module:Factory'")
    if not hasattr(be, "complete") or not getattr(be, "model_id", None):
        raise TypeError(f"backend {kind!r} must provide .complete(request) and a non-empty .model_id")
    return be


def build_llm(spec: dict, cache: ResponseCache | None = None, ledger: Ledger | None = None, name: str | None = None) -> LLM:
    """``spec`` keys: ``backend`` (see :func:`build_backend`), ``model``, backend options, and wrapper options
    (``reasoning``, ``concurrency``, ``budget_usd``, ``price_in``, ``price_out``, ``cache_contract_failures``,
    ``legacy_cache_keys``, ``max_consecutive_failures``, ``budget_scope``)."""
    s = dict(spec)
    reasoning = bool(s.pop("reasoning", False))
    concurrency = int(s.pop("concurrency", 2))
    budget = s.pop("budget_usd", None)
    pin, pout = s.pop("price_in", None), s.pop("price_out", None)
    if (pin is None) != (pout is None):
        raise ValueError("set both price_in and price_out (USD per million tokens), or neither")
    extra = {k: s.pop(k) for k in ("cache_contract_failures", "legacy_cache_keys", "max_consecutive_failures", "budget_scope")
             if k in s}
    if "backend" not in s:
        raise ValueError(f"model {name or ''} needs `backend:` (one of {', '.join(backend_kinds())}, or 'package.module:Factory')")
    be = build_backend(s)
    price = Price(pin, pout) if pin is not None else None
    return LLM(be, name=name, cache=cache, max_concurrency=concurrency, reasoning=reasoning, budget_usd=budget, ledger=ledger,
               price=price, **extra)


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
