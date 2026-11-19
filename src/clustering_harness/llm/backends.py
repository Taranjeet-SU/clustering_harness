"""Provider adapters. Each adapter only performs one chat request; retries on contract failures, caching
and budgets are handled by :class:`~clustering_harness.llm.client.LLM`.

* :class:`OllamaBackend` — local models through the Ollama ``/api/chat`` endpoint.
* :class:`OpenAICompatibleBackend` — OpenAI and any OpenAI-compatible server (vLLM, LM Studio, TGI, ...).
* :class:`AnthropicBackend` — Claude models through the official ``anthropic`` SDK.
* :class:`CallableBackend` — wrap any ``fn(ChatRequest) -> str`` (custom providers, tests).
* :class:`OfflineBackend` — raises on every call; use with a pre-filled cache to guarantee replay.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

from .base import BackendError, ChatRequest, ChatResult

__all__ = ["OllamaBackend", "OpenAICompatibleBackend", "AnthropicBackend", "CallableBackend", "OfflineBackend",
           "OfflineCallError"]


class OllamaBackend:
    """Ollama ``/api/chat``.

    ``num_ctx`` is pinned per backend so that requests never trigger a model reload. ``think`` is passed
    through for reasoning models (``"low"``/``"medium"``/``"high"``, ``False`` to disable, ``None`` to omit).
    """

    def __init__(self, model: str, url: str = "http://localhost:11434", num_ctx: int = 8192, think=None,
                 timeout: float = 600, keep_alive: str | None = None):
        self.model = model
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.model_id = f"ollama:{model}"
        self._session = None
        self._lock = threading.Lock()

    def _http(self):
        import requests
        with self._lock:
            if self._session is None:
                self._session = requests.Session()
        return self._session

    def complete(self, request: ChatRequest) -> ChatResult:
        body = {"model": self.model, "messages": request.messages, "stream": False,
                "options": {"temperature": request.temperature, "num_predict": request.max_tokens,
                            "num_ctx": self.num_ctx, "seed": request.seed}}
        if self.think is not None:
            body["think"] = self.think
        if request.json_mode:
            body["format"] = "json"
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        try:
            r = self._http().post(self.url + "/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            j = r.json()
        except Exception as e:  # noqa: BLE001
            raise BackendError(f"ollama {self.model}: {str(e)[:200]}") from e
        return ChatResult(text=j.get("message", {}).get("content", "") or "", input_tokens=int(j.get("prompt_eval_count") or 0),
                          output_tokens=int(j.get("eval_count") or 0))

    def available(self) -> bool:
        try:
            tags = self._http().get(self.url + "/api/tags", timeout=10).json()
            return any(m["name"] == self.model for m in tags.get("models", []))
        except Exception:  # noqa: BLE001
            return False


class OpenAICompatibleBackend:
    """OpenAI Chat Completions (``openai`` SDK). Works with OpenAI and OpenAI-compatible servers.

    ``api_key`` defaults to the ``api_key_env`` environment variable. ``json_mode`` requests use
    ``response_format={"type": "json_object"}`` unless ``supports_json_mode=False``.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 timeout: float = 120, max_retries: int = 5, supports_json_mode: bool = True, supports_seed: bool = True,
                 token_param: str = "max_completion_tokens"):
        self.model = model
        self.base_url = base_url
        self._api_key = api_key or os.environ.get(api_key_env)
        self.timeout = timeout
        self.max_retries = max_retries
        self.supports_json_mode = supports_json_mode
        self.supports_seed = supports_seed
        self.token_param = token_param
        self.model_id = f"openai:{model}" if base_url is None else f"openai-compatible:{base_url}:{model}"
        self._client = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key, base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def complete(self, request: ChatRequest) -> ChatResult:
        kw = {"model": self.model, "messages": request.messages, "temperature": request.temperature,
              self.token_param: request.max_tokens}
        if self.supports_seed:
            kw["seed"] = request.seed
        if request.json_mode and self.supports_json_mode:
            kw["response_format"] = {"type": "json_object"}
        try:
            r = self._get().chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            raise BackendError(f"openai {self.model}: {str(e)[:200]}") from e
        u = r.usage
        return ChatResult(text=r.choices[0].message.content or "", input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                          output_tokens=getattr(u, "completion_tokens", 0) or 0)


class AnthropicBackend:
    """Claude through the ``anthropic`` SDK.

    System messages are passed as the ``system`` parameter. Sampling parameters (temperature, seed) are not
    sent: current Claude models reject non-default sampling values, so runs are not seed-reproducible on
    this backend (cached answers still make reruns reproducible). ``thinking``/``effort`` default to a
    low-latency configuration suitable for short classification and judging calls.
    """

    def __init__(self, model: str = "claude-haiku-5-5", api_key: str | None = None, thinking: dict | None = None,
                 effort: str | None = "low", max_tokens_cap: int = 4000, min_max_tokens: int = 256, timeout: float = 120,
                 max_retries: int = 8):
        self.model = model
        self._api_key = api_key
        self.thinking = thinking if thinking is not None else {"type": "disabled"}
        self.effort = effort
        self.max_tokens_cap = max_tokens_cap
        self.min_max_tokens = min_max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_id = f"anthropic:{model}"
        self._client = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._client is None:
                import anthropic
                kw = {"max_retries": self.max_retries, "timeout": self.timeout,
                      # some environments ship broken brotli/zstd decoders; plain gzip is always safe
                      "default_headers": {"Accept-Encoding": "gzip"}}
                if self._api_key:
                    kw["api_key"] = self._api_key
                self._client = anthropic.Anthropic(**kw)
        return self._client

    def complete(self, request: ChatRequest) -> ChatResult:
        import anthropic
        system = "\n\n".join(m["content"] for m in request.messages if m["role"] == "system")
        msgs = [{"role": m["role"], "content": m["content"]} for m in request.messages if m["role"] != "system"]
        kw = {"model": self.model, "max_tokens": max(self.min_max_tokens, min(int(request.max_tokens), self.max_tokens_cap)),
              "messages": msgs}
        if system:
            kw["system"] = system
        if self.thinking:
            kw["thinking"] = self.thinking
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        try:
            r = self._get().messages.create(**kw)
        except anthropic.APIError as e:
            raise BackendError(f"anthropic {self.model}: {str(e)[:200]}") from e
        text = "" if r.stop_reason == "refusal" else "".join(b.text for b in r.content if b.type == "text")
        return ChatResult(text=text, input_tokens=r.usage.input_tokens or 0, output_tokens=r.usage.output_tokens or 0,
                          raw={"stop_reason": r.stop_reason})


class CallableBackend:
    """Adapter for a plain function ``fn(request) -> str`` (or ``ChatResult``)."""

    def __init__(self, fn: Callable[[ChatRequest], str | ChatResult], model_id: str = "callable"):
        self.fn = fn
        self.model_id = model_id
        self.model = model_id

    def complete(self, request: ChatRequest) -> ChatResult:
        out = self.fn(request)
        return out if isinstance(out, ChatResult) else ChatResult(text=str(out))


class OfflineCallError(BackendError):
    pass


class OfflineBackend:
    """Never calls a model. Used to replay a run strictly from a cache: any cache miss raises
    :class:`OfflineCallError` carrying the request, which pinpoints prompts that differ from the reference."""

    def __init__(self, model_id: str, on_miss: Callable[[ChatRequest], None] | None = None):
        self.model_id = model_id
        self.model = model_id
        self.on_miss = on_miss
        self.misses: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResult:
        self.misses.append(request)
        if self.on_miss:
            self.on_miss(request)
        raise OfflineCallError(f"offline backend {self.model_id}: cache miss")
