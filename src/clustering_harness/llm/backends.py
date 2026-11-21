"""Provider adapters. Each adapter only performs one chat request; retries on contract failures, caching
and budgets are handled by :class:`~clustering_harness.llm.client.LLM`.

* :class:`OllamaBackend` — local models through the Ollama ``/api/chat`` endpoint.
* :class:`OpenAICompatibleBackend` — OpenAI, Azure OpenAI, and any OpenAI-compatible server (vLLM, LM Studio,
  TGI, Gemini's OpenAI-compatible endpoint, ...).
* :class:`AnthropicBackend` — Claude models through the official ``anthropic`` SDK.
* :class:`CallableBackend` — wrap any ``fn(ChatRequest) -> str`` (custom providers, tests).
* :class:`OfflineBackend` — raises on every call; use with a pre-filled cache to guarantee replay.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

from .base import BackendConfigError, BackendError, ChatRequest, ChatResult, RequestRejectedError

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

    def cache_settings(self) -> dict:
        """Options that change answers (non-default values only, so default keys stay stable)."""
        out = {}
        if self.num_ctx != 8192:
            out["num_ctx"] = self.num_ctx
        if self.think is not None:
            out["think"] = self.think
        return out

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
        except Exception as e:  # noqa: BLE001 - connection problems are retryable
            raise BackendError(f"ollama {self.model} at {self.url}: {str(e)[:200]}") from e
        if r.status_code in (400, 413, 422):
            raise RequestRejectedError(f"ollama {self.model}: HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code == 404:
            raise BackendConfigError(f"ollama: model {self.model!r} not found at {self.url} (run `ollama pull {self.model}`): "
                                     f"{r.text[:200]}")
        try:
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
    """OpenAI Chat Completions (``openai`` SDK). Works with OpenAI, Azure OpenAI and OpenAI-compatible servers.

    The key is taken from ``api_key``, else from ``api_key_provider`` (a callable or ``"module:function"`` returning
    the key, called once when the client is created), else from the ``api_key_env`` environment variable.
    For Azure OpenAI pass ``azure_endpoint`` and ``api_version``; ``model`` is then the *deployment* name.
    ``json_mode`` requests use ``response_format={"type": "json_object"}`` unless ``supports_json_mode=False``.
    """

    #: the SDK retries transient errors with backoff, so LLM does not add its own retries
    retries_internally = True

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 timeout: float = 120, max_retries: int = 5, supports_json_mode: bool = True, supports_seed: bool = True,
                 token_param: str = "max_completion_tokens", api_key_provider=None, azure_endpoint: str | None = None,
                 api_version: str | None = None, default_headers: dict | None = None):
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._api_key_provider = api_key_provider
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.default_headers = default_headers
        self.timeout = timeout
        self.max_retries = max_retries
        self.supports_json_mode = supports_json_mode
        self.supports_seed = supports_seed
        self.token_param = token_param
        if azure_endpoint:
            self.model_id = f"azure-openai:{azure_endpoint.rstrip('/')}:{model}"
        else:
            self.model_id = f"openai:{model}" if base_url is None else f"openai-compatible:{base_url}:{model}"
        self._client = None
        self._lock = threading.Lock()

    def cache_settings(self) -> dict:
        return {} if self.supports_json_mode else {"json_mode": False}

    def _key(self):
        if self._api_key:
            return self._api_key
        if self._api_key_provider is not None:
            from .bedrock import resolve_callable
            return resolve_callable(self._api_key_provider)()
        return os.environ.get(self._api_key_env)

    def _get(self):
        with self._lock:
            if self._client is None:
                kw = {"api_key": self._key(), "timeout": self.timeout, "max_retries": self.max_retries}
                if self.default_headers:
                    kw["default_headers"] = self.default_headers
                if self.azure_endpoint:
                    from openai import AzureOpenAI
                    self._client = AzureOpenAI(azure_endpoint=self.azure_endpoint, api_version=self.api_version, **kw)
                else:
                    from openai import OpenAI
                    self._client = OpenAI(base_url=self.base_url, **kw)
        return self._client

    def complete(self, request: ChatRequest) -> ChatResult:
        kw = {"model": self.model, "messages": request.messages, "temperature": request.temperature,
              self.token_param: request.max_tokens}
        if self.supports_seed:
            kw["seed"] = request.seed
        if request.json_mode and self.supports_json_mode:
            kw["response_format"] = {"type": "json_object"}
        try:
            client = self._get()
        except Exception as e:  # noqa: BLE001 - e.g. no API key
            raise BackendConfigError(f"openai {self.model}: {str(e)[:300]}") from e
        try:
            r = client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            # 401 invalid key, 403 no access, 404 unknown model/deployment: retrying cannot help
            if getattr(e, "status_code", None) in (401, 403, 404):
                raise BackendConfigError(f"openai {self.model}: {str(e)[:300]}") from e
            if getattr(e, "status_code", None) in (400, 413, 422):
                raise RequestRejectedError(f"openai {self.model}: {str(e)[:300]}") from e
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

    #: the SDK retries transient errors with backoff, so LLM does not add its own retries
    retries_internally = True

    def __init__(self, model: str = "claude-haiku-5-5", api_key: str | None = None, thinking: dict | None = None,
                 effort: str | None = "low", max_tokens_cap: int = 4000, min_max_tokens: int = 256, timeout: float = 120,
                 max_retries: int = 8, api_key_provider=None, api_key_env: str = "ANTHROPIC_API_KEY", base_url: str | None = None):
        self.model = model
        self._api_key = api_key
        self._api_key_provider = api_key_provider
        self._api_key_env = api_key_env
        self.base_url = base_url
        self.thinking = thinking if thinking is not None else {"type": "disabled"}
        self.effort = effort
        self.max_tokens_cap = max_tokens_cap
        self.min_max_tokens = min_max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_id = f"anthropic:{model}"
        self._client = None
        self._lock = threading.Lock()

    def cache_settings(self) -> dict:
        out = {}
        if self.thinking != {"type": "disabled"}:
            out["thinking"] = self.thinking
        if self.effort != "low":
            out["effort"] = self.effort
        if self.max_tokens_cap != 4000:
            out["max_tokens_cap"] = self.max_tokens_cap
        return out

    def _get(self):
        with self._lock:
            if self._client is None:
                import anthropic
                kw = {"max_retries": self.max_retries, "timeout": self.timeout,
                      # some environments ship broken brotli/zstd decoders; plain gzip is always safe
                      "default_headers": {"Accept-Encoding": "gzip"}}
                key = self._api_key
                if not key and self._api_key_provider is not None:
                    from .bedrock import resolve_callable
                    key = resolve_callable(self._api_key_provider)()
                key = key or os.environ.get(self._api_key_env)
                if key:
                    kw["api_key"] = key
                if self.base_url:
                    kw["base_url"] = self.base_url
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
            client = self._get()
        except Exception as e:  # noqa: BLE001
            raise BackendConfigError(f"anthropic {self.model}: {str(e)[:300]}") from e
        try:
            r = client.messages.create(**kw)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
            raise BackendConfigError(f"anthropic {self.model}: {str(e)[:300]}") from e
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", None) in (400, 413, 422):
                raise RequestRejectedError(f"anthropic {self.model}: {str(e)[:300]}") from e
            raise BackendError(f"anthropic {self.model}: {str(e)[:200]}") from e
        except anthropic.APIError as e:
            raise BackendError(f"anthropic {self.model}: {str(e)[:200]}") from e
        except TypeError as e:   # raised by the SDK when no API key / auth method can be resolved
            raise BackendConfigError(f"anthropic {self.model}: {str(e)[:300]} (set ANTHROPIC_API_KEY)") from e
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
