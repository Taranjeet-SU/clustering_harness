"""Backend protocol: a backend turns one chat request into text. Everything else (JSON contract, retries,
caching, budgets, concurrency) lives in :class:`clustering_harness.llm.client.LLM` so that every provider
behaves the same way inside the algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["ChatRequest", "ChatResult", "Backend", "BackendError", "BackendConfigError", "RequestRejectedError",
           "ModelUnavailableError"]


class BackendError(RuntimeError):
    """A backend could not produce a response (after its own transport retries)."""


class BackendConfigError(BackendError):
    """A failure that retrying cannot fix: missing or invalid credentials, no permission, unknown model,
    missing region. Backends raise it so that a run stops at once with the provider's message."""


class RequestRejectedError(BackendError):
    """The provider refused *this request* and would refuse it again (HTTP 400/413/422: input too long, content
    filter, invalid request). It is not retried and does not block the run: the algorithm gets an error answer for
    that one request, it is counted (``LLM.stats["rejected"]``) and reported. Many in a row still stop the model."""


class ModelUnavailableError(BackendError):
    """Raised by :class:`~clustering_harness.llm.client.LLM` when a model cannot be used: a configuration error,
    or too many consecutive failed requests. The pipeline stops; nothing about the failed requests is cached."""


@dataclass(frozen=True)
class ChatRequest:
    messages: list            # [{"role": "system"|"user"|"assistant", "content": str}, ...]
    max_tokens: int = 800     # generation budget (num_predict / max_tokens)
    temperature: float = 0.0
    seed: int = 0
    json_mode: bool = False   # ask the provider for a JSON object if it supports it


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Minimal provider interface. Implementations must be thread-safe."""

    #: stable identity used in cache keys, e.g. ``"ollama:qwen3:8b"`` or ``"openai:gpt-4o-mini"``
    model_id: str

    def complete(self, request: ChatRequest) -> ChatResult:  # pragma: no cover - protocol
        ...
