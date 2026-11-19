"""Backend protocol: a backend turns one chat request into text. Everything else (JSON contract, retries,
caching, budgets, concurrency) lives in :class:`clustering_harness.llm.client.LLM` so that every provider
behaves the same way inside the algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["ChatRequest", "ChatResult", "Backend", "BackendError"]


class BackendError(RuntimeError):
    """A backend could not produce a response (after its own transport retries)."""


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
