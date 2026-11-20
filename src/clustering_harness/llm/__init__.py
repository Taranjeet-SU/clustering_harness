"""Pluggable LLM layer: provider backends, a uniform JSON contract, response cache and spend caps."""
from .backends import (AnthropicBackend, CallableBackend, OfflineBackend, OfflineCallError, OllamaBackend,
                       OpenAICompatibleBackend)
from .bedrock import BedrockBackend
from .base import Backend, BackendError, ChatRequest, ChatResult
from .budget import PRICES_PER_MTOK, Budget, BudgetExceeded, Ledger, Price, price_for
from .cache import ResponseCache, request_key
from .client import LLM, extract_json, run_parallel

__all__ = ["LLM", "run_parallel", "extract_json", "Backend", "BackendError", "ChatRequest", "ChatResult",
           "OllamaBackend", "OpenAICompatibleBackend", "AnthropicBackend", "BedrockBackend", "CallableBackend", "OfflineBackend",
           "OfflineCallError", "ResponseCache", "request_key", "Ledger", "Budget", "BudgetExceeded", "Price",
           "PRICES_PER_MTOK", "price_for"]
