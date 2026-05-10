from collections.abc import Generator
from typing import Protocol, runtime_checkable


# ── Error hierarchy ────────────────────────────────────────────────────────────

class ProviderError(Exception):
    """Base exception for all LLM provider errors."""

class AuthError(ProviderError):
    """Invalid or missing API credentials."""

class NetworkError(ProviderError):
    """Provider endpoint unreachable or timed out."""

class RateLimitError(ProviderError):
    """Request rate limit exceeded."""


# ── Provider protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """
    Minimal interface every LLM provider must satisfy.

    Implementing this protocol requires:
      - `name`  : short identifier string (e.g. "claude", "gemini")
      - `model` : full model ID string
      - `stream`: yields text chunks given a system prompt and message list
    """

    name:  str
    model: str

    def stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> Generator[str, None, None]:
        ...
