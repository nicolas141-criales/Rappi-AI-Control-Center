import os

from src.providers.base import LLMProvider

_REGISTRY: dict[str, type] = {}


def _register():
    from src.providers.claude import ClaudeProvider
    from src.providers.gemini import GeminiProvider
    _REGISTRY["claude"] = ClaudeProvider
    _REGISTRY["gemini"] = GeminiProvider


def get_provider(name: str | None = None) -> LLMProvider:
    """
    Return an LLMProvider instance for the requested backend.

    Resolution order:
      1. `name` argument
      2. LLM_PROVIDER environment variable
      3. "claude" (default)
    """
    if not _REGISTRY:
        _register()

    key = (name or os.environ.get("LLM_PROVIDER", "claude")).lower().strip()
    cls = _REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown provider {key!r}. Available: {available}")
    return cls()
