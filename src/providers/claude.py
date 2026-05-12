import time
from collections.abc import Generator

import anthropic as _sdk

from src.providers.base import AuthError, NetworkError, ProviderError, RateLimitError

_DEFAULT_MODEL = "claude-sonnet-4-6"
_RETRY_DELAYS  = [8]  # seconds between retries on rate limit


class ClaudeProvider:
    """Anthropic Claude provider — streams text via the Messages API."""

    name = "claude"

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self._client: _sdk.Anthropic | None = None

    @property
    def _api(self) -> _sdk.Anthropic:
        if self._client is None:
            self._client = _sdk.Anthropic()
        return self._client

    def stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1200,
    ) -> Generator[str, None, None]:
        """Yield response text chunks, translating SDK errors to ProviderError."""
        last_exc = None
        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                yield from self._do_stream(system, messages, max_tokens)
                return
            except RateLimitError as exc:
                last_exc = exc
                if attempt < len(_RETRY_DELAYS):
                    continue
                break
            except Exception:
                raise
        raise last_exc  # type: ignore[misc]

    def _do_stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> Generator[str, None, None]:
        try:
            with self._api.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            ) as stream:
                yield from stream.text_stream
        except _sdk.AuthenticationError as exc:
            raise AuthError(str(exc)) from exc
        except _sdk.APIConnectionError as exc:
            raise NetworkError(str(exc)) from exc
        except _sdk.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except _sdk.APIError as exc:
            raise ProviderError(str(exc)) from exc
