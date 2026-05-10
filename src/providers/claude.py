from collections.abc import Generator

import anthropic as _sdk

from src.providers.base import AuthError, NetworkError, ProviderError, RateLimitError

_DEFAULT_MODEL = "claude-sonnet-4-6"


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
        try:
            yield from self._do_stream(system, messages, max_tokens)
        except _sdk.AuthenticationError as exc:
            raise AuthError(str(exc)) from exc
        except _sdk.APIConnectionError as exc:
            raise NetworkError(str(exc)) from exc
        except _sdk.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except _sdk.APIError as exc:
            raise ProviderError(str(exc)) from exc

    def _do_stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> Generator[str, None, None]:
        with self._api.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            yield from stream.text_stream
