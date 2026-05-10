import os
import time
from collections.abc import Generator

from src.providers.base import AuthError, NetworkError, ProviderError, RateLimitError

_DEFAULT_MODEL = "gemini-2.5-flash"
_RETRY_DELAYS  = [5, 15, 30]  # seconds between retries on rate limit


class GeminiProvider:
    """
    Google Gemini provider via google-generativeai.
    Free tier available at aistudio.google.com — set GEMINI_API_KEY in .env.
    """

    name = "gemini"

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise AuthError(
                "GEMINI_API_KEY no configurada. "
                "Obtén una gratis en aistudio.google.com y agrégala al .env"
            )

    def stream(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1200,
    ) -> Generator[str, None, None]:
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
                    continue  # retry
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
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderError(
                "Instala google-generativeai: pip install google-generativeai"
            ) from exc

        try:
            genai.configure(api_key=self._api_key)

            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.3,
                ),
            )

            gemini_messages = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})

            response = model.generate_content(gemini_messages, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except AuthError:
            raise
        except RateLimitError:
            raise
        except Exception as exc:
            err = str(exc).lower()
            if "api key" in err or "permission" in err or "unauthorized" in err:
                raise AuthError(f"Clave Gemini invalida: {exc}") from exc
            if "quota" in err or "rate" in err or "429" in err or "resource_exhausted" in err:
                raise RateLimitError(f"Limite de tasa Gemini: {exc}") from exc
            if "connect" in err or "timeout" in err or "network" in err:
                raise NetworkError(f"Error de red Gemini: {exc}") from exc
            raise ProviderError(str(exc)) from exc
