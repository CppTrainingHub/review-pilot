from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from review_pilot.config import LLMConfig
from review_pilot.context_pack import ReviewContextPack

from .base import LLMConfigurationError, LLMRequestError, LLMResponse
from .prompt_builder import build_review_prompt


UrlOpen = Callable[..., Any]


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    config: LLMConfig
    urlopen: UrlOpen = urllib.request.urlopen
    name: str = "openai-compatible"

    @property
    def model(self) -> str:
        return self.config.model

    def review(self, context_pack: ReviewContextPack) -> LLMResponse:
        if not self.config.api_key:
            raise LLMConfigurationError(
                "missing API key: set REVIEW_PILOT_API_KEY or OPENAI_API_KEY"
            )

        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        prompt = build_review_prompt(context_pack)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt.system,
                },
                {
                    "role": "user",
                    "content": prompt.user,
                },
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with self.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMRequestError(f"llm request failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise LLMRequestError(f"llm request failed: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise LLMRequestError(f"llm request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMRequestError("llm response was not valid JSON") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("llm response did not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("llm response did not contain message content")
        return LLMResponse(
            provider=self.name,
            model=self.config.model,
            content=content.strip(),
        )
