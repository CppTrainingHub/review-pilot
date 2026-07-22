from __future__ import annotations

import json
from http.client import IncompleteRead
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from review_pilot.config import LLMConfig
from review_pilot.context_pack import ReviewContextPack
from review_pilot.evidence_guard import EvidenceGuardResult
from review_pilot.reflection import build_reflection_prompt
from review_pilot.report_models import Finding

from .base import (
    LLMConfigurationError,
    LLMRequestError,
    LLMResponse,
    LLMToolCall,
)
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
        prompt = build_review_prompt(context_pack)
        return self._request(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
        )

    def review_with_snippet_location(
        self,
        context_pack: ReviewContextPack,
    ) -> LLMResponse:
        prompt = build_review_prompt(context_pack, snippet_location=True)
        return self._request(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
        )

    def review_with_tools(
        self,
        context_pack: ReviewContextPack,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del context_pack
        return self._request(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            response_format=(
                {"type": "json_object"}
                if max_tokens is not None and not tools
                else None
            ),
        )

    def reflect_finding(
        self,
        *,
        finding: Finding,
        context_pack: ReviewContextPack,
        evidence: EvidenceGuardResult,
    ) -> LLMResponse:
        system, user = build_reflection_prompt(finding, context_pack, evidence)
        return self._request(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )

    def _request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.config.api_key:
            raise LLMConfigurationError(
                "missing API key: set REVIEW_PILOT_API_KEY or OPENAI_API_KEY"
            )

        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
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
        except (IncompleteRead, TimeoutError, OSError) as exc:
            raise LLMRequestError(f"llm request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMRequestError("llm response was not valid JSON") from exc

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(
                "llm response did not contain a message or message content"
            ) from exc
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise LLMRequestError("llm response message content must be a string or null")
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        if not content.strip() and not tool_calls:
            raise LLMRequestError("llm response did not contain message content or tool calls")
        return LLMResponse(
            provider=self.name,
            model=self.config.model,
            content=content.strip(),
            tool_calls=tuple(tool_calls),
            finish_reason=choice.get("finish_reason"),
            usage=_parse_usage(body.get("usage")),
        )


def _parse_tool_calls(raw: Any) -> list[LLMToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LLMRequestError("llm response tool_calls must be an array")
    calls: list[LLMToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LLMRequestError(f"llm response tool_calls[{index}] must be an object")
        function = item.get("function")
        if not isinstance(function, dict):
            raise LLMRequestError(f"llm response tool_calls[{index}] missing function")
        call_id = item.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if not all(isinstance(value, str) and value.strip() for value in (call_id, name, arguments)):
            raise LLMRequestError(
                f"llm response tool_calls[{index}] has invalid id, name, or arguments"
            )
        try:
            calls.append(
                LLMToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    type=str(item.get("type", "function")),
                )
            )
        except ValueError as exc:
            raise LLMRequestError(str(exc)) from exc
    return calls


def _parse_usage(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            usage[key] = value
    return usage or None
