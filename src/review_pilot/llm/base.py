from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from review_pilot.context_pack import ReviewContextPack


class LLMProviderError(RuntimeError):
    """Base error for provider configuration and request failures."""


class LLMConfigurationError(LLMProviderError):
    """Raised before a request when provider configuration is incomplete."""


class LLMRequestError(LLMProviderError):
    """Raised when a provider request or response cannot be completed."""


@dataclass(frozen=True)
class LLMToolCall:
    call_id: str
    name: str
    arguments: str
    type: str = "function"

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("tool call id must be a non-empty string")
        if not self.name.strip():
            raise ValueError("tool call name must be a non-empty string")
        if self.type != "function":
            raise ValueError("only function tool calls are supported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    content: str
    tool_calls: tuple[LLMToolCall, ...] = ()
    finish_reason: str | None = None
    usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not self.content.strip() and not self.tool_calls:
            raise ValueError("content must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
        }
        if self.usage is not None:
            payload["usage"] = dict(self.usage)
        return payload


class LLMProvider(Protocol):
    name: str
    model: str

    def review(self, context_pack: ReviewContextPack) -> LLMResponse:
        """Review an auditable context pack and return provider-neutral text."""
        ...

    def review_with_tools(
        self,
        context_pack: ReviewContextPack,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Continue a review conversation with a bounded tool list."""
        ...

    def reflect_finding(self, *, finding: Any, context_pack: ReviewContextPack, evidence: Any) -> LLMResponse:
        """Review one existing finding without creating a new finding."""
        ...
