from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from review_pilot.context_pack import ReviewContextPack


class LLMProviderError(RuntimeError):
    """Base error for provider configuration and request failures."""


class LLMConfigurationError(LLMProviderError):
    """Raised before a request when provider configuration is incomplete."""


class LLMRequestError(LLMProviderError):
    """Raised when a provider request or response cannot be completed."""


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    content: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not self.content.strip():
            raise ValueError("content must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "content": self.content,
        }


class LLMProvider(Protocol):
    name: str
    model: str

    def review(self, context_pack: ReviewContextPack) -> LLMResponse:
        """Review an auditable context pack and return provider-neutral text."""
        ...
