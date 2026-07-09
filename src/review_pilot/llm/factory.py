from __future__ import annotations

from review_pilot.config import LLMConfig

from .base import LLMConfigurationError, LLMProvider
from .fake_provider import FakeProvider
from .openai_compatible import OpenAICompatibleProvider


def supported_providers() -> tuple[str, ...]:
    return ("fake", "openai-compatible")


def create_provider(name: str, config: LLMConfig | None = None) -> LLMProvider:
    normalized = name.strip().lower()
    if normalized == "fake":
        return FakeProvider()
    if normalized == "openai-compatible":
        return OpenAICompatibleProvider(
            config or LLMConfig.from_env("openai-compatible")
        )
    raise LLMConfigurationError(
        f"unsupported provider: {name}; expected one of {supported_providers()}"
    )
