from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from review_pilot.context_pack import ReviewContextPack
from review_pilot.evidence_guard import (
    EvidenceGuardResult,
    guard_llm_findings,
)

from .base import LLMProvider, LLMResponse
from .schema import LLMFindingsEnvelope, parse_llm_findings


@dataclass(frozen=True)
class StructuredReviewResult:
    response: LLMResponse
    envelope: LLMFindingsEnvelope
    evidence: EvidenceGuardResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.response.provider,
            "model": self.response.model,
            "schema_version": self.envelope.schema_version,
            **self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class StructuredReviewer:
    provider: LLMProvider
    snippet_location: bool = False

    def review(
        self,
        context_pack: ReviewContextPack,
        *,
        snippet_location: bool | None = None,
    ) -> StructuredReviewResult:
        enabled = self.snippet_location if snippet_location is None else snippet_location
        if enabled:
            specialized = getattr(self.provider, "review_with_snippet_location", None)
            response = (
                specialized(context_pack)
                if callable(specialized)
                else self.provider.review(context_pack)
            )
        else:
            response = self.provider.review(context_pack)
        return self.parse_response(
            response,
            context_pack,
            snippet_location=enabled,
        )

    def review_with_tools(
        self,
        context_pack: ReviewContextPack,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return self.provider.review_with_tools(
            context_pack,
            messages,
            tools,
            max_tokens=max_tokens,
        )

    @staticmethod
    def parse_response(
        response: LLMResponse,
        context_pack: ReviewContextPack,
        *,
        snippet_location: bool = False,
    ) -> StructuredReviewResult:
        if response.tool_calls:
            raise ValueError("tool-call response must be resolved before parsing findings")
        envelope = parse_llm_findings(
            response.content,
            require_existing_code=snippet_location,
        )
        evidence = guard_llm_findings(
            envelope.findings,
            context_pack,
            snippet_location=snippet_location,
        )
        return StructuredReviewResult(
            response=response,
            envelope=envelope,
            evidence=evidence,
        )
