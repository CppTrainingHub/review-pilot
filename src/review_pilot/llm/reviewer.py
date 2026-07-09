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

    def review(
        self,
        context_pack: ReviewContextPack,
    ) -> StructuredReviewResult:
        response = self.provider.review(context_pack)
        envelope = parse_llm_findings(response.content)
        evidence = guard_llm_findings(
            envelope.findings,
            context_pack,
        )
        return StructuredReviewResult(
            response=response,
            envelope=envelope,
            evidence=evidence,
        )
