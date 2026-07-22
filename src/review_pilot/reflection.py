from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import ReflectionConfig
from .context_pack import ReviewContextPack
from .evidence_guard import EvidenceGuardResult
from .report_models import Finding, SEVERITY_RANK


REFLECTION_SCHEMA_VERSION = "review-pilot.reflection.v1"
REFLECTION_ACTIONS = {"keep", "downgrade", "drop"}
REFLECTION_ROOT_FIELDS = {"schema_version", "decision", "reason", "severity"}


class ReflectionOutputError(ValueError):
    """Raised when a reflection response is outside the decision contract."""


@dataclass(frozen=True)
class ParsedReflectionDecision:
    action: str
    reason: str
    severity: str | None


@dataclass(frozen=True)
class ReflectionDecision:
    finding_index: int
    action: str
    reason: str
    original_severity: str
    final_severity: str | None = None
    error: str | None = None
    kept_original: bool = False

    def __post_init__(self) -> None:
        if self.action not in {*REFLECTION_ACTIONS, "skip", "error"}:
            raise ValueError(f"invalid reflection action: {self.action!r}")
        if not self.reason.strip():
            raise ValueError("reflection reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_index": self.finding_index,
            "action": self.action,
            "reason": self.reason,
            "original_severity": self.original_severity,
            "final_severity": self.final_severity,
            "error": self.error,
            "reflection_error": self.error,
            "kept_original": self.kept_original,
        }


@dataclass(frozen=True)
class ReflectionResult:
    findings: tuple[Finding, ...]
    decisions: tuple[ReflectionDecision, ...]
    summary: dict[str, Any]


class ReflectionFilter:
    """Apply a second review pass to eligible LLM findings."""

    def __init__(self, provider: Any, config: ReflectionConfig) -> None:
        self.provider = provider
        self.config = config

    def apply(
        self,
        findings: list[Finding] | tuple[Finding, ...],
        *,
        context_pack: ReviewContextPack,
        evidence: EvidenceGuardResult,
    ) -> ReflectionResult:
        output: list[Finding] = []
        decisions: list[ReflectionDecision] = []
        eligible_count = 0
        reviewed_count = 0
        token_count = 0

        for index, finding in enumerate(findings):
            if not self._eligible(finding):
                output.append(finding)
                decisions.append(
                    ReflectionDecision(
                        finding_index=index,
                        action="skip",
                        reason="finding does not meet reflection policy",
                        original_severity=finding.severity,
                    )
                )
                continue

            eligible_count += 1
            if reviewed_count >= self.config.max_findings:
                output.append(finding)
                decisions.append(
                    ReflectionDecision(
                        finding_index=index,
                        action="skip",
                        reason="reflection max_findings limit reached",
                        original_severity=finding.severity,
                        kept_original=True,
                    )
                )
                continue
            if token_count >= self.config.max_tokens:
                output.append(finding)
                decisions.append(
                    ReflectionDecision(
                        finding_index=index,
                        action="skip",
                        reason="reflection max_tokens limit reached",
                        original_severity=finding.severity,
                        kept_original=True,
                    )
                )
                continue

            reviewed_count += 1
            try:
                response = self._request(finding, context_pack, evidence)
                token_count += int((response.usage or {}).get("total_tokens", 0))
                parsed = parse_reflection_decision(response.content)
                transformed, decision = self._apply_decision(index, finding, parsed)
                output.extend(transformed)
                decisions.append(decision)
            except Exception as exc:  # preserve the finding on all provider/schema failures
                output.append(finding)
                decisions.append(
                    ReflectionDecision(
                        finding_index=index,
                        action="error",
                        reason="reflection failed; original finding preserved",
                        original_severity=finding.severity,
                        error=str(exc),
                        kept_original=True,
                    )
                )

        counts = {action: 0 for action in ("keep", "downgrade", "drop", "error", "skip")}
        for decision in decisions:
            counts[decision.action] += 1
        return ReflectionResult(
            findings=tuple(output),
            decisions=tuple(decisions),
            summary={
                "enabled": True,
                "eligible": eligible_count,
                "reviewed": reviewed_count,
                "keep": counts["keep"],
                "downgrade": counts["downgrade"],
                "drop": counts["drop"],
                "errors": counts["error"],
                "skipped": counts["skip"],
                "kept_original_on_error": sum(
                    int(item.action == "error" and item.kept_original)
                    for item in decisions
                ),
                "token_usage": {"total_tokens": token_count},
                "max_findings": self.config.max_findings,
                "max_tokens": self.config.max_tokens,
            },
        )

    def _eligible(self, finding: Finding) -> bool:
        if finding.source != "llm":
            return False
        if self.config.review_all:
            return True
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        return (
            confidence_rank[finding.confidence]
            <= confidence_rank[self.config.confidence_threshold]
            or SEVERITY_RANK[finding.severity]
            <= SEVERITY_RANK[self.config.severity_threshold]
        )

    def _request(
        self,
        finding: Finding,
        context_pack: ReviewContextPack,
        evidence: EvidenceGuardResult,
    ) -> Any:
        method = getattr(self.provider, "reflect_finding", None)
        if not callable(method):
            raise ReflectionOutputError("provider does not support reflection requests")
        return method(finding=finding, context_pack=context_pack, evidence=evidence)

    @staticmethod
    def _apply_decision(
        index: int,
        finding: Finding,
        decision: ParsedReflectionDecision,
    ) -> tuple[list[Finding], ReflectionDecision]:
        if decision.action == "keep":
            return [finding], ReflectionDecision(
                finding_index=index,
                action="keep",
                reason=decision.reason,
                original_severity=finding.severity,
                final_severity=finding.severity,
            )
        if decision.action == "drop":
            return [], ReflectionDecision(
                finding_index=index,
                action="drop",
                reason=decision.reason,
                original_severity=finding.severity,
            )

        target = decision.severity or _next_lower_severity(finding.severity)
        if target not in SEVERITY_RANK:
            raise ReflectionOutputError("downgrade severity must be P0, P1, P2, or P3")
        if SEVERITY_RANK[target] <= SEVERITY_RANK[finding.severity]:
            raise ReflectionOutputError(
                "downgrade severity must be less severe than the original severity"
            )
        downgraded = Finding(
            message=finding.message,
            file_path=finding.file_path,
            line_no=finding.line_no,
            severity=target,
            category=finding.category,
            source=finding.source,
            confidence=finding.confidence,
            rule_id=finding.rule_id,
            evidence=finding.evidence,
            suggestion=finding.suggestion,
            existing_code=finding.existing_code,
        )
        return [downgraded], ReflectionDecision(
            finding_index=index,
            action="downgrade",
            reason=decision.reason,
            original_severity=finding.severity,
            final_severity=target,
        )


def parse_reflection_decision(content: str) -> ParsedReflectionDecision:
    stripped = content.strip()
    if not stripped or "```" in stripped:
        raise ReflectionOutputError(
            "reflection output must be plain JSON without markdown fences"
        )
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ReflectionOutputError("reflection output is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != REFLECTION_ROOT_FIELDS:
        raise ReflectionOutputError(
            "reflection output fields must be exactly "
            f"{sorted(REFLECTION_ROOT_FIELDS)}"
        )
    if payload["schema_version"] != REFLECTION_SCHEMA_VERSION:
        raise ReflectionOutputError(
            f"schema_version must be {REFLECTION_SCHEMA_VERSION!r}"
        )
    action = payload["decision"]
    if action not in REFLECTION_ACTIONS:
        raise ReflectionOutputError("decision must be keep, downgrade, or drop")
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ReflectionOutputError("reason must be a non-empty string")
    severity = payload["severity"]
    if severity is not None and severity not in SEVERITY_RANK:
        raise ReflectionOutputError("severity must be null or P0, P1, P2, P3")
    if action == "downgrade" and severity is None:
        raise ReflectionOutputError("downgrade requires a target severity")
    if action != "downgrade" and severity is not None:
        raise ReflectionOutputError("keep and drop decisions must set severity to null")
    return ParsedReflectionDecision(action, reason.strip(), severity)


def reflection_prompt_payload(
    finding: Finding,
    context_pack: ReviewContextPack,
    evidence: EvidenceGuardResult,
) -> dict[str, Any]:
    return {
        "finding": finding.to_dict(),
        "evidence_guard": evidence.to_dict(),
        "diff": {
            "files": [
                item
                for item in context_pack.diff.get("files", [])
                if item.get("path") == finding.file_path
            ]
        },
        "context_used": [
            item.to_dict()
            for item in context_pack.context.context_used
            if item.path == finding.file_path
        ],
    }


def build_reflection_prompt(
    finding: Finding,
    context_pack: ReviewContextPack,
    evidence: EvidenceGuardResult,
) -> tuple[str, str]:
    payload = reflection_prompt_payload(finding, context_pack, evidence)
    system = (
        "You are the final quality filter inside review-pilot. "
        "Review exactly one existing LLM finding using only the supplied evidence. "
        "Return one plain JSON object and no markdown. "
        "Choose keep when the finding is useful and supported, downgrade when it is "
        "directionally useful but overstated, and drop when it is unsupported, "
        "duplicate, or too generic. Never invent a new finding. "
        "Write reason in natural Simplified Chinese."
        " The JSON object must contain exactly these four keys: "
        "schema_version, decision, reason, severity. Do not add any other key."
    )
    contract = {
        "schema_version": REFLECTION_SCHEMA_VERSION,
        "decision": ["keep", "downgrade", "drop"],
        "reason": "non-empty string",
        "severity": "null unless decision is downgrade; then P0/P1/P2/P3",
    }
    user = (
        "## REFLECTION_INPUT\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n## OUTPUT_CONTRACT\n"
        + json.dumps(contract, ensure_ascii=False, indent=2)
    )
    return system, user


def _next_lower_severity(severity: str) -> str:
    rank = SEVERITY_RANK[severity]
    target = min(rank + 1, max(SEVERITY_RANK.values()))
    return next(name for name, value in SEVERITY_RANK.items() if value == target)
