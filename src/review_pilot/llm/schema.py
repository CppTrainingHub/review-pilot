from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from typing import Any

from review_pilot.report_models import (
    VALID_CATEGORIES,
    VALID_CONFIDENCES,
    VALID_SEVERITIES,
    Finding,
)


LLM_FINDINGS_SCHEMA_VERSION = "review-pilot.llm-findings.v1"
ROOT_FIELDS = {"schema_version", "findings"}
FINDING_FIELDS = {
    "message",
    "file_path",
    "line_no",
    "severity",
    "category",
    "source",
    "confidence",
    "evidence",
    "suggestion",
}
SNIPPET_FINDING_FIELDS = FINDING_FIELDS | {"existing_code"}
EVIDENCE_FIELDS = {"reason"}


class LLMOutputError(ValueError):
    """Raised when model content does not satisfy the LLM findings contract."""


@dataclass(frozen=True)
class LLMFindingsEnvelope:
    schema_version: str
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def parse_llm_findings(
    content: str,
    *,
    require_existing_code: bool = False,
) -> LLMFindingsEnvelope:
    stripped = content.strip()
    if not stripped:
        raise LLMOutputError("llm output must be a non-empty JSON object")
    if "```" in stripped:
        raise LLMOutputError(
            "llm output must be plain JSON without markdown fences"
        )
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(
            f"llm output is not valid JSON: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise LLMOutputError("llm output root must be an object")
    _require_exact_fields(payload, ROOT_FIELDS, "llm output root")
    if payload["schema_version"] != LLM_FINDINGS_SCHEMA_VERSION:
        raise LLMOutputError(
            "schema_version must be "
            f"{LLM_FINDINGS_SCHEMA_VERSION!r}"
        )
    raw_findings = payload["findings"]
    if not isinstance(raw_findings, list):
        raise LLMOutputError("findings must be an array")
    findings = tuple(
        _parse_finding(raw, index, require_existing_code=require_existing_code)
        for index, raw in enumerate(raw_findings)
    )
    return LLMFindingsEnvelope(
        schema_version=LLM_FINDINGS_SCHEMA_VERSION,
        findings=findings,
    )


def _parse_finding(
    raw: Any,
    index: int,
    *,
    require_existing_code: bool,
) -> Finding:
    path = f"findings[{index}]"
    if not isinstance(raw, dict):
        raise LLMOutputError(f"{path} must be an object")
    expected_fields = SNIPPET_FINDING_FIELDS if require_existing_code else FINDING_FIELDS
    missing = expected_fields - set(raw)
    if missing:
        raise LLMOutputError(f"{path} is missing fields: {sorted(missing)}")
    extra = set(raw) - SNIPPET_FINDING_FIELDS
    if extra:
        raise LLMOutputError(f"{path} has unexpected fields: {sorted(extra)}")

    message = _require_non_empty_string(raw["message"], f"{path}.message")
    file_path = _require_non_empty_string(
        raw["file_path"],
        f"{path}.file_path",
    )
    line_no = _coerce_provider_line_no(raw["line_no"])
    if not isinstance(line_no, int) or isinstance(line_no, bool) or line_no < 1:
        if line_no is not None:
            raise LLMOutputError(f"{path}.line_no must be a positive integer")
    severity = _require_enum(
        raw["severity"],
        VALID_SEVERITIES,
        f"{path}.severity",
    )
    category = _require_enum(
        raw["category"],
        VALID_CATEGORIES,
        f"{path}.category",
    )
    source = _require_non_empty_string(raw["source"], f"{path}.source")
    if source != "llm":
        raise LLMOutputError(f"{path}.source must be 'llm'")
    confidence = _require_enum(
        raw["confidence"],
        VALID_CONFIDENCES,
        f"{path}.confidence",
    )
    evidence = raw["evidence"]
    if not isinstance(evidence, dict):
        raise LLMOutputError(f"{path}.evidence must be an object")
    _require_exact_fields(evidence, EVIDENCE_FIELDS, f"{path}.evidence")
    reason = _require_non_empty_string(
        evidence["reason"],
        f"{path}.evidence.reason",
    )
    suggestion = _require_non_empty_string(
        raw["suggestion"],
        f"{path}.suggestion",
    )
    existing_code = None
    if "existing_code" in raw:
        existing_code = _require_non_empty_string(
            raw["existing_code"],
            f"{path}.existing_code",
        )
    try:
        return Finding(
            message=message,
            file_path=file_path,
            line_no=line_no,
            severity=severity,
            category=category,
            source=source,
            confidence=confidence,
            evidence={"reason": reason},
            suggestion=suggestion,
            existing_code=existing_code,
        )
    except ValueError as exc:
        raise LLMOutputError(f"{path} is invalid: {exc}") from exc


def _coerce_provider_line_no(value: Any) -> int | None:
    """Keep malformed numeric positions out of evidence matching.

    Some OpenAI-compatible providers occasionally serialize a calculated line
    position as a non-integral float.  Rounding it would create a false code
    reference, so the parser keeps that finding without a line number and
    lets EvidenceGuard drop it explicitly.
    """

    if isinstance(value, float):
        if not isfinite(value) or value < 1:
            return None
        if not value.is_integer():
            return None
        return int(value)
    return value


def _require_exact_fields(
    payload: dict[str, Any],
    expected: set[str],
    path: str,
) -> None:
    missing = sorted(expected - set(payload))
    if missing:
        raise LLMOutputError(f"{path} is missing fields: {missing}")
    extra = sorted(set(payload) - expected)
    if extra:
        raise LLMOutputError(f"{path} has unexpected fields: {extra}")


def _require_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMOutputError(f"{path} must be a non-empty string")
    return value.strip()


def _require_enum(value: Any, allowed: set[str], path: str) -> str:
    text = _require_non_empty_string(value, path)
    if text not in allowed:
        raise LLMOutputError(
            f"{path} must be one of {sorted(allowed)}"
        )
    return text
