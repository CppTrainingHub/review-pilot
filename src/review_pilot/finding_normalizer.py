from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from .report_models import Finding


RawFinding = Finding | Mapping[str, Any]
FindingKey = tuple[str, int, str, str, str]

CONFIDENCE_RANK = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


def coerce_finding(raw: RawFinding) -> Finding:
    """Convert a raw mapping or Finding object into a validated Finding."""
    if isinstance(raw, Finding):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"finding must be a Finding or mapping, got {type(raw).__name__}")
    try:
        return Finding.from_dict(dict(raw))
    except TypeError as exc:
        raise ValueError(f"invalid finding payload: {exc}") from exc


def finding_key(finding: Finding) -> FindingKey:
    """Return the identity used for deduping findings from one source."""
    identity = finding.rule_id or finding.message.strip().lower()
    return (
        finding.file_path or "",
        finding.line_no or 0,
        finding.category,
        finding.source,
        identity,
    )


def sort_key(finding: Finding) -> tuple[int, str, int, str, str, str]:
    """Sort by review priority, then stable location and source fields."""
    return (
        finding.severity_rank,
        finding.file_path or "",
        finding.line_no or 0,
        finding.category,
        finding.source,
        finding.rule_id or finding.message,
    )


def _merge_duplicate(current: Finding, incoming: Finding) -> Finding:
    winner = min(
        (current, incoming),
        key=lambda item: (item.severity_rank, CONFIDENCE_RANK[item.confidence]),
    )
    duplicate_count = 1
    evidence: dict[str, Any] = {}
    if current.evidence:
        evidence.update(current.evidence)
        duplicate_count = int(current.evidence.get("duplicate_count", duplicate_count))
    if incoming.evidence:
        for key, value in incoming.evidence.items():
            if key not in evidence:
                evidence[key] = value
    evidence["duplicate_count"] = duplicate_count + 1
    return replace(winner, evidence=evidence)


def normalize_findings(raw_findings: Iterable[RawFinding]) -> list[Finding]:
    """Validate, dedupe, and sort findings before report rendering."""
    deduped: dict[FindingKey, Finding] = {}
    for raw in raw_findings:
        finding = coerce_finding(raw)
        key = finding_key(finding)
        if key in deduped:
            deduped[key] = _merge_duplicate(deduped[key], finding)
        else:
            deduped[key] = finding
    return sorted(deduped.values(), key=sort_key)
