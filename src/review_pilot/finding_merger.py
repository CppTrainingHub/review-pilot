from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from .finding_normalizer import CONFIDENCE_RANK, normalize_findings, sort_key
from .report_models import Finding


SOURCE_PRIORITY = {
    "rule": 0,
    "semgrep": 1,
    "llm": 2,
    "simple-check": 3,
}

MergeKey = tuple[str, int, str]


@dataclass(frozen=True)
class FindingMergeSummary:
    total_input_findings: int
    total_output_findings: int
    merged_groups: int
    conflict_groups: int
    source_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_findings": self.total_input_findings,
            "total_output_findings": self.total_output_findings,
            "merged_groups": self.merged_groups,
            "conflict_groups": self.conflict_groups,
            "source_counts": self.source_counts,
        }


@dataclass(frozen=True)
class FindingMergeResult:
    findings: tuple[Finding, ...]
    summary: FindingMergeSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "merge_summary": self.summary.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def merge_findings(
    *,
    rule_findings: Iterable[Finding] = (),
    tool_findings: Iterable[Finding] = (),
    llm_findings: Iterable[Finding] = (),
) -> FindingMergeResult:
    """Merge rule, tool, and LLM findings into final report findings."""
    normalized = [
        *normalize_findings(rule_findings),
        *normalize_findings(tool_findings),
        *normalize_findings(llm_findings),
    ]

    source_counts: dict[str, int] = {}
    keyed: dict[MergeKey, list[Finding]] = {}
    unkeyed: list[Finding] = []
    for finding in normalized:
        source_counts[finding.source] = source_counts.get(finding.source, 0) + 1
        key = merge_key(finding)
        if key is None:
            unkeyed.append(_with_merge_evidence((finding,)))
        else:
            keyed.setdefault(key, []).append(finding)

    merged: list[Finding] = []
    merged_groups = 0
    conflict_groups = 0
    for group in keyed.values():
        merged_finding = _with_merge_evidence(tuple(group))
        if len(group) > 1:
            merged_groups += 1
        if _has_conflict(group):
            conflict_groups += 1
        merged.append(merged_finding)

    merged.extend(unkeyed)
    merged = sorted(merged, key=sort_key)
    summary = FindingMergeSummary(
        total_input_findings=len(normalized),
        total_output_findings=len(merged),
        merged_groups=merged_groups,
        conflict_groups=conflict_groups,
        source_counts=dict(sorted(source_counts.items())),
    )
    return FindingMergeResult(
        findings=tuple(merged),
        summary=summary,
    )


def merge_key(finding: Finding) -> MergeKey | None:
    if not finding.file_path or finding.line_no is None:
        return None
    return (finding.file_path, finding.line_no, finding.category)


def _with_merge_evidence(group: tuple[Finding, ...]) -> Finding:
    winner = _winner(group)
    sources = _ordered_unique(finding.source for finding in group)
    rule_ids = _ordered_unique(
        finding.rule_id for finding in group if finding.rule_id
    )
    source_evidence = [
        {
            "source": finding.source,
            "rule_id": finding.rule_id,
            "message": finding.message,
            "severity": finding.severity,
            "confidence": finding.confidence,
            "evidence": finding.evidence or {},
        }
        for finding in group
    ]
    evidence = dict(winner.evidence or {})
    evidence["merge"] = {
        "sources": sources,
        "source_count": len(sources),
        "input_count": len(group),
        "winner_source": winner.source,
        "conflict": _has_conflict(group),
        "severity_candidates": _ordered_unique(
            finding.severity for finding in group
        ),
        "confidence_candidates": _ordered_unique(
            finding.confidence for finding in group
        ),
        "rule_ids": rule_ids,
    }
    evidence["source_evidence"] = source_evidence
    return replace(winner, evidence=evidence)


def _winner(group: tuple[Finding, ...]) -> Finding:
    return min(
        group,
        key=lambda finding: (
            finding.severity_rank,
            CONFIDENCE_RANK[finding.confidence],
            SOURCE_PRIORITY[finding.source],
            finding.message,
        ),
    )


def _has_conflict(group: Iterable[Finding]) -> bool:
    findings = tuple(group)
    severities = {finding.severity for finding in findings}
    messages = {finding.message.strip().lower() for finding in findings}
    suggestions = {
        (finding.suggestion or "").strip().lower()
        for finding in findings
        if finding.suggestion
    }
    return len(severities) > 1 or len(messages) > 1 or len(suggestions) > 1


def _ordered_unique(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return sorted(
        result,
        key=lambda source: (SOURCE_PRIORITY.get(source, 99), source),
    )
