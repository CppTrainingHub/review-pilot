from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any

from review_pilot.context_pack import ReviewContextPack
from review_pilot.report_models import Finding


VERIFIED = "verified"
DOWNGRADED = "downgraded"
DROPPED = "dropped"


@dataclass(frozen=True)
class EvidenceReference:
    source: str
    file_path: str
    line_no: int
    line_content: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "file_path": self.file_path,
            "line_no": self.line_no,
            "line_content": self.line_content,
        }


@dataclass(frozen=True)
class EvidenceDecision:
    status: str
    reason: str
    finding: Finding
    reference: EvidenceReference | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "reason": self.reason,
            "finding": self.finding.to_dict(),
        }
        if self.reference is not None:
            payload["reference"] = self.reference.to_dict()
        return payload


@dataclass(frozen=True)
class EvidenceGuardResult:
    decisions: tuple[EvidenceDecision, ...]

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(
            decision.finding
            for decision in self.decisions
            if decision.status != DROPPED
        )

    @property
    def dropped_findings(self) -> tuple[EvidenceDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.status == DROPPED
        )

    @property
    def summary(self) -> dict[str, int]:
        verified = sum(
            decision.status == VERIFIED for decision in self.decisions
        )
        downgraded = sum(
            decision.status == DOWNGRADED for decision in self.decisions
        )
        dropped = sum(
            decision.status == DROPPED for decision in self.decisions
        )
        return {
            "total": len(self.decisions),
            "kept": verified + downgraded,
            "verified": verified,
            "downgraded": downgraded,
            "dropped": dropped,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_summary": self.summary,
            "findings": [finding.to_dict() for finding in self.findings],
            "dropped_findings": [
                decision.to_dict()
                for decision in self.dropped_findings
            ],
        }


@dataclass(frozen=True)
class EvidenceIndex:
    added_lines: dict[str, dict[int, str]]
    context_lines: dict[str, dict[int, str]]

    @property
    def paths(self) -> set[str]:
        return set(self.added_lines) | set(self.context_lines)


def build_evidence_index(context_pack: ReviewContextPack) -> EvidenceIndex:
    payload = context_pack.to_dict()
    added_lines: dict[str, dict[int, str]] = {}
    for diff_file in payload["diff"]["files"]:
        path = _normalize_path(diff_file.get("path"))
        if path is None:
            continue
        file_lines = added_lines.setdefault(path, {})
        for hunk in diff_file.get("hunks", []):
            for line in hunk.get("lines", []):
                if line.get("kind") != "added":
                    continue
                line_no = line.get("new_line_no")
                content = line.get("content")
                if isinstance(line_no, int) and isinstance(content, str):
                    file_lines[line_no] = content

    context_lines: dict[str, dict[int, str]] = {}
    for context_slice in payload["context"]["context_used"]:
        path = _normalize_path(context_slice.get("path"))
        start_line = context_slice.get("start_line")
        end_line = context_slice.get("end_line")
        content = context_slice.get("content")
        if (
            path is None
            or not isinstance(start_line, int)
            or not isinstance(end_line, int)
            or not isinstance(content, str)
        ):
            continue
        file_lines = context_lines.setdefault(path, {})
        for offset, line_content in enumerate(content.split("\n")):
            line_no = start_line + offset
            if line_no > end_line:
                break
            file_lines[line_no] = line_content

    return EvidenceIndex(
        added_lines=added_lines,
        context_lines=context_lines,
    )


def guard_llm_findings(
    findings: tuple[Finding, ...],
    context_pack: ReviewContextPack,
) -> EvidenceGuardResult:
    index = build_evidence_index(context_pack)
    decisions = tuple(
        _guard_finding(finding, index)
        for finding in findings
    )
    return EvidenceGuardResult(decisions=decisions)


def _guard_finding(
    finding: Finding,
    index: EvidenceIndex,
) -> EvidenceDecision:
    path = _normalize_path(finding.file_path)
    if path is None:
        return EvidenceDecision(
            status=DROPPED,
            reason="file_path must be a repository-relative POSIX path",
            finding=finding,
        )

    if path not in index.paths:
        return EvidenceDecision(
            status=DROPPED,
            reason=(
                "file_path is not present in the supplied diff or context"
            ),
            finding=finding,
        )

    if finding.line_no is None:
        return EvidenceDecision(
            status=DROPPED,
            reason="line_no is required for LLM evidence verification",
            finding=finding,
        )

    added_content = index.added_lines.get(path, {}).get(finding.line_no)
    if added_content is not None:
        reference = EvidenceReference(
            source="diff_added_line",
            file_path=path,
            line_no=finding.line_no,
            line_content=added_content,
        )
        return EvidenceDecision(
            status=VERIFIED,
            reason="file_path and line_no match a diff added line",
            finding=_verified_finding(
                finding,
                path=path,
                status=VERIFIED,
                reference=reference,
                confidence=finding.confidence,
            ),
            reference=reference,
        )

    context_content = index.context_lines.get(path, {}).get(
        finding.line_no
    )
    if context_content is not None:
        reference = EvidenceReference(
            source="context_used",
            file_path=path,
            line_no=finding.line_no,
            line_content=context_content,
        )
        return EvidenceDecision(
            status=DOWNGRADED,
            reason=(
                "file_path and line_no match supplied context, "
                "but not a diff added line"
            ),
            finding=_verified_finding(
                finding,
                path=path,
                status=DOWNGRADED,
                reference=reference,
                confidence="low",
            ),
            reference=reference,
        )

    return EvidenceDecision(
        status=DROPPED,
        reason="line_no is not present in the supplied diff or context",
        finding=finding,
    )


def _verified_finding(
    finding: Finding,
    *,
    path: str,
    status: str,
    reference: EvidenceReference,
    confidence: str,
) -> Finding:
    evidence = dict(finding.evidence or {})
    evidence["verification"] = {
        "status": status,
        "source": reference.source,
        "line_content": reference.line_content,
    }
    return replace(
        finding,
        file_path=path,
        confidence=confidence,
        evidence=evidence,
    )


def _normalize_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or "\\" in stripped:
        return None
    path = PurePosixPath(stripped)
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        return None
    return str(PurePosixPath(*parts))
