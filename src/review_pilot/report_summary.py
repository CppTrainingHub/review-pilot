from __future__ import annotations

from dataclasses import dataclass, field

from .report_models import Finding, SEVERITY_RANK


SEVERITY_ORDER = ("P0", "P1", "P2", "P3")


@dataclass(frozen=True)
class ReportSummary:
    total_findings: int = 0
    severity_counts: dict[str, int] = field(
        default_factory=lambda: {severity: 0 for severity in SEVERITY_ORDER}
    )
    category_counts: dict[str, int] = field(default_factory=dict)
    highest_severity: str | None = None

    @classmethod
    def from_findings(cls, findings: list[Finding]) -> "ReportSummary":
        severity_counts = {severity: 0 for severity in SEVERITY_ORDER}
        category_counts: dict[str, int] = {}
        for finding in findings:
            severity_counts[finding.severity] += 1
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

        highest = None
        if findings:
            highest = min(findings, key=lambda finding: finding.severity_rank).severity

        return cls(
            total_findings=len(findings),
            severity_counts=severity_counts,
            category_counts=category_counts,
            highest_severity=highest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "total_findings": self.total_findings,
            "severity_counts": self.severity_counts,
            "category_counts": self.category_counts,
            "highest_severity": self.highest_severity,
        }

    def should_fail(self, fail_on: str | None) -> bool:
        if fail_on is None or self.highest_severity is None:
            return False
        validate_severity_threshold(fail_on)
        return SEVERITY_RANK[self.highest_severity] <= SEVERITY_RANK[fail_on]


def validate_severity_threshold(value: str) -> None:
    if value not in SEVERITY_RANK:
        raise ValueError(f"invalid severity threshold: {value!r}; expected one of {list(SEVERITY_ORDER)}")


def build_report_summary(findings: list[Finding]) -> ReportSummary:
    return ReportSummary.from_findings(findings)


def group_findings_by_severity(findings: list[Finding]) -> dict[str, list[Finding]]:
    groups = {severity: [] for severity in SEVERITY_ORDER}
    for finding in findings:
        groups[finding.severity].append(finding)
    return groups


def should_fail_findings(findings: list[Finding], fail_on: str | None) -> bool:
    return build_report_summary(findings).should_fail(fail_on)
