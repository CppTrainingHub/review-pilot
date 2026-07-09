from __future__ import annotations

import pytest

from review_pilot.report_models import Finding
from review_pilot.report_summary import (
    build_report_summary,
    group_findings_by_severity,
    should_fail_findings,
    validate_severity_threshold,
)


def findings() -> list[Finding]:
    return [
        Finding(
            message="Critical workflow changed",
            file_path=".github/workflows/ci.yml",
            severity="P1",
            category="maintainability",
            source="rule",
            confidence="high",
            rule_id="rule.sensitive-path",
        ),
        Finding(
            message="Debug output",
            file_path="src/app.py",
            line_no=3,
            severity="P3",
            category="maintainability",
            source="rule",
            confidence="medium",
            rule_id="rule.debug-output",
        ),
        Finding(
            message="Missing tests",
            file_path="src/app.py",
            severity="P2",
            category="test",
            source="rule",
            confidence="medium",
            rule_id="rule.missing-tests",
        ),
    ]


def test_build_report_summary_counts_total_severity_and_categories() -> None:
    summary = build_report_summary(findings())

    assert summary.total_findings == 3
    assert summary.highest_severity == "P1"
    assert summary.severity_counts == {"P0": 0, "P1": 1, "P2": 1, "P3": 1}
    assert summary.category_counts == {"maintainability": 2, "test": 1}
    assert summary.to_dict()["highest_severity"] == "P1"


def test_empty_summary_has_stable_zero_counts() -> None:
    summary = build_report_summary([])

    assert summary.total_findings == 0
    assert summary.highest_severity is None
    assert summary.severity_counts == {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    assert summary.category_counts == {}


def test_group_findings_by_severity_keeps_all_severity_buckets() -> None:
    groups = group_findings_by_severity(findings())

    assert list(groups) == ["P0", "P1", "P2", "P3"]
    assert groups["P0"] == []
    assert [finding.severity for finding in groups["P1"]] == ["P1"]
    assert [finding.severity for finding in groups["P2"]] == ["P2"]
    assert [finding.severity for finding in groups["P3"]] == ["P3"]


def test_should_fail_uses_threshold_rank() -> None:
    data = findings()

    assert should_fail_findings(data, "P0") is False
    assert should_fail_findings(data, "P1") is True
    assert should_fail_findings(data, "P2") is True
    assert should_fail_findings(data, "P3") is True
    assert should_fail_findings(data, None) is False


def test_should_fail_passes_when_no_findings() -> None:
    assert should_fail_findings([], "P3") is False


def test_validate_severity_threshold_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="invalid severity threshold"):
        validate_severity_threshold("HIGH")
