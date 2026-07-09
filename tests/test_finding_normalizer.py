from __future__ import annotations

import pytest

from review_pilot.finding_normalizer import (
    coerce_finding,
    finding_key,
    normalize_findings,
)
from review_pilot.report_models import Finding


def test_coerce_finding_applies_model_defaults() -> None:
    finding = coerce_finding(
        {
            "message": "Missing focused test",
            "file_path": "src/app.py",
            "line_no": 12,
            "severity": "P2",
            "category": "test",
            "source": "rule",
            "rule_id": "rule.missing-tests",
        }
    )

    assert finding.confidence == "high"
    assert finding.rule_id == "rule.missing-tests"


def test_coerce_finding_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="unexpected finding fields"):
        coerce_finding(
            {
                "message": "Bad payload",
                "source": "rule",
                "unknown": True,
            }
        )


def test_coerce_finding_rejects_invalid_enum_values() -> None:
    with pytest.raises(ValueError, match="invalid severity"):
        coerce_finding(
            {
                "message": "Bad severity",
                "severity": "HIGH",
                "source": "rule",
            }
        )


def test_finding_key_uses_location_category_source_and_rule_identity() -> None:
    finding = Finding(
        message="Debug output",
        file_path="src/app.py",
        line_no=4,
        category="maintainability",
        source="rule",
        rule_id="rule.debug-output",
    )

    assert finding_key(finding) == (
        "src/app.py",
        4,
        "maintainability",
        "rule",
        "rule.debug-output",
    )


def test_normalize_findings_dedupes_duplicate_rule_findings() -> None:
    first = Finding(
        message="Debug output appears in added code",
        file_path="src/app.py",
        line_no=3,
        severity="P3",
        category="maintainability",
        source="rule",
        confidence="medium",
        rule_id="rule.debug-output",
        evidence={"matched_pattern": "print("},
    )
    second = Finding(
        message="Debug output appears in added code",
        file_path="src/app.py",
        line_no=3,
        severity="P2",
        category="maintainability",
        source="rule",
        confidence="high",
        rule_id="rule.debug-output",
        evidence={"line": "print('debug')"},
    )

    findings = normalize_findings([first, second])

    assert len(findings) == 1
    assert findings[0].severity == "P2"
    assert findings[0].confidence == "high"
    assert findings[0].evidence == {
        "line": "print('debug')",
        "matched_pattern": "print(",
        "duplicate_count": 2,
    }


def test_normalize_findings_does_not_merge_different_rules_same_location() -> None:
    findings = normalize_findings(
        [
            Finding(
                message="Debug output",
                file_path="src/app.py",
                line_no=1,
                category="maintainability",
                source="rule",
                rule_id="rule.debug-output",
            ),
            Finding(
                message="Sensitive path",
                file_path="src/app.py",
                line_no=1,
                category="maintainability",
                source="rule",
                rule_id="rule.sensitive-path",
            ),
        ]
    )

    assert [finding.rule_id for finding in findings] == [
        "rule.debug-output",
        "rule.sensitive-path",
    ]


def test_normalize_findings_sorts_by_severity_file_line_and_identity() -> None:
    findings = normalize_findings(
        [
            Finding(
                message="Low priority",
                file_path="z.py",
                line_no=1,
                severity="P3",
                source="rule",
                rule_id="rule.low",
            ),
            Finding(
                message="High priority later file",
                file_path="b.py",
                line_no=5,
                severity="P1",
                source="rule",
                rule_id="rule.high-b",
            ),
            Finding(
                message="High priority earlier file",
                file_path="a.py",
                line_no=8,
                severity="P1",
                source="rule",
                rule_id="rule.high-a",
            ),
            Finding(
                message="High priority earlier line",
                file_path="a.py",
                line_no=2,
                severity="P1",
                source="rule",
                rule_id="rule.high-line",
            ),
        ]
    )

    assert [finding.rule_id for finding in findings] == [
        "rule.high-line",
        "rule.high-a",
        "rule.high-b",
        "rule.low",
    ]
