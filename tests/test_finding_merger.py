from __future__ import annotations

from review_pilot.finding_merger import merge_findings, merge_key
from review_pilot.report_models import Finding


def test_merge_findings_combines_same_location_category_sources() -> None:
    rule = Finding(
        message="Debug output appears in added code: print(",
        file_path="src/app.py",
        line_no=4,
        severity="P3",
        category="maintainability",
        source="rule",
        confidence="medium",
        rule_id="rule.debug-output",
        evidence={"matched_pattern": "print("},
    )
    llm = Finding(
        message="Debug print should be removed before merge.",
        file_path="src/app.py",
        line_no=4,
        severity="P2",
        category="maintainability",
        source="llm",
        confidence="medium",
        evidence={
            "verification": {
                "status": "verified",
                "source": "diff_added_line",
            }
        },
    )

    result = merge_findings(rule_findings=[rule], llm_findings=[llm])

    assert result.summary.to_dict() == {
        "total_input_findings": 2,
        "total_output_findings": 1,
        "merged_groups": 1,
        "conflict_groups": 1,
        "source_counts": {"llm": 1, "rule": 1},
    }
    merged = result.findings[0]
    assert merged.severity == "P2"
    assert merged.source == "llm"
    assert merged.evidence["merge"]["sources"] == ["rule", "llm"]
    assert merged.evidence["merge"]["input_count"] == 2
    assert merged.evidence["merge"]["conflict"] is True
    assert len(merged.evidence["source_evidence"]) == 2


def test_merge_findings_keeps_unlocated_findings_separate() -> None:
    file_level = Finding(
        message="Production code changed without a staged test change.",
        file_path="src/app.py",
        line_no=None,
        severity="P2",
        category="test",
        source="rule",
        confidence="medium",
        rule_id="rule.missing-tests",
    )
    line_level = Finding(
        message="Line-level test concern.",
        file_path="src/app.py",
        line_no=4,
        severity="P2",
        category="test",
        source="llm",
        confidence="low",
        evidence={"verification": {"status": "verified"}},
    )

    result = merge_findings(
        rule_findings=[file_level],
        llm_findings=[line_level],
    )

    assert len(result.findings) == 2
    assert result.summary.merged_groups == 0
    assert result.summary.total_output_findings == 2


def test_merge_findings_uses_highest_severity_then_source_tiebreak() -> None:
    rule = Finding(
        message="Rule issue",
        file_path="src/app.py",
        line_no=2,
        severity="P2",
        category="security",
        source="rule",
        confidence="high",
        rule_id="rule.security",
    )
    semgrep = Finding(
        message="Semgrep issue",
        file_path="src/app.py",
        line_no=2,
        severity="P1",
        category="security",
        source="semgrep",
        confidence="medium",
        rule_id="semgrep.security",
    )

    result = merge_findings(rule_findings=[rule], tool_findings=[semgrep])

    merged = result.findings[0]
    assert merged.source == "semgrep"
    assert merged.severity == "P1"
    assert merged.evidence["merge"]["sources"] == ["rule", "semgrep"]


def test_merge_findings_normalizes_each_source_before_multi_source_merge() -> None:
    first = Finding(
        message="Duplicate rule",
        file_path="src/app.py",
        line_no=1,
        severity="P3",
        category="maintainability",
        source="rule",
        confidence="medium",
        rule_id="rule.debug-output",
    )
    second = Finding(
        message="Duplicate rule",
        file_path="src/app.py",
        line_no=1,
        severity="P2",
        category="maintainability",
        source="rule",
        confidence="high",
        rule_id="rule.debug-output",
    )

    result = merge_findings(rule_findings=[first, second])

    assert len(result.findings) == 1
    assert result.summary.total_input_findings == 1
    assert result.findings[0].severity == "P2"
    assert result.findings[0].evidence["merge"]["sources"] == ["rule"]


def test_merge_key_requires_file_line_and_category() -> None:
    assert merge_key(
        Finding(
            message="Located",
            file_path="src/app.py",
            line_no=1,
            category="bug",
        )
    ) == ("src/app.py", 1, "bug")
    assert merge_key(Finding(message="Repo-level")) is None
