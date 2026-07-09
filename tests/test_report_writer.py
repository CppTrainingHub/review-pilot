from __future__ import annotations

import json

import pytest

from review_pilot.report_models import Finding, ReviewReport
from review_pilot.report_writer import ReportWriter, write_report


def sample_report() -> ReviewReport:
    return ReviewReport(
        findings=[
            Finding(
                message="File is too large",
                file_path="app.py",
                line_no=1,
                severity="P2",
                category="size",
                source="simple-check",
                confidence="high",
                rule_id="simple-check.file-too-large",
                evidence={"added_lines": 250, "threshold": 200},
                suggestion="Split the change.",
            ),
            Finding(
                message="Missing test",
                file_path="lib.py",
                severity="P1",
                category="test",
                source="rule",
                confidence="medium",
            ),
        ],
        repo_info={"branch": "main"},
        config_source="default",
    )


def test_json_writer_outputs_dict_structure() -> None:
    report = sample_report()
    report.sort_findings()
    text = ReportWriter(report).write_json()
    payload = json.loads(text)

    assert payload["summary"]["total_findings"] == 2
    assert payload["summary"]["highest_severity"] == "P1"
    assert payload["summary"]["severity_counts"]["P2"] == 1
    assert len(payload["findings"]) == 2
    assert payload["findings"][0]["severity"] == "P1"
    assert payload["repo_info"]["branch"] == "main"
    assert payload["config_source"] == "default"


def test_markdown_writer_includes_findings_and_summary() -> None:
    report = sample_report()
    md = ReportWriter(report).write_markdown()

    assert "# Review Pilot Report" in md
    assert "**Total findings:** 2" in md
    assert "### P1 Findings" in md
    assert "### P2 Findings" in md
    assert "File is too large" in md
    assert "Missing test" in md
    assert "app.py:1" in md
    assert "lib.py" in md
    assert "P1" in md
    assert "P2" in md


def test_write_report_dispatcher_uses_format() -> None:
    report = sample_report()

    json_output = write_report(report, "json")
    assert json.loads(json_output)["summary"]["total_findings"] == 2

    md_output = write_report(report, "markdown")
    assert "# Review Pilot Report" in md_output


def test_write_report_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unsupported report format"):
        write_report(ReviewReport(), "xml")


def test_report_summary_is_empty_when_no_findings() -> None:
    report = ReviewReport()

    assert report.summary["total_findings"] == 0
    assert report.summary["highest_severity"] is None
    assert report.summary["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0, "P3": 0}


def test_json_writer_keeps_severity_count_order_stable() -> None:
    report = sample_report()

    text = ReportWriter(report).write_json()
    p0_index = text.index('"P0"')
    p1_index = text.index('"P1"')
    p2_index = text.index('"P2"')
    p3_index = text.index('"P3"')

    assert p0_index < p1_index < p2_index < p3_index


def test_markdown_writer_groups_findings_by_severity_order() -> None:
    md = ReportWriter(sample_report()).write_markdown()

    assert md.index("### P1 Findings") < md.index("### P2 Findings")
    assert "### P0 Findings" not in md
    assert "### P3 Findings" not in md
