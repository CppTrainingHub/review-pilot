from __future__ import annotations

import json

from review_pilot.finding_merger import merge_findings
from review_pilot.report_models import Finding, ReviewReport
from review_pilot.report_writer import ReportWriter


def test_final_report_json_contains_merge_summary_and_sources() -> None:
    merge_result = merge_findings(
        rule_findings=[
            Finding(
                message="Debug output appears in added code: print(",
                file_path="src/app.py",
                line_no=3,
                severity="P3",
                category="maintainability",
                source="rule",
                confidence="medium",
                rule_id="rule.debug-output",
            )
        ],
        llm_findings=[
            Finding(
                message="Debug output should be removed.",
                file_path="src/app.py",
                line_no=3,
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
        ],
    )
    report = ReviewReport(
        findings=list(merge_result.findings),
        merge_summary=merge_result.summary.to_dict(),
    )

    payload = json.loads(ReportWriter(report).write_json())

    assert payload["merge_summary"]["merged_groups"] == 1
    assert payload["findings"][0]["evidence"]["merge"]["sources"] == [
        "rule",
        "llm",
    ]
    assert payload["findings"][0]["evidence"]["source_evidence"][0]["source"] == "rule"


def test_final_report_markdown_renders_merge_summary_and_source_list() -> None:
    merge_result = merge_findings(
        rule_findings=[
            Finding(
                message="Rule issue",
                file_path="src/app.py",
                line_no=1,
                severity="P2",
                category="bug",
                source="rule",
                confidence="high",
                rule_id="rule.demo",
            )
        ],
        tool_findings=[
            Finding(
                message="Tool issue",
                file_path="src/app.py",
                line_no=1,
                severity="P2",
                category="bug",
                source="semgrep",
                confidence="medium",
                rule_id="semgrep.demo",
            )
        ],
    )
    report = ReviewReport(
        findings=list(merge_result.findings),
        merge_summary=merge_result.summary.to_dict(),
    )

    markdown = ReportWriter(report).write_markdown()

    assert "### Merge Summary" in markdown
    assert "- Merged groups: 1" in markdown
    assert "- **Sources:** rule, semgrep" in markdown
    assert "- **Conflict:** true" in markdown
