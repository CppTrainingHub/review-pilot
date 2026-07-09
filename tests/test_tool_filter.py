from __future__ import annotations

from review_pilot.diff_line_map import ChangedLineMap
from review_pilot.report_models import Finding
from review_pilot.tool_filter import filter_tool_findings
from review_pilot.tool_models import ToolResult


def test_filter_tool_findings_keeps_only_changed_lines_by_default() -> None:
    result = filter_tool_findings(
        [
            ToolResult(
                tool_name="semgrep",
                status="success",
                findings=(
                    Finding(
                        message="new issue",
                        file_path="src/app.py",
                        line_no=10,
                        source="semgrep",
                        rule_id="semgrep.new",
                    ),
                    Finding(
                        message="old issue",
                        file_path="src/app.py",
                        line_no=4,
                        source="semgrep",
                        rule_id="semgrep.old",
                    ),
                ),
            )
        ],
        ChangedLineMap({"src/app.py": frozenset({10})}),
    )

    assert [finding.rule_id for finding in result.included_findings] == ["semgrep.new"]
    assert [finding.rule_id for finding in result.out_of_diff_findings] == ["semgrep.old"]
    assert result.total_tool_findings == 2
    assert result.included_count == 1
    assert result.out_of_diff_count == 1


def test_filter_tool_findings_can_include_out_of_diff_findings() -> None:
    result = filter_tool_findings(
        [
            ToolResult(
                tool_name="semgrep",
                status="success",
                findings=(
                    Finding(
                        message="new issue",
                        file_path="src/app.py",
                        line_no=10,
                        source="semgrep",
                        rule_id="semgrep.new",
                    ),
                    Finding(
                        message="old issue",
                        file_path="src/app.py",
                        line_no=4,
                        source="semgrep",
                        rule_id="semgrep.old",
                    ),
                ),
            )
        ],
        ChangedLineMap({"src/app.py": frozenset({10})}),
        include_out_of_diff=True,
    )

    assert [finding.rule_id for finding in result.included_findings] == [
        "semgrep.new",
        "semgrep.old",
    ]
    assert result.out_of_diff_count == 1


def test_filter_tool_findings_treats_missing_location_as_out_of_diff() -> None:
    result = filter_tool_findings(
        [
            ToolResult(
                tool_name="semgrep",
                status="success",
                findings=(
                    Finding(
                        message="repo level issue",
                        source="semgrep",
                        rule_id="semgrep.repo",
                    ),
                ),
            )
        ],
        ChangedLineMap({}),
    )

    assert result.included_findings == ()
    assert [finding.rule_id for finding in result.out_of_diff_findings] == ["semgrep.repo"]
