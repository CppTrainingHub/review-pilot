from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .diff_line_map import ChangedLineMap
from .report_models import Finding
from .tool_models import ToolResult


@dataclass(frozen=True)
class ToolFilterResult:
    included_findings: tuple[Finding, ...]
    out_of_diff_findings: tuple[Finding, ...]
    total_tool_findings: int

    @property
    def included_count(self) -> int:
        return len(self.included_findings)

    @property
    def out_of_diff_count(self) -> int:
        return len(self.out_of_diff_findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tool_findings": self.total_tool_findings,
            "included_count": self.included_count,
            "out_of_diff_count": self.out_of_diff_count,
            "out_of_diff_findings": [
                finding.to_dict()
                for finding in self.out_of_diff_findings
            ],
        }


def filter_tool_findings(
    tool_results: list[ToolResult],
    changed_lines: ChangedLineMap,
    *,
    include_out_of_diff: bool = False,
) -> ToolFilterResult:
    included: list[Finding] = []
    out_of_diff: list[Finding] = []
    total = 0

    for result in tool_results:
        for finding in result.findings:
            total += 1
            if changed_lines.contains(finding.file_path, finding.line_no):
                included.append(finding)
            else:
                out_of_diff.append(finding)

    if include_out_of_diff:
        included.extend(out_of_diff)

    return ToolFilterResult(
        included_findings=tuple(included),
        out_of_diff_findings=tuple(out_of_diff),
        total_tool_findings=total,
    )
