from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .report_models import Finding, ReviewReport
from .report_summary import SEVERITY_ORDER, build_report_summary, group_findings_by_severity


@dataclass(frozen=True)
class ReportWriter:
    """Render a ReviewReport as JSON or Markdown."""

    report: ReviewReport

    def write_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.report.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    def write_markdown(self) -> str:
        lines: list[str] = []
        summary = build_report_summary(self.report.findings)

        lines.append("# Review Pilot Report")
        lines.append("")
        lines.append(f"**Total findings:** {summary.total_findings}")
        if summary.highest_severity:
            lines.append(f"**Highest severity:** {summary.highest_severity}")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        for severity in SEVERITY_ORDER:
            count = summary.severity_counts[severity]
            lines.append(f"- {severity}: {count}")
        lines.append("")

        if self.report.merge_summary:
            lines.append("### Merge Summary")
            lines.append("")
            lines.extend(_render_merge_summary(self.report.merge_summary))
            lines.append("")

        if summary.category_counts:
            lines.append("### Categories")
            lines.append("")
            for category, count in sorted(summary.category_counts.items()):
                lines.append(f"- {category}: {count}")
            lines.append("")

        lines.append("## Findings")
        lines.append("")
        if not self.report.findings:
            lines.append("No findings detected.")
            lines.append("")
        else:
            index = 1
            grouped = group_findings_by_severity(self.report.findings)
            for severity in SEVERITY_ORDER:
                findings = grouped[severity]
                if not findings:
                    continue
                lines.append(f"### {severity} Findings")
                lines.append("")
                for finding in findings:
                    lines.append(self._render_finding(index, finding))
                    index += 1

        lines.append("## Metadata")
        lines.append("")
        if self.report.repo_info:
            for key, value in sorted(self.report.repo_info.items()):
                lines.append(f"- **{key}:** {value}")
        else:
            lines.append("- No repository metadata available.")
        if self.report.config_source:
            lines.append(f"- **config_source:** {self.report.config_source}")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_finding(index: int, finding: Finding) -> str:
        location = finding.file_path or "unknown"
        if finding.line_no:
            location = f"{location}:{finding.line_no}"

        lines: list[str] = []
        lines.append(f"#### {index}. [{finding.severity}] {finding.message}")
        lines.append("")
        lines.append(f"- **Location:** `{location}`")
        lines.append(f"- **Category:** {finding.category}")
        lines.append(f"- **Source:** {finding.source}")
        lines.append(f"- **Confidence:** {finding.confidence}")
        merge = (finding.evidence or {}).get("merge")
        if isinstance(merge, dict):
            sources = merge.get("sources")
            if isinstance(sources, list):
                lines.append(f"- **Sources:** {', '.join(str(item) for item in sources)}")
            lines.append(f"- **Merged inputs:** {merge.get('input_count', 1)}")
            lines.append(f"- **Conflict:** {str(bool(merge.get('conflict'))).lower()}")
        if finding.rule_id:
            lines.append(f"- **Rule:** {finding.rule_id}")
        if finding.suggestion:
            lines.append(f"- **Suggestion:** {finding.suggestion}")
        if finding.evidence:
            lines.append("- **Evidence:**")
            for key, value in finding.evidence.items():
                lines.append(f"  - {key}: {_format_evidence_value(value)}")
        lines.append("")
        return "\n".join(lines)


def write_report(report: ReviewReport, fmt: str) -> str:
    """Convenience dispatcher for JSON and Markdown formats."""
    fmt = fmt.lower()
    if fmt == "json":
        return ReportWriter(report).write_json()
    if fmt == "markdown":
        return ReportWriter(report).write_markdown()
    raise ValueError(f"unsupported report format: {fmt!r}; use 'json' or 'markdown'")


def _render_merge_summary(summary: dict[str, Any]) -> list[str]:
    lines = [
        f"- Input findings: {summary.get('total_input_findings', 0)}",
        f"- Output findings: {summary.get('total_output_findings', 0)}",
        f"- Merged groups: {summary.get('merged_groups', 0)}",
        f"- Conflict groups: {summary.get('conflict_groups', 0)}",
    ]
    source_counts = summary.get("source_counts")
    if isinstance(source_counts, dict) and source_counts:
        rendered = ", ".join(
            f"{source}: {count}"
            for source, count in sorted(source_counts.items())
        )
        lines.append(f"- Sources: {rendered}")
    return lines


def _format_evidence_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)
