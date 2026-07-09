from __future__ import annotations

from dataclasses import dataclass

from review_pilot.report_models import Finding
from review_pilot.rules.base import RuleContext, RuleMetadata


def added_line_count(context: RuleContext) -> int:
    return sum(
        1
        for diff_file in context.review_files
        for hunk in diff_file.hunks
        for line in hunk.lines
        if line.kind == "added"
    )


@dataclass(frozen=True)
class FileTooLargeRule:
    max_added_lines: int = 200

    @property
    def metadata(self) -> RuleMetadata:
        return RuleMetadata(
            rule_id="rule.file-too-large",
            name="File too large",
            description="Flags a file when one staged file adds too many lines.",
            category="size",
            default_severity="P2",
        )

    def run(self, context: RuleContext) -> list[Finding]:
        findings: list[Finding] = []
        for diff_file in context.review_files:
            count = sum(
                1
                for hunk in diff_file.hunks
                for line in hunk.lines
                if line.kind == "added"
            )
            if count > self.max_added_lines:
                findings.append(
                    Finding(
                        message=(
                            f"File '{diff_file.path}' adds {count} lines, exceeding "
                            f"threshold of {self.max_added_lines}."
                        ),
                        file_path=diff_file.path,
                        severity=self.metadata.default_severity,
                        category=self.metadata.category,
                        source="rule",
                        confidence="high",
                        rule_id=self.metadata.rule_id,
                        evidence={
                            "added_lines": count,
                            "threshold": self.max_added_lines,
                        },
                        suggestion="Split the file into smaller, focused changes.",
                    )
                )
        return findings


@dataclass(frozen=True)
class ChangeTooLargeRule:
    max_total_added_lines: int = 400

    @property
    def metadata(self) -> RuleMetadata:
        return RuleMetadata(
            rule_id="rule.change-too-large",
            name="Change too large",
            description="Flags the staged change when total added lines are too high.",
            category="size",
            default_severity="P2",
        )

    def run(self, context: RuleContext) -> list[Finding]:
        total = added_line_count(context)
        if total <= self.max_total_added_lines:
            return []
        return [
            Finding(
                message=(
                    f"Staged change adds {total} lines, exceeding threshold of "
                    f"{self.max_total_added_lines}."
                ),
                severity=self.metadata.default_severity,
                category=self.metadata.category,
                source="rule",
                confidence="high",
                rule_id=self.metadata.rule_id,
                evidence={
                    "added_lines": total,
                    "threshold": self.max_total_added_lines,
                },
                suggestion="Split the staged change before requesting review.",
            )
        ]
