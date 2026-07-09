from __future__ import annotations

from dataclasses import dataclass, field

from .models import ParsedDiff
from .report_models import Finding


@dataclass(frozen=True)
class FileSizeCheck:
    """Flag files whose number of added lines exceeds a threshold."""

    max_added_lines: int = 200
    rule_id: str = "simple-check.file-too-large"

    def run(self, parsed_diff: ParsedDiff) -> list[Finding]:
        findings: list[Finding] = []
        for diff_file in parsed_diff.files:
            added_count = sum(
                1
                for hunk in diff_file.hunks
                for line in hunk.lines
                if line.kind == "added"
            )
            if added_count > self.max_added_lines:
                findings.append(
                    Finding(
                        message=(
                            f"File '{diff_file.path}' adds {added_count} lines, "
                            f"exceeding threshold of {self.max_added_lines}. "
                            "Consider splitting the change into smaller reviews."
                        ),
                        file_path=diff_file.path,
                        severity="P2",
                        category="size",
                        source="simple-check",
                        confidence="high",
                        rule_id=self.rule_id,
                        evidence={
                            "added_lines": added_count,
                            "threshold": self.max_added_lines,
                        },
                        suggestion="Break the file into smaller, focused changes.",
                    )
                )
        return findings


@dataclass(frozen=True)
class SimpleCheckRunner:
    """Run all minimal deterministic checks against a parsed diff."""

    file_size_check: FileSizeCheck = field(default_factory=FileSizeCheck)

    def run(self, parsed_diff: ParsedDiff) -> list[Finding]:
        return self.file_size_check.run(parsed_diff)
