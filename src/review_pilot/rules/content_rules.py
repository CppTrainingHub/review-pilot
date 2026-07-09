from __future__ import annotations

from dataclasses import dataclass, field

from review_pilot.report_models import Finding
from review_pilot.rules.base import RuleContext, RuleMetadata


DEFAULT_DEBUG_PATTERNS = (
    "console.log(",
    "debugger;",
    "pdb.set_trace(",
    "print(",
)


@dataclass(frozen=True)
class DebugOutputRule:
    patterns: tuple[str, ...] = field(default_factory=lambda: DEFAULT_DEBUG_PATTERNS)

    @property
    def metadata(self) -> RuleMetadata:
        return RuleMetadata(
            rule_id="rule.debug-output",
            name="Debug output left in code",
            description="Flags common debug output patterns in added lines.",
            category="maintainability",
            default_severity="P3",
        )

    def run(self, context: RuleContext) -> list[Finding]:
        findings: list[Finding] = []
        lowered_patterns = tuple(pattern.lower() for pattern in self.patterns)
        for diff_file in context.review_files:
            for hunk in diff_file.hunks:
                for line in hunk.lines:
                    if line.kind != "added":
                        continue
                    content = line.content.lower()
                    matched = next(
                        (pattern for pattern in lowered_patterns if pattern in content),
                        None,
                    )
                    if not matched:
                        continue
                    findings.append(
                        Finding(
                            message=f"Debug output appears in added code: {matched}",
                            file_path=diff_file.path,
                            line_no=line.new_line_no,
                            severity=self.metadata.default_severity,
                            category=self.metadata.category,
                            source="rule",
                            confidence="medium",
                            rule_id=self.metadata.rule_id,
                            evidence={"matched_pattern": matched},
                            suggestion="Remove debug output or replace it with structured logging.",
                        )
                    )
        return findings
