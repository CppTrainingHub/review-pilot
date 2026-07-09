from __future__ import annotations

from dataclasses import dataclass, field

from review_pilot.report_models import Finding
from review_pilot.rules.base import RuleContext, RuleMetadata


DEFAULT_SENSITIVE_MARKERS = (
    ".github/workflows/",
    "dockerfile",
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "infra/",
    "deploy/",
    "migrations/",
)


@dataclass(frozen=True)
class SensitivePathRule:
    markers: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SENSITIVE_MARKERS)

    @property
    def metadata(self) -> RuleMetadata:
        return RuleMetadata(
            rule_id="rule.sensitive-path",
            name="Sensitive path changed",
            description="Flags changes to CI, dependency, deployment, or migration paths.",
            category="maintainability",
            default_severity="P2",
        )

    def run(self, context: RuleContext) -> list[Finding]:
        findings: list[Finding] = []
        lowered_markers = tuple(marker.lower() for marker in self.markers)
        for path in context.changed_paths:
            normalized = path.lower()
            matched = next(
                (marker for marker in lowered_markers if marker in normalized),
                None,
            )
            if not matched:
                continue
            findings.append(
                Finding(
                    message=f"Sensitive path changed: {path}",
                    file_path=path,
                    severity=self.metadata.default_severity,
                    category=self.metadata.category,
                    source="rule",
                    confidence="medium",
                    rule_id=self.metadata.rule_id,
                    evidence={"matched_marker": matched},
                    suggestion="Request focused review for build, dependency, deployment, or migration impact.",
                )
            )
        return findings
