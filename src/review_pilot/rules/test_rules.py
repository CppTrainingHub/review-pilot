from __future__ import annotations

from dataclasses import dataclass

from review_pilot.report_models import Finding
from review_pilot.rules.base import RuleContext, RuleMetadata


TEST_PATH_MARKERS = ("test", "tests/", "_test.", "spec", "__tests__")


def is_test_path(path: str) -> bool:
    normalized = path.lower()
    return any(marker in normalized for marker in TEST_PATH_MARKERS)


def is_production_path(path: str) -> bool:
    normalized = path.lower()
    if is_test_path(normalized):
        return False
    return normalized.endswith((".py", ".js", ".ts", ".tsx", ".jsx"))


@dataclass(frozen=True)
class MissingTestChangeRule:
    @property
    def metadata(self) -> RuleMetadata:
        return RuleMetadata(
            rule_id="rule.missing-tests",
            name="Production change without tests",
            description="Flags production code changes when no test files changed.",
            category="test",
            default_severity="P2",
        )

    def run(self, context: RuleContext) -> list[Finding]:
        paths = context.changed_paths
        production_paths = [path for path in paths if is_production_path(path)]
        test_paths = [path for path in paths if is_test_path(path)]
        if not production_paths or test_paths:
            return []
        return [
            Finding(
                message="Production code changed without a staged test change.",
                file_path=production_paths[0],
                severity=self.metadata.default_severity,
                category=self.metadata.category,
                source="rule",
                confidence="medium",
                rule_id=self.metadata.rule_id,
                evidence={
                    "production_files": production_paths,
                    "test_files": test_paths,
                },
                suggestion="Add or update a focused test, or document why no test is needed.",
            )
        ]
