from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable

from review_pilot.config import ReviewPilotConfig
from review_pilot.models import ParsedDiff, RepoInfo
from review_pilot.finding_normalizer import normalize_findings
from review_pilot.report_models import Finding
from review_pilot.rules.content_rules import DEFAULT_DEBUG_PATTERNS
from review_pilot.rules.path_rules import DEFAULT_SENSITIVE_MARKERS
from review_pilot.rules import (
    ChangeTooLargeRule,
    DebugOutputRule,
    FileTooLargeRule,
    MissingTestChangeRule,
    Rule,
    RuleContext,
    SensitivePathRule,
)


def default_rules(config: ReviewPilotConfig | None = None) -> list[Rule]:
    active_config = config or ReviewPilotConfig.default()
    rules: list[Rule] = []

    file_too_large = active_config.rule("rule.file-too-large")
    if file_too_large.enabled:
        rules.append(
            FileTooLargeRule(
                max_added_lines=file_too_large.max_added_lines
                or FileTooLargeRule().max_added_lines
            )
        )

    change_too_large = active_config.rule("rule.change-too-large")
    if change_too_large.enabled:
        rules.append(
            ChangeTooLargeRule(
                max_total_added_lines=change_too_large.max_total_added_lines
                or ChangeTooLargeRule().max_total_added_lines
            )
        )

    missing_tests = active_config.rule("rule.missing-tests")
    if missing_tests.enabled:
        rules.append(MissingTestChangeRule())

    debug_output = active_config.rule("rule.debug-output")
    if debug_output.enabled:
        rules.append(DebugOutputRule(patterns=debug_output.patterns or DEFAULT_DEBUG_PATTERNS))

    sensitive_path = active_config.rule("rule.sensitive-path")
    if sensitive_path.enabled:
        rules.append(
            SensitivePathRule(markers=sensitive_path.markers or DEFAULT_SENSITIVE_MARKERS)
        )

    return rules


@dataclass(frozen=True)
class RuleEngine:
    rules: tuple[Rule, ...] = field(default_factory=lambda: tuple(default_rules()))
    config: ReviewPilotConfig = field(default_factory=ReviewPilotConfig.default)

    def __init__(
        self,
        rules: Iterable[Rule] | None = None,
        config: ReviewPilotConfig | None = None,
    ) -> None:
        active_config = config or ReviewPilotConfig.default()
        object.__setattr__(self, "config", active_config)
        object.__setattr__(
            self,
            "rules",
            tuple(default_rules(active_config) if rules is None else rules),
        )

    def run(self, parsed_diff: ParsedDiff, repo_info: RepoInfo | None = None) -> list[Finding]:
        context = RuleContext(
            parsed_diff=parsed_diff,
            repo_info=repo_info,
            config=self.config,
        )
        findings: list[Finding] = []
        for rule in self.rules:
            findings.extend(rule.run(context))
        return normalize_findings(findings)


def default_rule_engine(config: ReviewPilotConfig | None = None) -> RuleEngine:
    return RuleEngine(config=config)
