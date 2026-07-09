from __future__ import annotations

from dataclasses import dataclass

from review_pilot.diff_parser import parse_unified_diff
from review_pilot.models import RawDiff
from review_pilot.config import ReviewPilotConfig, RuleConfig
from review_pilot.report_models import Finding
from review_pilot.rule_engine import RuleEngine, default_rules
from review_pilot.rules.base import RuleContext, RuleMetadata


def parsed_diff(text: str):
    return parse_unified_diff(RawDiff(text))


@dataclass(frozen=True)
class FakeRule:
    rule_id: str
    file_path: str

    @property
    def metadata(self) -> RuleMetadata:
        return RuleMetadata(
            rule_id=self.rule_id,
            name=self.rule_id,
            description="fake rule",
            category="other",
            default_severity="P2",
        )

    def run(self, context: RuleContext) -> list[Finding]:
        return [
            Finding(
                message=f"{self.rule_id} hit",
                file_path=self.file_path,
                source="rule",
                rule_id=self.rule_id,
            )
        ]


def test_rule_engine_runs_all_registered_rules() -> None:
    diff = parsed_diff(
        "\n".join(
            [
                "diff --git a/app.py b/app.py",
                "--- a/app.py",
                "+++ b/app.py",
                "@@ -1 +1 @@",
                "+print('hello')",
            ]
        )
    )
    engine = RuleEngine([FakeRule("rule.one", "app.py"), FakeRule("rule.two", "app.py")])

    findings = engine.run(diff)

    assert [finding.rule_id for finding in findings] == ["rule.one", "rule.two"]
    assert all(finding.source == "rule" for finding in findings)


def test_rule_engine_allows_empty_rule_list() -> None:
    diff = parsed_diff("")

    findings = RuleEngine([]).run(diff)

    assert findings == []


def test_rule_engine_normalizes_duplicate_findings() -> None:
    diff = parsed_diff(
        "\n".join(
            [
                "diff --git a/app.py b/app.py",
                "--- a/app.py",
                "+++ b/app.py",
                "@@ -1 +1 @@",
                "+print('hello')",
            ]
        )
    )
    engine = RuleEngine([FakeRule("rule.one", "app.py"), FakeRule("rule.one", "app.py")])

    findings = engine.run(diff)

    assert len(findings) == 1
    assert findings[0].rule_id == "rule.one"
    assert findings[0].evidence == {"duplicate_count": 2}


def test_default_rules_have_stable_order_and_ids() -> None:
    rule_ids = [rule.metadata.rule_id for rule in default_rules()]

    assert rule_ids == [
        "rule.file-too-large",
        "rule.change-too-large",
        "rule.missing-tests",
        "rule.debug-output",
        "rule.sensitive-path",
    ]


def test_default_rules_respect_disabled_rule_config() -> None:
    config = ReviewPilotConfig(
        rules={
            "rule.debug-output": RuleConfig(enabled=False),
        }
    )

    rule_ids = [rule.metadata.rule_id for rule in default_rules(config)]

    assert "rule.debug-output" not in rule_ids


def test_default_rules_use_threshold_override() -> None:
    diff = parsed_diff(
        "\n".join(
            [
                "diff --git a/app.py b/app.py",
                "--- /dev/null",
                "+++ b/app.py",
                "@@ -0,0 +1,2 @@",
                "+a = 1",
                "+b = 2",
            ]
        )
    )
    config = ReviewPilotConfig(
        rules={
            "rule.file-too-large": RuleConfig(max_added_lines=1),
        }
    )

    findings = RuleEngine(config=config).run(diff)

    assert any(finding.rule_id == "rule.file-too-large" for finding in findings)
