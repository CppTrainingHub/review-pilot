from __future__ import annotations

from review_pilot.diff_parser import parse_unified_diff
from review_pilot.models import RawDiff
from review_pilot.rules import (
    ChangeTooLargeRule,
    DebugOutputRule,
    FileTooLargeRule,
    MissingTestChangeRule,
    RuleContext,
    SensitivePathRule,
)


def parse(lines: list[str]):
    return parse_unified_diff(RawDiff("\n".join(lines)))


def added_file(path: str, added_lines: list[str]):
    return [
        f"diff --git a/{path} b/{path}",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(added_lines)} @@",
        *[f"+{line}" for line in added_lines],
    ]


def run_rule(rule, lines: list[str]):
    return rule.run(RuleContext(parse(lines)))


def test_file_too_large_rule_flags_large_single_file() -> None:
    lines = added_file("app.py", [f"x = {i}" for i in range(6)])

    findings = run_rule(FileTooLargeRule(max_added_lines=5), lines)

    assert len(findings) == 1
    assert findings[0].rule_id == "rule.file-too-large"
    assert findings[0].source == "rule"
    assert findings[0].evidence == {"added_lines": 6, "threshold": 5}


def test_change_too_large_rule_flags_total_added_lines() -> None:
    lines = added_file("app.py", [f"x = {i}" for i in range(3)])
    lines += added_file("lib.py", [f"y = {i}" for i in range(3)])

    findings = run_rule(ChangeTooLargeRule(max_total_added_lines=5), lines)

    assert len(findings) == 1
    assert findings[0].rule_id == "rule.change-too-large"
    assert findings[0].file_path is None
    assert findings[0].evidence == {"added_lines": 6, "threshold": 5}


def test_missing_test_change_rule_flags_source_without_tests() -> None:
    lines = added_file("src/service.py", ["def run():", "    return 1"])

    findings = run_rule(MissingTestChangeRule(), lines)

    assert len(findings) == 1
    assert findings[0].rule_id == "rule.missing-tests"
    assert findings[0].confidence == "medium"
    assert findings[0].evidence == {
        "production_files": ["src/service.py"],
        "test_files": [],
    }


def test_missing_test_change_rule_ignores_when_test_file_changed() -> None:
    lines = added_file("src/service.py", ["def run():", "    return 1"])
    lines += added_file("tests/test_service.py", ["def test_run():", "    assert True"])

    findings = run_rule(MissingTestChangeRule(), lines)

    assert findings == []


def test_debug_output_rule_flags_added_debug_patterns() -> None:
    lines = added_file("src/app.py", ["def run():", "    print('debug')"])

    findings = run_rule(DebugOutputRule(), lines)

    assert len(findings) == 1
    assert findings[0].rule_id == "rule.debug-output"
    assert findings[0].line_no == 2
    assert findings[0].evidence == {"matched_pattern": "print("}


def test_sensitive_path_rule_flags_dependency_or_deploy_paths() -> None:
    lines = added_file("requirements.txt", ["requests==2.0.0"])

    findings = run_rule(SensitivePathRule(), lines)

    assert len(findings) == 1
    assert findings[0].rule_id == "rule.sensitive-path"
    assert findings[0].file_path == "requirements.txt"
    assert findings[0].evidence == {"matched_marker": "requirements.txt"}
