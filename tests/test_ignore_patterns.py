from __future__ import annotations

from review_pilot.config import ReviewPilotConfig
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.models import RawDiff
from review_pilot.rule_engine import default_rule_engine


def parse(lines: list[str]):
    return parse_unified_diff(RawDiff("\n".join(lines)))


def added_file(path: str, added_lines: list[str]) -> list[str]:
    return [
        f"diff --git a/{path} b/{path}",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(added_lines)} @@",
        *[f"+{line}" for line in added_lines],
    ]


def test_ignore_pattern_skips_rule_input_without_changing_parsed_diff() -> None:
    diff = parse(
        added_file("generated/client.py", ["print('generated')"])
        + added_file("src/app.py", ["print('debug')"])
    )
    config = ReviewPilotConfig(ignore_paths=("generated/**",))

    findings = default_rule_engine(config).run(diff)

    assert len(diff.files) == 2
    assert [finding.file_path for finding in findings if finding.rule_id == "rule.debug-output"] == [
        "src/app.py"
    ]


def test_ignore_pattern_can_suppress_all_rule_findings() -> None:
    diff = parse(added_file("generated/client.py", ["print('generated')"]))
    config = ReviewPilotConfig(ignore_paths=("generated/**",))

    findings = default_rule_engine(config).run(diff)

    assert findings == []
