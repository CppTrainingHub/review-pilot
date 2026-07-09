from __future__ import annotations

from review_pilot.diff_parser import parse_unified_diff
from review_pilot.models import RawDiff
from review_pilot.simple_checks import FileSizeCheck, SimpleCheckRunner


def build_diff(line_count: int) -> RawDiff:
    lines = [
        "diff --git a/app.py b/app.py",
        "--- a/app.py",
        "+++ b/app.py",
        f"@@ -1 +1,{line_count} @@",
    ]
    lines.extend(f"+line {i}" for i in range(line_count))
    return RawDiff("\n".join(lines))


def test_file_size_check_flags_file_exceeding_threshold() -> None:
    parsed = parse_unified_diff(build_diff(250))

    findings = FileSizeCheck(max_added_lines=200).run(parsed)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.file_path == "app.py"
    assert finding.severity == "P2"
    assert finding.category == "size"
    assert finding.source == "simple-check"
    assert finding.rule_id == "simple-check.file-too-large"
    assert finding.evidence == {"added_lines": 250, "threshold": 200}


def test_file_size_check_ignores_files_under_threshold() -> None:
    parsed = parse_unified_diff(build_diff(50))

    findings = FileSizeCheck(max_added_lines=200).run(parsed)

    assert findings == []


def test_file_size_check_uses_threshold_at_boundary() -> None:
    parsed = parse_unified_diff(build_diff(201))

    findings = FileSizeCheck(max_added_lines=200).run(parsed)

    assert len(findings) == 1
    assert findings[0].evidence["added_lines"] == 201


def test_runner_returns_combined_findings() -> None:
    parsed = parse_unified_diff(build_diff(250))

    findings = SimpleCheckRunner(FileSizeCheck(max_added_lines=200)).run(parsed)

    assert len(findings) == 1
    assert findings[0].category == "size"
