from __future__ import annotations

from pathlib import Path

from review_pilot.tool_models import CommandSpec, RegisteredTool, ToolSpec
from review_pilot.tools import semgrep_tool
from review_pilot.tools.semgrep_tool import parse_semgrep_json, run_semgrep_tool


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "semgrep"


def test_parse_semgrep_json_maps_results_to_findings() -> None:
    text = (FIXTURE_DIR / "simple_results.json").read_text(encoding="utf-8")

    findings, raw_findings = parse_semgrep_json(text)

    assert len(findings) == 2
    assert len(raw_findings) == 2
    assert findings[0].source == "semgrep"
    assert findings[0].rule_id == "python.lang.security.audit.subprocess-shell-true.subprocess-shell-true"
    assert findings[0].file_path == "src/app.py"
    assert findings[0].line_no == 12
    assert findings[0].severity == "P1"
    assert findings[0].category == "security"
    assert findings[0].confidence == "high"
    assert findings[1].severity == "P2"
    assert findings[1].category == "bug"
    assert findings[1].confidence == "medium"


def test_parse_semgrep_json_rejects_invalid_json() -> None:
    try:
        parse_semgrep_json("{")
    except ValueError as exc:
        assert "invalid semgrep JSON" in str(exc)
    else:
        raise AssertionError("expected invalid JSON to fail")


def test_parse_semgrep_json_rejects_missing_path() -> None:
    try:
        parse_semgrep_json(
            '{"results":[{"check_id":"x","start":{"line":1},"extra":{"message":"m"}}]}'
        )
    except ValueError as exc:
        assert "result path is required" in str(exc)
    else:
        raise AssertionError("expected missing path to fail")


def test_run_semgrep_tool_reports_missing_executable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(semgrep_tool, "is_semgrep_available", lambda: False)
    tool = make_semgrep_tool()

    result = run_semgrep_tool(tool, str(tmp_path))

    assert result.tool_name == "semgrep"
    assert result.status == "missing"
    assert result.findings == ()
    assert result.error == "semgrep executable not found"


def test_run_semgrep_tool_parses_command_stdout(monkeypatch, tmp_path) -> None:
    fixture = FIXTURE_DIR / "simple_results.json"
    fake_semgrep = tmp_path / "semgrep"
    fake_semgrep.write_text(
        "#!/bin/sh\n"
        f"cat {str(fixture)!r}\n",
        encoding="utf-8",
    )
    fake_semgrep.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/bin:/usr/bin")
    monkeypatch.setattr(semgrep_tool, "is_semgrep_available", lambda: True)
    tool = make_semgrep_tool()

    result = run_semgrep_tool(tool, str(tmp_path))

    assert result.status == "success"
    assert len(result.findings) == 2
    assert len(result.raw_findings) == 2
    assert result.command_result is not None


def make_semgrep_tool() -> RegisteredTool:
    return RegisteredTool(
        spec=ToolSpec(
            name="semgrep",
            description="test semgrep",
            command=CommandSpec(("semgrep", "--config", "auto", "--json", ".")),
            project_types=("python",),
        ),
        enabled=True,
        timeout_seconds=10,
        reason="test",
    )
