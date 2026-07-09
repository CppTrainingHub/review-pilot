from __future__ import annotations

from pathlib import Path

from review_pilot.command_runner import CommandRunnerError, run_registered_tool
from review_pilot.tool_models import CommandSpec, RegisteredTool, ToolSpec


def make_tool(
    name: str,
    argv: tuple[str, ...],
    *,
    enabled: bool = True,
    timeout_seconds: int = 5,
    cwd: str = ".",
) -> RegisteredTool:
    return RegisteredTool(
        spec=ToolSpec(
            name=name,
            description="test tool",
            command=CommandSpec(argv=argv, cwd=cwd),
            project_types=("python",),
        ),
        enabled=enabled,
        timeout_seconds=timeout_seconds,
        reason="test",
    )


def test_run_registered_tool_captures_stdout_stderr_and_exit_code(tmp_path) -> None:
    tool = make_tool(
        "python-script",
        (
            "python",
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
        ),
    )

    result = run_registered_tool(tool, tmp_path, raw_output_dir=tmp_path / "raw")

    assert result.tool_name == "python-script"
    assert result.exit_code == 3
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.timed_out is False
    assert Path(result.stdout_path).read_text(encoding="utf-8") == "out\n"
    assert Path(result.stderr_path).read_text(encoding="utf-8") == "err\n"


def test_run_registered_tool_records_timeout(tmp_path) -> None:
    tool = make_tool(
        "slow",
        ("python", "-c", "import time; time.sleep(2)"),
        timeout_seconds=1,
    )

    result = run_registered_tool(tool, tmp_path, raw_output_dir=tmp_path / "raw")

    assert result.exit_code == 124
    assert result.timed_out is True
    assert "command timed out after 1s" in result.stderr


def test_run_registered_tool_refuses_disabled_tool(tmp_path) -> None:
    tool = make_tool("disabled", ("python", "--version"), enabled=False)

    try:
        run_registered_tool(tool, tmp_path)
    except CommandRunnerError as exc:
        assert "tool is disabled: disabled" in str(exc)
    else:
        raise AssertionError("expected disabled tool to fail")


def test_run_registered_tool_refuses_cwd_escape(tmp_path) -> None:
    tool = make_tool("escape", ("python", "--version"), cwd="..")

    try:
        run_registered_tool(tool, tmp_path)
    except ValueError as exc:
        assert "escapes repository root" in str(exc)
    else:
        raise AssertionError("expected cwd escape to fail")


def test_command_spec_requires_non_empty_argv() -> None:
    try:
        CommandSpec(())
    except ValueError as exc:
        assert "command argv must contain non-empty items" in str(exc)
    else:
        raise AssertionError("expected empty argv to fail")
