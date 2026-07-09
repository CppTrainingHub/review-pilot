from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

from .tool_models import CommandResult, RegisteredTool


class CommandRunnerError(ValueError):
    """Raised when a registered tool cannot be executed."""


def run_registered_tool(
    tool: RegisteredTool,
    repo_root: str | Path,
    raw_output_dir: str | Path | None = None,
) -> CommandResult:
    if not tool.enabled:
        raise CommandRunnerError(f"tool is disabled: {tool.spec.name}")

    root = Path(repo_root).resolve()
    cwd = tool.spec.command.resolved_cwd(root)
    if not cwd.exists() or not cwd.is_dir():
        raise CommandRunnerError(f"command cwd does not exist: {tool.spec.command.cwd}")

    output_dir = Path(raw_output_dir) if raw_output_dir is not None else root / ".review-pilot" / "raw-outputs"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    stdout = ""
    stderr = ""
    exit_code = 0
    timed_out = False
    try:
        completed = subprocess.run(
            list(tool.spec.command.argv),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=tool.timeout_seconds,
            shell=False,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = _decode_timeout_output(exc.stdout)
        stderr = _decode_timeout_output(exc.stderr)
        if stderr:
            stderr = f"{stderr.rstrip()}\n"
        stderr += f"command timed out after {tool.timeout_seconds}s\n"

    duration_ms = max(round((time.monotonic() - started) * 1000), 0)
    prefix = _safe_output_prefix(tool.spec.name)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    stdout_path = output_dir / f"{stamp}-{prefix}.stdout"
    stderr_path = output_dir / f"{stamp}-{prefix}.stderr"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    return CommandResult(
        tool_name=tool.spec.name,
        command=tool.spec.command.argv,
        cwd=str(cwd),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        timed_out=timed_out,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _decode_timeout_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _safe_output_prefix(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in name)
