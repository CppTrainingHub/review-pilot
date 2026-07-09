from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report_models import Finding


@dataclass(frozen=True)
class ProjectDetection:
    root: str
    project_types: tuple[str, ...]
    markers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "project_types": list(self.project_types),
            "markers": list(self.markers),
        }


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: str = "."

    def __post_init__(self) -> None:
        if not self.argv or any(not item for item in self.argv):
            raise ValueError("command argv must contain non-empty items")

    def resolved_cwd(self, repo_root: str | Path) -> Path:
        root = Path(repo_root).resolve()
        cwd_path = (root / self.cwd).resolve()
        if cwd_path != root and root not in cwd_path.parents:
            raise ValueError(f"command cwd escapes repository root: {self.cwd}")
        return cwd_path

    def to_dict(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "cwd": self.cwd}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    command: CommandSpec
    project_types: tuple[str, ...]
    default_timeout_seconds: int = 30
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "command": self.command.to_dict(),
            "project_types": list(self.project_types),
            "default_timeout_seconds": self.default_timeout_seconds,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    enabled: bool
    timeout_seconds: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = self.spec.to_dict()
        payload.update(
            {
                "enabled": self.enabled,
                "timeout_seconds": self.timeout_seconds,
                "reason": self.reason,
            }
        )
        return payload


@dataclass(frozen=True)
class CommandResult:
    tool_name: str
    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool
    stdout_path: str
    stderr_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "command": list(self.command),
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "raw_outputs": {
                "stdout": self.stdout_path,
                "stderr": self.stderr_path,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    status: str
    findings: tuple[Finding, ...] = ()
    raw_findings: tuple[dict[str, Any], ...] = ()
    command_result: CommandResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "raw_findings": list(self.raw_findings),
            "command_result": (
                self.command_result.to_dict()
                if self.command_result is not None
                else None
            ),
            "error": self.error,
        }
