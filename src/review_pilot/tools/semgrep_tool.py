from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from typing import Any

from ..command_runner import CommandRunnerError, run_registered_tool
from ..report_models import Finding
from ..tool_models import RegisteredTool, ToolResult


SEMGREP_TOOL_NAME = "semgrep"


def is_semgrep_available() -> bool:
    return shutil.which("semgrep") is not None


def run_semgrep_tool(tool: RegisteredTool, repo_root: str) -> ToolResult:
    if tool.spec.name != SEMGREP_TOOL_NAME:
        raise ValueError(f"expected semgrep tool, got {tool.spec.name}")

    if not is_semgrep_available():
        return ToolResult(
            tool_name=SEMGREP_TOOL_NAME,
            status="missing",
            error="semgrep executable not found",
        )

    try:
        command_result = run_registered_tool(tool, repo_root)
    except (CommandRunnerError, ValueError) as exc:
        return ToolResult(
            tool_name=SEMGREP_TOOL_NAME,
            status="failed",
            error=str(exc),
        )

    if command_result.timed_out:
        return ToolResult(
            tool_name=SEMGREP_TOOL_NAME,
            status="timeout",
            command_result=command_result,
            error="semgrep scan timed out",
        )

    if command_result.exit_code not in {0, 1}:
        return ToolResult(
            tool_name=SEMGREP_TOOL_NAME,
            status="failed",
            command_result=command_result,
            error=f"semgrep exited with code {command_result.exit_code}",
        )

    try:
        findings, raw_findings = parse_semgrep_json(command_result.stdout)
    except ValueError as exc:
        return ToolResult(
            tool_name=SEMGREP_TOOL_NAME,
            status="failed",
            command_result=command_result,
            error=str(exc),
        )

    return ToolResult(
        tool_name=SEMGREP_TOOL_NAME,
        status="success",
        findings=tuple(findings),
        raw_findings=tuple(raw_findings),
        command_result=command_result,
    )


def parse_semgrep_json(text: str) -> tuple[list[Finding], list[dict[str, Any]]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid semgrep JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("invalid semgrep JSON: root must be an object")

    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("invalid semgrep JSON: results must be a list")

    findings: list[Finding] = []
    raw_findings: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid semgrep JSON: results[{index}] must be an object")
        raw_dict = dict(raw)
        raw_findings.append(raw_dict)
        findings.append(_semgrep_result_to_finding(raw_dict))
    return findings, raw_findings


def _semgrep_result_to_finding(result: Mapping[str, Any]) -> Finding:
    extra = _require_mapping(result.get("extra", {}), "extra")
    start = _require_mapping(result.get("start", {}), "start")
    metadata = _require_mapping(extra.get("metadata", {}), "extra.metadata")
    check_id = str(result.get("check_id") or "semgrep.unknown")
    path = str(result.get("path") or "")
    if not path:
        raise ValueError("invalid semgrep JSON: result path is required")

    message = str(extra.get("message") or check_id)
    severity = _map_semgrep_severity(str(extra.get("severity") or "WARNING"))
    category = _map_semgrep_category(metadata)
    line_no = _coerce_line_no(start.get("line"))
    confidence = _map_confidence(metadata)

    return Finding(
        message=message,
        file_path=path,
        line_no=line_no,
        severity=severity,
        category=category,
        source="semgrep",
        confidence=confidence,
        rule_id=check_id,
        evidence={
            "semgrep_check_id": check_id,
            "semgrep_severity": extra.get("severity"),
            "metadata": dict(metadata),
        },
    )


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid semgrep JSON: {path} must be an object")
    return value


def _coerce_line_no(value: object) -> int | None:
    if isinstance(value, int) and value >= 1:
        return value
    return None


def _map_semgrep_severity(severity: str) -> str:
    normalized = severity.upper()
    if normalized == "ERROR":
        return "P1"
    if normalized == "WARNING":
        return "P2"
    return "P3"


def _map_semgrep_category(metadata: Mapping[str, Any]) -> str:
    category = str(metadata.get("category") or "").lower()
    if category == "security":
        return "security"
    if category in {"bug", "correctness"}:
        return "bug"
    if category in {"maintainability", "performance"}:
        return "maintainability"
    if category == "style":
        return "style"
    return "other"


def _map_confidence(metadata: Mapping[str, Any]) -> str:
    confidence = str(metadata.get("confidence") or "").lower()
    if confidence in {"high", "medium", "low"}:
        return confidence
    return "medium"
