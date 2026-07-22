from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .models import DiffFile, ParsedDiff


MAX_FILE_READ_LINES = 500
MAX_SEARCH_RESULTS = 100
MAX_FIND_RESULTS = 100
MAX_DIFF_LINES = 500
MAX_SEARCH_LINE_CHARS = 400

SUPPORTED_CONTEXT_TOOLS = (
    "file_read",
    "code_search",
    "file_read_diff",
    "file_find",
)


class ContextToolError(ValueError):
    """A user-visible, structured error from a read-only context tool."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ContextToolRequest:
    tool_name: str
    arguments: Mapping[str, Any]
    call_id: str | None = None

    def __post_init__(self) -> None:
        if self.tool_name not in SUPPORTED_CONTEXT_TOOLS:
            raise ContextToolError(
                "unknown_tool",
                f"unknown context tool: {self.tool_name}",
            )
        if not isinstance(self.arguments, Mapping):
            raise ContextToolError(
                "invalid_arguments",
                "tool arguments must be a JSON object",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "call_id": self.call_id,
        }


@dataclass(frozen=True)
class ContextToolResult:
    tool_name: str
    content: str
    returned_count: int = 0
    truncated: bool = False
    error_code: str | None = None
    error_message: str | None = None
    call_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "returned_count": self.returned_count,
            "truncated": self.truncated,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "content": self.content,
        }


@dataclass(frozen=True)
class DynamicContextTrace:
    round: int
    tool_name: str
    arguments: dict[str, Any]
    call_id: str | None
    status: str
    returned_count: int
    truncated: bool
    elapsed_ms: float
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "status": self.status,
            "returned_count": self.returned_count,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


class ContextToolExecutor:
    """Execute the four read-only context tools inside one repository root."""

    def __init__(self, root: str | Path, parsed_diff: ParsedDiff | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.parsed_diff = parsed_diff
        if not self.root.is_dir():
            raise ContextToolError(
                "workspace_not_found",
                f"workspace is not a directory: {self.root}",
            )

    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "file_read",
                    "description": (
                        "Read a bounded text range from one repository-relative file. "
                        "Use this when the current review evidence needs exact file content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "code_search",
                    "description": (
                        "Search literal text in repository files. The result is capped at "
                        "100 matches and each match includes a relative path and line number."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read_diff",
                    "description": (
                        "Read the parsed unified diff for one changed repository-relative file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_find",
                    "description": (
                        "Find repository files by a glob pattern. The result is capped at "
                        "100 relative paths."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(self, request: ContextToolRequest) -> ContextToolResult:
        try:
            if request.tool_name == "file_read":
                payload, count, truncated = self._file_read(request.arguments)
            elif request.tool_name == "code_search":
                payload, count, truncated = self._code_search(request.arguments)
            elif request.tool_name == "file_read_diff":
                payload, count, truncated = self._file_read_diff(request.arguments)
            elif request.tool_name == "file_find":
                payload, count, truncated = self._file_find(request.arguments)
            else:  # The dataclass validates this; keep the dispatch defensive.
                raise ContextToolError("unknown_tool", request.tool_name)
        except ContextToolError as exc:
            content = json.dumps(
                {"error": {"code": exc.code, "message": str(exc)}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return ContextToolResult(
                tool_name=request.tool_name,
                call_id=request.call_id,
                content=content,
                error_code=exc.code,
                error_message=str(exc),
            )
        return ContextToolResult(
            tool_name=request.tool_name,
            call_id=request.call_id,
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            returned_count=count,
            truncated=truncated,
        )

    def _file_read(self, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], int, bool]:
        path = self._required_path(arguments, "path")
        start_line = self._positive_int(arguments.get("start_line", 1), "start_line")
        end_line_value = arguments.get("end_line")
        end_line = (
            self._positive_int(end_line_value, "end_line")
            if end_line_value is not None
            else start_line + MAX_FILE_READ_LINES - 1
        )
        if end_line < start_line:
            raise ContextToolError("invalid_arguments", "end_line must be >= start_line")
        target = self._resolve_file(path)
        lines = self._read_text_lines(target)
        requested_end = min(end_line, start_line + MAX_FILE_READ_LINES - 1)
        selected = lines[start_line - 1 : requested_end]
        truncated = end_line > requested_end or requested_end < len(lines) and end_line >= len(lines)
        payload = {
            "path": path,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else start_line - 1,
            "total_lines": len(lines),
            "lines": [
                {"line": start_line + index, "text": text}
                for index, text in enumerate(selected)
            ],
            "truncated": truncated,
        }
        return payload, len(selected), truncated

    def _code_search(self, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], int, bool]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            raise ContextToolError("invalid_arguments", "query must be a non-empty string")
        scope = self._resolve_scope(arguments.get("path", "."))
        matches: list[dict[str, Any]] = []
        truncated = False
        for target in self._iter_text_candidates(scope):
            relative = self._relative(target)
            try:
                lines = self._read_text_lines(target)
            except ContextToolError as exc:
                if exc.code == "binary_file":
                    continue
                raise
            for line_no, text in enumerate(lines, start=1):
                if query not in text:
                    continue
                matches.append(
                    {
                        "path": relative,
                        "line": line_no,
                        "text": text[:MAX_SEARCH_LINE_CHARS],
                    }
                )
                if len(matches) >= MAX_SEARCH_RESULTS:
                    truncated = True
                    break
            if truncated:
                break
        return {
            "query": query,
            "matches": matches,
            "truncated": truncated,
        }, len(matches), truncated

    def _file_read_diff(self, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], int, bool]:
        path = self._required_path(arguments, "path")
        if self.parsed_diff is None:
            raise ContextToolError(
                "diff_unavailable",
                "file_read_diff requires a parsed diff",
            )
        normalized = _normalize_relative(path)
        for diff_file in self.parsed_diff.files:
            if _normalize_relative(diff_file.path) != normalized:
                continue
            payload, count, truncated = _limited_diff_payload(diff_file, path)
            return payload, count, truncated
        raise ContextToolError("not_changed", f"file is not in the parsed diff: {path}")

    def _file_find(self, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], int, bool]:
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ContextToolError("invalid_arguments", "pattern must be a non-empty string")
        scope = self._resolve_scope(arguments.get("path", "."))
        matches: list[str] = []
        truncated = False
        for target in self._iter_files(scope):
            relative = self._relative(target)
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(target.name, pattern):
                matches.append(relative)
                if len(matches) >= MAX_FIND_RESULTS:
                    truncated = True
                    break
        return {
            "pattern": pattern,
            "files": matches,
            "truncated": truncated,
        }, len(matches), truncated

    def _required_path(self, arguments: Mapping[str, Any], name: str) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ContextToolError("invalid_arguments", f"{name} must be a non-empty string")
        return value.strip().replace("\\", "/")

    def _positive_int(self, value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ContextToolError("invalid_arguments", f"{name} must be a positive integer")
        return value

    def _resolve_file(self, path: str) -> Path:
        candidate = self._resolve_path(path)
        if not candidate.exists():
            raise ContextToolError("not_found", f"file does not exist: {path}")
        if candidate.is_dir():
            raise ContextToolError("directory_not_allowed", f"path is a directory: {path}")
        return candidate

    def _resolve_scope(self, path: Any) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ContextToolError("invalid_arguments", "path must be a non-empty string")
        candidate = self._resolve_path(path.strip().replace("\\", "/"), allow_dot=True)
        if not candidate.exists():
            raise ContextToolError("not_found", f"path does not exist: {path}")
        return candidate

    def _resolve_path(self, path: str, *, allow_dot: bool = False) -> Path:
        normalized = _normalize_relative(path, allow_dot=allow_dot)
        if normalized == ".":
            return self.root
        candidate = (self.root / Path(*PurePosixPath(normalized).parts)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ContextToolError(
                "path_out_of_workspace",
                f"path escapes workspace: {path}",
            )
        return candidate

    def _iter_files(self, scope: Path):
        if scope.is_file():
            yield scope
            return
        for target in sorted(scope.rglob("*")):
            if ".git" in target.relative_to(self.root).parts:
                continue
            resolved = target.resolve()
            if resolved != self.root and self.root not in resolved.parents:
                continue
            if resolved.is_file():
                yield resolved

    def _iter_text_candidates(self, scope: Path):
        for target in self._iter_files(scope):
            if target.stat().st_size > 2 * 1024 * 1024:
                continue
            yield target

    def _read_text_lines(self, target: Path) -> list[str]:
        try:
            raw = target.read_bytes()
        except OSError as exc:
            raise ContextToolError("read_error", f"cannot read {self._relative(target)}: {exc}") from exc
        if b"\x00" in raw:
            raise ContextToolError("binary_file", f"binary file is not readable: {self._relative(target)}")
        try:
            return raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ContextToolError("binary_file", f"binary file is not readable: {self._relative(target)}") from exc

    def _relative(self, target: Path) -> str:
        return target.resolve().relative_to(self.root).as_posix()


def _normalize_relative(path: str, *, allow_dot: bool = False) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ContextToolError("invalid_arguments", "path must be a non-empty relative path")
    normalized = path.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ContextToolError(
            "path_out_of_workspace",
            f"path must stay inside workspace: {path}",
        )
    cleaned = "/".join(part for part in pure.parts if part not in ("", "."))
    if not cleaned:
        if allow_dot:
            return "."
        raise ContextToolError("invalid_arguments", "file path must name a file")
    return cleaned


def _limited_diff_payload(
    diff_file: DiffFile,
    requested_path: str,
) -> tuple[dict[str, Any], int, bool]:
    total_lines = sum(len(hunk.lines) for hunk in diff_file.hunks)
    if total_lines <= MAX_DIFF_LINES:
        return {
            "path": requested_path,
            "diff": diff_file.to_dict(),
            "truncated": False,
        }, total_lines, False

    remaining = MAX_DIFF_LINES
    hunks: list[dict[str, Any]] = []
    for hunk in diff_file.hunks:
        lines = list(hunk.lines[:remaining])
        if lines:
            payload = hunk.to_dict()
            payload["lines"] = [line.to_dict() for line in lines]
            hunks.append(payload)
            remaining -= len(lines)
        if remaining == 0:
            break
    diff_payload = diff_file.to_dict()
    diff_payload["hunks"] = hunks
    return {
        "path": requested_path,
        "diff": diff_payload,
        "truncated": True,
    }, MAX_DIFF_LINES, True
