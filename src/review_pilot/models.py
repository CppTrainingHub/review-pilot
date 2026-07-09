from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RepoInfo:
    root: str
    branch: str
    head: str
    has_staged_changes: bool
    has_unstaged_changes: bool


@dataclass(frozen=True)
class RawDiff:
    text: str

    @property
    def is_empty(self) -> bool:
        return self.text == ""


@dataclass(frozen=True)
class DiffLine:
    kind: str
    content: str
    old_line_no: int | None
    new_line_no: int | None
    no_newline_at_eof: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "content": self.content,
            "old_line_no": self.old_line_no,
            "new_line_no": self.new_line_no,
            "no_newline_at_eof": self.no_newline_at_eof,
        }


@dataclass(frozen=True)
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: tuple[DiffLine, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "old_start": self.old_start,
            "old_count": self.old_count,
            "new_start": self.new_start,
            "new_count": self.new_count,
            "section": self.section,
            "lines": [line.to_dict() for line in self.lines],
        }


@dataclass(frozen=True)
class DiffFile:
    old_path: str | None
    new_path: str | None
    change_type: str
    hunks: tuple[DiffHunk, ...]

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    def to_dict(self) -> dict[str, object]:
        return {
            "old_path": self.old_path,
            "new_path": self.new_path,
            "path": self.path,
            "change_type": self.change_type,
            "hunks": [hunk.to_dict() for hunk in self.hunks],
        }


@dataclass(frozen=True)
class ParsedDiff:
    files: tuple[DiffFile, ...]

    @property
    def is_empty(self) -> bool:
        return len(self.files) == 0

    def to_dict(self) -> dict[str, object]:
        return {"files": [file.to_dict() for file in self.files]}


@dataclass(frozen=True)
class IndexedFile:
    path: str
    language: str
    size_bytes: int
    is_test: bool
    imports: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "size_bytes": self.size_bytes,
            "is_test": self.is_test,
            "imports": list(self.imports),
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True)
class CodeIndex:
    root: str
    files: tuple[IndexedFile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files": [file.to_dict() for file in self.files],
        }


@dataclass(frozen=True)
class ContextCandidate:
    path: str
    reason: str
    priority: int
    language: str
    is_changed: bool = False
    is_test: bool = False
    matched_symbols: tuple[str, ...] = ()
    matched_imports: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "priority": self.priority,
            "language": self.language,
            "is_changed": self.is_changed,
            "is_test": self.is_test,
            "matched_symbols": list(self.matched_symbols),
            "matched_imports": list(self.matched_imports),
        }


@dataclass(frozen=True)
class ContextCandidateManifest:
    changed_paths: tuple[str, ...]
    candidates: tuple[ContextCandidate, ...]
    index_file_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_paths": list(self.changed_paths),
            "index_file_count": self.index_file_count,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class ContextSlice:
    path: str
    reason: str
    priority: int
    language: str
    start_line: int
    end_line: int
    estimated_tokens: int
    content: str
    is_changed: bool = False
    is_test: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "priority": self.priority,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "estimated_tokens": self.estimated_tokens,
            "content": self.content,
            "is_changed": self.is_changed,
            "is_test": self.is_test,
        }


@dataclass(frozen=True)
class OmittedContext:
    path: str
    reason: str
    priority: int
    language: str
    omitted_reason: str
    estimated_tokens: int = 0
    omitted_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "priority": self.priority,
            "language": self.language,
            "omitted_reason": self.omitted_reason,
            "estimated_tokens": self.estimated_tokens,
            "omitted_lines": self.omitted_lines,
        }


@dataclass(frozen=True)
class ContextBudgetManifest:
    changed_paths: tuple[str, ...]
    max_context_tokens: int
    used_tokens: int
    index_file_count: int
    context_used: tuple[ContextSlice, ...]
    context_omitted: tuple[OmittedContext, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_paths": list(self.changed_paths),
            "max_context_tokens": self.max_context_tokens,
            "used_tokens": self.used_tokens,
            "remaining_tokens": max(self.max_context_tokens - self.used_tokens, 0),
            "index_file_count": self.index_file_count,
            "context_used": [item.to_dict() for item in self.context_used],
            "context_omitted": [item.to_dict() for item in self.context_omitted],
        }
