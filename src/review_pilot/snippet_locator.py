from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


MATCHED = "matched"
DOWNGRADED = "downgraded"
DROPPED = "dropped"


@dataclass(frozen=True)
class LocationDecision:
    """The deterministic result of locating a finding's source snippet."""

    status: str
    file_path: str | None
    line_no: int | None
    source: str | None
    match_count: int
    reason: str
    matched_text: str | None = None
    match_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "file_path": self.file_path,
            "line_no": self.line_no,
            "source": self.source,
            "match_count": self.match_count,
            "reason": self.reason,
            "matched_text": self.matched_text,
            "match_mode": self.match_mode,
        }


@dataclass(frozen=True)
class _SourceLine:
    content: str
    line_no: int
    kind: str


class SnippetLocator:
    """Locate exact-ish copied code in a diff hunk or the new file.

    The matching rule is deliberately small: line boundaries must stay in the
    same order, line endings are normalized, and leading/trailing indentation
    plus repeated horizontal whitespace are ignored. Tokens are never changed
    or guessed. A unique hunk match is verified; a unique full-file match is
    retained as a downgraded decision because it is no longer tied to a changed
    line.
    """

    def __init__(
        self,
        diff: Mapping[str, Any] | Any,
        workspace_root: str | Path | None = None,
    ) -> None:
        self._diff = _as_diff_payload(diff)
        self._workspace_root = Path(workspace_root or ".").resolve()

    def locate(
        self,
        file_path: str | Any,
        existing_code: str | None = None,
    ) -> LocationDecision:
        if existing_code is None and not isinstance(file_path, str):
            finding = file_path
            file_path = getattr(finding, "file_path", None)
            existing_code = getattr(finding, "existing_code", None)

        path = _normalize_path(file_path)
        if path is None:
            return LocationDecision(
                status=DROPPED,
                file_path=None,
                line_no=None,
                source=None,
                match_count=0,
                reason="file_path must be a repository-relative POSIX path",
            )
        snippet_lines = _split_code(existing_code)
        if not snippet_lines:
            return LocationDecision(
                status=DROPPED,
                file_path=path,
                line_no=None,
                source=None,
                match_count=0,
                reason="existing_code must contain at least one non-empty line",
            )

        diff_file = self._find_diff_file(path)
        if diff_file is not None:
            hunk_matches = _find_hunk_matches(diff_file, snippet_lines)
            if len(hunk_matches) > 1:
                return self._ambiguous(path, "multiple_matches_in_hunk", len(hunk_matches))
            if len(hunk_matches) == 1:
                match = hunk_matches[0]
                line_no = _location_line(match)
                matched_text = "\n".join(item.content for item in match)
                if any(item.kind == "added" for item in match):
                    return LocationDecision(
                        status=MATCHED,
                        file_path=path,
                        line_no=line_no,
                        source="hunk",
                        match_count=1,
                        reason="unique existing_code match in diff hunk",
                        matched_text=matched_text,
                        match_mode="whitespace-normalized",
                    )
                return LocationDecision(
                    status=DOWNGRADED,
                    file_path=path,
                    line_no=line_no,
                    source="hunk",
                    match_count=1,
                    reason="hunk match does not include an added line",
                    matched_text=matched_text,
                    match_mode="whitespace-normalized",
                )

        file_lines = self._read_new_file(path)
        if file_lines is None:
            return LocationDecision(
                status=DROPPED,
                file_path=path,
                line_no=None,
                source=None,
                match_count=0,
                reason="existing_code was not found in the diff hunk or new file",
            )
        file_matches = _find_matches(file_lines, snippet_lines)
        if len(file_matches) > 1:
            return self._ambiguous(path, "multiple_matches_in_file", len(file_matches))
        if len(file_matches) == 1:
            match = file_matches[0]
            return LocationDecision(
                status=DOWNGRADED,
                file_path=path,
                line_no=match[0].line_no,
                source="file",
                match_count=1,
                reason="hunk match failed; unique match found in new file",
                matched_text="\n".join(item.content for item in match),
                match_mode="whitespace-normalized",
            )
        return LocationDecision(
            status=DROPPED,
            file_path=path,
            line_no=None,
            source=None,
            match_count=0,
            reason="existing_code was not found in the diff hunk or new file",
        )

    def _find_diff_file(self, path: str) -> Mapping[str, Any] | None:
        for item in self._diff.get("files", []):
            if not isinstance(item, Mapping):
                continue
            candidate = _normalize_path(item.get("path") or item.get("new_path"))
            if candidate == path:
                return item
        return None

    def _read_new_file(self, path: str) -> list[_SourceLine] | None:
        candidate = self._workspace_root / Path(*PurePosixPath(path).parts)
        try:
            candidate.relative_to(self._workspace_root)
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            return None
        return [
            _SourceLine(content=line, line_no=index, kind="file")
            for index, line in enumerate(_split_physical_lines(text), start=1)
        ]

    @staticmethod
    def _ambiguous(path: str, reason: str, count: int) -> LocationDecision:
        return LocationDecision(
            status=DROPPED,
            file_path=path,
            line_no=None,
            source=None,
            match_count=count,
            reason=reason,
        )


def _as_diff_payload(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if hasattr(value, "diff"):
        value = value.diff
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("SnippetLocator requires a diff payload or ReviewContextPack")
    return value


def _hunk_lines(diff_file: Mapping[str, Any]) -> list[_SourceLine]:
    result: list[_SourceLine] = []
    for hunk in diff_file.get("hunks", []):
        if not isinstance(hunk, Mapping):
            continue
        for line in hunk.get("lines", []):
            if not isinstance(line, Mapping):
                continue
            line_no = line.get("new_line_no")
            kind = str(line.get("kind", ""))
            content = line.get("content")
            if isinstance(line_no, int) and kind != "deleted" and isinstance(content, str):
                result.append(_SourceLine(content=content, line_no=line_no, kind=kind))
    return result


def _find_hunk_matches(
    diff_file: Mapping[str, Any],
    snippet_lines: list[str],
) -> list[list[_SourceLine]]:
    matches: list[list[_SourceLine]] = []
    for hunk in diff_file.get("hunks", []):
        if not isinstance(hunk, Mapping):
            continue
        matches.extend(
            _find_matches(
                _hunk_lines({"hunks": [hunk]}),
                snippet_lines,
            )
        )
    return matches


def _find_matches(
    lines: list[_SourceLine],
    snippet_lines: list[str],
) -> list[list[_SourceLine]]:
    if len(snippet_lines) > len(lines):
        return []
    expected = [_normalize_line(line) for line in snippet_lines]
    matches: list[list[_SourceLine]] = []
    for start in range(len(lines) - len(expected) + 1):
        actual = [_normalize_line(item.content) for item in lines[start : start + len(expected)]]
        if actual == expected:
            matches.append(lines[start : start + len(expected)])
    return matches


def _location_line(match: list[_SourceLine]) -> int:
    for item in match:
        if item.kind == "added":
            return item.line_no
    return match[0].line_no


def _split_code(text: str | None) -> list[str]:
    if not isinstance(text, str):
        return []
    lines = _split_physical_lines(text)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _split_physical_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _normalize_line(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line.strip())


def _normalize_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        return None
    raw = value.strip()
    if raw.startswith("/"):
        return None
    path = PurePosixPath(raw)
    if ".." in path.parts:
        return None
    normalized = str(path)
    if normalized in {"", "."}:
        return None
    return normalized[2:] if normalized.startswith("./") else normalized
