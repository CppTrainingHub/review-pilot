from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import DiffFile, DiffHunk, DiffLine, ParsedDiff, RawDiff


class DiffParseError(ValueError):
    pass


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<section>.*)$"
)


@dataclass
class _FileBuilder:
    old_path: str | None
    new_path: str | None
    change_type: str = "modified"
    hunks: list[DiffHunk] = field(default_factory=list)

    def to_file(self) -> DiffFile:
        return DiffFile(
            old_path=self.old_path,
            new_path=self.new_path,
            change_type=self.change_type,
            hunks=tuple(self.hunks),
        )


@dataclass
class _HunkBuilder:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: list[DiffLine] = field(default_factory=list)
    old_cursor: int = field(init=False)
    new_cursor: int = field(init=False)

    def __post_init__(self) -> None:
        self.old_cursor = self.old_start
        self.new_cursor = self.new_start

    def add_line(self, raw_line: str) -> None:
        if raw_line.startswith(" "):
            self.lines.append(
                DiffLine(
                    kind="context",
                    content=raw_line[1:],
                    old_line_no=self.old_cursor,
                    new_line_no=self.new_cursor,
                )
            )
            self.old_cursor += 1
            self.new_cursor += 1
            return

        if raw_line.startswith("+"):
            self.lines.append(
                DiffLine(
                    kind="added",
                    content=raw_line[1:],
                    old_line_no=None,
                    new_line_no=self.new_cursor,
                )
            )
            self.new_cursor += 1
            return

        if raw_line.startswith("-"):
            self.lines.append(
                DiffLine(
                    kind="deleted",
                    content=raw_line[1:],
                    old_line_no=self.old_cursor,
                    new_line_no=None,
                )
            )
            self.old_cursor += 1
            return

        raise DiffParseError(f"invalid hunk line: {raw_line!r}")

    def mark_no_newline_at_eof(self) -> None:
        if not self.lines:
            raise DiffParseError("no newline marker appeared before any hunk line")
        previous = self.lines[-1]
        self.lines[-1] = DiffLine(
            kind=previous.kind,
            content=previous.content,
            old_line_no=previous.old_line_no,
            new_line_no=previous.new_line_no,
            no_newline_at_eof=True,
        )

    def to_hunk(self) -> DiffHunk:
        return DiffHunk(
            old_start=self.old_start,
            old_count=self.old_count,
            new_start=self.new_start,
            new_count=self.new_count,
            section=self.section,
            lines=tuple(self.lines),
        )


def parse_unified_diff(raw_diff: RawDiff | str) -> ParsedDiff:
    text = raw_diff.text if isinstance(raw_diff, RawDiff) else raw_diff
    files: list[DiffFile] = []
    current_file: _FileBuilder | None = None
    current_hunk: _HunkBuilder | None = None

    def flush_hunk() -> None:
        nonlocal current_hunk
        if current_file is not None and current_hunk is not None:
            current_file.hunks.append(current_hunk.to_hunk())
        current_hunk = None

    def flush_file() -> None:
        nonlocal current_file
        flush_hunk()
        if current_file is not None:
            files.append(current_file.to_file())
        current_file = None

    for line in text.splitlines():
        diff_match = _DIFF_HEADER_RE.match(line)
        if diff_match:
            flush_file()
            current_file = _FileBuilder(
                old_path=diff_match.group(1),
                new_path=diff_match.group(2),
            )
            continue

        if current_file is None:
            if line.strip():
                raise DiffParseError(f"diff content before file header: {line!r}")
            continue

        if line.startswith("new file mode "):
            current_file.change_type = "added"
            continue

        if line.startswith("deleted file mode "):
            current_file.change_type = "deleted"
            continue

        if line.startswith("rename from "):
            current_file.change_type = "renamed"
            current_file.old_path = line.removeprefix("rename from ")
            continue

        if line.startswith("rename to "):
            current_file.change_type = "renamed"
            current_file.new_path = line.removeprefix("rename to ")
            continue

        if current_hunk is None and line.startswith("--- "):
            path = _parse_file_marker_path(line.removeprefix("--- "))
            current_file.old_path = path
            if path is None:
                current_file.change_type = "added"
            continue

        if current_hunk is None and line.startswith("+++ "):
            path = _parse_file_marker_path(line.removeprefix("+++ "))
            current_file.new_path = path
            if path is None:
                current_file.change_type = "deleted"
            continue

        hunk_match = _HUNK_HEADER_RE.match(line)
        if hunk_match:
            flush_hunk()
            current_hunk = _HunkBuilder(
                old_start=int(hunk_match.group("old_start")),
                old_count=int(hunk_match.group("old_count") or "1"),
                new_start=int(hunk_match.group("new_start")),
                new_count=int(hunk_match.group("new_count") or "1"),
                section=hunk_match.group("section").strip(),
            )
            continue

        if line.startswith("\\ No newline at end of file"):
            if current_hunk is None:
                raise DiffParseError("no newline marker appeared outside a hunk")
            current_hunk.mark_no_newline_at_eof()
            continue

        if line.startswith(("index ", "old mode ", "new mode ", "similarity index ")):
            continue

        if line.startswith("Binary files "):
            current_file.change_type = "binary"
            continue

        if current_hunk is None:
            if line.strip():
                raise DiffParseError(f"file metadata line is not supported: {line!r}")
            continue

        current_hunk.add_line(line)

    flush_file()
    return ParsedDiff(files=tuple(files))


def _parse_file_marker_path(marker: str) -> str | None:
    if marker == "/dev/null":
        return None
    if marker.startswith("a/") or marker.startswith("b/"):
        return marker[2:]
    return marker
