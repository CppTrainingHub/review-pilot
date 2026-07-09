from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import ParsedDiff


@dataclass(frozen=True)
class ChangedLineMap:
    lines_by_path: dict[str, frozenset[int]]

    def contains(self, path: str | None, line_no: int | None) -> bool:
        if not path or line_no is None:
            return False
        return line_no in self.lines_by_path.get(path, frozenset())

    def to_dict(self) -> dict[str, Any]:
        return {
            path: sorted(lines)
            for path, lines in sorted(self.lines_by_path.items())
        }


def build_changed_line_map(parsed_diff: ParsedDiff) -> ChangedLineMap:
    lines_by_path: dict[str, set[int]] = {}
    for file in parsed_diff.files:
        if not file.path:
            continue
        changed_lines = lines_by_path.setdefault(file.path, set())
        for hunk in file.hunks:
            for line in hunk.lines:
                if line.kind == "added" and line.new_line_no is not None:
                    changed_lines.add(line.new_line_no)

    return ChangedLineMap(
        {
            path: frozenset(lines)
            for path, lines in lines_by_path.items()
            if lines
        }
    )
