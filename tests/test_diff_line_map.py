from __future__ import annotations

from pathlib import Path

from review_pilot.diff_line_map import build_changed_line_map
from review_pilot.diff_parser import parse_unified_diff


FIXTURES = Path(__file__).parent / "fixtures" / "diffs"


def test_changed_line_map_tracks_added_lines_across_hunks() -> None:
    parsed = parse_unified_diff((FIXTURES / "modified_multi_hunk.diff").read_text(encoding="utf-8"))

    line_map = build_changed_line_map(parsed)

    assert line_map.to_dict() == {"app.py": [2, 11]}
    assert line_map.contains("app.py", 2) is True
    assert line_map.contains("app.py", 11) is True
    assert line_map.contains("app.py", 1) is False
    assert line_map.contains("other.py", 2) is False


def test_changed_line_map_tracks_added_file_lines() -> None:
    parsed = parse_unified_diff((FIXTURES / "added_file.diff").read_text(encoding="utf-8"))

    line_map = build_changed_line_map(parsed)

    assert line_map.to_dict() == {"app.py": [1, 2]}


def test_changed_line_map_ignores_deleted_file_lines() -> None:
    parsed = parse_unified_diff((FIXTURES / "deleted_file.diff").read_text(encoding="utf-8"))

    line_map = build_changed_line_map(parsed)

    assert line_map.to_dict() == {}
