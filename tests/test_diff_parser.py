from __future__ import annotations

from pathlib import Path

import pytest

from review_pilot.diff_parser import DiffParseError, parse_unified_diff
from review_pilot.models import RawDiff


FIXTURES = Path(__file__).parent / "fixtures" / "diffs"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_added_file_tracks_new_lines() -> None:
    parsed = parse_unified_diff(RawDiff(read_fixture("added_file.diff")))

    file = parsed.files[0]
    hunk = file.hunks[0]

    assert file.old_path is None
    assert file.new_path == "app.py"
    assert file.change_type == "added"
    assert hunk.old_start == 0
    assert hunk.old_count == 0
    assert hunk.new_start == 1
    assert hunk.new_count == 2
    assert [line.kind for line in hunk.lines] == ["added", "added"]
    assert [line.new_line_no for line in hunk.lines] == [1, 2]
    assert [line.old_line_no for line in hunk.lines] == [None, None]


def test_parse_modified_file_with_multiple_hunks_and_line_numbers() -> None:
    parsed = parse_unified_diff(read_fixture("modified_multi_hunk.diff"))

    file = parsed.files[0]
    first_hunk, second_hunk = file.hunks

    assert file.old_path == "app.py"
    assert file.new_path == "app.py"
    assert file.change_type == "modified"
    assert first_hunk.old_start == 1
    assert first_hunk.new_start == 1
    assert [line.kind for line in first_hunk.lines] == [
        "context",
        "deleted",
        "added",
        "context",
        "context",
    ]
    assert first_hunk.lines[1].old_line_no == 2
    assert first_hunk.lines[1].new_line_no is None
    assert first_hunk.lines[2].old_line_no is None
    assert first_hunk.lines[2].new_line_no == 2
    assert second_hunk.old_start == 8
    assert second_hunk.new_start == 8
    assert second_hunk.lines[-1].kind == "added"
    assert second_hunk.lines[-1].new_line_no == 11


def test_parse_deleted_file_tracks_old_lines() -> None:
    parsed = parse_unified_diff(read_fixture("deleted_file.diff"))

    file = parsed.files[0]
    hunk = file.hunks[0]

    assert file.old_path == "old.py"
    assert file.new_path is None
    assert file.change_type == "deleted"
    assert [line.kind for line in hunk.lines] == ["deleted", "deleted"]
    assert [line.old_line_no for line in hunk.lines] == [1, 2]
    assert [line.new_line_no for line in hunk.lines] == [None, None]


def test_parse_renamed_file_preserves_both_paths() -> None:
    parsed = parse_unified_diff(read_fixture("renamed_file.diff"))

    file = parsed.files[0]

    assert file.old_path == "old_name.py"
    assert file.new_path == "new_name.py"
    assert file.path == "new_name.py"
    assert file.change_type == "renamed"
    assert file.hunks[0].lines[0].content == "name = 'review-pilot'"


def test_parse_no_newline_marker_marks_previous_diff_line() -> None:
    parsed = parse_unified_diff(read_fixture("no_newline.diff"))

    line1, line2 = parsed.files[0].hunks[0].lines

    assert line1.kind == "deleted"
    assert line1.no_newline_at_eof is True
    assert line2.kind == "added"
    assert line2.no_newline_at_eof is True


def test_parse_empty_diff_returns_empty_parsed_diff() -> None:
    parsed = parse_unified_diff("")

    assert parsed.files == ()
    assert parsed.is_empty is True
    assert parsed.to_dict() == {"files": []}


def test_parse_rejects_content_before_diff_header() -> None:
    with pytest.raises(DiffParseError, match="before file header"):
        parse_unified_diff("@@ -1 +1 @@\n+broken\n")
