from __future__ import annotations

from dataclasses import dataclass

from review_pilot.diff_reader import DiffReader


def test_staged_raw_diff_wraps_git_output() -> None:
    reader = DiffReader(FakeGit("diff --git a/app.py b/app.py\n+print('hello')\n"))

    raw_diff = reader.staged_raw_diff()

    assert raw_diff.text.startswith("diff --git")
    assert raw_diff.is_empty is False


def test_staged_raw_diff_marks_empty_diff() -> None:
    reader = DiffReader(FakeGit(""))

    raw_diff = reader.staged_raw_diff()

    assert raw_diff.text == ""
    assert raw_diff.is_empty is True


def test_staged_parsed_diff_parses_git_output() -> None:
    reader = DiffReader(FakeGit("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"))

    parsed_diff = reader.staged_parsed_diff()

    assert parsed_diff.files[0].path == "app.py"
    assert parsed_diff.files[0].hunks[0].lines[0].kind == "deleted"
    assert parsed_diff.files[0].hunks[0].lines[1].kind == "added"


@dataclass(frozen=True)
class FakeGit:
    diff: str

    def staged_raw_diff(self) -> str:
        return self.diff
