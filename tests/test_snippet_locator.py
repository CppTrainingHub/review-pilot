from __future__ import annotations

from pathlib import Path

import pytest

from review_pilot.snippet_locator import DOWNGRADED, DROPPED, MATCHED, SnippetLocator


def _diff(*lines: tuple[str, str, int | None]) -> dict:
    return {
        "files": [
            {
                "path": "src/app.py",
                "hunks": [
                    {
                        "lines": [
                            {
                                "kind": kind,
                                "content": content,
                                "new_line_no": line_no,
                            }
                            for kind, content, line_no in lines
                        ]
                    }
                ],
            }
        ]
    }


def test_locates_single_line_in_hunk_and_uses_added_line() -> None:
    decision = SnippetLocator(
        _diff(
            ("context", "def run():", 1),
            ("added", "    print('debug')", 2),
        )
    ).locate("src/app.py", "print('debug')")

    assert decision.status == MATCHED
    assert decision.source == "hunk"
    assert decision.line_no == 2
    assert decision.match_count == 1


def test_locates_multi_line_snippet_and_tolerates_indent_and_crlf() -> None:
    decision = SnippetLocator(
        _diff(
            ("added", "    result = run()", 8),
            ("added", "\treturn result", 9),
        )
    ).locate("src/app.py", "result = run()\r\n  return result\r\n")

    assert decision.status == MATCHED
    assert decision.line_no == 8


def test_falls_back_to_new_file_as_downgraded(tmp_path: Path) -> None:
    target = tmp_path / "src/app.py"
    target.parent.mkdir()
    target.write_text("def run():\n    return safe()\n", encoding="utf-8")

    decision = SnippetLocator(_diff(("added", "    return changed()", 20)), tmp_path).locate(
        "src/app.py",
        "return safe()",
    )

    assert decision.status == DOWNGRADED
    assert decision.source == "file"
    assert decision.line_no == 2
    assert "new file" in decision.reason


def test_drops_multiple_matches_in_hunk() -> None:
    decision = SnippetLocator(
        _diff(
            ("added", "    return value", 2),
            ("context", "", 3),
            ("added", "    return value", 4),
        )
    ).locate("src/app.py", "return value")

    assert decision.status == DROPPED
    assert decision.line_no is None
    assert decision.match_count == 2
    assert "multiple_matches" in decision.reason


def test_does_not_join_two_separate_hunks() -> None:
    diff = {
        "files": [
            {
                "path": "src/app.py",
                "hunks": [
                    {
                        "lines": [
                            {"kind": "added", "content": "first()", "new_line_no": 2},
                        ]
                    },
                    {
                        "lines": [
                            {"kind": "added", "content": "second()", "new_line_no": 20},
                        ]
                    },
                ],
            }
        ]
    }

    decision = SnippetLocator(diff).locate("src/app.py", "first()\nsecond()")

    assert decision.status == DROPPED
    assert decision.line_no is None


@pytest.mark.parametrize("snippet", ["", "rewritten call()"])
def test_drops_empty_or_rewritten_snippet(tmp_path: Path, snippet: str) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_text("return value\n", encoding="utf-8")
    decision = SnippetLocator(_diff(("added", "return value", 1)), tmp_path).locate(
        "src/app.py",
        snippet,
    )

    assert decision.status == DROPPED
    assert decision.line_no is None


def test_drops_when_no_match_exists(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_text("return value\n", encoding="utf-8")

    decision = SnippetLocator(_diff(("added", "return value", 1)), tmp_path).locate(
        "src/app.py",
        "return missing()",
    )

    assert decision.status == DROPPED
    assert "not found" in decision.reason
