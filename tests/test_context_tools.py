from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_pilot.context_tools import (
    ContextToolError,
    ContextToolExecutor,
    ContextToolRequest,
)
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.models import RawDiff


def test_file_read_returns_bounded_number_of_lines(tmp_path: Path) -> None:
    target = tmp_path / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("".join(f"value_{i} = {i}\n" for i in range(1, 601)), encoding="utf-8")
    executor = ContextToolExecutor(tmp_path, _diff("src/service.py"))

    result = executor.execute(
        ContextToolRequest(
            "file_read",
            {"path": "src/service.py", "start_line": 1, "end_line": 600},
            call_id="call-1",
        )
    )

    payload = json.loads(result.content)
    assert result.ok is True
    assert result.returned_count == 500
    assert result.truncated is True
    assert payload["lines"][0] == {"line": 1, "text": "value_1 = 1"}
    assert payload["lines"][-1]["line"] == 500


def test_code_search_and_file_find_are_capped(tmp_path: Path) -> None:
    for index in range(105):
        path = tmp_path / "src" / f"service_{index}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text("needle = True\n", encoding="utf-8")
    executor = ContextToolExecutor(tmp_path, _diff("src/service_0.py"))

    search = executor.execute(
        ContextToolRequest("code_search", {"query": "needle"})
    )
    find = executor.execute(
        ContextToolRequest("file_find", {"pattern": "src/service_*.py"})
    )

    assert search.returned_count == 100
    assert search.truncated is True
    assert find.returned_count == 100
    assert find.truncated is True


def test_file_read_diff_returns_only_the_requested_changed_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text("return 2\n", encoding="utf-8")
    executor = ContextToolExecutor(
        tmp_path,
        parse_unified_diff(
            RawDiff(
                "\n".join(
                    [
                        "diff --git a/src/service.py b/src/service.py",
                        "--- a/src/service.py",
                        "+++ b/src/service.py",
                        "@@ -1 +1 @@",
                        "-return 1",
                        "+return 2",
                    ]
                )
            )
        ),
    )

    result = executor.execute(
        ContextToolRequest("file_read_diff", {"path": "src/service.py"})
    )

    payload = json.loads(result.content)
    assert payload["path"] == "src/service.py"
    assert payload["diff"]["path"] == "src/service.py"
    assert result.returned_count == 2


@pytest.mark.parametrize(
    ("tool_name", "arguments", "error_code"),
    [
        ("file_read", {"path": "/tmp/outside.py"}, "path_out_of_workspace"),
        ("file_read", {"path": "../outside.py"}, "path_out_of_workspace"),
        ("file_read", {"path": "src"}, "directory_not_allowed"),
        ("file_read", {"path": "missing.py"}, "not_found"),
        ("file_read", {"path": "binary.dat"}, "binary_file"),
        ("file_read_diff", {"path": "other.py"}, "not_changed"),
    ],
)
def test_context_tools_return_structured_errors(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
    error_code: str,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "binary.dat").write_bytes(b"\x00\x01")
    executor = ContextToolExecutor(tmp_path, _diff("src/service.py"))

    result = executor.execute(ContextToolRequest(tool_name, arguments))

    assert result.ok is False
    assert result.error_code == error_code
    assert json.loads(result.content)["error"]["code"] == error_code


def test_context_tool_request_rejects_unknown_tool() -> None:
    with pytest.raises(ContextToolError, match="unknown context tool"):
        ContextToolRequest("shell", {})


def _diff(path: str):
    return parse_unified_diff(
        RawDiff(
            "\n".join(
                [
                    f"diff --git a/{path} b/{path}",
                    f"--- a/{path}",
                    f"+++ b/{path}",
                    "@@ -1 +1 @@",
                    "-return 1",
                    "+return 2",
                ]
            )
        )
    )
