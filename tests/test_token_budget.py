from __future__ import annotations

from pathlib import Path

import pytest

from review_pilot.code_index import build_code_index
from review_pilot.context_selector import select_context_candidates
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.models import RawDiff
from review_pilot.token_budget import apply_token_budget, estimate_tokens


def test_estimate_tokens_is_stable_and_non_zero() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcd efgh") == 3


def test_apply_token_budget_includes_candidate_slices_when_budget_allows(tmp_path: Path) -> None:
    _write_project(tmp_path)
    manifest = _select_manifest(tmp_path, _added_file_diff("src/service.py"))

    budgeted = apply_token_budget(
        manifest,
        parse_unified_diff(_added_file_diff("src/service.py")),
        tmp_path,
        max_context_tokens=400,
    )

    used_paths = [item.path for item in budgeted.context_used]
    assert budgeted.max_context_tokens == 400
    assert budgeted.used_tokens <= 400
    assert used_paths[:3] == ["src/service.py", "tests/test_service.py", "src/helpers.py"]
    assert "README.md" in used_paths
    assert "pyproject.toml" in used_paths


def test_apply_token_budget_omits_lower_priority_context_when_budget_is_tight(tmp_path: Path) -> None:
    _write_project(tmp_path)
    manifest = _select_manifest(tmp_path, _added_file_diff("src/service.py"))

    budgeted = apply_token_budget(
        manifest,
        parse_unified_diff(_added_file_diff("src/service.py")),
        tmp_path,
        max_context_tokens=12,
    )

    assert [item.path for item in budgeted.context_used] == ["src/service.py"]
    omitted_paths = {item.path: item.omitted_reason for item in budgeted.context_omitted}
    assert omitted_paths["tests/test_service.py"] == "budget_exhausted"
    assert omitted_paths["src/helpers.py"] == "budget_exhausted"
    assert omitted_paths["README.md"] == "budget_exhausted"


def test_changed_file_slice_keeps_changed_line_window(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    lines = [f"value_{index} = {index}" for index in range(1, 21)]
    lines[9] = "value_10 = run_with_new_branch()"
    (tmp_path / "src" / "service.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diff = RawDiff(
        "\n".join(
            [
                "diff --git a/src/service.py b/src/service.py",
                "--- a/src/service.py",
                "+++ b/src/service.py",
                "@@ -8,5 +8,5 @@",
                " value_8 = 8",
                " value_9 = 9",
                "-value_10 = 10",
                "+value_10 = run_with_new_branch()",
                " value_11 = 11",
                " value_12 = 12",
            ]
        )
    )
    manifest = _select_manifest(tmp_path, diff)

    budgeted = apply_token_budget(manifest, parse_unified_diff(diff), tmp_path, max_context_tokens=80)

    slice_ = budgeted.context_used[0]
    assert slice_.path == "src/service.py"
    assert slice_.start_line <= 10 <= slice_.end_line
    assert "value_10 = run_with_new_branch()" in slice_.content
    assert "value_1 = 1" not in slice_.content
    assert budgeted.context_omitted[0].omitted_reason == "truncated"


def test_apply_token_budget_rejects_non_positive_budget(tmp_path: Path) -> None:
    _write_project(tmp_path)
    manifest = _select_manifest(tmp_path, _added_file_diff("src/service.py"))

    with pytest.raises(ValueError, match="positive"):
        apply_token_budget(
            manifest,
            parse_unified_diff(_added_file_diff("src/service.py")),
            tmp_path,
            max_context_tokens=0,
        )


def _write_project(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "helpers.py").write_text("def load():\n    return 1\n", encoding="utf-8")
    (root / "src" / "service.py").write_text(
        "from .helpers import load\n\n"
        "def run():\n"
        "    return load()\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_service.py").write_text(
        "from src.service import run\n\n"
        "def test_run():\n"
        "    assert run() == 1\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# demo\n\nReview context docs.\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")


def _select_manifest(root: Path, diff: RawDiff):
    return select_context_candidates(parse_unified_diff(diff), build_code_index(root))


def _added_file_diff(path: str) -> RawDiff:
    return RawDiff(
        "\n".join(
            [
                f"diff --git a/{path} b/{path}",
                "--- /dev/null",
                f"+++ b/{path}",
                "@@ -0,0 +1,4 @@",
                "+from .helpers import load",
                "+",
                "+def run():",
                "+    return load()",
            ]
        )
    )
