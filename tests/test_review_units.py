from __future__ import annotations

from pathlib import Path

import pytest

from review_pilot.code_index import build_code_index
from review_pilot.config import ReviewPilotConfig
from review_pilot.models import DiffFile, DiffHunk, DiffLine, ParsedDiff
from review_pilot.review_units import build_review_plan


def test_plan_groups_related_tests_imports_siblings_and_config(tmp_path: Path) -> None:
    _write_project(tmp_path)
    parsed_diff = ParsedDiff(
        files=(
            _diff_file("src/parser.py", 1),
            _diff_file("src/other.py", 1),
        )
    )
    plan = build_review_plan(
        parsed_diff,
        build_code_index(tmp_path, ReviewPilotConfig.default()),
        max_context_tokens=101,
    )

    assert plan.changed_files == ("src/other.py", "src/parser.py")
    assert [unit.unit_id for unit in plan.units] == ["unit-001", "unit-002"]
    parser_unit = next(unit for unit in plan.units if "src/parser.py" in unit.changed_files)
    assert "tests/test_parser.py" in parser_unit.context_files
    assert "src/helpers.py" in parser_unit.context_files
    assert "src/parser_impl.py" in parser_unit.context_files
    assert "pyproject.toml" in parser_unit.context_files
    assert parser_unit.reasons["tests/test_parser.py"] == ("相关测试",)
    assert parser_unit.reasons["src/helpers.py"] == ("直接 import 的本地文件",)
    assert parser_unit.reasons["src/parser_impl.py"] == ("同目录接口或实现",)

    assert sum(unit.budget_tokens for unit in plan.units) == 101
    assert plan.to_dict() == build_review_plan(
        parsed_diff,
        build_code_index(tmp_path, ReviewPilotConfig.default()),
        max_context_tokens=101,
    ).to_dict()


def test_plan_rejects_zero_budget_duplicate_and_unknown_changed_path(tmp_path: Path) -> None:
    _write_project(tmp_path)
    index = build_code_index(tmp_path, ReviewPilotConfig.default())
    parsed_diff = ParsedDiff(files=(_diff_file("src/parser.py", 1),))

    with pytest.raises(ValueError, match="max_context_tokens must be positive"):
        build_review_plan(parsed_diff, index, max_context_tokens=0)

    with pytest.raises(ValueError, match="duplicate changed file path"):
        build_review_plan(
            ParsedDiff(
                files=(
                    _diff_file("src/parser.py", 1),
                    _diff_file("src/parser.py", 2),
                )
            ),
            index,
            max_context_tokens=20,
        )

    with pytest.raises(ValueError, match="changed file path is required"):
        build_review_plan(
            ParsedDiff(
                files=(
                    DiffFile(
                        old_path=None,
                        new_path=None,
                        change_type="modified",
                        hunks=(),
                    ),
                )
            ),
            index,
            max_context_tokens=20,
        )


def _write_project(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "parser.py").write_text(
        "from .helpers import parse\n\nvalue = parse('x')\n",
        encoding="utf-8",
    )
    (root / "src" / "parser_impl.py").write_text(
        "def parse_impl(value):\n    return value\n",
        encoding="utf-8",
    )
    (root / "src" / "helpers.py").write_text(
        "def parse(value):\n    return value\n",
        encoding="utf-8",
    )
    (root / "src" / "other.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_parser.py").write_text(
        "from src.parser import value\n\ndef test_value():\n    assert value\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[tool.review_pilot]\nname = 'fixture'\n",
        encoding="utf-8",
    )


def _diff_file(path: str, line_no: int) -> DiffFile:
    return DiffFile(
        old_path=path,
        new_path=path,
        change_type="modified",
        hunks=(
            DiffHunk(
                old_start=line_no,
                old_count=1,
                new_start=line_no,
                new_count=1,
                section="",
                lines=(
                    DiffLine(
                        kind="added",
                        content="value = 2",
                        old_line_no=None,
                        new_line_no=line_no,
                    ),
                ),
            ),
        ),
    )
