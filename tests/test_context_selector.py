from __future__ import annotations

from pathlib import Path

from review_pilot.code_index import build_code_index
from review_pilot.config import ReviewPilotConfig
from review_pilot.context_selector import select_context_candidates
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.models import RawDiff


def parse_added_file(path: str) -> RawDiff:
    return RawDiff(
        "\n".join(
            [
                f"diff --git a/{path} b/{path}",
                "--- /dev/null",
                f"+++ b/{path}",
                "@@ -0,0 +1,2 @@",
                "+from .helpers import load",
                "+def run(): return load()",
            ]
        )
    )


def test_select_context_candidates_includes_changed_test_import_doc_and_config(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "from .helpers import load\n\ndef run():\n    return load()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "helpers.py").write_text("def load():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_service.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    index = build_code_index(tmp_path)
    manifest = select_context_candidates(parse_unified_diff(parse_added_file("src/service.py")), index)

    by_path = {candidate.path: candidate for candidate in manifest.candidates}
    assert manifest.changed_paths == ("src/service.py",)
    assert by_path["src/service.py"].reason == "changed_file"
    assert by_path["tests/test_service.py"].reason == "related_test"
    assert by_path["src/helpers.py"].reason == "local_import"
    assert by_path["README.md"].reason == "project_doc"
    assert by_path["pyproject.toml"].reason == "project_config"
    assert [candidate.path for candidate in manifest.candidates[:3]] == [
        "src/service.py",
        "tests/test_service.py",
        "src/helpers.py",
    ]


def test_select_context_candidates_continues_without_related_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lonely.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    index = build_code_index(tmp_path)
    manifest = select_context_candidates(parse_unified_diff(parse_added_file("src/lonely.py")), index)

    assert [candidate.path for candidate in manifest.candidates] == ["src/lonely.py"]


def test_select_context_candidates_respects_ignored_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "service.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_service.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")

    index = build_code_index(
        tmp_path,
        ReviewPilotConfig(ignore_paths=("tests/**",)),
    )
    manifest = select_context_candidates(parse_unified_diff(parse_added_file("src/service.py")), index)

    assert [candidate.path for candidate in manifest.candidates] == ["src/service.py"]
