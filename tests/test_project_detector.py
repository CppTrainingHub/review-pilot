from __future__ import annotations

from review_pilot.project_detector import detect_project


def test_detect_project_finds_python_markers(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()

    detection = detect_project(tmp_path)

    assert detection.project_types == ("python",)
    assert detection.markers == ("pyproject.toml",)
    assert detection.root == str(tmp_path.resolve())


def test_detect_project_finds_multiple_project_types(tmp_path) -> None:
    (tmp_path / "package.json").write_text('{"scripts":{"test":"node test.js"}}\n', encoding="utf-8")
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")

    detection = detect_project(tmp_path)

    assert detection.project_types == ("node", "go")
    assert detection.markers == ("package.json", "go.mod")


def test_detect_project_uses_tests_directory_as_python_hint(tmp_path) -> None:
    (tmp_path / "tests").mkdir()

    detection = detect_project(tmp_path)

    assert detection.project_types == ("python",)
    assert detection.markers == ("tests/",)
