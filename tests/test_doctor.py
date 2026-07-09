from __future__ import annotations

from review_pilot.doctor import check_git_available, check_python_version


def test_python_version_check_accepts_supported_version() -> None:
    result = check_python_version((3, 11, 0))

    assert result.ok is True
    assert result.name == "python"
    assert ">= 3.11" in result.message


def test_python_version_check_rejects_old_version() -> None:
    result = check_python_version((3, 10, 9))

    assert result.ok is False
    assert result.name == "python"
    assert "< 3.11" in result.message


def test_git_check_accepts_existing_path() -> None:
    result = check_git_available("/usr/bin/git")

    assert result.ok is True
    assert result.name == "git"
    assert "/usr/bin/git" in result.message


def test_git_check_rejects_missing_path() -> None:
    result = check_git_available("")

    assert result.ok is False
    assert result.name == "git"
    assert "not found" in result.message
