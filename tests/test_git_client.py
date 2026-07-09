from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from review_pilot.git_client import GitClient, NotGitRepositoryError


def test_repo_info_reads_root_branch_head_and_change_flags(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    (repo / "staged.py").write_text("print('staged')\n", encoding="utf-8")
    git(repo, "add", "staged.py")
    (repo / "README.md").write_text("# demo\n\nchanged\n", encoding="utf-8")

    info = GitClient.from_cwd(repo / "src").repo_info()

    assert info.root == str(repo)
    assert info.branch == "main"
    assert len(info.head) == 40
    assert info.has_staged_changes is True
    assert info.has_unstaged_changes is True


def test_repo_root_rejects_non_git_directory(tmp_path: Path) -> None:
    with pytest.raises(NotGitRepositoryError):
        GitClient.from_cwd(tmp_path).repo_root()


def test_staged_raw_diff_reads_unified_diff(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")

    raw_diff = GitClient.from_cwd(repo).staged_raw_diff()

    assert "diff --git a/app.py b/app.py" in raw_diff
    assert "new file mode" in raw_diff
    assert "+print('hello')" in raw_diff


def test_staged_raw_diff_returns_empty_string_when_nothing_is_staged(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)

    assert GitClient.from_cwd(repo).staged_raw_diff() == ""


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    git(repo, "init")
    git(repo, "checkout", "-b", "main")
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "-c", "user.name=Test User", "-c", "user.email=test@example.com", "commit", "-m", "init")
    return repo


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
