from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from review_pilot.benchmark.dataset import AACRCase
from review_pilot.benchmark.repository import RepositoryCache, RepositoryError


def test_prepare_compares_source_to_merge_base_not_moved_target(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test User")

    (repository / "README.md").write_text("base\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "base")
    base_commit = _git(repository, "rev-parse", "HEAD")

    _git(repository, "checkout", "-b", "target")
    (repository / "target.txt").write_text("target branch change\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "target branch change")
    target_commit = _git(repository, "rev-parse", "HEAD")

    _git(repository, "checkout", "-b", "feature", base_commit)
    (repository / "src").mkdir()
    (repository / "src" / "app.py").write_text("print('feature')\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "feature change")
    source_commit = _git(repository, "rev-parse", "HEAD")

    case = AACRCase(
        case_id="merge-base-case",
        dataset_index=0,
        pr_url="https://github.com/example/review/pull/1",
        repo_url=str(repository),
        source_commit=source_commit,
        target_commit=target_commit,
        project_main_language="Python",
        change_line_count=1,
        comments=(),
        context_level="diff",
    )

    prepared = RepositoryCache(
        cache_root=tmp_path / "cache",
        workspace_root=tmp_path / "workspaces",
    ).prepare(case)

    assert {diff_file.path for diff_file in prepared.parsed_diff.files} == {"src/app.py"}


def test_prepare_falls_back_to_github_patch_when_git_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = AACRCase(
        case_id="github-patch-case",
        dataset_index=0,
        pr_url="https://github.com/example/review/pull/7",
        repo_url="https://github.com/example/review.git",
        source_commit="a" * 40,
        target_commit="b" * 40,
        project_main_language="Python",
        change_line_count=1,
        comments=(),
        context_level="diff",
    )
    patch = "\n".join(
        [
            "diff --git a/src/app.py b/src/app.py",
            "index 1111111..2222222 100644",
            "--- a/src/app.py",
            "+++ b/src/app.py",
            "@@ -1 +1,2 @@",
            " print('base')",
            "+print('feature')",
        ]
    )

    cache = RepositoryCache(
        cache_root=tmp_path / "cache",
        workspace_root=tmp_path / "workspaces",
    )

    def unavailable_git(_case: AACRCase):
        raise RepositoryError("git unavailable")

    monkeypatch.setattr(cache, "_prepare_git", unavailable_git)
    monkeypatch.setattr(
        "review_pilot.benchmark.repository.urlopen",
        lambda request, timeout: _FakeResponse(
            patch if request.full_url.endswith("/pull/7.diff") else "print('base')\nprint('feature')\n"
        ),
    )

    prepared = cache.prepare(case)

    assert prepared.repository_source == "github-patch-fallback"
    assert {diff_file.path for diff_file in prepared.parsed_diff.files} == {"src/app.py"}
    assert (
        Path(prepared.workspace_path, "src/app.py").read_text(encoding="utf-8")
        == "print('base')\nprint('feature')\n"
    )


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content.encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.content


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()
