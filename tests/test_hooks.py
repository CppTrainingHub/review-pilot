from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from review_pilot.hooks import (
    HookError,
    hook_path,
    hook_statuses,
    install_hooks,
    is_review_pilot_hook,
    render_hook_script,
    selected_hooks,
    uninstall_hooks,
)
from review_pilot.review_profiles import profile_for_hook


def test_review_profiles_define_hook_commands() -> None:
    pre_commit = profile_for_hook("pre-commit")
    pre_push = profile_for_hook("pre-push")

    assert pre_commit.name == "pre_commit"
    assert pre_commit.command() == (
        "review-pilot",
        "review",
        "--staged",
        "--no-ai",
        "--fail-on",
        "P1",
    )
    assert pre_push.name == "pre_push"
    assert pre_push.command() == (
        "review-pilot",
        "review",
        "--staged",
        "--no-ai",
        "--with-tools",
        "--fail-on",
        "P2",
    )


def test_render_hook_script_is_readable_and_contains_profile() -> None:
    script = render_hook_script("pre-commit", profile_for_hook("pre-commit"))

    assert script.startswith("#!/bin/sh\n")
    assert "# review-pilot managed hook" in script
    assert "# hook: pre-commit" in script
    assert "# profile: pre_commit" in script
    assert "review-pilot review --staged --no-ai --fail-on P1" in script


def test_install_hooks_writes_executable_managed_scripts(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)

    changes = install_hooks(repo, ("pre-commit", "pre-push"))

    assert [change.action for change in changes] == ["installed", "installed"]
    pre_commit_path = hook_path(repo, "pre-commit")
    pre_push_path = hook_path(repo, "pre-push")
    assert is_review_pilot_hook(pre_commit_path) is True
    assert is_review_pilot_hook(pre_push_path) is True
    assert os.access(pre_commit_path, os.X_OK)
    assert os.access(pre_push_path, os.X_OK)
    assert pre_commit_path.stat().st_mode & stat.S_IXUSR
    assert "pre_commit" in pre_commit_path.read_text(encoding="utf-8")
    assert "--with-tools" not in pre_commit_path.read_text(encoding="utf-8")
    assert "pre_push" in pre_push_path.read_text(encoding="utf-8")
    assert "--with-tools" in pre_push_path.read_text(encoding="utf-8")


def test_hook_statuses_report_managed_and_missing_hooks(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    install_hooks(repo, ("pre-commit",))

    statuses = {status.hook_name: status for status in hook_statuses(repo)}

    assert statuses["pre-commit"].exists is True
    assert statuses["pre-commit"].managed is True
    assert statuses["pre-commit"].profile == "pre_commit"
    assert "managed by review-pilot" in statuses["pre-commit"].format_line()
    assert statuses["pre-push"].exists is False
    assert statuses["pre-push"].managed is False
    assert "not installed" in statuses["pre-push"].format_line()


def test_install_blocks_existing_non_review_pilot_hook(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = hook_path(repo, "pre-commit")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho custom\n", encoding="utf-8")

    with pytest.raises(HookError, match="already exists"):
        install_hooks(repo, ("pre-commit",))

    assert path.read_text(encoding="utf-8") == "#!/bin/sh\necho custom\n"


def test_install_force_replaces_existing_non_review_pilot_hook(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = hook_path(repo, "pre-commit")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho custom\n", encoding="utf-8")

    install_hooks(repo, ("pre-commit",), force=True)

    assert is_review_pilot_hook(path) is True
    assert "review-pilot review --staged --no-ai --fail-on P1" in path.read_text(encoding="utf-8")


def test_uninstall_removes_only_managed_hooks(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    install_hooks(repo, ("pre-commit", "pre-push"))

    changes = uninstall_hooks(repo, ("pre-commit", "pre-push"))

    assert [change.action for change in changes] == ["removed", "removed"]
    assert not hook_path(repo, "pre-commit").exists()
    assert not hook_path(repo, "pre-push").exists()


def test_uninstall_skips_missing_hook(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)

    changes = uninstall_hooks(repo, ("pre-commit",))

    assert changes[0].action == "skipped"
    assert "not installed" in changes[0].message


def test_uninstall_blocks_existing_non_review_pilot_hook(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    path = hook_path(repo, "pre-commit")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho custom\n", encoding="utf-8")

    with pytest.raises(HookError, match="not managed"):
        uninstall_hooks(repo, ("pre-commit",))

    assert path.exists()


def test_selected_hooks_requires_at_least_one_hook() -> None:
    with pytest.raises(HookError, match="select at least one hook"):
        selected_hooks(pre_commit=False, pre_push=False)


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
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
