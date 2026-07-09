from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .git_client import GitClient
from .review_profiles import HookName, ReviewProfile, profile_for_hook


REVIEW_PILOT_HOOK_MARKER = "# review-pilot managed hook"
HOOK_NAMES: tuple[HookName, ...] = ("pre-commit", "pre-push")


class HookError(ValueError):
    pass


HookAction = Literal["installed", "removed", "skipped", "blocked"]


@dataclass(frozen=True)
class HookStatus:
    hook_name: HookName
    path: str
    exists: bool
    managed: bool
    profile: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "hook_name": self.hook_name,
            "path": self.path,
            "exists": self.exists,
            "managed": self.managed,
            "profile": self.profile,
        }

    def format_line(self) -> str:
        if not self.exists:
            state = "not installed"
        elif self.managed:
            profile = self.profile or "unknown"
            state = f"managed by review-pilot ({profile})"
        else:
            state = "exists but is not managed by review-pilot"
        return f"{self.hook_name}: {state} [{self.path}]"


@dataclass(frozen=True)
class HookChange:
    hook_name: HookName
    path: str
    action: HookAction
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "hook_name": self.hook_name,
            "path": self.path,
            "action": self.action,
            "message": self.message,
        }

    def format_line(self) -> str:
        return f"{self.action} {self.hook_name}: {self.message}"


def selected_hooks(*, pre_commit: bool, pre_push: bool) -> tuple[HookName, ...]:
    hooks: list[HookName] = []
    if pre_commit:
        hooks.append("pre-commit")
    if pre_push:
        hooks.append("pre-push")
    if not hooks:
        raise HookError("select at least one hook: --pre-commit or --pre-push")
    return tuple(hooks)


def hooks_dir(repo_root: str | Path) -> Path:
    return Path(repo_root) / ".git" / "hooks"


def hook_path(repo_root: str | Path, hook_name: HookName) -> Path:
    return hooks_dir(repo_root) / hook_name


def is_review_pilot_hook(path: Path) -> bool:
    return path.exists() and REVIEW_PILOT_HOOK_MARKER in path.read_text(encoding="utf-8")


def read_hook_profile(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# profile:"):
            return line.split(":", 1)[1].strip() or None
    return None


def render_hook_script(hook_name: HookName, profile: ReviewProfile) -> str:
    command = " ".join(profile.command())
    return (
        "#!/bin/sh\n"
        f"{REVIEW_PILOT_HOOK_MARKER}\n"
        f"# hook: {hook_name}\n"
        f"# profile: {profile.name}\n"
        "\n"
        f'echo "review-pilot {hook_name}: running {profile.name}"\n'
        f"{command}\n"
    )


def install_hooks(
    repo_root: str | Path,
    hook_names: tuple[HookName, ...],
    *,
    force: bool = False,
) -> list[HookChange]:
    directory = hooks_dir(repo_root)
    directory.mkdir(parents=True, exist_ok=True)
    changes: list[HookChange] = []

    for hook_name in hook_names:
        path = hook_path(repo_root, hook_name)
        if path.exists() and not is_review_pilot_hook(path) and not force:
            raise HookError(
                f"{hook_name} already exists and is not managed by review-pilot; "
                "use --force to replace it"
            )
        profile = profile_for_hook(hook_name)
        path.write_text(render_hook_script(hook_name, profile), encoding="utf-8")
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        changes.append(
            HookChange(
                hook_name=hook_name,
                path=str(path),
                action="installed",
                message=f"installed {hook_name} with {profile.name} profile",
            )
        )
    return changes


def uninstall_hooks(
    repo_root: str | Path,
    hook_names: tuple[HookName, ...],
) -> list[HookChange]:
    changes: list[HookChange] = []
    for hook_name in hook_names:
        path = hook_path(repo_root, hook_name)
        if not path.exists():
            changes.append(
                HookChange(
                    hook_name=hook_name,
                    path=str(path),
                    action="skipped",
                    message=f"{hook_name} is not installed",
                )
            )
            continue
        if not is_review_pilot_hook(path):
            raise HookError(
                f"{hook_name} exists but is not managed by review-pilot; "
                "remove it manually if needed"
            )
        path.unlink()
        changes.append(
            HookChange(
                hook_name=hook_name,
                path=str(path),
                action="removed",
                message=f"removed {hook_name}",
            )
        )
    return changes


def hook_statuses(repo_root: str | Path) -> list[HookStatus]:
    statuses: list[HookStatus] = []
    for hook_name in HOOK_NAMES:
        path = hook_path(repo_root, hook_name)
        managed = is_review_pilot_hook(path)
        statuses.append(
            HookStatus(
                hook_name=hook_name,
                path=str(path),
                exists=path.exists(),
                managed=managed,
                profile=read_hook_profile(path) if managed else None,
            )
        )
    return statuses


def repo_root_from_cwd() -> Path:
    return GitClient.from_cwd().repo_root()
