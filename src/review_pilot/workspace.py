from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pr_models import PullRequestInfo


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspacePlan:
    workspace_path: str
    repo_clone_url: str
    base_sha: str
    head_sha: str
    source: str
    dry_run: bool
    commands: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_path": self.workspace_path,
            "repo_clone_url": self.repo_clone_url,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "source": self.source,
            "dry_run": self.dry_run,
            "commands": [list(command) for command in self.commands],
        }


def build_workspace_plan(
    pr_info: PullRequestInfo,
    *,
    parent_dir: str | Path | None = None,
    dry_run: bool = True,
) -> WorkspacePlan:
    root = Path(parent_dir or ".review-pilot/workspaces").resolve()
    safe_name = f"{pr_info.owner}-{pr_info.repo}-pr-{pr_info.number}-{pr_info.head.sha[:8]}"
    workspace_path = root / safe_name
    repo_url = pr_info.head.repo_clone_url or pr_info.base.repo_clone_url
    commands: tuple[tuple[str, ...], ...] = (
        ("git", "clone", repo_url, str(workspace_path)),
        ("git", "-C", str(workspace_path), "checkout", pr_info.head.sha),
    )
    return WorkspacePlan(
        workspace_path=str(workspace_path),
        repo_clone_url=repo_url,
        base_sha=pr_info.base.sha,
        head_sha=pr_info.head.sha,
        source=f"github:{pr_info.full_name}#{pr_info.number}",
        dry_run=dry_run,
        commands=commands,
    )


def build_existing_workspace_plan(
    pr_info: PullRequestInfo,
    *,
    workspace_path: str | Path,
    dry_run: bool = False,
) -> WorkspacePlan:
    return WorkspacePlan(
        workspace_path=str(Path(workspace_path).resolve()),
        repo_clone_url=pr_info.head.repo_clone_url or pr_info.base.repo_clone_url,
        base_sha=pr_info.base.sha,
        head_sha=pr_info.head.sha,
        source=f"github-actions-checkout:{pr_info.full_name}#{pr_info.number}",
        dry_run=dry_run,
        commands=(),
    )


def prepare_workspace(plan: WorkspacePlan) -> WorkspacePlan:
    if plan.dry_run:
        return plan

    workspace_path = Path(plan.workspace_path)
    if not plan.commands:
        if not workspace_path.exists():
            raise WorkspaceError(f"workspace does not exist: {workspace_path}")
        return plan

    if workspace_path.exists():
        raise WorkspaceError(f"workspace already exists: {workspace_path}")
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    for command in plan.commands:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise WorkspaceError(f"workspace command failed: {' '.join(command)}: {message}")
    return plan
