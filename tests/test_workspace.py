from __future__ import annotations

from pathlib import Path

import pytest

from review_pilot.pr_models import PullRequestFile, PullRequestInfo, PullRequestRef
from review_pilot.workspace import (
    WorkspaceError,
    build_existing_workspace_plan,
    build_workspace_plan,
    prepare_workspace,
)


def test_build_workspace_plan_uses_head_repo_and_sha(tmp_path: Path) -> None:
    pr_info = _pr_info()

    plan = build_workspace_plan(pr_info, parent_dir=tmp_path, dry_run=True)

    assert plan.source == "github:octo-org/review-demo#42"
    assert plan.repo_clone_url == "https://github.com/contrib/review-demo.git"
    assert plan.base_sha == "1111111111111111111111111111111111111111"
    assert plan.head_sha == "2222222222222222222222222222222222222222"
    assert plan.workspace_path.endswith("octo-org-review-demo-pr-42-22222222")
    assert plan.commands == (
        (
            "git",
            "clone",
            "https://github.com/contrib/review-demo.git",
            plan.workspace_path,
        ),
        ("git", "-C", plan.workspace_path, "checkout", pr_info.head.sha),
    )


def test_prepare_workspace_dry_run_does_not_create_directory(tmp_path: Path) -> None:
    plan = build_workspace_plan(_pr_info(), parent_dir=tmp_path, dry_run=True)

    returned = prepare_workspace(plan)

    assert returned == plan
    assert not Path(plan.workspace_path).exists()


def test_existing_workspace_plan_reuses_checked_out_repo(tmp_path: Path) -> None:
    plan = build_existing_workspace_plan(_pr_info(), workspace_path=tmp_path)

    returned = prepare_workspace(plan)

    assert returned == plan
    assert plan.workspace_path == str(tmp_path.resolve())
    assert plan.source == "github-actions-checkout:octo-org/review-demo#42"
    assert plan.commands == ()


def test_existing_workspace_plan_rejects_missing_directory(tmp_path: Path) -> None:
    plan = build_existing_workspace_plan(_pr_info(), workspace_path=tmp_path / "missing")

    with pytest.raises(WorkspaceError, match="workspace does not exist"):
        prepare_workspace(plan)


def test_prepare_workspace_rejects_existing_workspace(tmp_path: Path) -> None:
    plan = build_workspace_plan(_pr_info(), parent_dir=tmp_path, dry_run=False)
    Path(plan.workspace_path).mkdir(parents=True)

    with pytest.raises(WorkspaceError, match="workspace already exists"):
        prepare_workspace(plan)


def _pr_info() -> PullRequestInfo:
    base = PullRequestRef(
        label="octo-org:main",
        ref="main",
        sha="1111111111111111111111111111111111111111",
        repo_full_name="octo-org/review-demo",
        repo_clone_url="https://github.com/octo-org/review-demo.git",
    )
    head = PullRequestRef(
        label="contrib:review-output",
        ref="review-output",
        sha="2222222222222222222222222222222222222222",
        repo_full_name="contrib/review-demo",
        repo_clone_url="https://github.com/contrib/review-demo.git",
    )
    return PullRequestInfo(
        provider="github",
        url="https://github.com/octo-org/review-demo/pull/42",
        owner="octo-org",
        repo="review-demo",
        number=42,
        title="Tighten review output",
        state="open",
        base=base,
        head=head,
        files=(
            PullRequestFile(
                filename="src/review.py",
                status="modified",
                additions=1,
                deletions=1,
                changes=2,
                patch="@@ -1 +1 @@\n-old\n+new\n",
            ),
        ),
    )
