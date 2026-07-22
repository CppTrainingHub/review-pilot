from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import ReviewPilotConfig
from .git_providers import GitLabProvider, GitProviderError
from .models import RepoInfo
from .pr_commenter import (
    CommentAction,
    CommentError,
    GitLabNoteClient,
    build_summary_comment,
    gitlab_note_target_from_report,
)
from .pr_models import PullRequestInfo
from .report_models import ReviewReport
from .report_summary import should_fail_findings
from .report_writer import write_report
from .review_engine import ReviewEngine, ReviewEngineError, ReviewEngineOptions, ReviewInput
from .workspace import WorkspaceError, WorkspacePlan, build_existing_workspace_plan, prepare_workspace


class GitLabCIError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitLabCIContext:
    project_id: str
    project_path: str
    project_url: str
    merge_request_iid: int
    merge_request_url: str
    source_branch: str
    target_branch: str
    commit_sha: str
    pipeline_id: str | None = None
    job_id: str | None = None

    @property
    def repository(self) -> str:
        return self.project_path

    @property
    def pull_request_number(self) -> int:
        return self.merge_request_iid

    @property
    def event_name(self) -> str:
        return "merge_request_event"

    @property
    def base_ref(self) -> str:
        return self.target_branch

    @property
    def head_ref(self) -> str:
        return self.source_branch

    @property
    def head_sha(self) -> str:
        return self.commit_sha

    @property
    def base_sha(self) -> str:
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_name": self.event_name,
            "project_id": self.project_id,
            "project_path": self.project_path,
            "project_url": self.project_url,
            "merge_request_iid": self.merge_request_iid,
            "merge_request_url": self.merge_request_url,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "commit_sha": self.commit_sha,
            "pipeline_id": self.pipeline_id,
            "job_id": self.job_id,
        }


@dataclass(frozen=True)
class GitLabCIResult:
    context: GitLabCIContext
    pull_request: PullRequestInfo
    workspace: WorkspacePlan
    report: ReviewReport
    output_dir: Path
    markdown_path: Path
    json_path: Path
    dry_run: bool
    exit_code: int
    summary_comment: CommentAction | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "mode": "gitlab-ci-dry-run" if self.dry_run else "gitlab-ci",
            "context": self.context.to_dict(),
            "pull_request": self.pull_request.to_dict(),
            "workspace": self.workspace.to_dict(),
            "artifacts": {
                "output_dir": str(self.output_dir),
                "markdown": str(self.markdown_path),
                "json": str(self.json_path),
            },
            "report_summary": self.report.summary,
            "exit_code": self.exit_code,
        }
        if self.summary_comment is not None:
            payload["summary_comment"] = self.summary_comment.to_dict()
        return payload


def load_gitlab_ci_context(
    *,
    env: Mapping[str, str] | None = None,
) -> GitLabCIContext:
    active_env = env or os.environ
    project_id = _required_env(active_env, "CI_PROJECT_ID")
    project_path = _required_env(active_env, "CI_PROJECT_PATH")
    project_url = _required_env(active_env, "CI_PROJECT_URL")
    iid_raw = _required_env(active_env, "CI_MERGE_REQUEST_IID")
    try:
        iid = int(iid_raw)
    except ValueError as exc:
        raise GitLabCIError("CI_MERGE_REQUEST_IID must be an integer") from exc
    if iid < 1:
        raise GitLabCIError("CI_MERGE_REQUEST_IID must be positive")

    return GitLabCIContext(
        project_id=project_id,
        project_path=project_path,
        project_url=project_url,
        merge_request_iid=iid,
        merge_request_url=f"{project_url}/-/merge_requests/{iid}",
        source_branch=_required_env(active_env, "CI_MERGE_REQUEST_SOURCE_BRANCH_NAME"),
        target_branch=_required_env(active_env, "CI_MERGE_REQUEST_TARGET_BRANCH_NAME"),
        commit_sha=_required_env(active_env, "CI_COMMIT_SHA"),
        pipeline_id=active_env.get("CI_PIPELINE_ID"),
        job_id=active_env.get("CI_JOB_ID"),
    )


def run_gitlab_ci(
    *,
    output_dir: str | Path = "review-pilot-artifacts",
    dry_run: bool = False,
    fail_on: str | None = None,
    provider: GitLabProvider | None = None,
    llm_provider: str | None = None,
    post_summary_comment: bool = False,
    report_url: str | None = None,
    comment_client: GitLabNoteClient | None = None,
) -> GitLabCIResult:
    context = load_gitlab_ci_context()
    active_provider = provider or GitLabProvider()
    try:
        pr_info = active_provider.fetch_merge_request(context.project_id, context.merge_request_iid)
        workspace = prepare_workspace(_build_gitlab_workspace_plan(pr_info, dry_run))
    except GitProviderError as exc:
        raise GitLabCIError(f"gitlab error: {exc}") from exc
    except WorkspaceError as exc:
        raise GitLabCIError(f"workspace error: {exc}") from exc

    report = _build_gitlab_report(
        context=context,
        pr_info=pr_info,
        workspace=workspace,
        dry_run=dry_run,
        llm_provider=llm_provider,
    )

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = artifact_dir / "review-report.md"
    json_path = artifact_dir / "review-report.json"
    markdown_path.write_text(write_report(report, "markdown") + "\n", encoding="utf-8")
    json_path.write_text(write_report(report, "json") + "\n", encoding="utf-8")

    exit_code = 1 if should_fail_findings(report.findings, fail_on) else 0
    summary_comment = None
    if post_summary_comment:
        try:
            summary_comment = (comment_client or GitLabNoteClient()).upsert_summary_note(
                gitlab_note_target_from_report(report),
                build_summary_comment(report, report_url=report_url),
                dry_run=dry_run,
            )
        except CommentError as exc:
            raise GitLabCIError(f"gitlab note error: {exc}") from exc

    return GitLabCIResult(
        context=context,
        pull_request=pr_info,
        workspace=workspace,
        report=report,
        output_dir=artifact_dir,
        markdown_path=markdown_path,
        json_path=json_path,
        dry_run=dry_run,
        exit_code=exit_code,
        summary_comment=summary_comment,
    )


def _build_gitlab_report(
    *,
    context: GitLabCIContext,
    pr_info: PullRequestInfo,
    workspace: WorkspacePlan,
    dry_run: bool,
    llm_provider: str | None,
) -> ReviewReport:
    if dry_run and llm_provider is not None:
        raise GitLabCIError("gitlab-ci --provider requires a real workspace; remove --dry-run")

    metadata = {
        "pipeline": "gitlab-ci-dry-run" if dry_run else "gitlab-ci",
        "event_name": context.event_name,
        "project_id": context.project_id,
        "repository": context.project_path,
        "pull_request": context.merge_request_iid,
        "base_ref": pr_info.base.ref,
        "base_sha": pr_info.base.sha,
        "head_ref": pr_info.head.ref,
        "head_sha": pr_info.head.sha,
        "merge_request_url": context.merge_request_url,
        "pipeline_id": context.pipeline_id,
        "job_id": context.job_id,
        "workspace_path": workspace.workspace_path,
        "artifact_markdown": "review-report.md",
        "artifact_json": "review-report.json",
    }
    review_input = ReviewInput(
        repo_info=RepoInfo(
            root=workspace.workspace_path,
            branch=pr_info.head.ref,
            head=pr_info.head.sha,
            has_staged_changes=False,
            has_unstaged_changes=False,
        ),
        config=ReviewPilotConfig.default(),
        parsed_diff=pr_info.parsed_diff,
        input_source="gitlab-mr",
        metadata=metadata,
    )
    try:
        return ReviewEngine(
            ReviewEngineOptions(provider=llm_provider)
        ).run(review_input).report
    except ReviewEngineError as exc:
        raise GitLabCIError(str(exc)) from exc


def _build_gitlab_workspace_plan(
    pr_info: PullRequestInfo,
    dry_run: bool,
) -> WorkspacePlan:
    gitlab_workspace = os.environ.get("CI_PROJECT_DIR")
    if gitlab_workspace and not dry_run:
        plan = build_existing_workspace_plan(
            pr_info,
            workspace_path=gitlab_workspace,
            dry_run=False,
        )
        return WorkspacePlan(
            workspace_path=plan.workspace_path,
            repo_clone_url=plan.repo_clone_url,
            base_sha=plan.base_sha,
            head_sha=plan.head_sha,
            source=f"gitlab-ci-checkout:{pr_info.full_name}!{pr_info.number}",
            dry_run=plan.dry_run,
            commands=plan.commands,
        )
    from .workspace import build_workspace_plan

    plan = build_workspace_plan(
        pr_info,
        parent_dir=Path(".review-pilot") / "workspaces",
        dry_run=dry_run,
    )
    return WorkspacePlan(
        workspace_path=plan.workspace_path,
        repo_clone_url=plan.repo_clone_url,
        base_sha=plan.base_sha,
        head_sha=plan.head_sha,
        source=f"gitlab:{pr_info.full_name}!{pr_info.number}",
        dry_run=plan.dry_run,
        commands=plan.commands,
    )


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise GitLabCIError(f"{name} is required")
    return value
