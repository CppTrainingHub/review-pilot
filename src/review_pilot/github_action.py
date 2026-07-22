from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import ReviewPilotConfig
from .git_providers import GitHubProvider, GitProviderError
from .models import RepoInfo
from .pr_commenter import CommentAction, build_summary_comment, comment_target_from_report
from .pr_commenter import GitHubCommentClient, CommentError
from .pr_models import PullRequestFile, PullRequestInfo, PullRequestRef
from .report_models import ReviewReport
from .report_summary import should_fail_findings
from .report_writer import write_report
from .review_engine import ReviewEngine, ReviewEngineError, ReviewEngineOptions, ReviewInput
from .workspace import (
    WorkspaceError,
    WorkspacePlan,
    build_existing_workspace_plan,
    build_workspace_plan,
    prepare_workspace,
)


class GitHubActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitHubActionContext:
    event_name: str
    repository: str
    owner: str
    repo: str
    pull_request_number: int
    pull_request_url: str
    base_ref: str
    base_sha: str
    head_ref: str
    head_sha: str
    action: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_name": self.event_name,
            "repository": self.repository,
            "owner": self.owner,
            "repo": self.repo,
            "pull_request_number": self.pull_request_number,
            "pull_request_url": self.pull_request_url,
            "base_ref": self.base_ref,
            "base_sha": self.base_sha,
            "head_ref": self.head_ref,
            "head_sha": self.head_sha,
            "action": self.action,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class GitHubActionResult:
    context: GitHubActionContext
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
            "mode": "github-action-dry-run" if self.dry_run else "github-action",
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


def load_github_action_context(
    event_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> GitHubActionContext:
    active_env = env or os.environ
    event = load_github_event(event_path)

    pull_request = event.get("pull_request")
    if not isinstance(pull_request, dict):
        raise GitHubActionError("GitHub event must contain pull_request")
    repository = event.get("repository")
    if not isinstance(repository, dict):
        raise GitHubActionError("GitHub event must contain repository")

    full_name = _required_str(repository, "full_name", "repository.full_name")
    owner, repo = _split_full_name(full_name)
    base = _required_dict(pull_request, "base", "pull_request.base")
    head = _required_dict(pull_request, "head", "pull_request.head")
    number = pull_request.get("number")
    if not isinstance(number, int) or number < 1:
        raise GitHubActionError("pull_request.number must be a positive integer")

    html_url = _required_str(pull_request, "html_url", "pull_request.html_url")
    return GitHubActionContext(
        event_name=active_env.get("GITHUB_EVENT_NAME", "pull_request"),
        repository=full_name,
        owner=owner,
        repo=repo,
        pull_request_number=number,
        pull_request_url=html_url,
        base_ref=_required_str(base, "ref", "pull_request.base.ref"),
        base_sha=_required_str(base, "sha", "pull_request.base.sha"),
        head_ref=_required_str(head, "ref", "pull_request.head.ref"),
        head_sha=_required_str(head, "sha", "pull_request.head.sha"),
        action=event.get("action") if isinstance(event.get("action"), str) else None,
        run_id=active_env.get("GITHUB_RUN_ID"),
    )


def run_github_action(
    *,
    event_path: str | Path,
    output_dir: str | Path = "review-pilot-artifacts",
    dry_run: bool = True,
    fail_on: str | None = None,
    provider: GitHubProvider | None = None,
    llm_provider: str | None = None,
    post_summary_comment: bool = False,
    report_url: str | None = None,
    comment_client: GitHubCommentClient | None = None,
) -> GitHubActionResult:
    context = load_github_action_context(event_path)
    active_provider = provider or GitHubProvider()
    try:
        pr_info = _fixture_pull_request(event_path, context) or active_provider.fetch_pull_request(
            context.pull_request_url
        )
        workspace = prepare_workspace(_build_action_workspace_plan(pr_info, output_dir, dry_run))
    except GitProviderError as exc:
        raise GitHubActionError(f"github error: {exc}") from exc
    except WorkspaceError as exc:
        raise GitHubActionError(f"workspace error: {exc}") from exc

    report = build_artifact_report(
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
            summary_comment = (comment_client or GitHubCommentClient()).upsert_summary_comment(
                comment_target_from_report(report),
                build_summary_comment(report, report_url=report_url),
                dry_run=dry_run,
            )
        except CommentError as exc:
            raise GitHubActionError(f"github comment error: {exc}") from exc

    return GitHubActionResult(
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


def load_github_event(event_path: str | Path) -> dict[str, Any]:
    path = Path(event_path)
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GitHubActionError(f"cannot read GitHub event file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GitHubActionError(f"GitHub event file is not valid JSON: {path}") from exc
    if not isinstance(event, dict):
        raise GitHubActionError("GitHub event file must contain a JSON object")
    return event


def _build_action_workspace_plan(
    pr_info: PullRequestInfo,
    output_dir: str | Path,
    dry_run: bool,
) -> WorkspacePlan:
    github_workspace = os.environ.get("GITHUB_WORKSPACE")
    if github_workspace and not dry_run:
        return build_existing_workspace_plan(
            pr_info,
            workspace_path=github_workspace,
            dry_run=False,
        )
    return build_workspace_plan(
        pr_info,
        parent_dir=Path(output_dir) / "workspaces",
        dry_run=dry_run,
    )


def build_artifact_report(
    *,
    context: GitHubActionContext,
    pr_info: PullRequestInfo,
    workspace: WorkspacePlan,
    dry_run: bool,
    llm_provider: str | None = None,
) -> ReviewReport:
    if dry_run and llm_provider is not None:
        raise GitHubActionError("github-action --provider requires a real workspace; remove --dry-run")

    metadata: dict[str, Any] = {
        "pipeline": "github-action-dry-run" if dry_run else "github-action",
        "event_name": context.event_name,
        "repository": context.repository,
        "pull_request": context.pull_request_number,
        "base_ref": context.base_ref,
        "base_sha": context.base_sha,
        "head_ref": context.head_ref,
        "head_sha": context.head_sha,
        "workspace_path": workspace.workspace_path,
        "artifact_markdown": "review-report.md",
        "artifact_json": "review-report.json",
    }
    review_input = ReviewInput(
        repo_info=RepoInfo(
            root=workspace.workspace_path,
            branch=context.head_ref,
            head=context.head_sha,
            has_staged_changes=False,
            has_unstaged_changes=False,
        ),
        config=ReviewPilotConfig.default(),
        parsed_diff=pr_info.parsed_diff,
        input_source="github-pr",
        metadata=metadata,
    )
    try:
        return ReviewEngine(
            ReviewEngineOptions(provider=llm_provider)
        ).run(review_input).report
    except ReviewEngineError as exc:
        raise GitHubActionError(str(exc)) from exc


def _required_str(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GitHubActionError(f"{label} must be a non-empty string")
    return value


def _required_dict(payload: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise GitHubActionError(f"{label} must be an object")
    return value


def _split_full_name(full_name: str) -> tuple[str, str]:
    parts = full_name.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise GitHubActionError("repository.full_name must look like OWNER/REPO")
    return parts[0], parts[1]


def _fixture_pull_request(
    event_path: str | Path,
    context: GitHubActionContext,
) -> PullRequestInfo | None:
    event = load_github_event(event_path)
    fixture = event.get("review_pilot_fixture")
    if not isinstance(fixture, dict):
        return None
    pr_payload = fixture.get("pull_request")
    files_payload = fixture.get("pull_request_files")
    if not isinstance(pr_payload, dict) or not isinstance(files_payload, list):
        raise GitHubActionError(
            "review_pilot_fixture must contain pull_request and pull_request_files"
        )

    return PullRequestInfo(
        provider="github",
        url=_required_str(pr_payload, "html_url", "review_pilot_fixture.pull_request.html_url"),
        owner=context.owner,
        repo=context.repo,
        number=context.pull_request_number,
        title=_required_str(pr_payload, "title", "review_pilot_fixture.pull_request.title"),
        state=_required_str(pr_payload, "state", "review_pilot_fixture.pull_request.state"),
        base=_fixture_ref(pr_payload, "base"),
        head=_fixture_ref(pr_payload, "head"),
        files=tuple(_fixture_file(item) for item in files_payload),
    )


def _fixture_ref(payload: dict[str, Any], key: str) -> PullRequestRef:
    value = _required_dict(payload, key, f"review_pilot_fixture.pull_request.{key}")
    repo = _required_dict(value, "repo", f"review_pilot_fixture.pull_request.{key}.repo")
    return PullRequestRef(
        label=_required_str(value, "label", f"review_pilot_fixture.pull_request.{key}.label"),
        ref=_required_str(value, "ref", f"review_pilot_fixture.pull_request.{key}.ref"),
        sha=_required_str(value, "sha", f"review_pilot_fixture.pull_request.{key}.sha"),
        repo_full_name=_required_str(repo, "full_name", f"review_pilot_fixture.pull_request.{key}.repo.full_name"),
        repo_clone_url=_required_str(repo, "clone_url", f"review_pilot_fixture.pull_request.{key}.repo.clone_url"),
    )


def _fixture_file(payload: Any) -> PullRequestFile:
    if not isinstance(payload, dict):
        raise GitHubActionError("review_pilot_fixture.pull_request_files items must be objects")
    return PullRequestFile(
        filename=_required_str(payload, "filename", "review_pilot_fixture.pull_request_files.filename"),
        status=_required_str(payload, "status", "review_pilot_fixture.pull_request_files.status"),
        additions=int(payload.get("additions", 0)),
        deletions=int(payload.get("deletions", 0)),
        changes=int(payload.get("changes", 0)),
        patch=payload.get("patch"),
        previous_filename=payload.get("previous_filename"),
    )
