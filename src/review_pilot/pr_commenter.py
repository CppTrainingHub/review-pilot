from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .report_models import Finding, ReviewReport
from .report_summary import SEVERITY_ORDER


SUMMARY_COMMENT_MARKER = "<!-- review-pilot-summary -->"


class CommentError(RuntimeError):
    pass


class HttpTransport(Protocol):
    def __call__(self, request: Request, timeout: float) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class CommentTarget:
    owner: str
    repo: str
    issue_number: int


@dataclass(frozen=True)
class GitLabNoteTarget:
    project_id_or_path: str | int
    merge_request_iid: int


@dataclass(frozen=True)
class IssueComment:
    id: int
    body: str
    user_login: str | None = None


@dataclass(frozen=True)
class CommentAction:
    action: str
    comment_id: int | None
    url: str | None
    body: str
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "comment_id": self.comment_id,
            "url": self.url,
            "dry_run": self.dry_run,
            "body": self.body,
        }


class GitHubCommentClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        api_base_url: str = "https://api.github.com",
        transport: HttpTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self.api_base_url = api_base_url.rstrip("/")
        self.transport = transport or urlopen
        self.timeout = timeout

    def list_comments(self, target: CommentTarget) -> list[IssueComment]:
        payload = self._request_json(
            "GET",
            f"/repos/{target.owner}/{target.repo}/issues/{target.issue_number}/comments?per_page=100",
        )
        if not isinstance(payload, list):
            raise CommentError("github comments response is not a list")
        return [_build_comment(item) for item in payload]

    def create_comment(self, target: CommentTarget, body: str) -> CommentAction:
        payload = self._request_json(
            "POST",
            f"/repos/{target.owner}/{target.repo}/issues/{target.issue_number}/comments",
            {"body": body},
        )
        return CommentAction(
            action="create",
            comment_id=_optional_int(payload, "id"),
            url=_optional_str(payload, "html_url"),
            body=body,
            dry_run=False,
        )

    def update_comment(
        self,
        target: CommentTarget,
        comment_id: int,
        body: str,
    ) -> CommentAction:
        payload = self._request_json(
            "PATCH",
            f"/repos/{target.owner}/{target.repo}/issues/comments/{comment_id}",
            {"body": body},
        )
        return CommentAction(
            action="update",
            comment_id=_optional_int(payload, "id") or comment_id,
            url=_optional_str(payload, "html_url"),
            body=body,
            dry_run=False,
        )

    def upsert_summary_comment(
        self,
        target: CommentTarget,
        body: str,
        *,
        dry_run: bool = True,
    ) -> CommentAction:
        if SUMMARY_COMMENT_MARKER not in body:
            body = f"{SUMMARY_COMMENT_MARKER}\n\n{body}"

        if dry_run:
            return CommentAction(
                action="dry-run",
                comment_id=None,
                url=None,
                body=body,
                dry_run=True,
            )

        existing = next(
            (
                comment
                for comment in self.list_comments(target)
                if SUMMARY_COMMENT_MARKER in comment.body
            ),
            None,
        )
        if existing is not None:
            return self.update_comment(target, existing.id, body)
        return self.create_comment(target, body)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body: bytes | None = None
        headers = self._headers()
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.api_base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self.transport(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise _http_error_to_comment_error(exc) from exc
        except URLError as exc:
            raise CommentError(f"github comment request failed: {exc.reason}") from exc

        if not response_body.strip():
            return None
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise CommentError("github comment response is not valid JSON") from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "review-pilot",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


class GitLabNoteClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        api_base_url: str | None = None,
        transport: HttpTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.token = token if token is not None else os.environ.get("GITLAB_TOKEN")
        self.api_base_url = (
            api_base_url
            or os.environ.get("GITLAB_API_BASE_URL")
            or os.environ.get("CI_API_V4_URL")
            or "https://gitlab.com/api/v4"
        ).rstrip("/")
        self.transport = transport or urlopen
        self.timeout = timeout

    def list_notes(self, target: GitLabNoteTarget) -> list[IssueComment]:
        payload = self._request_json(
            "GET",
            f"/projects/{_quote_gitlab_project(target.project_id_or_path)}"
            f"/merge_requests/{target.merge_request_iid}/notes?per_page=100",
        )
        if not isinstance(payload, list):
            raise CommentError("gitlab notes response is not a list")
        return [_build_gitlab_note(item) for item in payload]

    def create_note(self, target: GitLabNoteTarget, body: str) -> CommentAction:
        payload = self._request_json(
            "POST",
            f"/projects/{_quote_gitlab_project(target.project_id_or_path)}"
            f"/merge_requests/{target.merge_request_iid}/notes",
            {"body": body},
        )
        return CommentAction(
            action="create",
            comment_id=_optional_int(payload, "id"),
            url=_optional_str(payload, "web_url"),
            body=body,
            dry_run=False,
        )

    def update_note(
        self,
        target: GitLabNoteTarget,
        note_id: int,
        body: str,
    ) -> CommentAction:
        payload = self._request_json(
            "PUT",
            f"/projects/{_quote_gitlab_project(target.project_id_or_path)}"
            f"/merge_requests/{target.merge_request_iid}/notes/{note_id}",
            {"body": body},
        )
        return CommentAction(
            action="update",
            comment_id=_optional_int(payload, "id") or note_id,
            url=_optional_str(payload, "web_url"),
            body=body,
            dry_run=False,
        )

    def upsert_summary_note(
        self,
        target: GitLabNoteTarget,
        body: str,
        *,
        dry_run: bool = True,
    ) -> CommentAction:
        if SUMMARY_COMMENT_MARKER not in body:
            body = f"{SUMMARY_COMMENT_MARKER}\n\n{body}"

        if dry_run:
            return CommentAction(
                action="dry-run",
                comment_id=None,
                url=None,
                body=body,
                dry_run=True,
            )

        existing = next(
            (
                note
                for note in self.list_notes(target)
                if SUMMARY_COMMENT_MARKER in note.body
            ),
            None,
        )
        if existing is not None:
            return self.update_note(target, existing.id, body)
        return self.create_note(target, body)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body: bytes | None = None
        headers = self._headers()
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.api_base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self.transport(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise _http_error_to_gitlab_comment_error(exc) from exc
        except URLError as exc:
            raise CommentError(f"gitlab note request failed: {exc.reason}") from exc

        if not response_body.strip():
            return None
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise CommentError("gitlab note response is not valid JSON") from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "review-pilot",
        }
        if self.token:
            headers["PRIVATE-TOKEN"] = self.token
        return headers


def build_summary_comment(
    report: ReviewReport,
    *,
    report_url: str | None = None,
) -> str:
    summary = report.summary
    repo_info = report.repo_info or {}
    repository = str(repo_info.get("repository") or repo_info.get("root") or "unknown")
    pull_request = repo_info.get("pull_request")
    highest = summary.get("highest_severity") or "none"
    total = int(summary.get("total_findings") or 0)
    risk = _risk_label(str(highest), total)

    lines = [
        SUMMARY_COMMENT_MARKER,
        "",
        "## Review Pilot Summary",
        "",
        f"- Repository: `{repository}`",
    ]
    if pull_request:
        lines.append(f"- Pull Request: `#{pull_request}`")
    lines.extend(
        [
            f"- Risk: **{risk}**",
            f"- Total findings: **{total}**",
            f"- Highest severity: **{highest}**",
            "",
            "### Severity",
            "",
        ]
    )

    severity_counts = summary.get("severity_counts")
    if not isinstance(severity_counts, dict):
        severity_counts = {}
    for severity in SEVERITY_ORDER:
        lines.append(f"- {severity}: {int(severity_counts.get(severity, 0))}")

    if report_url:
        lines.extend(["", f"[Open full report]({report_url})"])

    top_findings = _top_findings(report.findings)
    if top_findings:
        lines.extend(["", "### Top findings", ""])
        for finding in top_findings:
            lines.append(f"- [{finding.severity}] {_format_location(finding)} {finding.message}")

    return "\n".join(lines)


def comment_target_from_report(report: ReviewReport) -> CommentTarget:
    repo_info = report.repo_info or {}
    repository = repo_info.get("repository")
    pull_request = repo_info.get("pull_request")
    if not isinstance(repository, str) or "/" not in repository:
        raise CommentError("report repo_info.repository must look like OWNER/REPO")
    if not isinstance(pull_request, int) or pull_request < 1:
        raise CommentError("report repo_info.pull_request must be a positive integer")
    owner, repo = repository.split("/", 1)
    return CommentTarget(owner=owner, repo=repo, issue_number=pull_request)


def gitlab_note_target_from_report(report: ReviewReport) -> GitLabNoteTarget:
    repo_info = report.repo_info or {}
    project_id = repo_info.get("project_id")
    repository = repo_info.get("repository")
    pull_request = repo_info.get("pull_request")
    target_project: str | int | None = None
    if isinstance(project_id, int):
        target_project = project_id
    elif isinstance(project_id, str) and project_id:
        target_project = project_id
    elif isinstance(repository, str) and "/" in repository:
        target_project = repository
    if target_project is None:
        raise CommentError("report repo_info.project_id or repository is required for GitLab")
    if not isinstance(pull_request, int) or pull_request < 1:
        raise CommentError("report repo_info.pull_request must be a positive integer")
    return GitLabNoteTarget(project_id_or_path=target_project, merge_request_iid=pull_request)


def _build_comment(payload: Any) -> IssueComment:
    if not isinstance(payload, dict):
        raise CommentError("github comment item is not an object")
    comment_id = _optional_int(payload, "id")
    body = _optional_str(payload, "body")
    if comment_id is None or body is None:
        raise CommentError("github comment item is missing id or body")
    user = payload.get("user")
    login = None
    if isinstance(user, dict):
        login = _optional_str(user, "login")
    return IssueComment(id=comment_id, body=body, user_login=login)


def _build_gitlab_note(payload: Any) -> IssueComment:
    if not isinstance(payload, dict):
        raise CommentError("gitlab note item is not an object")
    note_id = _optional_int(payload, "id")
    body = _optional_str(payload, "body")
    if note_id is None or body is None:
        raise CommentError("gitlab note item is missing id or body")
    author = payload.get("author")
    login = None
    if isinstance(author, dict):
        login = _optional_str(author, "username")
    return IssueComment(id=note_id, body=body, user_login=login)


def _optional_int(payload: Any, key: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, int) else None


def _optional_str(payload: Any, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _http_error_to_comment_error(exc: HTTPError) -> CommentError:
    if exc.code in {401, 403}:
        return CommentError(
            "github comment permission denied; check GITHUB_TOKEN and pull-requests write permission"
        )
    if exc.code == 404:
        return CommentError("github comment target not found or token has no access")
    return CommentError(f"github comment request failed with HTTP {exc.code}")


def _http_error_to_gitlab_comment_error(exc: HTTPError) -> CommentError:
    if exc.code in {401, 403}:
        return CommentError(
            "gitlab note permission denied; check GITLAB_TOKEN merge request write permission"
        )
    if exc.code == 404:
        return CommentError("gitlab note target not found or token has no access")
    return CommentError(f"gitlab note request failed with HTTP {exc.code}")


def _quote_gitlab_project(project_id_or_path: str | int) -> str:
    from urllib.parse import quote

    if isinstance(project_id_or_path, int):
        return str(project_id_or_path)
    return quote(project_id_or_path, safe="")


def _risk_label(highest: str, total: int) -> str:
    if total == 0:
        return "clean"
    if highest in {"P0", "P1"}:
        return "high"
    if highest == "P2":
        return "medium"
    return "low"


def _top_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda finding: finding.severity_rank)[:3]


def _format_location(finding: Finding) -> str:
    if finding.file_path and finding.line_no:
        return f"`{finding.file_path}:{finding.line_no}`:"
    if finding.file_path:
        return f"`{finding.file_path}`:"
    return ""
