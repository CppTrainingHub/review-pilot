from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from review_pilot.git_providers.base import GitProviderError
from review_pilot.pr_models import (
    PullRequestFile,
    PullRequestInfo,
    PullRequestRef,
)


_GITLAB_MR_RE = re.compile(
    r"^(?P<base>https?://[^/]+)/(?P<namespace>.+)/-/merge_requests/(?P<iid>\d+)/?$"
)


@dataclass(frozen=True)
class GitLabMergeRequestURL:
    base_url: str
    namespace: str
    iid: int

    @property
    def project_path(self) -> str:
        return self.namespace


class HttpTransport(Protocol):
    def __call__(self, request: Request, timeout: float) -> Any:
        raise NotImplementedError


class GitLabProvider:
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

    def fetch_pull_request(self, url: str) -> PullRequestInfo:
        parsed = parse_gitlab_mr_url(url)
        return self.fetch_merge_request(parsed.project_path, parsed.iid)

    def fetch_merge_request(self, project_id_or_path: str | int, iid: int) -> PullRequestInfo:
        project_id = _quote_project(project_id_or_path)
        mr_payload = self._get_json(
            f"/projects/{project_id}/merge_requests/{iid}"
        )
        changes_payload = self._get_json(
            f"/projects/{project_id}/merge_requests/{iid}/changes"
        )
        changes = changes_payload.get("changes") if isinstance(changes_payload, dict) else None
        if not isinstance(changes, list):
            raise GitProviderError("gitlab response for merge request changes is not a list")

        target_project_id = _required_int(mr_payload, "target_project_id")
        project_payload = self._get_json(f"/projects/{target_project_id}")
        path_with_namespace = _required_str(project_payload, "path_with_namespace")
        web_url = _required_str(project_payload, "web_url")
        http_url_to_repo = _required_str(project_payload, "http_url_to_repo")

        source_project_id = mr_payload.get("source_project_id")
        source_project_payload = project_payload
        if source_project_id != target_project_id and source_project_id is not None:
            source_project_payload = self._get_json(f"/projects/{source_project_id}")

        source_path = _required_str(source_project_payload, "path_with_namespace")
        source_http_url = _required_str(source_project_payload, "http_url_to_repo")
        diff_refs = mr_payload.get("diff_refs")
        if not isinstance(diff_refs, dict):
            diff_refs = {}
        return PullRequestInfo(
            provider="gitlab",
            url=_required_str(mr_payload, "web_url"),
            owner=path_with_namespace.rsplit("/", 1)[0],
            repo=path_with_namespace.rsplit("/", 1)[1],
            number=int(_required_int(mr_payload, "iid")),
            title=_required_str(mr_payload, "title"),
            state=_required_str(mr_payload, "state"),
            base=PullRequestRef(
                label=f"{path_with_namespace}:{_required_str(mr_payload, 'target_branch')}",
                ref=_required_str(mr_payload, "target_branch"),
                sha=_required_ref_sha(
                    diff_refs,
                    ("base_sha", "start_sha"),
                    mr_payload,
                    ("diff_base_sha",),
                ),
                repo_full_name=path_with_namespace,
                repo_clone_url=http_url_to_repo,
            ),
            head=PullRequestRef(
                label=f"{source_path}:{_required_str(mr_payload, 'source_branch')}",
                ref=_required_str(mr_payload, "source_branch"),
                sha=_required_ref_sha(
                    diff_refs,
                    ("head_sha",),
                    mr_payload,
                    ("sha", "merge_commit_sha", "squash_commit_sha"),
                ),
                repo_full_name=source_path,
                repo_clone_url=source_http_url,
            ),
            files=tuple(_build_file(item) for item in changes),
        )

    def _get_json(self, path: str) -> Any:
        request = Request(
            f"{self.api_base_url}{path}",
            headers=self._headers(),
            method="GET",
        )
        try:
            with self.transport(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise _http_error_to_provider_error(exc) from exc
        except URLError as exc:
            raise GitProviderError(f"gitlab request failed: {exc.reason}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise GitProviderError("gitlab response is not valid JSON") from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "review-pilot",
        }
        if self.token:
            headers["PRIVATE-TOKEN"] = self.token
        return headers


def parse_gitlab_mr_url(url: str) -> GitLabMergeRequestURL:
    match = _GITLAB_MR_RE.match(url)
    if not match:
        raise GitProviderError(
            "expected GitLab MR URL like https://gitlab.com/GROUP/PROJECT/-/merge_requests/123"
        )
    return GitLabMergeRequestURL(
        base_url=match.group("base"),
        namespace=match.group("namespace"),
        iid=int(match.group("iid")),
    )


def _build_file(payload: Any) -> PullRequestFile:
    if not isinstance(payload, dict):
        raise GitProviderError("gitlab change item is not an object")
    old_path = _required_str(payload, "old_path")
    new_path = _required_str(payload, "new_path")
    renamed = bool(payload.get("renamed_file"))
    deleted = bool(payload.get("deleted_file"))
    added = bool(payload.get("new_file"))
    if added:
        status = "added"
    elif deleted:
        status = "removed"
    elif renamed:
        status = "renamed"
    else:
        status = "modified"
    patch = _optional_str(payload, "diff")
    additions, deletions = _count_patch_lines(patch)
    return PullRequestFile(
        filename=new_path,
        status=status,
        additions=additions,
        deletions=deletions,
        changes=additions + deletions,
        patch=patch,
        previous_filename=old_path if renamed else None,
    )


def _quote_project(project_id_or_path: str | int) -> str:
    if isinstance(project_id_or_path, int):
        return str(project_id_or_path)
    return quote(str(project_id_or_path), safe="")


def _required_str(payload: Any, key: str) -> str:
    if not isinstance(payload, dict):
        raise GitProviderError("gitlab response item is not an object")
    value = payload.get(key)
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str) or not value:
        raise GitProviderError(f"gitlab response is missing string field: {key}")
    return value


def _required_int(payload: Any, key: str) -> int:
    if not isinstance(payload, dict):
        raise GitProviderError("gitlab response item is not an object")
    value = payload.get(key)
    if not isinstance(value, int):
        raise GitProviderError(f"gitlab response is missing integer field: {key}")
    return value


def _required_ref_sha(
    diff_refs: dict[str, Any],
    diff_keys: tuple[str, ...],
    payload: dict[str, Any],
    fallback_keys: tuple[str, ...],
) -> str:
    for key in diff_keys:
        value = diff_refs.get(key)
        if isinstance(value, str) and value:
            return value
    for key in fallback_keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    expected = ", ".join((*diff_keys, *fallback_keys))
    raise GitProviderError(f"gitlab response is missing ref sha field: {expected}")


def _optional_str(payload: Any, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _count_patch_lines(patch: str | None) -> tuple[int, int]:
    additions = 0
    deletions = 0
    for line in (patch or "").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def _http_error_to_provider_error(exc: HTTPError) -> GitProviderError:
    if exc.code in {401, 403}:
        return GitProviderError(
            "gitlab permission denied; check GITLAB_TOKEN scope and project access"
        )
    if exc.code == 404:
        return GitProviderError("gitlab merge request not found or token has no access")
    return GitProviderError(f"gitlab request failed with HTTP {exc.code}")
