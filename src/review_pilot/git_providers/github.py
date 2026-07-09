from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from review_pilot.git_providers.base import GitProviderError
from review_pilot.pr_models import (
    PullRequestFile,
    PullRequestInfo,
    PullRequestRef,
)


_GITHUB_PR_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)/?$"
)


@dataclass(frozen=True)
class GitHubPullRequestURL:
    owner: str
    repo: str
    number: int


class HttpTransport(Protocol):
    def __call__(self, request: Request, timeout: float) -> Any:
        raise NotImplementedError


class GitHubProvider:
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

    def fetch_pull_request(self, url: str) -> PullRequestInfo:
        parsed = parse_github_pr_url(url)
        pr_payload = self._get_json(
            f"/repos/{parsed.owner}/{parsed.repo}/pulls/{parsed.number}"
        )
        files_payload = self._get_json(
            f"/repos/{parsed.owner}/{parsed.repo}/pulls/{parsed.number}/files?per_page=100"
        )
        if not isinstance(files_payload, list):
            raise GitProviderError("github response for pull request files is not a list")

        return PullRequestInfo(
            provider="github",
            url=pr_payload.get("html_url") or url,
            owner=parsed.owner,
            repo=parsed.repo,
            number=parsed.number,
            title=_required_str(pr_payload, "title"),
            state=_required_str(pr_payload, "state"),
            base=_build_ref(pr_payload, "base"),
            head=_build_ref(pr_payload, "head"),
            files=tuple(_build_file(item) for item in files_payload),
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
            raise GitProviderError(f"github request failed: {exc.reason}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise GitProviderError("github response is not valid JSON") from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "review-pilot",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


def parse_github_pr_url(url: str) -> GitHubPullRequestURL:
    match = _GITHUB_PR_RE.match(url)
    if not match:
        raise GitProviderError(
            "expected GitHub PR URL like https://github.com/OWNER/REPO/pull/123"
        )
    return GitHubPullRequestURL(
        owner=match.group("owner"),
        repo=match.group("repo"),
        number=int(match.group("number")),
    )


def _build_ref(payload: dict[str, Any], key: str) -> PullRequestRef:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise GitProviderError(f"github response is missing {key} ref")
    repo = value.get("repo")
    if not isinstance(repo, dict):
        raise GitProviderError(f"github response is missing {key}.repo")
    return PullRequestRef(
        label=_required_str(value, "label"),
        ref=_required_str(value, "ref"),
        sha=_required_str(value, "sha"),
        repo_full_name=_required_str(repo, "full_name"),
        repo_clone_url=_required_str(repo, "clone_url"),
    )


def _build_file(payload: dict[str, Any]) -> PullRequestFile:
    return PullRequestFile(
        filename=_required_str(payload, "filename"),
        status=_required_str(payload, "status"),
        additions=int(payload.get("additions", 0)),
        deletions=int(payload.get("deletions", 0)),
        changes=int(payload.get("changes", 0)),
        patch=payload.get("patch"),
        previous_filename=payload.get("previous_filename"),
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GitProviderError(f"github response is missing string field: {key}")
    return value


def _http_error_to_provider_error(exc: HTTPError) -> GitProviderError:
    headers = exc.headers or {}
    if headers.get("x-ratelimit-remaining") == "0":
        reset = headers.get("x-ratelimit-reset")
        suffix = f"; reset={reset}" if reset else ""
        return GitProviderError(f"github rate limit exceeded{suffix}")

    if exc.code in {401, 403}:
        return GitProviderError(
            "github permission denied; check GITHUB_TOKEN scope and repository access"
        )

    if exc.code == 404:
        return GitProviderError("github pull request not found or token has no access")

    return GitProviderError(f"github request failed with HTTP {exc.code}")
