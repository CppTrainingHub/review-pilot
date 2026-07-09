from __future__ import annotations

import json
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from review_pilot.git_providers import (
    GitLabProvider,
    GitProviderError,
    parse_gitlab_mr_url,
)


FIXTURES = Path("tests/fixtures/gitlab")


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeTransport:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        url = request.full_url
        if url.endswith("/projects/cpp-camp%2Fspeed-logger/merge_requests/7"):
            return FakeResponse(_fixture("merge_request.json"))
        if url.endswith("/projects/cpp-camp%2Fspeed-logger/merge_requests/7/changes"):
            return FakeResponse(_fixture("changes.json"))
        if url.endswith("/projects/84181645"):
            return FakeResponse(_fixture("project.json"))
        raise AssertionError(f"unexpected url: {url}")


def test_parse_gitlab_mr_url_accepts_canonical_url() -> None:
    parsed = parse_gitlab_mr_url(
        "https://gitlab.com/cpp-camp/speed-logger/-/merge_requests/7"
    )

    assert parsed.base_url == "https://gitlab.com"
    assert parsed.project_path == "cpp-camp/speed-logger"
    assert parsed.iid == 7


def test_parse_gitlab_mr_url_rejects_non_mr_url() -> None:
    with pytest.raises(GitProviderError, match="expected GitLab MR URL"):
        parse_gitlab_mr_url("https://gitlab.com/cpp-camp/speed-logger/-/issues/7")


def test_fetch_merge_request_builds_pr_info_and_diff() -> None:
    transport = FakeTransport()
    provider = GitLabProvider(token="glpat_secret", transport=transport)

    pr_info = provider.fetch_merge_request("cpp-camp/speed-logger", 7)

    assert pr_info.provider == "gitlab"
    assert pr_info.full_name == "cpp-camp/speed-logger"
    assert pr_info.title == "Remove debug output from logger fanout path"
    assert pr_info.base.sha == "1111111111111111111111111111111111111111"
    assert pr_info.head.sha == "2222222222222222222222222222222222222222"
    assert pr_info.head.repo_clone_url == "https://gitlab.com/cpp-camp/speed-logger.git"
    assert len(pr_info.files) == 2
    assert pr_info.files[0].additions == 1
    assert pr_info.files[1].status == "added"
    assert pr_info.raw_diff.text.startswith("diff --git a/src/logger.cpp b/src/logger.cpp")
    assert pr_info.parsed_diff.files[0].path == "src/logger.cpp"
    assert all(
        request.headers["Private-token"] == "glpat_secret"
        for request in transport.requests
    )
    assert "glpat_secret" not in json.dumps(pr_info.to_dict())


def test_fetch_merge_request_accepts_start_sha_when_base_sha_is_null() -> None:
    class StartShaTransport(FakeTransport):
        def __call__(self, request, timeout: float) -> FakeResponse:
            response = super().__call__(request, timeout)
            if request.full_url.endswith("/projects/cpp-camp%2Fspeed-logger/merge_requests/7"):
                payload = dict(response.payload)
                payload.pop("diff_base_sha", None)
                payload["diff_refs"] = {
                    "base_sha": None,
                    "head_sha": "2222222222222222222222222222222222222222",
                    "start_sha": "1111111111111111111111111111111111111111",
                }
                return FakeResponse(payload)
            return response

    provider = GitLabProvider(token="glpat_secret", transport=StartShaTransport())

    pr_info = provider.fetch_merge_request("cpp-camp/speed-logger", 7)

    assert pr_info.base.sha == "1111111111111111111111111111111111111111"
    assert pr_info.head.sha == "2222222222222222222222222222222222222222"


def test_gitlab_permission_error_is_readable() -> None:
    def transport(request, timeout: float):
        raise HTTPError(request.full_url, 403, "forbidden", Message(), None)

    provider = GitLabProvider(token="token", transport=transport)

    with pytest.raises(GitProviderError, match="gitlab permission denied"):
        provider.fetch_merge_request("cpp-camp/speed-logger", 7)


def _fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
