from __future__ import annotations

import json
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from review_pilot.git_providers import (
    GitHubProvider,
    GitProviderError,
    parse_github_pr_url,
)


FIXTURES = Path("tests/fixtures/github")


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
        if url.endswith("/pulls/42"):
            return FakeResponse(_fixture("pull_request.json"))
        if url.endswith("/pulls/42/files?per_page=100"):
            return FakeResponse(_fixture("pull_request_files.json"))
        raise AssertionError(f"unexpected url: {url}")


def test_parse_github_pr_url_accepts_canonical_url() -> None:
    parsed = parse_github_pr_url("https://github.com/octo-org/review-demo/pull/42")

    assert parsed.owner == "octo-org"
    assert parsed.repo == "review-demo"
    assert parsed.number == 42


def test_parse_github_pr_url_rejects_non_pr_url() -> None:
    with pytest.raises(GitProviderError, match="expected GitHub PR URL"):
        parse_github_pr_url("https://github.com/octo-org/review-demo/issues/42")


def test_fetch_pull_request_builds_pr_info_and_diff(monkeypatch) -> None:
    transport = FakeTransport()
    provider = GitHubProvider(token="ghp_secret", transport=transport)

    pr_info = provider.fetch_pull_request(
        "https://github.com/octo-org/review-demo/pull/42"
    )

    assert pr_info.provider == "github"
    assert pr_info.full_name == "octo-org/review-demo"
    assert pr_info.title == "Tighten review output"
    assert pr_info.base.sha == "1111111111111111111111111111111111111111"
    assert pr_info.head.repo_clone_url == "https://github.com/contrib/review-demo.git"
    assert len(pr_info.files) == 2
    assert pr_info.raw_diff.text.startswith("diff --git a/src/review.py b/src/review.py")
    assert "new file mode 100644" in pr_info.raw_diff.text
    assert pr_info.parsed_diff.files[0].path == "src/review.py"
    assert pr_info.parsed_diff.files[1].change_type == "added"
    assert all(
        request.headers["Authorization"] == "Bearer ghp_secret"
        for request in transport.requests
    )
    assert "ghp_secret" not in json.dumps(pr_info.to_dict())


def test_fetch_pull_request_uses_public_headers_without_token() -> None:
    transport = FakeTransport()
    provider = GitHubProvider(token="", transport=transport)

    provider.fetch_pull_request("https://github.com/octo-org/review-demo/pull/42")

    assert all("Authorization" not in request.headers for request in transport.requests)


def test_rate_limit_error_is_readable() -> None:
    def transport(request, timeout: float):
        headers = Message()
        headers["x-ratelimit-remaining"] = "0"
        headers["x-ratelimit-reset"] = "1893456000"
        raise HTTPError(
            request.full_url,
            403,
            "rate limited",
            headers,
            None,
        )

    provider = GitHubProvider(token="token", transport=transport)

    with pytest.raises(GitProviderError, match="github rate limit exceeded"):
        provider.fetch_pull_request("https://github.com/octo-org/review-demo/pull/42")


def test_permission_error_is_readable() -> None:
    def transport(request, timeout: float):
        raise HTTPError(request.full_url, 403, "forbidden", Message(), None)

    provider = GitHubProvider(token="token", transport=transport)

    with pytest.raises(GitProviderError, match="permission denied"):
        provider.fetch_pull_request("https://github.com/octo-org/review-demo/pull/42")


def _fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
