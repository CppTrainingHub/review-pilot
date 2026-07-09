from __future__ import annotations

import json
from urllib.request import Request

import pytest

from review_pilot.pr_commenter import (
    SUMMARY_COMMENT_MARKER,
    CommentTarget,
    GitLabNoteClient,
    GitLabNoteTarget,
    GitHubCommentClient,
    build_summary_comment,
    comment_target_from_report,
    gitlab_note_target_from_report,
)
from review_pilot.report_models import Finding, ReviewReport


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[Request] = []
        self.bodies: list[dict[str, object] | None] = []

    def __call__(self, request: Request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        data = request.data
        self.bodies.append(json.loads(data.decode("utf-8")) if data else None)
        return FakeResponse(self.responses.pop(0))


def test_build_summary_comment_contains_marker_counts_and_report_url() -> None:
    report = _report()

    body = build_summary_comment(report, report_url="https://ci.example/report")

    assert SUMMARY_COMMENT_MARKER in body
    assert "Review Pilot Summary" in body
    assert "Total findings: **2**" in body
    assert "Highest severity: **P1**" in body
    assert "[Open full report](https://ci.example/report)" in body
    assert "`src/app.py:12`" in body


def test_comment_target_from_report_reads_repository_and_pr() -> None:
    target = comment_target_from_report(_report())

    assert target == CommentTarget(owner="octo-org", repo="review-demo", issue_number=42)


def test_upsert_summary_comment_dry_run_does_not_call_transport() -> None:
    transport = FakeTransport([])
    client = GitHubCommentClient(token="token", transport=transport)

    action = client.upsert_summary_comment(
        CommentTarget("octo-org", "review-demo", 42),
        build_summary_comment(_report()),
        dry_run=True,
    )

    assert action.action == "dry-run"
    assert action.dry_run is True
    assert SUMMARY_COMMENT_MARKER in action.body
    assert transport.requests == []


def test_upsert_summary_comment_updates_existing_marker_comment() -> None:
    transport = FakeTransport(
        [
            [
                {
                    "id": 1001,
                    "body": f"{SUMMARY_COMMENT_MARKER}\nold body",
                    "user": {"login": "github-actions[bot]"},
                }
            ],
            {
                "id": 1001,
                "html_url": "https://github.com/octo-org/review-demo/pull/42#issuecomment-1001",
            },
        ]
    )
    client = GitHubCommentClient(token="token", transport=transport)

    action = client.upsert_summary_comment(
        CommentTarget("octo-org", "review-demo", 42),
        build_summary_comment(_report()),
        dry_run=False,
    )

    assert action.action == "update"
    assert action.comment_id == 1001
    assert len(transport.requests) == 2
    assert transport.requests[0].get_method() == "GET"
    assert transport.requests[1].get_method() == "PATCH"
    assert transport.bodies[1] is not None
    assert SUMMARY_COMMENT_MARKER in str(transport.bodies[1]["body"])


def test_upsert_summary_comment_creates_when_marker_is_missing() -> None:
    transport = FakeTransport(
        [
            [{"id": 7, "body": "human comment", "user": {"login": "alice"}}],
            {
                "id": 1002,
                "html_url": "https://github.com/octo-org/review-demo/pull/42#issuecomment-1002",
            },
        ]
    )
    client = GitHubCommentClient(token="token", transport=transport)

    action = client.upsert_summary_comment(
        CommentTarget("octo-org", "review-demo", 42),
        build_summary_comment(_report()),
        dry_run=False,
    )

    assert action.action == "create"
    assert action.comment_id == 1002
    assert transport.requests[1].get_method() == "POST"


def test_upsert_gitlab_summary_note_updates_existing_marker_note() -> None:
    transport = FakeTransport(
        [
            [
                {
                    "id": 2001,
                    "body": f"{SUMMARY_COMMENT_MARKER}\nold body",
                    "author": {"username": "review-pilot"},
                }
            ],
            {
                "id": 2001,
                "web_url": "https://gitlab.com/cpp-camp/speed-logger/-/merge_requests/7#note_2001",
            },
        ]
    )
    client = GitLabNoteClient(token="token", transport=transport)

    action = client.upsert_summary_note(
        GitLabNoteTarget("cpp-camp/speed-logger", 7),
        build_summary_comment(_gitlab_report()),
        dry_run=False,
    )

    assert action.action == "update"
    assert action.comment_id == 2001
    assert len(transport.requests) == 2
    assert transport.requests[0].get_method() == "GET"
    assert transport.requests[1].get_method() == "PUT"
    assert "cpp-camp%2Fspeed-logger" in transport.requests[0].full_url
    assert transport.bodies[1] is not None
    assert SUMMARY_COMMENT_MARKER in str(transport.bodies[1]["body"])


def test_upsert_gitlab_summary_note_creates_when_marker_is_missing() -> None:
    transport = FakeTransport(
        [
            [{"id": 8, "body": "human note", "author": {"username": "alice"}}],
            {
                "id": 2002,
                "web_url": "https://gitlab.com/cpp-camp/speed-logger/-/merge_requests/7#note_2002",
            },
        ]
    )
    client = GitLabNoteClient(token="token", transport=transport)

    action = client.upsert_summary_note(
        GitLabNoteTarget(84181645, 7),
        build_summary_comment(_gitlab_report()),
        dry_run=False,
    )

    assert action.action == "create"
    assert action.comment_id == 2002
    assert transport.requests[1].get_method() == "POST"
    assert "/projects/84181645/merge_requests/7/notes" in transport.requests[1].full_url


def test_gitlab_note_client_uses_ci_api_base_url(monkeypatch) -> None:
    monkeypatch.setenv("CI_API_V4_URL", "http://git.cpptrain.top/api/v4")
    transport = FakeTransport(
        [
            [],
            {
                "id": 2003,
                "web_url": "http://git.cpptrain.top/cpp-camp/speed-logger/-/merge_requests/7#note_2003",
            },
        ]
    )
    client = GitLabNoteClient(token="token", transport=transport)

    client.upsert_summary_note(
        GitLabNoteTarget("cpp-camp/speed-logger", 7),
        build_summary_comment(_gitlab_report()),
        dry_run=False,
    )

    assert transport.requests[0].full_url.startswith("http://git.cpptrain.top/api/v4/")
    assert transport.requests[1].full_url.startswith("http://git.cpptrain.top/api/v4/")


def test_gitlab_note_target_from_report_prefers_project_id() -> None:
    target = gitlab_note_target_from_report(_gitlab_report())

    assert target == GitLabNoteTarget(project_id_or_path=84181645, merge_request_iid=7)


def test_comment_target_from_report_rejects_missing_pr_metadata() -> None:
    with pytest.raises(Exception, match="repository"):
        comment_target_from_report(ReviewReport(findings=[]))


def _report() -> ReviewReport:
    return ReviewReport(
        findings=[
            Finding(
                message="Debug print leaked into review path",
                file_path="src/app.py",
                line_no=12,
                severity="P1",
                category="bug",
                source="rule",
            ),
            Finding(
                message="Missing regression test",
                file_path="tests/test_app.py",
                line_no=4,
                severity="P2",
                category="test",
                source="rule",
            ),
        ],
        repo_info={
            "repository": "octo-org/review-demo",
            "pull_request": 42,
        },
        config_source="default",
    )


def _gitlab_report() -> ReviewReport:
    report = _report()
    report.repo_info = {
        "repository": "cpp-camp/speed-logger",
        "project_id": 84181645,
        "pull_request": 7,
    }
    return report
