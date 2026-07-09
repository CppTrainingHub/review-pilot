from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from review_pilot.cli import main
from review_pilot.git_providers import GitProviderError
from review_pilot.github_action import (
    GitHubActionError,
    load_github_action_context,
    run_github_action,
)
from review_pilot.pr_models import PullRequestFile, PullRequestInfo, PullRequestRef
from review_pilot.workspace import WorkspacePlan


EVENT = Path("tests/fixtures/github/pull_request_event.json")


class FakeProvider:
    def fetch_pull_request(self, url: str) -> PullRequestInfo:
        assert url == "https://github.com/octo-org/review-demo/pull/42"
        return _pr_info()


def run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_load_github_action_context_reads_pull_request_event(monkeypatch) -> None:
    context = load_github_action_context(
        EVENT,
        env={
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_RUN_ID": "1001",
        },
    )

    assert context.repository == "octo-org/review-demo"
    assert context.owner == "octo-org"
    assert context.repo == "review-demo"
    assert context.pull_request_number == 42
    assert context.pull_request_url == "https://github.com/octo-org/review-demo/pull/42"
    assert context.base_ref == "main"
    assert context.head_ref == "review-output"
    assert context.run_id == "1001"


def test_load_github_action_context_rejects_non_pr_event(tmp_path: Path) -> None:
    event = tmp_path / "push.json"
    event.write_text('{"repository": {"full_name": "octo-org/review-demo"}}', encoding="utf-8")

    with pytest.raises(GitHubActionError, match="pull_request"):
        load_github_action_context(event)


def test_run_github_action_dry_run_writes_artifacts(tmp_path: Path) -> None:
    result = run_github_action(
        event_path=EVENT,
        output_dir=tmp_path / "artifacts",
        dry_run=True,
        provider=FakeProvider(),
    )

    assert result.exit_code == 0
    assert result.workspace.dry_run is True
    assert result.markdown_path.exists()
    assert result.json_path.exists()
    markdown = result.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert "# Review Pilot Report" in markdown
    assert payload["repo_info"]["pipeline"] == "github-action-dry-run"
    assert payload["repo_info"]["repository"] == "octo-org/review-demo"
    assert payload["repo_info"]["pull_request"] == 42
    assert payload["summary"]["total_findings"] >= 1


def test_run_github_action_rejects_provider_in_dry_run(tmp_path: Path) -> None:
    with pytest.raises(GitHubActionError, match="requires a real workspace"):
        run_github_action(
            event_path=EVENT,
            output_dir=tmp_path / "artifacts",
            dry_run=True,
            provider=FakeProvider(),
            llm_provider="fake",
        )


def test_run_github_action_reuses_github_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "checked-out"
    workspace.mkdir()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))

    result = run_github_action(
        event_path=EVENT,
        output_dir=tmp_path / "artifacts",
        dry_run=False,
        provider=FakeProvider(),
    )

    assert result.workspace.workspace_path == str(workspace.resolve())
    assert result.workspace.source == "github-actions-checkout:octo-org/review-demo#42"
    assert result.workspace.commands == ()


def test_run_github_action_provider_merges_llm_findings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "review.py").write_text(
        "def review():\n    result = build_report()\n    return result\n",
        encoding="utf-8",
    )

    def fake_prepare(plan):
        return plan

    monkeypatch.setattr("review_pilot.github_action.prepare_workspace", fake_prepare)
    monkeypatch.setattr(
        "review_pilot.github_action.build_workspace_plan",
        lambda pr_info, parent_dir=None, dry_run=True: WorkspacePlan(
            workspace_path=str(workspace),
            repo_clone_url=pr_info.head.repo_clone_url,
            base_sha=pr_info.base.sha,
            head_sha=pr_info.head.sha,
            source=f"github:{pr_info.full_name}#{pr_info.number}",
            dry_run=False,
            commands=(),
        ),
    )

    result = run_github_action(
        event_path=EVENT,
        output_dir=tmp_path / "artifacts",
        dry_run=False,
        provider=FakeProvider(),
        llm_provider="fake",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["repo_info"]["pipeline"] == "github-action"
    assert payload["repo_info"]["provider"] == "fake"
    assert payload["repo_info"]["model"] == "fake-review-model"
    assert payload["merge_summary"]["source_counts"]["llm"] == 1
    assert any(item["source"] == "llm" for item in payload["findings"])


def test_run_github_action_translates_provider_error(tmp_path: Path) -> None:
    class FailingProvider:
        def fetch_pull_request(self, url: str) -> PullRequestInfo:
            raise GitProviderError("github permission denied")

    event = tmp_path / "pull_request_event.json"
    payload = json.loads(EVENT.read_text(encoding="utf-8"))
    payload.pop("review_pilot_fixture")
    event.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GitHubActionError, match="github permission denied"):
        run_github_action(
            event_path=event,
            output_dir=tmp_path / "artifacts",
            dry_run=True,
            provider=FailingProvider(),
        )


def test_github_action_cli_outputs_dry_run_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("review_pilot.cli.GitHubProvider", FakeProvider)

    exit_code, stdout, stderr = run_cli(
        [
            "github-action",
            "--event-path",
            str(EVENT),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--dry-run",
        ]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["mode"] == "github-action-dry-run"
    assert payload["context"]["repository"] == "octo-org/review-demo"
    assert payload["artifacts"]["markdown"].endswith("review-report.md")
    assert payload["artifacts"]["json"].endswith("review-report.json")
    assert stderr == ""


def test_github_action_cli_requires_dry_run() -> None:
    exit_code, stdout, stderr = run_cli(["github-action", "--help"])

    assert exit_code == 0
    assert "review-pilot github-action" in stdout
    assert "--dry-run" in stdout
    assert "--provider" in stdout
    assert "required" not in stdout.lower()
    assert stderr == ""


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
                additions=2,
                deletions=1,
                changes=3,
                patch="@@ -1,2 +1,3 @@\n def review():\n-    print('debug')\n+    result = build_report()\n+    return result\n",
            ),
        ),
    )
