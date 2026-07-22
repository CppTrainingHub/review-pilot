from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from review_pilot.cli import main
from review_pilot.git_providers import GitProviderError
from review_pilot.gitlab_ci import GitLabCIError, load_gitlab_ci_context, run_gitlab_ci
from review_pilot.pr_models import PullRequestFile, PullRequestInfo, PullRequestRef


class FakeProvider:
    def fetch_merge_request(self, project_id_or_path: str | int, iid: int) -> PullRequestInfo:
        assert str(project_id_or_path) == "84181645"
        assert iid == 7
        return _pr_info()


def run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def gitlab_env(tmp_path: Path) -> dict[str, str]:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    return {
        "CI_PROJECT_ID": "84181645",
        "CI_PROJECT_PATH": "cpp-camp/speed-logger",
        "CI_PROJECT_URL": "https://gitlab.com/cpp-camp/speed-logger",
        "CI_MERGE_REQUEST_IID": "7",
        "CI_MERGE_REQUEST_SOURCE_BRANCH_NAME": "fix/feishu-card-debug-output-demo",
        "CI_MERGE_REQUEST_TARGET_BRANCH_NAME": "main",
        "CI_COMMIT_SHA": "2222222222222222222222222222222222222222",
        "CI_PROJECT_DIR": str(workspace),
        "CI_PIPELINE_ID": "9001",
        "CI_JOB_ID": "8001",
    }


def test_load_gitlab_ci_context_reads_merge_request_env(tmp_path: Path) -> None:
    context = load_gitlab_ci_context(env=gitlab_env(tmp_path))

    assert context.project_id == "84181645"
    assert context.project_path == "cpp-camp/speed-logger"
    assert context.merge_request_iid == 7
    assert context.merge_request_url == "https://gitlab.com/cpp-camp/speed-logger/-/merge_requests/7"
    assert context.source_branch == "fix/feishu-card-debug-output-demo"


def test_load_gitlab_ci_context_requires_mr_env(tmp_path: Path) -> None:
    env = gitlab_env(tmp_path)
    env.pop("CI_MERGE_REQUEST_IID")

    with pytest.raises(GitLabCIError, match="CI_MERGE_REQUEST_IID"):
        load_gitlab_ci_context(env=env)


def test_run_gitlab_ci_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    for key, value in gitlab_env(tmp_path).items():
        monkeypatch.setenv(key, value)

    result = run_gitlab_ci(
        output_dir=tmp_path / "artifacts",
        dry_run=False,
        provider=FakeProvider(),
    )

    assert result.exit_code == 0
    assert result.workspace.source == "gitlab-ci-checkout:cpp-camp/speed-logger!7"
    assert result.markdown_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["repo_info"]["pipeline"] == "gitlab-ci"
    report = json.loads(
        (tmp_path / "artifacts" / "review-report.json").read_text(encoding="utf-8")
    )
    assert report["repo_info"]["engine"] == "review-engine"
    assert report["repo_info"]["input_source"] == "gitlab-mr"
    assert payload["repo_info"]["project_id"] == "84181645"
    assert payload["repo_info"]["repository"] == "cpp-camp/speed-logger"
    assert payload["repo_info"]["pull_request"] == 7
    assert payload["repo_info"]["head_ref"] == "fix/feishu-card-debug-output-demo"
    assert isinstance(payload["summary"]["total_findings"], int)


def test_run_gitlab_ci_rejects_provider_in_dry_run(tmp_path: Path, monkeypatch) -> None:
    for key, value in gitlab_env(tmp_path).items():
        monkeypatch.setenv(key, value)

    with pytest.raises(GitLabCIError, match="requires a real workspace"):
        run_gitlab_ci(
            output_dir=tmp_path / "artifacts",
            dry_run=True,
            provider=FakeProvider(),
            llm_provider="fake",
        )


def test_run_gitlab_ci_translates_provider_error(tmp_path: Path, monkeypatch) -> None:
    class FailingProvider:
        def fetch_merge_request(self, project_id_or_path: str | int, iid: int) -> PullRequestInfo:
            raise GitProviderError("gitlab permission denied")

    for key, value in gitlab_env(tmp_path).items():
        monkeypatch.setenv(key, value)

    with pytest.raises(GitLabCIError, match="gitlab permission denied"):
        run_gitlab_ci(
            output_dir=tmp_path / "artifacts",
            dry_run=True,
            provider=FailingProvider(),
        )


def test_gitlab_ci_cli_outputs_json(tmp_path: Path, monkeypatch) -> None:
    for key, value in gitlab_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("review_pilot.cli.GitLabProvider", FakeProvider)

    exit_code, stdout, stderr = run_cli(
        [
            "gitlab-ci",
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["mode"] == "gitlab-ci"
    assert payload["context"]["project_path"] == "cpp-camp/speed-logger"
    assert payload["artifacts"]["json"].endswith("review-report.json")
    assert stderr == ""


def test_gitlab_ci_help() -> None:
    exit_code, stdout, stderr = run_cli(["gitlab-ci", "--help"])

    assert exit_code == 0
    assert "review-pilot gitlab-ci" in stdout
    assert "--provider" in stdout
    assert stderr == ""


def _pr_info() -> PullRequestInfo:
    base = PullRequestRef(
        label="cpp-camp/speed-logger:main",
        ref="main",
        sha="1111111111111111111111111111111111111111",
        repo_full_name="cpp-camp/speed-logger",
        repo_clone_url="https://gitlab.com/cpp-camp/speed-logger.git",
    )
    head = PullRequestRef(
        label="cpp-camp/speed-logger:fix/feishu-card-debug-output-demo",
        ref="fix/feishu-card-debug-output-demo",
        sha="2222222222222222222222222222222222222222",
        repo_full_name="cpp-camp/speed-logger",
        repo_clone_url="https://gitlab.com/cpp-camp/speed-logger.git",
    )
    return PullRequestInfo(
        provider="gitlab",
        url="https://gitlab.com/cpp-camp/speed-logger/-/merge_requests/7",
        owner="cpp-camp",
        repo="speed-logger",
        number=7,
        title="Remove debug output from logger fanout path",
        state="opened",
        base=base,
        head=head,
        files=(
            PullRequestFile(
                filename="src/logger.cpp",
                status="modified",
                additions=1,
                deletions=0,
                changes=1,
                patch="@@ -1,3 +1,4 @@\n void log_message() {\n+  std::cout << \"debug fanout\" << std::endl;\n   write_sink();\n }\n",
            ),
        ),
    )
