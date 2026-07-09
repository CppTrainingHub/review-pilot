from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from review_pilot.report_models import Finding
from review_pilot.review_pipeline import (
    ReviewPipeline,
    ReviewPipelineError,
    ReviewPipelineOptions,
)
from review_pilot.tool_models import ToolResult


def test_pipeline_no_ai_builds_final_report(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    result = ReviewPipeline(
        ReviewPipelineOptions(staged=True, no_ai=True, output_format="json")
    ).run()

    payload = json.loads(result.rendered_output)
    rule_ids = {finding["rule_id"] for finding in payload["findings"]}
    assert result.exit_code == 0
    assert payload["repo_info"]["profile"] == "manual"
    assert payload["repo_info"]["pipeline"] == "local-staged"
    assert payload["repo_info"]["ai_enabled"] is False
    assert payload["merge_summary"]["source_counts"] == {"rule": 2}
    assert "rule.debug-output" in rule_ids
    assert "rule.missing-tests" in rule_ids


def test_pipeline_fake_provider_merges_llm_with_rules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    result = ReviewPipeline(
        ReviewPipelineOptions(provider="fake", output_format="json")
    ).run()

    payload = json.loads(result.rendered_output)
    assert result.exit_code == 0
    assert payload["repo_info"]["provider"] == "fake"
    assert payload["repo_info"]["model"] == "fake-review-model"
    assert payload["repo_info"]["evidence_summary"]["verified"] == 1
    assert payload["merge_summary"]["source_counts"] == {
        "llm": 1,
        "rule": 2,
    }
    assert any(
        finding["evidence"]["merge"]["sources"] == ["rule", "llm"]
        for finding in payload["findings"]
    )


def test_pipeline_pre_push_profile_enables_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    import review_pilot.review_pipeline as review_pipeline

    monkeypatch.setattr(
        review_pipeline,
        "run_semgrep_tool",
        lambda tool, repo_root: ToolResult(
            tool_name="semgrep",
            status="success",
            findings=(
                Finding(
                    message="Semgrep issue",
                    file_path="src/app.py",
                    line_no=1,
                    severity="P1",
                    category="security",
                    source="semgrep",
                    rule_id="semgrep.demo",
                ),
            ),
        ),
    )

    result = ReviewPipeline(
        ReviewPipelineOptions(
            no_ai=True,
            profile="pre-push",
            output_format="json",
        )
    ).run()

    payload = json.loads(result.rendered_output)
    assert result.exit_code == 0
    assert payload["repo_info"]["profile"] == "pre-push"
    assert payload["repo_info"]["tools_enabled"] is True
    assert payload["repo_info"]["fail_on"] is None
    assert any(
        finding["source"] == "semgrep"
        for finding in payload["findings"]
    )


def test_pipeline_fail_on_returns_one(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    result = ReviewPipeline(
        ReviewPipelineOptions(no_ai=True, fail_on="P2")
    ).run()

    assert result.exit_code == 1
    assert result.report.summary["highest_severity"] == "P2"


def test_pipeline_rejects_include_out_of_diff_without_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    with pytest.raises(ReviewPipelineError, match="include-out-of-diff"):
        ReviewPipeline(
            ReviewPipelineOptions(no_ai=True, include_out_of_diff=True)
        ).run()


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "checkout", "-b", "main")
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "init",
    )
    return repo


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
