from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

from review_pilot import __version__
from review_pilot.cli import main
from review_pilot.pr_models import PullRequestFile, PullRequestInfo, PullRequestRef


def run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_help_shows_doctor_command() -> None:
    exit_code, stdout, stderr = run_cli(["--help"])

    assert exit_code == 0
    assert "Usage" in stdout or "usage" in stdout
    assert "doctor" in stdout
    assert "review" in stdout
    assert "config" in stdout
    assert "context" in stdout
    assert "--version" in stdout
    assert stderr == ""


def test_version_uses_package_version() -> None:
    exit_code, stdout, stderr = run_cli(["--version"])

    assert exit_code == 0
    assert stdout.strip() == __version__
    assert stderr == ""


def test_doctor_command_succeeds_in_normal_environment() -> None:
    exit_code, stdout, stderr = run_cli(["doctor"])

    assert exit_code == 0
    assert "[OK] python:" in stdout
    assert "[OK] git:" in stdout
    assert stderr == ""


def test_no_subcommand_prints_help() -> None:
    exit_code, stdout, stderr = run_cli([])

    assert exit_code == 0
    assert "doctor" in stdout
    assert "review" in stdout
    assert "config" in stdout
    assert "context" in stdout
    assert stderr == ""


def test_review_help_shows_future_local_review_shape() -> None:
    exit_code, stdout, stderr = run_cli(["review", "--help"])

    assert exit_code == 0
    assert "review-pilot review" in stdout
    assert "--staged" in stdout
    assert "--base" in stdout
    assert "--no-ai" in stdout
    assert "--format" in stdout
    assert "--output" in stdout
    assert "--debug-findings" in stdout
    assert "--fail-on" in stdout
    assert "--with-tools" in stdout
    assert "--include-out-of-diff" in stdout
    assert "--profile" in stdout
    assert "Local review workflow entry" in stdout
    assert stderr == ""


def test_review_pr_help_shows_github_pr_shape() -> None:
    exit_code, stdout, stderr = run_cli(["review-pr", "--help"])

    assert exit_code == 0
    assert "review-pilot review-pr" in stdout
    assert "--dry-run" in stdout
    assert "--no-ai" in stdout
    assert "GitHub pull request" in stdout
    assert stderr == ""


def test_context_help_shows_staged_json_shape() -> None:
    exit_code, stdout, stderr = run_cli(["context", "--help"])

    assert exit_code == 0
    assert "review-pilot context" in stdout
    assert "--staged" in stdout
    assert "--json" in stdout
    assert "--max-context-tokens" in stdout
    assert "context candidates" in stdout
    assert stderr == ""


def test_context_pack_help_shows_staged_output_shape() -> None:
    exit_code, stdout, stderr = run_cli(["context-pack", "--help"])

    assert exit_code == 0
    assert "review-pilot context-pack" in stdout
    assert "--staged" in stdout
    assert "--output" in stdout
    assert "--max-context-tokens" in stdout
    assert "context pack" in stdout
    assert stderr == ""


def test_hooks_help_shows_install_status_uninstall_shape() -> None:
    exit_code, stdout, stderr = run_cli(["hooks", "--help"])

    assert exit_code == 0
    assert "review-pilot hooks" in stdout
    assert "install" in stdout
    assert "status" in stdout
    assert "uninstall" in stdout
    assert stderr == ""


def test_llm_help_shows_doctor_shape() -> None:
    exit_code, stdout, stderr = run_cli(["llm", "--help"])

    assert exit_code == 0
    assert "review-pilot llm" in stdout
    assert "doctor" in stdout
    assert "validate-output" in stdout
    assert "provider configuration" in stdout
    assert stderr == ""


def test_prompt_preview_help_shows_staged_provider_shape() -> None:
    exit_code, stdout, stderr = run_cli(["prompt-preview", "--help"])

    assert exit_code == 0
    assert "review-pilot prompt-preview" in stdout
    assert "--staged" in stdout
    assert "--provider" in stdout
    assert "structured review prompt" in stdout
    assert stderr == ""


def test_evidence_check_help_shows_staged_input_shape() -> None:
    exit_code, stdout, stderr = run_cli(["evidence-check", "--help"])

    assert exit_code == 0
    assert "review-pilot evidence-check" in stdout
    assert "--staged" in stdout
    assert "--input" in stdout
    assert "staged evidence" in stdout
    assert stderr == ""


def test_llm_doctor_reports_status_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_PILOT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("REVIEW_PILOT_LLM_MODEL", "review-model")
    monkeypatch.setenv("REVIEW_PILOT_API_KEY", "secret-value")

    exit_code, stdout, stderr = run_cli(["llm", "doctor"])

    assert exit_code == 0
    assert "provider: openai-compatible" in stdout
    assert "model: review-model" in stdout
    assert "api_key: configured" in stdout
    assert "secret-value" not in stdout
    assert stderr == ""


def test_llm_doctor_reports_missing_key(monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_PILOT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("REVIEW_PILOT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code, stdout, stderr = run_cli(["llm", "doctor"])

    assert exit_code == 0
    assert "provider: openai-compatible" in stdout
    assert "api_key: missing" in stdout
    assert stderr == ""


def test_repo_info_prints_json_inside_git_repo(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["repo-info", "--json"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["root"] == str(repo)
    assert payload["branch"] in {"main", "master"}
    assert len(payload["head"]) == 40
    assert payload["has_staged_changes"] is False
    assert payload["has_unstaged_changes"] is False
    assert stderr == ""


def test_diff_staged_raw_prints_unified_diff(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["diff", "--staged", "--raw"])

    assert exit_code == 0
    assert "diff --git a/app.py b/app.py" in stdout
    assert "+print('hello')" in stdout
    assert stderr == ""


def test_diff_staged_json_prints_parsed_diff(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["diff", "--staged", "--json"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["files"][0]["path"] == "app.py"
    assert payload["files"][0]["change_type"] == "added"
    assert payload["files"][0]["hunks"][0]["lines"][0]["kind"] == "added"
    assert payload["files"][0]["hunks"][0]["lines"][0]["new_line_no"] == 1
    assert stderr == ""


def test_diff_requires_one_output_format() -> None:
    exit_code, stdout, stderr = run_cli(["diff", "--staged"])

    assert exit_code == 2
    assert stdout == ""
    assert "exactly one output format" in stderr


def test_config_show_prints_effective_json(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / ".review-pilot.toml").write_text(
        """
ignore_paths = ["generated/**"]

[tools.semgrep]
enabled = true
timeout_seconds = 20
severity_threshold = "P1"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["config", "show"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["source"] == str(repo / ".review-pilot.toml")
    assert payload["ignore_paths"] == ["generated/**"]
    assert payload["tools"]["semgrep"]["enabled"] is True
    assert payload["tools"]["semgrep"]["timeout_seconds"] == 20
    assert stderr == ""


def test_config_validate_reports_source(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / ".review-pilot.toml").write_text(
        """
[rules."rule.debug-output"]
enabled = false
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["config", "validate"])

    assert exit_code == 0
    assert stdout.strip() == f"config ok: {repo / '.review-pilot.toml'}"
    assert stderr == ""


def test_config_validate_reports_invalid_config(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / ".review-pilot.toml").write_text("unknown = true\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["config", "validate"])

    assert exit_code == 2
    assert stdout == ""
    assert "config error:" in stderr


def test_diff_staged_raw_reports_empty_staged_diff(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["diff", "--staged", "--raw"])

    assert exit_code == 1
    assert stdout == ""
    assert "no staged changes" in stderr


def test_review_staged_runs_local_pipeline_by_default(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "review.py").write_text("risk = True\n", encoding="utf-8")
    git(repo, "add", "review.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged"])

    assert exit_code == 0
    assert "# Review Pilot Report" in stdout
    assert "**profile:** manual" in stdout
    assert "**pipeline:** local-staged" in stdout
    assert stderr == ""


def test_review_base_reports_current_supported_input() -> None:
    exit_code, stdout, stderr = run_cli(["review", "--base", "main"])

    assert exit_code == 2
    assert stdout == ""
    assert "supports --staged" in stderr


def test_review_pr_dry_run_outputs_pr_and_workspace_plan(monkeypatch) -> None:
    class FakeProvider:
        def fetch_pull_request(self, url: str) -> PullRequestInfo:
            assert url == "https://github.com/octo-org/review-demo/pull/42"
            return _pr_info()

    monkeypatch.setattr("review_pilot.cli.GitHubProvider", FakeProvider)

    exit_code, stdout, stderr = run_cli(
        [
            "review-pr",
            "https://github.com/octo-org/review-demo/pull/42",
            "--dry-run",
            "--no-ai",
        ]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["mode"] == "github-pr-dry-run"
    assert payload["pull_request"]["full_name"] == "octo-org/review-demo"
    assert payload["pull_request"]["diff"]["changed_paths"] == ["src/review.py"]
    assert payload["workspace"]["dry_run"] is True
    assert payload["workspace"]["source"] == "github:octo-org/review-demo#42"
    assert payload["workspace"]["commands"][0][0:2] == ["git", "clone"]
    assert stderr == ""


def test_review_pr_requires_no_ai_for_input_preparation() -> None:
    exit_code, stdout, stderr = run_cli(
        [
            "review-pr",
            "https://github.com/octo-org/review-demo/pull/42",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "pass --no-ai" in stderr


def test_review_fake_provider_outputs_structured_findings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--provider", "fake"]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["provider"] == "fake"
    assert payload["schema_version"] == "review-pilot.llm-findings.v1"
    assert payload["evidence_summary"] == {
        "total": 1,
        "kept": 1,
        "verified": 1,
        "downgraded": 0,
        "dropped": 0,
    }
    assert payload["findings"][0]["file_path"] == "app.py"
    assert payload["findings"][0]["source"] == "llm"
    assert (
        payload["findings"][0]["evidence"]["verification"]["source"]
        == "diff_added_line"
    )
    assert stderr == ""


def test_review_fake_provider_json_outputs_final_merged_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--provider", "fake", "--format", "json"]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["merge_summary"]["merged_groups"] == 1
    assert payload["merge_summary"]["source_counts"] == {
        "llm": 1,
        "rule": 2,
    }
    merged = next(
        finding
        for finding in payload["findings"]
        if finding["evidence"]["merge"]["sources"] == ["rule", "llm"]
    )
    assert merged["evidence"]["merge"]["sources"] == [
        "rule",
        "llm",
    ]
    assert payload["repo_info"]["provider"] == "fake"
    assert stderr == ""


def test_review_fake_provider_with_tools_writes_final_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        [
            "review",
            "--staged",
            "--provider",
            "fake",
            "--with-tools",
            "--output",
            "report.md",
        ]
    )

    report = (repo / "report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert stdout.strip() == "wrote report: report.md"
    assert "# Review Pilot Report" in report
    assert "### Merge Summary" in report
    assert "- **Sources:** rule, llm" in report
    assert stderr == ""


def test_review_rejects_no_ai_with_provider() -> None:
    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--provider", "fake"]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "either --no-ai or --provider" in stderr


def test_prompt_preview_uses_context_pack_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["prompt-preview", "--staged", "--provider", "fake"]
    )

    assert exit_code == 0
    assert "PROVIDER\nfake / fake-review-model" in stdout
    assert "\nSYSTEM\n" in stdout
    assert "## REPOSITORY" in stdout
    assert "## DIFF" in stdout
    assert "## OUTPUT_CONTRACT" in stdout
    assert stderr == ""


def test_llm_validate_output_accepts_valid_fixture() -> None:
    exit_code, stdout, stderr = run_cli(
        [
            "llm",
            "validate-output",
            "--input",
            "tests/fixtures/llm/valid_findings.json",
        ]
    )

    assert exit_code == 0
    assert "valid llm output" in stdout
    assert "findings=1" in stdout
    assert stderr == ""


def test_llm_validate_output_rejects_markdown_fixture() -> None:
    exit_code, stdout, stderr = run_cli(
        [
            "llm",
            "validate-output",
            "--input",
            "tests/fixtures/llm/invalid_markdown.txt",
        ]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "markdown fences" in stderr


def test_evidence_check_keeps_added_line(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    output = repo / "llm.json"
    output.write_text(
        json.dumps(_llm_payload("app.py", 1)),
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["evidence-check", "--staged", "--input", str(output)]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["evidence_summary"]["verified"] == 1
    assert payload["evidence_summary"]["dropped"] == 0
    assert payload["dropped_findings"] == []
    assert stderr == ""


def test_evidence_check_returns_one_for_hallucinated_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    output = repo / "llm.json"
    output.write_text(
        json.dumps(_llm_payload("missing.py", 99)),
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["evidence-check", "--staged", "--input", str(output)]
    )

    payload = json.loads(stdout)
    assert exit_code == 1
    assert payload["evidence_summary"]["dropped"] == 1
    assert payload["findings"] == []
    assert "file_path is not present" in (
        payload["dropped_findings"][0]["reason"]
    )
    assert stderr == ""


def test_naive_review_fake_provider_prints_unstructured_review(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["naive-review", "--staged", "--provider", "fake"])

    assert exit_code == 0
    assert "Naive Review Result" in stdout
    assert "app.py" in stdout
    assert "自由文本输出" in stdout
    assert stderr == ""


def test_naive_review_openai_provider_reports_missing_api_key(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)
    monkeypatch.delenv("REVIEW_PILOT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code, stdout, stderr = run_cli(["naive-review", "--staged", "--provider", "openai"])

    assert exit_code == 2
    assert stdout == ""
    assert "missing API key" in stderr


def test_naive_review_reports_empty_staged_diff(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["naive-review", "--staged", "--provider", "fake"])

    assert exit_code == 1
    assert stdout == ""
    assert "no staged changes" in stderr


def test_review_no_ai_markdown_outputs_report(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged", "--no-ai"])

    assert exit_code == 0
    assert "# Review Pilot Report" in stdout
    assert "Total findings:" in stdout
    assert stderr == ""


def test_review_no_ai_json_outputs_report(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged", "--no-ai", "--format", "json"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert "summary" in payload
    assert "findings" in payload
    assert payload["config_source"] == "default"
    assert stderr == ""


def test_review_no_ai_reports_file_too_large(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    content = "\n".join(f"x = {i}" for i in range(250))
    (repo / "big.py").write_text(content + "\n", encoding="utf-8")
    git(repo, "add", "big.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged", "--no-ai", "--format", "json"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert any(
        finding.get("rule_id") == "rule.file-too-large"
        for finding in payload["findings"]
    )
    assert stderr == ""


def test_review_no_ai_reports_multiple_rule_findings(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("requests==2.0.0\n", encoding="utf-8")
    git(repo, "add", "src/app.py", "requirements.txt")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged", "--no-ai", "--format", "json"])

    payload = json.loads(stdout)
    rule_ids = {finding.get("rule_id") for finding in payload["findings"]}
    assert exit_code == 0
    assert "rule.debug-output" in rule_ids
    assert "rule.missing-tests" in rule_ids
    assert "rule.sensitive-path" in rule_ids


def test_review_no_ai_records_project_config_source(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / ".review-pilot.toml").write_text(
        """
[rules."rule.debug-output"]
enabled = false
""",
        encoding="utf-8",
    )
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", ".review-pilot.toml", "src/app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged", "--no-ai", "--format", "json"])

    payload = json.loads(stdout)
    rule_ids = {finding.get("rule_id") for finding in payload["findings"]}
    assert exit_code == 0
    assert payload["config_source"] == str(repo / ".review-pilot.toml")
    assert "rule.debug-output" not in rule_ids
    assert stderr == ""


def test_context_staged_json_outputs_candidates(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "helpers.py").write_text("def load():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "service.py").write_text(
        "from .helpers import load\n\ndef run():\n    return load()\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_service.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
    git(repo, "add", "src/service.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["context", "--staged", "--json"])

    payload = json.loads(stdout)
    by_path = {candidate["path"]: candidate for candidate in payload["candidates"]}
    assert exit_code == 0
    assert payload["changed_paths"] == ["src/service.py"]
    assert by_path["src/service.py"]["reason"] == "changed_file"
    assert by_path["tests/test_service.py"]["reason"] == "related_test"
    assert by_path["src/helpers.py"]["reason"] == "local_import"
    assert stderr == ""


def test_context_staged_json_with_budget_outputs_used_and_omitted(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "helpers.py").write_text("def load():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "service.py").write_text(
        "from .helpers import load\n\n"
        "def run():\n"
        "    return load()\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_service.py").write_text(
        "from src.service import run\n\n"
        "def test_run():\n"
        "    assert run() == 1\n",
        encoding="utf-8",
    )
    git(repo, "add", "src/service.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["context", "--staged", "--max-context-tokens", "14", "--json"]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["max_context_tokens"] == 14
    assert payload["used_tokens"] <= 14
    assert payload["context_used"][0]["path"] == "src/service.py"
    assert payload["context_omitted"]
    assert "candidates" not in payload
    assert stderr == ""


def test_context_staged_json_rejects_invalid_budget(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["context", "--staged", "--max-context-tokens", "0", "--json"]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "context budget error" in stderr


def test_context_staged_json_reports_empty_diff(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["context", "--staged", "--json"])

    assert exit_code == 1
    assert stdout == ""
    assert "no staged changes" in stderr


def test_context_pack_staged_writes_json_file(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "helpers.py").write_text("def load():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "service.py").write_text(
        "from .helpers import load\n\n"
        "def run():\n"
        "    print('debug')\n"
        "    return load()\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_service.py").write_text(
        "from src.service import run\n\n"
        "def test_run():\n"
        "    assert run() == 1\n",
        encoding="utf-8",
    )
    git(repo, "add", "src/service.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["context-pack", "--staged", "--output", ".review-pilot/context-pack.json"]
    )

    output_path = repo / ".review-pilot" / "context-pack.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "wrote context pack:" in stdout
    assert "findings=" in stdout
    assert payload["schema_version"] == "review-pilot.context-pack.v1"
    assert payload["diff"]["files"][0]["path"] == "src/service.py"
    assert payload["context"]["context_used"][0]["path"] == "src/service.py"
    assert any(finding["rule_id"] == "rule.debug-output" for finding in payload["rule_findings"])
    assert stderr == ""


def test_context_pack_staged_preserves_omitted_context_with_low_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "helpers.py").write_text("def load():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "service.py").write_text(
        "from .helpers import load\n\n"
        "def run():\n"
        "    return load()\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_service.py").write_text(
        "from src.service import run\n\n"
        "def test_run():\n"
        "    assert run() == 1\n",
        encoding="utf-8",
    )
    git(repo, "add", "src/service.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        [
            "context-pack",
            "--staged",
            "--max-context-tokens",
            "12",
            "--output",
            ".review-pilot/context-pack-small.json",
        ]
    )

    payload = json.loads((repo / ".review-pilot" / "context-pack-small.json").read_text(encoding="utf-8"))
    omitted_reasons = {item["omitted_reason"] for item in payload["context"]["context_omitted"]}
    assert exit_code == 0
    assert "wrote context pack:" in stdout
    assert "budget_exhausted" in omitted_reasons
    assert stderr == ""


def test_context_pack_reports_empty_staged_diff(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["context-pack", "--staged", "--output", ".review-pilot/context-pack.json"]
    )

    assert exit_code == 1
    assert stdout == ""
    assert "no staged changes" in stderr


def test_review_no_ai_uses_ignore_patterns(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / ".review-pilot.toml").write_text(
        """
ignore_paths = ["generated/**"]
""",
        encoding="utf-8",
    )
    (repo / "generated").mkdir()
    (repo / "generated" / "client.py").write_text("print('generated')\n", encoding="utf-8")
    git(repo, "add", ".review-pilot.toml", "generated/client.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged", "--no-ai", "--format", "json"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["findings"] == []
    assert stderr == ""


def test_review_no_ai_debug_findings_outputs_normalized_findings(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("requests==2.0.0\n", encoding="utf-8")
    git(repo, "add", "src/app.py", "requirements.txt")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--debug-findings"]
    )

    payload = json.loads(stdout)
    rule_ids = [finding["rule_id"] for finding in payload["findings"]]
    assert exit_code == 0
    assert rule_ids == [
        "rule.sensitive-path",
        "rule.missing-tests",
        "rule.debug-output",
    ]
    assert all(finding.get("source") == "rule" for finding in payload["findings"])
    assert stderr == ""


def test_review_no_ai_reports_empty_staged_diff(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged", "--no-ai"])

    assert exit_code == 1
    assert stdout == ""
    assert "no staged changes" in stderr


def test_repo_info_reports_non_git_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code, stdout, stderr = run_cli(["repo-info", "--json"])

    assert exit_code == 2
    assert stdout == ""
    assert "not a git repository" in stderr


def test_detect_project_json_prints_detected_markers(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["detect-project", "--json"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["root"] == str(repo)
    assert payload["project_types"] == ["python"]
    assert payload["markers"] == ["pyproject.toml"]
    assert stderr == ""


def test_tools_list_json_prints_registry(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["tools", "list", "--json"])

    payload = json.loads(stdout)
    by_name = {tool["name"]: tool for tool in payload["tools"]}
    assert exit_code == 0
    assert payload["project"]["project_types"] == ["python"]
    assert by_name["python-version"]["enabled"] is True
    assert by_name["python-tests"]["enabled"] is True
    assert by_name["semgrep"]["enabled"] is True
    assert by_name["npm-test"]["enabled"] is False
    assert stderr == ""


def test_run_tool_json_runs_registered_tool_and_saves_raw_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["run-tool", "--name", "python-version", "--json"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["tool_name"] == "python-version"
    assert payload["command"] == ["python", "--version"]
    assert "Python" in (payload["stdout"] + payload["stderr"])
    assert Path(payload["raw_outputs"]["stdout"]).exists()
    assert Path(payload["raw_outputs"]["stderr"]).exists()
    assert stderr == ""


def test_run_tool_rejects_unknown_tool(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["run-tool", "--name", "shell"])

    assert exit_code == 2
    assert stdout == ""
    assert "unknown tool: shell" in stderr


def test_hooks_install_status_and_uninstall(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["hooks", "install", "--pre-commit", "--pre-push"]
    )

    assert exit_code == 0
    assert "installed pre-commit" in stdout
    assert "installed pre-push" in stdout
    assert (repo / ".git" / "hooks" / "pre-commit").exists()
    assert (repo / ".git" / "hooks" / "pre-push").exists()
    assert stderr == ""

    exit_code, stdout, stderr = run_cli(["hooks", "status"])

    assert exit_code == 0
    assert "pre-commit: managed by review-pilot (pre_commit)" in stdout
    assert "pre-push: managed by review-pilot (pre_push)" in stdout
    assert stderr == ""

    exit_code, stdout, stderr = run_cli(
        ["hooks", "uninstall", "--pre-commit", "--pre-push"]
    )

    assert exit_code == 0
    assert "removed pre-commit" in stdout
    assert "removed pre-push" in stdout
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()
    assert not (repo / ".git" / "hooks" / "pre-push").exists()
    assert stderr == ""


def test_hooks_install_requires_selected_hook(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["hooks", "install"])

    assert exit_code == 2
    assert stdout == ""
    assert "select at least one hook" in stderr


def test_hooks_install_blocks_existing_non_review_pilot_hook(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho custom\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["hooks", "install", "--pre-commit"])

    assert exit_code == 2
    assert stdout == ""
    assert "already exists" in stderr
    assert hook.read_text(encoding="utf-8") == "#!/bin/sh\necho custom\n"


def test_hooks_reports_non_git_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code, stdout, stderr = run_cli(["hooks", "status"])

    assert exit_code == 2
    assert stdout == ""
    assert "not a git repository" in stderr


def test_run_tool_semgrep_reports_missing_executable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    import review_pilot.tools.semgrep_tool as semgrep_tool

    monkeypatch.setattr(semgrep_tool, "is_semgrep_available", lambda: False)

    exit_code, stdout, stderr = run_cli(["run-tool", "--name", "semgrep", "--json"])

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["tool_name"] == "semgrep"
    assert payload["status"] == "missing"
    assert payload["error"] == "semgrep executable not found"
    assert stderr == ""


def test_review_with_tools_includes_semgrep_findings_in_debug_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    import review_pilot.review_pipeline as review_pipeline
    from review_pilot.report_models import Finding
    from review_pilot.tool_models import ToolResult

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
            raw_findings=({"check_id": "semgrep.demo"},),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--with-tools", "--debug-findings"]
    )

    payload = json.loads(stdout)
    sources = {finding["source"] for finding in payload["findings"]}
    assert exit_code == 0
    assert "rule" in sources
    assert "semgrep" in sources
    assert payload["tool_results"][0]["status"] == "success"
    assert payload["tool_results"][0]["findings"][0]["rule_id"] == "semgrep.demo"
    assert stderr == ""


def test_review_include_out_of_diff_requires_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--include-out-of-diff"]
    )

    assert exit_code == 2
    assert stdout == ""
    assert "requires --with-tools" in stderr


def test_review_with_tools_filters_out_of_diff_tool_findings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("old = True\n", encoding="utf-8")
    git(repo, "add", "pyproject.toml", "src/app.py")
    git(repo, "-c", "user.name=Test User", "-c", "user.email=test@example.com", "commit", "-m", "base")
    (repo / "src" / "app.py").write_text("old = True\nnew = True\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    import review_pilot.review_pipeline as review_pipeline
    from review_pilot.report_models import Finding
    from review_pilot.tool_models import ToolResult

    monkeypatch.setattr(
        review_pipeline,
        "run_semgrep_tool",
        lambda tool, repo_root: ToolResult(
            tool_name="semgrep",
            status="success",
            findings=(
                Finding(
                    message="old issue",
                    file_path="src/app.py",
                    line_no=1,
                    severity="P1",
                    category="security",
                    source="semgrep",
                    rule_id="semgrep.old",
                ),
                Finding(
                    message="new issue",
                    file_path="src/app.py",
                    line_no=2,
                    severity="P1",
                    category="security",
                    source="semgrep",
                    rule_id="semgrep.new",
                ),
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--with-tools", "--debug-findings"]
    )

    payload = json.loads(stdout)
    semgrep_rule_ids = [
        finding["rule_id"]
        for finding in payload["findings"]
        if finding["source"] == "semgrep"
    ]
    assert exit_code == 0
    assert semgrep_rule_ids == ["semgrep.new"]
    assert payload["tool_filter"]["total_tool_findings"] == 2
    assert payload["tool_filter"]["included_count"] == 1
    assert payload["tool_filter"]["out_of_diff_count"] == 1
    assert payload["tool_filter"]["out_of_diff_findings"][0]["rule_id"] == "semgrep.old"
    assert stderr == ""


def test_review_with_tools_can_include_out_of_diff_tool_findings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("old = True\n", encoding="utf-8")
    git(repo, "add", "pyproject.toml", "src/app.py")
    git(repo, "-c", "user.name=Test User", "-c", "user.email=test@example.com", "commit", "-m", "base")
    (repo / "src" / "app.py").write_text("old = True\nnew = True\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    import review_pilot.review_pipeline as review_pipeline
    from review_pilot.report_models import Finding
    from review_pilot.tool_models import ToolResult

    monkeypatch.setattr(
        review_pipeline,
        "run_semgrep_tool",
        lambda tool, repo_root: ToolResult(
            tool_name="semgrep",
            status="success",
            findings=(
                Finding(
                    message="old issue",
                    file_path="src/app.py",
                    line_no=1,
                    severity="P1",
                    category="security",
                    source="semgrep",
                    rule_id="semgrep.old",
                ),
                Finding(
                    message="new issue",
                    file_path="src/app.py",
                    line_no=2,
                    severity="P1",
                    category="security",
                    source="semgrep",
                    rule_id="semgrep.new",
                ),
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        [
            "review",
            "--staged",
            "--no-ai",
            "--with-tools",
            "--include-out-of-diff",
            "--debug-findings",
        ]
    )

    payload = json.loads(stdout)
    semgrep_rule_ids = [
        finding["rule_id"]
        for finding in payload["findings"]
        if finding["source"] == "semgrep"
    ]
    assert exit_code == 0
    assert semgrep_rule_ids == ["semgrep.new", "semgrep.old"]
    assert payload["tool_filter"]["out_of_diff_count"] == 1
    assert stderr == ""


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "checkout", "-b", "main")
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "-c", "user.name=Test User", "-c", "user.email=test@example.com", "commit", "-m", "init")
    return repo


def _llm_payload(file_path: str, line_no: int) -> dict:
    return {
        "schema_version": "review-pilot.llm-findings.v1",
        "findings": [
            {
                "message": "Review the referenced line.",
                "file_path": file_path,
                "line_no": line_no,
                "severity": "P2",
                "category": "bug",
                "source": "llm",
                "confidence": "high",
                "evidence": {
                    "reason": "The referenced line may be incorrect.",
                },
                "suggestion": "Update the referenced line.",
            }
        ],
    }


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
                additions=1,
                deletions=1,
                changes=2,
                patch="@@ -1 +1 @@\n-old\n+new\n",
            ),
        ),
    )


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
