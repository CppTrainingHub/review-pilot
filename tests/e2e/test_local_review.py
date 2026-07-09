from __future__ import annotations

import json
import subprocess
from pathlib import Path

from review_pilot.cli import main


def test_local_review_no_ai_with_tools_writes_markdown_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    exit_code = main(
        [
            "review",
            "--staged",
            "--no-ai",
            "--with-tools",
            "--output",
            "report.md",
        ]
    )

    report = (repo / "report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert "# Review Pilot Report" in report
    assert "### Merge Summary" in report
    assert "- **profile:** manual" in report
    assert "- **tools_enabled:** True" in report


def test_local_review_fake_provider_pre_push_profile_writes_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    exit_code = main(
        [
            "review",
            "--staged",
            "--provider",
            "fake",
            "--profile",
            "pre-push",
            "--output",
            "report.md",
        ]
    )

    report = (repo / "report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert "# Review Pilot Report" in report
    assert "- **profile:** pre-push" in report
    assert "- **provider:** fake" in report
    assert "- **tools_enabled:** True" in report


def test_local_review_json_report_contains_pipeline_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
    git(repo, "add", "src/app.py")
    monkeypatch.chdir(repo)

    exit_code = main(
        [
            "review",
            "--staged",
            "--no-ai",
            "--format",
            "json",
            "--output",
            "report.json",
        ]
    )

    payload = json.loads((repo / "report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["repo_info"]["pipeline"] == "local-staged"
    assert payload["repo_info"]["profile"] == "manual"
    assert payload["repo_info"]["ai_enabled"] is False


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
