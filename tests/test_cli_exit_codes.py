from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

from review_pilot.cli import main


def run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_fail_on_returns_one_when_threshold_is_met(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "requirements.txt").write_text("requests==2.0.0\n", encoding="utf-8")
    git(repo, "add", "requirements.txt")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--format", "json", "--fail-on", "P2"]
    )

    payload = json.loads(stdout)
    assert exit_code == 1
    assert payload["summary"]["highest_severity"] == "P2"
    assert stderr == ""


def test_fail_on_returns_zero_when_findings_are_below_threshold(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text("def test_app():\n    assert True\n", encoding="utf-8")
    git(repo, "add", "src/app.py", "tests/test_app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--format", "json", "--fail-on", "P2"]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["summary"]["highest_severity"] == "P3"
    assert stderr == ""


def test_fail_on_returns_zero_when_no_findings(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "docs.md").write_text("notes\n", encoding="utf-8")
    git(repo, "add", "docs.md")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--format", "json", "--fail-on", "P3"]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["summary"]["total_findings"] == 0
    assert stderr == ""


def test_fail_on_also_applies_to_debug_findings(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "requirements.txt").write_text("requests==2.0.0\n", encoding="utf-8")
    git(repo, "add", "requirements.txt")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(
        ["review", "--staged", "--no-ai", "--debug-findings", "--fail-on", "P2"]
    )

    payload = json.loads(stdout)
    assert exit_code == 1
    assert payload["findings"][0]["severity"] == "P2"
    assert stderr == ""


def test_fail_on_can_run_with_default_local_pipeline(tmp_path: Path, monkeypatch) -> None:
    repo = init_repo(tmp_path)
    (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
    git(repo, "add", "app.py")
    monkeypatch.chdir(repo)

    exit_code, stdout, stderr = run_cli(["review", "--staged", "--fail-on", "P2"])

    assert exit_code == 1
    assert "# Review Pilot Report" in stdout
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


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
