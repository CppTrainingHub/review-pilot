from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

from review_pilot.benchmark.metrics import (
    evaluate_case,
    metrics_from_counts,
    normalize_finding,
)
from review_pilot.benchmark.runner import run_aacr_benchmark
from review_pilot.cli import main
from review_pilot.benchmark.dataset import BenchmarkComment, case_id_for


def test_metrics_cover_precision_recall_f1_line_accuracy_and_noise() -> None:
    assert metrics_from_counts(
        expected_count=2,
        generated_count=3,
        matched_count=1,
        line_match_count=1,
    ) == {
        "expected_comments": 2,
        "generated_comments": 3,
        "positive_matches": 1,
        "line_matches": 1,
        "precision": 0.333333,
        "recall": 0.5,
        "f1": 0.4,
        "line_precision": 0.333333,
        "line_recall": 0.5,
        "line_accuracy": 1.0,
        "noise_rate": 0.666667,
        "latency_ms": 0.0,
        "token_cost_usd": None,
    }


def test_evaluate_case_matches_same_path_and_line() -> None:
    evaluation = evaluate_case(
        case_id="case-1",
        language="Python",
        context_level="file level",
        expected_comments=(
            BenchmarkComment(
                note="Add a focused test for this changed line.",
                path="src/app.py",
                side="right",
                from_line=2,
                to_line=3,
                category="maintainability",
                context="file level",
            ),
        ),
        predicted_findings=(
            {
                "message": "Add a focused test for this changed line.",
                "file_path": "src/app.py",
                "line_no": 3,
                "category": "maintainability",
            },
        ),
    )

    assert evaluation.matched_count == 1
    assert evaluation.line_match_count == 1


def test_normalize_finding_uses_aacr_comment_shape() -> None:
    assert normalize_finding(
        {
            "message": "Use the changed path consistently.",
            "file_path": "b/src/app.py",
            "line_no": 7,
            "category": "maintainability",
        }
    ) == {
        "path": "src/app.py",
        "side": "right",
        "from_line": 7,
        "to_line": 7,
        "note": "Use the changed path consistently.",
        "category": "maintainability",
        "context": "diff",
    }


def test_runner_completes_fake_provider_and_writes_resumable_reports(tmp_path: Path) -> None:
    repo, target_commit, source_commit = _make_fixture_repo(tmp_path / "repo")
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, repo, target_commit, source_commit, case_number=1)
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"

    result = run_aacr_benchmark(
        dataset_path=dataset,
        output_dir=output_dir,
        provider="fake",
        cache_dir=cache_dir,
    )

    assert result.exit_code == 0
    assert result.report["summary"]["completed_cases"] == 1
    assert result.report["summary"]["failed_cases"] == 0
    assert result.report["summary"]["precision"] == 1.0
    assert result.report["summary"]["recall"] == 1.0
    assert result.report["summary"]["f1"] == 1.0
    assert result.report["summary"]["line_accuracy"] == 1.0
    assert result.report["run_config"]["prompt_version"]
    case_record = json.loads(
        (output_dir / "cases" / f"{result.report['cases'][0]['case_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert case_record["predicted_comments"]
    assert set(case_record["predicted_comments"][0]) == {
        "path",
        "side",
        "from_line",
        "to_line",
        "note",
        "category",
        "context",
    }
    assert Path(result.report_json_path).exists()
    assert Path(result.report_markdown_path).exists()

    resumed = run_aacr_benchmark(
        dataset_path=dataset,
        output_dir=output_dir,
        provider="fake",
        cache_dir=cache_dir,
        resume=True,
    )

    assert resumed.exit_code == 0
    assert resumed.report["summary"]["resumed_cases"] == 1
    assert resumed.report["summary"]["completed_cases"] == 1


def test_runner_records_one_failed_case_and_continues(tmp_path: Path) -> None:
    repo, target_commit, source_commit = _make_fixture_repo(tmp_path / "repo")
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, repo, target_commit, source_commit, case_number=1)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload.append(
        {
            "githubPrUrl": "https://github.com/example/review-fixture/pull/2",
            "repo_url": str(repo),
            "source_commit": "0" * 40,
            "target_commit": target_commit,
            "project_main_language": "Python",
            "comments": payload[0]["comments"],
        }
    )
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    result = run_aacr_benchmark(
        dataset_path=dataset,
        output_dir=tmp_path / "results",
        provider="fake",
        cache_dir=tmp_path / "cache",
    )

    assert result.exit_code == 1
    assert result.report["summary"]["completed_cases"] == 1
    assert result.report["summary"]["failed_cases"] == 1
    assert any(case["status"] == "failed" for case in result.report["cases"])


def test_runner_review_units_writes_strategy_comparison(tmp_path: Path) -> None:
    repo, target_commit, source_commit = _make_fixture_repo(tmp_path / "repo")
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, repo, target_commit, source_commit, case_number=4)

    result = run_aacr_benchmark(
        dataset_path=dataset,
        output_dir=tmp_path / "review-units",
        provider="fake",
        strategy="review-units",
        cache_dir=tmp_path / "cache",
    )

    assert result.exit_code == 0
    assert result.report["run_config"]["strategy"] == "review-units"
    assert set(result.report["comparison"]) == {"baseline", "review-units"}
    assert result.report["comparison"]["review-units"]["summary"]["completed_cases"] == 1
    markdown = Path(result.report_markdown_path).read_text(encoding="utf-8")
    assert "baseline" in markdown
    assert "review-units" in markdown
    assert "Precision" in markdown
    assert "Recall" in markdown
    assert "F1" in markdown


def test_runner_dynamic_context_writes_recall_breakdown(tmp_path: Path) -> None:
    repo, target_commit, source_commit = _make_fixture_repo(tmp_path / "repo")
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, repo, target_commit, source_commit, case_number=5)

    result = run_aacr_benchmark(
        dataset_path=dataset,
        output_dir=tmp_path / "dynamic-context",
        provider="fake",
        dynamic_context=True,
        cache_dir=tmp_path / "cache",
    )

    assert result.exit_code == 0
    assert result.report["run_config"]["dynamic_context"] is True
    assert result.report["dynamic_context"]["tool_calls"] == 1
    assert result.report["context_recall"]["diff"]["tool_calls"] == 1
    markdown = Path(result.report_markdown_path).read_text(encoding="utf-8")
    assert "Recall by Context Level" in markdown
    assert "Diff" in markdown
    assert "tool_calls" in markdown
    assert "dynamic_tokens" in markdown
    assert "token_cost" in markdown


def test_runner_records_reflection_summary_and_tokens(tmp_path: Path) -> None:
    repo, target_commit, source_commit = _make_fixture_repo(tmp_path / "repo")
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, repo, target_commit, source_commit, case_number=6)
    output_dir = tmp_path / "reflection"

    result = run_aacr_benchmark(
        dataset_path=dataset,
        output_dir=output_dir,
        provider="fake",
        reflection=True,
        cache_dir=tmp_path / "cache",
    )

    assert result.report["run_config"]["reflection"] is True
    assert result.report["summary"]["reflection"]["enabled"] is True
    assert result.report["summary"]["reflection"]["reviewed"] == 1
    case = result.report["cases"][0]
    assert case["reflection"]["reviewed"] == 1
    assert case["reflection_tokens"] == 0
    case_record = json.loads(
        (output_dir / "cases" / f"{case['case_id']}.json").read_text(encoding="utf-8")
    )
    assert case_record["reflection_tokens"] == 0
    assert "Reflection" in Path(result.report_markdown_path).read_text(encoding="utf-8")


def test_runner_uses_configured_provider_when_argument_is_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, target_commit, source_commit = _make_fixture_repo(tmp_path / "repo")
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, repo, target_commit, source_commit, case_number=3)
    monkeypatch.setenv("REVIEW_PILOT_LLM_PROVIDER", "fake")

    result = run_aacr_benchmark(
        dataset_path=dataset,
        output_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
    )

    assert result.exit_code == 0
    assert result.report["run_config"]["provider"] == "fake"
    assert result.report["run_config"]["provider_source"] == "environment"


def test_benchmark_cli_prints_stable_summary_fields(tmp_path: Path) -> None:
    repo, target_commit, source_commit = _make_fixture_repo(tmp_path / "repo")
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset, repo, target_commit, source_commit, case_number=1)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "benchmark",
            "aacr",
            "--dataset",
            str(dataset),
            "--provider",
            "fake",
            "--output-dir",
            str(tmp_path / "results"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert "completed_cases: 1" in stdout.getvalue()
    assert "precision: 1.0" in stdout.getvalue()
    assert stderr.getvalue() == ""


def _make_fixture_repo(path: Path) -> tuple[Path, str, str]:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")
    (path / "src").mkdir()
    (path / "tests").mkdir()
    (path / "src" / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (path / "tests" / "test_app.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "base")
    target_commit = _git(path, "rev-parse", "HEAD")
    (path / "src" / "app.py").write_text("def run():\n    return 2\n", encoding="utf-8")
    (path / "tests" / "test_app.py").write_text("def test_run():\n    assert run() == 2\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "change")
    source_commit = _git(path, "rev-parse", "HEAD")
    return path, target_commit, source_commit


def _write_dataset(
    path: Path,
    repo: Path,
    target_commit: str,
    source_commit: str,
    *,
    case_number: int,
) -> None:
    payload = [
        {
            "githubPrUrl": f"https://github.com/example/review-fixture/pull/{case_number}",
            "repo_url": str(repo),
            "source_commit": source_commit,
            "target_commit": target_commit,
            "project_main_language": "Python",
            "comments": [
                {
                    "note": "Fake provider found a deterministic review issue.",
                    "path": "src/app.py",
                    "side": "right",
                    "from_line": 2,
                    "to_line": 2,
                    "category": "maintainability",
                    "context": "diff",
                }
            ],
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()
