from __future__ import annotations

import json
import io
from pathlib import Path

import pytest

from review_pilot.benchmark.reports import compare_benchmark_reports
from review_pilot.cli import main


GROUPS = (
    "baseline-30",
    "review-units",
    "dynamic-context",
    "snippet-location",
    "reflection",
)


def test_compare_benchmark_reports_writes_markdown_and_json(tmp_path: Path) -> None:
    result_dirs = []
    for name in GROUPS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "completed_cases": 30,
                        "precision": 0.5,
                        "recall": 0.4,
                        "f1": 0.444444,
                        "noise_rate": 0.2,
                        "average_latency_ms": 12.3,
                        "token_cost_usd": None,
                    },
                    "run_config": {"provider": "fake"},
                }
            ),
            encoding="utf-8",
        )
        result_dirs.append(str(directory))

    output = tmp_path / "final-comparison.md"
    payload = compare_benchmark_reports(result_dirs, str(output))

    assert tuple(payload["groups"]) == GROUPS
    assert output.exists()
    assert output.with_suffix(".json").exists()
    markdown = output.read_text(encoding="utf-8")
    assert "AACR-Bench Final Comparison" in markdown
    assert "Noise Rate" in markdown
    assert "reflection" in markdown


def test_compare_benchmark_reports_rejects_missing_group(tmp_path: Path) -> None:
    result_dirs = []
    for name in GROUPS[:-1]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "report.json").write_text(
            json.dumps({"summary": {}}),
            encoding="utf-8",
        )
        result_dirs.append(str(directory))

    with pytest.raises(ValueError, match="missing: reflection"):
        compare_benchmark_reports(result_dirs, str(tmp_path / "missing.md"))


def test_compare_cli_writes_final_reports(tmp_path: Path) -> None:
    result_dirs = []
    for name in GROUPS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "report.json").write_text(
            json.dumps({"summary": {"completed_cases": 1}}),
            encoding="utf-8",
        )
        result_dirs.append(str(directory))

    stdout = io.StringIO()
    stderr = io.StringIO()
    output = tmp_path / "final.md"
    exit_code = main(
        ["benchmark", "compare", *result_dirs, "--output", str(output)],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "groups:" in stdout.getvalue()
    assert output.exists()
    assert output.with_suffix(".json").exists()


def test_compare_rejects_incompatible_dataset_hashes(tmp_path: Path) -> None:
    result_dirs = []
    for index, name in enumerate(GROUPS):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "dataset": {"sha256": f"dataset-{index}"},
                    "sample_set": {"sha256": "sample"},
                    "run_config": {
                        "provider": "openai-compatible",
                        "model": "deepseek-v4-pro",
                        "prompt_version": "prompt",
                        "matcher": "matcher",
                    },
                    "summary": {},
                }
            ),
            encoding="utf-8",
        )
        result_dirs.append(str(directory))

    with pytest.raises(ValueError, match="dataset.sha256"):
        compare_benchmark_reports(result_dirs, str(tmp_path / "incompatible.md"))
