from __future__ import annotations

from pathlib import Path
import json

import pytest

from review_pilot.config import ReviewPilotConfig
from review_pilot.models import DiffFile, DiffHunk, DiffLine, ParsedDiff, RepoInfo
from review_pilot.review_engine import (
    ReviewEngine,
    ReviewEngineError,
    ReviewEngineOptions,
    ReviewInput,
)
from review_pilot.llm import LLMResponse


def test_engine_runs_deterministic_review_and_records_input_metadata(tmp_path: Path) -> None:
    review_input = make_input(tmp_path, metadata={"pipeline": "test-entry"})

    result = ReviewEngine().run(review_input)

    assert result.report.repo_info["engine"] == "review-engine"
    assert result.report.repo_info["input_source"] == "local-staged"
    assert result.report.repo_info["pipeline"] == "test-entry"
    assert result.report.merge_summary["source_counts"] == {
        "rule": 2,
    }
    assert result.debug_payload["rule_findings"]
    assert result.debug_payload["llm_findings"] == []


def test_engine_runs_fake_provider_and_preserves_raw_llm_findings(tmp_path: Path) -> None:
    result = ReviewEngine(ReviewEngineOptions(provider="fake")).run(
        make_input(tmp_path)
    )

    assert result.report.repo_info["provider"] == "fake"
    assert result.report.repo_info["model"] == "fake-review-model"
    assert result.report.repo_info["evidence_summary"]["verified"] == 1
    assert result.report.merge_summary["source_counts"] == {
        "llm": 1,
        "rule": 2,
    }
    assert len(result.debug_payload["llm_findings"]) == 1
    assert any(
        finding["source"] == "llm" for finding in result.debug_payload["merged_findings"]
    )


def test_engine_dynamic_context_records_trace_and_usage_metadata(tmp_path: Path) -> None:
    result = ReviewEngine(
        ReviewEngineOptions(provider="fake", dynamic_context=True)
    ).run(make_input(tmp_path))

    dynamic = result.report.repo_info["dynamic_context"]
    assert dynamic["enabled"] is True
    assert dynamic["tool_calls"] == 1
    assert dynamic["rounds"] == 2
    assert dynamic["dynamic_context_trace"][0]["tool_name"] == "file_read"
    assert result.debug_payload["dynamic_context_trace"][0]["returned_count"] >= 1


def test_engine_rejects_empty_diff_with_input_error(tmp_path: Path) -> None:
    empty = make_input(tmp_path, parsed_diff=ParsedDiff(files=()))

    with pytest.raises(ReviewEngineError, match="review input diff is empty") as error:
        ReviewEngine().run(empty)

    assert error.value.exit_code == 1


def test_engine_rejects_unknown_input_source(tmp_path: Path) -> None:
    invalid = make_input(tmp_path, input_source="unknown")

    with pytest.raises(ReviewEngineError, match="unsupported input source"):
        ReviewEngine().run(invalid)


def test_engine_reports_context_build_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "review_pilot.review_engine.build_code_index",
        lambda root, config: (_ for _ in ()).throw(ValueError("index unavailable")),
    )

    with pytest.raises(ReviewEngineError, match="context pack error: index unavailable"):
        ReviewEngine(ReviewEngineOptions(provider="fake")).run(make_input(tmp_path))


def test_engine_runs_review_units_and_records_unit_metadata(tmp_path: Path) -> None:
    parsed_diff = ParsedDiff(
        files=(
            parsed_app_diff().files[0],
            DiffFile(
                old_path="src/second.py",
                new_path="src/second.py",
                change_type="modified",
                hunks=parsed_app_diff().files[0].hunks,
            ),
        )
    )

    review_input = make_input(tmp_path, parsed_diff=parsed_diff)
    (tmp_path / "src" / "second.py").write_text("value = 2\n", encoding="utf-8")
    result = ReviewEngine(
        ReviewEngineOptions(
            provider="fake",
            strategy="review-units",
            max_context_tokens=100,
            review_unit_workers=2,
        )
    ).run(review_input)

    metadata = result.report.repo_info
    assert metadata["review_strategy"] == "review-units"
    assert metadata["unit_count"] == 2
    assert len(metadata["review_plan"]["review_units"]) == 2
    assert len(metadata["review_unit_summary"]) == 2
    assert all("changed_files" in unit for unit in result.debug_payload["review_units"])
    assert all("context_files" in unit for unit in result.debug_payload["review_units"])
    assert result.debug_payload["review_plan"]["unit_count"] == 2


def test_engine_keeps_malformed_review_unit_as_failed_unit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parsed_diff = ParsedDiff(
        files=(
            parsed_app_diff().files[0],
            DiffFile(
                old_path="src/second.py",
                new_path="src/second.py",
                change_type="modified",
                hunks=parsed_app_diff().files[0].hunks,
            ),
        )
    )

    class MalformedProvider:
        name = "openai-compatible"
        model = "deepseek-v4-pro"

        def review(self, context_pack):
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=json.dumps(
                    {
                        "schema_version": "review-pilot.llm-findings.v1",
                        "findings": [{"file_path": "src/app.py"}],
                    }
                ),
            )

    monkeypatch.setattr(
        "review_pilot.review_engine.create_provider",
        lambda name: MalformedProvider(),
    )
    review_input = make_input(tmp_path, parsed_diff=parsed_diff)
    (tmp_path / "src" / "second.py").write_text("value = 2\n", encoding="utf-8")

    result = ReviewEngine(
        ReviewEngineOptions(
            provider="openai-compatible",
            strategy="review-units",
            review_unit_workers=2,
        )
    ).run(review_input)

    metadata = result.report.repo_info
    assert metadata["unit_count"] == 2
    assert metadata["unit_completed_count"] == 0
    assert metadata["unit_failed_count"] == 2
    assert all(unit["status"] == "failed" for unit in metadata["review_unit_summary"])


def test_engine_merges_duplicate_findings_from_shared_context_units(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parsed_diff = ParsedDiff(
        files=(
            parsed_app_diff().files[0],
            DiffFile(
                old_path="src/second.py",
                new_path="src/second.py",
                change_type="modified",
                hunks=parsed_app_diff().files[0].hunks,
            ),
        )
    )

    class SharedContextProvider:
        name = "static"
        model = "static-model"

        def review(self, context_pack):
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=json.dumps(
                    {
                        "schema_version": "review-pilot.llm-findings.v1",
                        "findings": [
                            {
                                "message": "检查项目配置是否与变更一致。",
                                "file_path": "pyproject.toml",
                                "line_no": 1,
                                "severity": "P2",
                                "category": "maintainability",
                                "source": "llm",
                                "confidence": "medium",
                                "evidence": {"reason": "配置文件属于共享上下文。"},
                                "suggestion": "保持配置和代码行为一致。",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            )

    review_input = make_input(tmp_path, parsed_diff=parsed_diff)
    (tmp_path / "src" / "second.py").write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.review_pilot]\nname='x'\n", encoding="utf-8")
    monkeypatch.setattr(
        "review_pilot.review_engine.create_provider",
        lambda name: SharedContextProvider(),
    )
    result = ReviewEngine(
        ReviewEngineOptions(provider="openai-compatible", strategy="review-units")
    ).run(review_input)

    assert result.report.merge_summary["source_counts"]["llm"] == 1
    assert result.report.findings[0].evidence["duplicate_count"] == 2
    assert result.report.findings[0].evidence["merge"]["input_count"] == 1


def make_input(
    tmp_path: Path,
    *,
    input_source: str = "local-staged",
    parsed_diff: ParsedDiff | None = None,
    metadata: dict[str, object] | None = None,
) -> ReviewInput:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    return ReviewInput(
        repo_info=RepoInfo(
            root=str(tmp_path),
            branch="main",
            head="abc123",
            has_staged_changes=True,
            has_unstaged_changes=False,
        ),
        config=ReviewPilotConfig.default(),
        parsed_diff=parsed_diff or parsed_app_diff(),
        input_source=input_source,
        metadata=metadata or {},
    )


def parsed_app_diff() -> ParsedDiff:
    return ParsedDiff(
        files=(
            DiffFile(
                old_path="src/app.py",
                new_path="src/app.py",
                change_type="modified",
                hunks=(
                    DiffHunk(
                        old_start=1,
                        old_count=1,
                        new_start=1,
                        new_count=1,
                        section="",
                        lines=(
                            DiffLine(
                                kind="added",
                                content="print('debug')",
                                old_line_no=None,
                                new_line_no=1,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
