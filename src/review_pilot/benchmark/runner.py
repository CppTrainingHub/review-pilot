from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import LLMConfig, ReflectionConfig, ReviewPilotConfig
from ..llm.schema import LLM_FINDINGS_SCHEMA_VERSION
from ..review_engine import ReviewEngine, ReviewEngineOptions, ReviewInput
from .dataset import AACRCase, DatasetError, load_aacr_dataset
from .metrics import (
    CaseEvaluation,
    aggregate_evaluations,
    case_evaluation_from_dict,
    evaluate_case,
    grouped_metrics,
    normalize_finding,
)
from .reports import write_benchmark_reports
from .repository import RepositoryCache, RepositoryError


BENCHMARK_REPORT_SCHEMA_VERSION = "review-pilot.aacr-benchmark.v1"
PROMPT_VERSION = "review-pilot.structured-review.v1"
MATCHER_VERSION = "token-overlap.v1"


class BenchmarkError(RuntimeError):
    """Raised when benchmark configuration cannot be executed."""


@dataclass(frozen=True)
class BenchmarkRunResult:
    report: dict[str, Any]
    output_dir: str
    report_json_path: str
    report_markdown_path: str
    exit_code: int


def run_aacr_benchmark(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    sample_set_path: str | Path | None = None,
    provider: str | None = None,
    limit: int | None = None,
    language: str | None = None,
    resume: bool = False,
    cache_dir: str | Path | None = None,
    max_context_tokens: int = 4000,
    strategy: str = "baseline",
    dynamic_context: bool = False,
    snippet_location: bool = False,
    reflection: bool = False,
) -> BenchmarkRunResult:
    if limit is not None and limit < 1:
        raise BenchmarkError("--limit must be a positive integer")
    if max_context_tokens < 1:
        raise BenchmarkError("--max-context-tokens must be a positive integer")
    if strategy not in {"baseline", "review-units"}:
        raise BenchmarkError("--strategy must be baseline or review-units")
    review_unit_workers = _review_unit_workers_from_env()

    dataset = load_aacr_dataset(dataset_path)
    sample_config, sample_sha256 = _load_sample_set(sample_set_path)
    selected_cases = _select_cases(
        dataset.cases,
        sample_config=sample_config,
        language=language,
        limit=limit,
    )
    if not selected_cases:
        raise BenchmarkError("benchmark selected no cases")

    target_dir = Path(output_dir).resolve()
    case_dir = target_dir / "cases"
    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else (
        Path.home() / ".cache" / "review-pilot" / "aacr-bench"
    )
    repository_cache = RepositoryCache(
        cache_root=cache_root,
        workspace_root=target_dir / "workspaces",
    )
    effective_provider, provider_source = _resolve_provider(provider)
    llm_config = LLMConfig.from_env(effective_provider)
    model = llm_config.model
    evaluations: list[CaseEvaluation] = []
    latency_by_case: dict[str, float] = {}
    token_cost_by_case: dict[str, float | None] = {}
    tool_calls_by_case: dict[str, int] = {}
    dynamic_tokens_by_case: dict[str, int] = {}
    location_stats_by_case: dict[str, dict[str, int | float]] = {}
    reflection_summary_by_case: dict[str, dict[str, Any]] = {}
    case_summaries: list[dict[str, Any]] = []
    completed_cases = 0
    failed_cases = 0
    resumed_cases = 0

    for case in selected_cases:
        existing = _load_resumable_case(case_dir / f"{case.case_id}.json") if resume else None
        if existing is not None:
            evaluation = case_evaluation_from_dict(existing["evaluation"])
            latency_by_case[case.case_id] = float(existing.get("latency_ms", 0.0))
            token_cost_by_case[case.case_id] = existing.get("token_cost_usd")
            tool_calls_by_case[case.case_id] = int(existing.get("tool_calls", 0))
            dynamic_tokens_by_case[case.case_id] = int(existing.get("dynamic_tokens", 0))
            location_stats_by_case[case.case_id] = _location_stats(existing.get("report"))
            reflection_summary_by_case[case.case_id] = _reflection_summary(
                existing.get("report"),
                existing.get("reflection"),
            )
            evaluations.append(evaluation)
            case_summaries.append(_case_summary(existing))
            resumed_cases += 1
            completed_cases += 1
            continue

        started = time.perf_counter()
        token_cost = 0.0 if effective_provider == "fake" else None
        try:
            prepared = repository_cache.prepare(case)
            engine_result = ReviewEngine(
                ReviewEngineOptions(
                    provider=effective_provider,
                    max_context_tokens=max_context_tokens,
                    strategy=strategy,
                    dynamic_context=dynamic_context,
                    snippet_location=snippet_location,
                    reflection=reflection,
                    review_unit_workers=review_unit_workers,
                )
            ).run(
                ReviewInput(
                    repo_info=prepared.repo_info,
                    config=ReviewPilotConfig(
                        reflection=ReflectionConfig(
                            enabled=reflection,
                            review_all=reflection,
                        )
                    ),
                    parsed_diff=prepared.parsed_diff,
                    input_source="aacr-bench",
                    metadata={
                        "benchmark": "aacr-bench",
                        "case_id": case.case_id,
                        "pr_url": case.pr_url,
                        "language": case.project_main_language,
                        "context_level": case.context_level,
                        "source_commit": case.source_commit,
                        "target_commit": case.target_commit,
                        "repository_source": prepared.repository_source,
                    },
                )
            )
            predicted = [finding.to_dict() for finding in engine_result.report.findings]
            predicted_comments = [normalize_finding(finding) for finding in predicted]
            evaluation = evaluate_case(
                case_id=case.case_id,
                language=case.project_main_language,
                context_level=case.context_level,
                expected_comments=case.comments,
                predicted_findings=predicted,
            )
            status = "completed"
            completed_cases += 1
            error = None
            estimated_input_tokens = (
                (engine_result.report.repo_info or {}).get("context", {}).get("used_tokens", 0)
            )
            report_payload = engine_result.report.to_dict()
            dynamic_metadata = (engine_result.report.repo_info or {}).get(
                "dynamic_context", {}
            )
            tool_calls = int(dynamic_metadata.get("tool_calls", 0))
            dynamic_tokens = int(
                (dynamic_metadata.get("token_usage", {}) or {}).get(
                    "total_tokens",
                    0,
                )
            )
            location_stats = _location_stats(report_payload)
            reflection_summary = _reflection_summary(report_payload)
        except (RepositoryError, DatasetError, ValueError, RuntimeError) as exc:
            predicted_comments = []
            evaluation = evaluate_case(
                case_id=case.case_id,
                language=case.project_main_language,
                context_level=case.context_level,
                expected_comments=case.comments,
                predicted_findings=[],
            )
            status = "failed"
            completed_cases += 0
            failed_cases += 1
            error = str(exc)
            estimated_input_tokens = 0
            report_payload = None
            tool_calls = 0
            dynamic_tokens = 0
            location_stats = _empty_location_stats()
            reflection_summary = _empty_reflection_summary(enabled=reflection)
        latency_ms = (time.perf_counter() - started) * 1000
        latency_by_case[case.case_id] = latency_ms
        token_cost_by_case[case.case_id] = token_cost
        tool_calls_by_case[case.case_id] = tool_calls
        dynamic_tokens_by_case[case.case_id] = dynamic_tokens
        location_stats_by_case[case.case_id] = location_stats
        reflection_summary_by_case[case.case_id] = reflection_summary
        record = {
            "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
            "status": status,
            "case": case.to_dict(),
            "latency_ms": round(latency_ms, 3),
            "estimated_input_tokens": estimated_input_tokens,
            "tool_calls": tool_calls,
            "dynamic_tokens": dynamic_tokens,
            "location_stats": location_stats,
            "reflection": reflection_summary,
            "reflection_tokens": reflection_summary["token_usage"]["total_tokens"],
            "token_cost_usd": token_cost,
            "error": error,
            "report": report_payload,
            "predicted_comments": predicted_comments,
            "evaluation": evaluation.to_dict(
                latency_ms=latency_ms,
                token_cost_usd=token_cost,
            ),
        }
        _write_case_record(case_dir / f"{case.case_id}.json", record)
        evaluations.append(evaluation)
        case_summaries.append(_case_summary(record))

    summary = aggregate_evaluations(
        evaluations,
        latency_by_case=latency_by_case,
        token_cost_by_case=token_cost_by_case,
    )
    summary.update(
        {
            "selected_cases": len(selected_cases),
            "completed_cases": completed_cases,
            "failed_cases": failed_cases,
            "resumed_cases": resumed_cases,
            "dataset_issues": len(dataset.issues),
            "average_latency_ms": summary.pop("latency_ms", 0.0),
        }
    )
    summary["reflection"] = _aggregate_reflection_summaries(
        reflection_summary_by_case.values()
    )
    if snippet_location:
        location_total = sum(
            int(item.get("location_total", 0))
            for item in location_stats_by_case.values()
        )
        location_matched = sum(
            int(item.get("location_matched", 0))
            for item in location_stats_by_case.values()
        )
        location_downgraded = sum(
            int(item.get("location_downgraded", 0))
            for item in location_stats_by_case.values()
        )
        location_dropped = sum(
            int(item.get("location_dropped", 0))
            for item in location_stats_by_case.values()
        )
        summary.update(
            {
                "location_total": location_total,
                "location_matched": location_matched,
                "location_downgraded": location_downgraded,
                "location_dropped": location_dropped,
                "location_failure_rate": (
                    (location_downgraded + location_dropped) / location_total
                    if location_total
                    else 0.0
                ),
            }
        )
    report: dict[str, Any] = {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "benchmark": "AACR-Bench",
        "dataset": {
            **dataset.to_dict(),
            "source_url": "https://github.com/alibaba/aacr-bench/blob/main/dataset/positive_samples.json",
        },
        "sample_set": {
            "path": str(sample_set_path) if sample_set_path else None,
            "sha256": sample_sha256,
            "name": sample_config.get("name") if sample_config else None,
            "case_ids": [case.case_id for case in selected_cases],
        },
        "run_config": {
            "provider": effective_provider,
            "provider_source": provider_source,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "llm_schema_version": LLM_FINDINGS_SCHEMA_VERSION,
            "matcher": MATCHER_VERSION,
            "max_context_tokens": max_context_tokens,
            "strategy": strategy,
            "dynamic_context": dynamic_context,
            "snippet_location": snippet_location,
            "reflection": reflection,
            "reflection_review_all": reflection,
            "review_unit_workers": review_unit_workers,
            "tools_enabled": dynamic_context,
            "token_cost_status": (
                "fake_provider_zero_cost"
                if effective_provider == "fake"
                else "provider_usage_not_reported"
            ),
        },
        "summary": summary,
        "by_language": grouped_metrics(
            evaluations,
            "language",
            latency_by_case=latency_by_case,
            token_cost_by_case=token_cost_by_case,
        ),
        "by_category": grouped_metrics(
            evaluations,
            "category",
            latency_by_case=latency_by_case,
            token_cost_by_case=token_cost_by_case,
        ),
        "by_context_level": grouped_metrics(
            evaluations,
            "context_level",
            latency_by_case=latency_by_case,
            token_cost_by_case=token_cost_by_case,
        ),
        "context_recall": _context_recall_metrics(
            evaluations,
            tool_calls_by_case=tool_calls_by_case,
            dynamic_tokens_by_case=dynamic_tokens_by_case,
            latency_by_case=latency_by_case,
            token_cost_by_case=token_cost_by_case,
        ),
        "dynamic_context": {
            "enabled": dynamic_context,
            "tool_calls": sum(tool_calls_by_case.values()),
            "dynamic_tokens": sum(dynamic_tokens_by_case.values()),
            "token_cost_usd": summary.get("token_cost_usd"),
        },
        "dataset_issues": [issue.to_dict() for issue in dataset.issues],
        "cases": sorted(case_summaries, key=lambda item: item["case_id"]),
    }
    if strategy == "review-units":
        report["comparison"] = _load_strategy_comparison(
            target_dir,
            review_units_summary=summary,
        )
    report_json_path, report_markdown_path = write_benchmark_reports(report, str(target_dir))
    return BenchmarkRunResult(
        report=report,
        output_dir=str(target_dir),
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        exit_code=1 if failed_cases else 0,
    )


def _resolve_provider(provider: str | None) -> tuple[str, str]:
    if provider:
        return provider, "argument"
    configured = os.getenv("REVIEW_PILOT_LLM_PROVIDER")
    if configured:
        return configured, "environment"
    if os.getenv("REVIEW_PILOT_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return "openai-compatible", "environment-api-key"
    raise BenchmarkError(
        "benchmark requires --provider or REVIEW_PILOT_LLM_PROVIDER; "
        "a real benchmark must not silently fall back to fake"
    )


def _review_unit_workers_from_env() -> int:
    raw = os.getenv("REVIEW_PILOT_REVIEW_UNIT_WORKERS", "1").strip()
    try:
        workers = int(raw)
    except ValueError as exc:
        raise BenchmarkError(
            "REVIEW_PILOT_REVIEW_UNIT_WORKERS must be a positive integer"
        ) from exc
    if workers < 1:
        raise BenchmarkError(
            "REVIEW_PILOT_REVIEW_UNIT_WORKERS must be a positive integer"
        )
    return workers


def _load_strategy_comparison(
    target_dir: Path,
    *,
    review_units_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the sibling real baseline when it is available."""

    baseline_path = target_dir.parent / "baseline-30-real" / "report.json"
    baseline_summary: dict[str, Any] | None = None
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
            baseline_summary = payload["summary"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        baseline_summary = None
    return {
        "baseline": {
            "available": baseline_summary is not None,
            "report_path": str(baseline_path),
            "summary": baseline_summary or {},
        },
        "review-units": {
            "available": True,
            "report_path": str(target_dir / "report.json"),
            "summary": dict(review_units_summary),
        },
    }


def _load_sample_set(path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    sample_path = Path(path)
    try:
        raw_bytes = sample_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read sample set {sample_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError("sample set root must be an object")
    case_ids = payload.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids or not all(isinstance(item, str) for item in case_ids):
        raise BenchmarkError("sample set case_ids must be a non-empty string array")
    if len(set(case_ids)) != len(case_ids):
        raise BenchmarkError("sample set case_ids must be unique")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def _select_cases(
    cases: tuple[AACRCase, ...],
    *,
    sample_config: dict[str, Any] | None,
    language: str | None,
    limit: int | None,
) -> list[AACRCase]:
    by_id = {case.case_id: case for case in cases}
    if sample_config is None:
        selected = list(cases)
    else:
        selected = []
        missing: list[str] = []
        for case_id in sample_config["case_ids"]:
            case = by_id.get(case_id)
            if case is None:
                missing.append(case_id)
            else:
                selected.append(case)
        if missing:
            raise BenchmarkError(f"sample set cases are missing from dataset: {missing}")
    if language:
        expected_language = language.strip().lower()
        selected = [
            case
            for case in selected
            if case.project_main_language.lower() == expected_language
        ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _load_resumable_case(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "completed" or not isinstance(payload.get("evaluation"), dict):
        return None
    return payload


def _write_case_record(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _case_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    case = record.get("case", {})
    evaluation = record.get("evaluation", {})
    return {
        "case_id": case.get("case_id", ""),
        "language": case.get("project_main_language", ""),
        "context_level": case.get("context_level", "unknown"),
        "status": record.get("status", "unknown"),
        "latency_ms": record.get("latency_ms", 0.0),
        "estimated_input_tokens": record.get("estimated_input_tokens", 0),
        "tool_calls": record.get("tool_calls", 0),
        "dynamic_tokens": record.get("dynamic_tokens", 0),
        "location_stats": record.get("location_stats", _empty_location_stats()),
        "reflection": record.get("reflection", _empty_reflection_summary()),
        "reflection_tokens": record.get("reflection_tokens", 0),
        "token_cost_usd": record.get("token_cost_usd"),
        "error": record.get("error"),
        "metrics": evaluation.get("metrics", {}),
    }


def _empty_reflection_summary(*, enabled: bool = False) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "eligible": 0,
        "reviewed": 0,
        "keep": 0,
        "downgrade": 0,
        "drop": 0,
        "errors": 0,
        "skipped": 0,
        "kept_original_on_error": 0,
        "token_usage": {"total_tokens": 0},
    }


def _reflection_summary(
    report: Any,
    fallback: Any = None,
) -> dict[str, Any]:
    source = fallback
    if isinstance(report, Mapping):
        repo_info = report.get("repo_info")
        if isinstance(repo_info, Mapping) and isinstance(repo_info.get("reflection"), Mapping):
            source = repo_info["reflection"]
    if not isinstance(source, Mapping):
        return _empty_reflection_summary()
    result = _empty_reflection_summary(enabled=bool(source.get("enabled", False)))
    for key in (
        "eligible",
        "reviewed",
        "keep",
        "downgrade",
        "drop",
        "errors",
        "skipped",
        "kept_original_on_error",
    ):
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
    usage = source.get("token_usage")
    if isinstance(usage, Mapping):
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, int) and not isinstance(total_tokens, bool) and total_tokens >= 0:
            result["token_usage"]["total_tokens"] = total_tokens
    return result


def _aggregate_reflection_summaries(
    summaries: Any,
) -> dict[str, Any]:
    result = _empty_reflection_summary()
    for summary in summaries:
        if not isinstance(summary, Mapping):
            continue
        result["enabled"] = result["enabled"] or bool(summary.get("enabled", False))
        for key in (
            "eligible",
            "reviewed",
            "keep",
            "downgrade",
            "drop",
            "errors",
            "skipped",
            "kept_original_on_error",
        ):
            value = summary.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                result[key] += value
        usage = summary.get("token_usage")
        if isinstance(usage, Mapping):
            value = usage.get("total_tokens")
            if isinstance(value, int) and not isinstance(value, bool):
                result["token_usage"]["total_tokens"] += value
    return result


def _empty_location_stats() -> dict[str, int | float]:
    return {
        "location_total": 0,
        "location_matched": 0,
        "location_downgraded": 0,
        "location_dropped": 0,
        "location_failure_rate": 0.0,
    }


def _location_stats(report: Any) -> dict[str, int | float]:
    if not isinstance(report, Mapping):
        return _empty_location_stats()
    repo_info = report.get("repo_info")
    if not isinstance(repo_info, Mapping):
        return _empty_location_stats()
    evidence_summary = repo_info.get("evidence_summary")
    if not isinstance(evidence_summary, Mapping):
        return _empty_location_stats()
    result = _empty_location_stats()
    for key in result:
        value = evidence_summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = value
    return result


def _context_recall_metrics(
    evaluations: list[CaseEvaluation],
    *,
    tool_calls_by_case: Mapping[str, int],
    dynamic_tokens_by_case: Mapping[str, int],
    latency_by_case: Mapping[str, float],
    token_cost_by_case: Mapping[str, float | None],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[CaseEvaluation]] = {}
    for evaluation in evaluations:
        groups.setdefault(evaluation.context_level, []).append(evaluation)
    output: dict[str, dict[str, Any]] = {}
    for level, items in sorted(groups.items()):
        metrics = aggregate_evaluations(
            items,
            latency_by_case=latency_by_case,
            token_cost_by_case=token_cost_by_case,
        )
        metrics["tool_calls"] = sum(
            tool_calls_by_case.get(item.case_id, 0) for item in items
        )
        metrics["dynamic_tokens"] = sum(
            dynamic_tokens_by_case.get(item.case_id, 0) for item in items
        )
        output[level] = metrics
    return output
