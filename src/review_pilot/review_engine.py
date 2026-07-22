from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from .code_index import build_code_index
from .config import ConfigError, ReviewPilotConfig
from .context_pack import build_review_context_pack, validate_context_pack_dict
from .context_tools import ContextToolExecutor
from .diff_line_map import build_changed_line_map
from .dynamic_context import (
    DynamicContextResult,
    run_dynamic_context,
)
from .evidence_guard import EvidenceGuardResult
from .finding_merger import merge_findings
from .llm import (
    LLMOutputError,
    LLMProviderError,
    StructuredReviewResult,
    StructuredReviewer,
    create_provider,
)
from .context_selector import select_context_candidates
from .language_detection import normalize_path
from .models import (
    ContextBudgetManifest,
    ContextCandidate,
    ContextCandidateManifest,
    OmittedContext,
    ParsedDiff,
    RepoInfo,
)
from .project_detector import detect_project
from .report_models import Finding, ReviewReport
from .reflection import ReflectionDecision, ReflectionFilter
from .review_units import ReviewPlan, ReviewUnit, build_review_plan
from .rule_engine import default_rule_engine
from .token_budget import apply_token_budget
from .tool_filter import ToolFilterResult, filter_tool_findings
from .tool_models import ToolResult
from .tool_registry import ToolRegistry
from .tools.semgrep_tool import SEMGREP_TOOL_NAME, run_semgrep_tool


SUPPORTED_INPUT_SOURCES = {"local-staged", "github-pr", "gitlab-mr", "aacr-bench"}


class ReviewEngineError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class ReviewInput:
    """All review data needed by the shared execution core."""

    repo_info: RepoInfo
    config: ReviewPilotConfig
    parsed_diff: ParsedDiff
    input_source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewEngineOptions:
    provider: str | None = None
    with_tools: bool = False
    include_out_of_diff: bool = False
    max_context_tokens: int = 4000
    tool_runner: Callable[[Any, str], ToolResult] | None = None
    strategy: str = "baseline"
    dynamic_context: bool = False
    dynamic_max_rounds: int = 5
    dynamic_max_tokens: int = 12_000
    snippet_location: bool = False
    review_unit_workers: int = 1
    reflection: bool = False

    def __post_init__(self) -> None:
        if self.review_unit_workers < 1:
            raise ValueError("review_unit_workers must be positive")


@dataclass(frozen=True)
class ReviewEngineResult:
    report: ReviewReport
    debug_payload: dict[str, Any]


@dataclass(frozen=True)
class ToolCollection:
    results: list[ToolResult]
    filter_result: ToolFilterResult | None


class ReviewEngine:
    """Run the review behavior shared by local and platform adapters."""

    def __init__(self, options: ReviewEngineOptions | None = None) -> None:
        self.options = options or ReviewEngineOptions()

    def run(self, review_input: ReviewInput) -> ReviewEngineResult:
        if self.options.strategy not in {"baseline", "review-units"}:
            raise ReviewEngineError(
                "unsupported review strategy: "
                f"{self.options.strategy!r}; expected 'baseline' or 'review-units'"
            )
        self._validate_input(review_input)
        rule_findings = self._run_rules(review_input)

        context: ContextBudgetManifest | None = None
        llm_result: StructuredReviewResult | None = None
        llm_results: list[StructuredReviewResult] = []
        review_plan: ReviewPlan | None = None
        unit_runs: list[dict[str, Any]] = []
        dynamic_trace: list[dict[str, Any]] = []
        dynamic_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        dynamic_rounds = 0
        dynamic_truncated = False
        llm_context_packs: list[Any] = []
        reflection_provider: Any | None = None
        if self.options.provider is not None:
            try:
                if self.options.strategy == "review-units":
                    review_plan = self._build_review_plan(review_input)
                    index = build_code_index(
                        review_input.repo_info.root,
                        review_input.config,
                    )
                    reflection_provider = create_provider(self.options.provider)
                    reviewer = StructuredReviewer(
                        reflection_provider,
                        snippet_location=self.options.snippet_location,
                    )

                    def run_unit(
                        unit: ReviewUnit,
                    ) -> tuple[
                        ContextBudgetManifest,
                        StructuredReviewResult | None,
                        dict[str, Any],
                        DynamicContextResult | None,
                        Any,
                    ]:
                        started = time.perf_counter()
                        unit_diff = _diff_for_unit(
                            review_input.parsed_diff,
                            unit,
                        )
                        unit_context = self._build_unit_context(
                            review_input,
                            unit,
                            unit_diff,
                            index,
                        )
                        pack = build_review_context_pack(
                            repo_info=review_input.repo_info,
                            config=review_input.config,
                            parsed_diff=unit_diff,
                            rule_findings=_rule_findings_for_unit(
                                rule_findings,
                                unit,
                            ),
                            context=unit_context,
                            metadata={
                                "strategy": "review-units",
                                "review_unit": unit.to_dict(),
                            },
                        )
                        validate_context_pack_dict(pack.to_dict())
                        dynamic_result: DynamicContextResult | None = None
                        try:
                            if self.options.dynamic_context:
                                dynamic_result, attempts = _dynamic_review_unit_with_retry(
                                    reviewer,
                                    pack,
                                    root=review_input.repo_info.root,
                                    parsed_diff=unit_diff,
                                    max_rounds=self.options.dynamic_max_rounds,
                                    max_tokens=self.options.dynamic_max_tokens,
                                )
                                result = dynamic_result.review
                            else:
                                result, attempts = _review_unit_with_retry(reviewer, pack)
                        except LLMOutputError as exc:
                            return (
                                unit_context,
                                None,
                                _failed_unit_run_summary(
                                    unit,
                                    unit_context,
                                    error=str(exc),
                                    elapsed_ms=(time.perf_counter() - started) * 1000,
                                ),
                                None,
                                pack,
                            )
                        return (
                            unit_context,
                            result,
                            _unit_run_summary(
                                unit,
                                unit_context,
                                result,
                                attempts=attempts,
                                elapsed_ms=(time.perf_counter() - started) * 1000,
                                dynamic_result=dynamic_result,
                            ),
                            dynamic_result,
                            pack,
                        )

                    if self.options.review_unit_workers == 1:
                        unit_results = [run_unit(unit) for unit in review_plan.units]
                    else:
                        with ThreadPoolExecutor(
                            max_workers=self.options.review_unit_workers,
                            thread_name_prefix="review-unit",
                        ) as executor:
                            unit_results = list(executor.map(run_unit, review_plan.units))
                    for unit_context, result, unit_run, dynamic_result, pack in unit_results:
                        if result is not None:
                            llm_results.append(result)
                            llm_context_packs.append(pack)
                        unit_runs.append(unit_run)
                        if result is not None and dynamic_result is not None:
                            dynamic_trace.extend(
                                _trace_with_unit_id(
                                    dynamic_result,
                                    unit_run["unit_id"],
                                )
                            )
                            _accumulate_dynamic_usage(
                                dynamic_token_usage,
                                dynamic_result,
                            )
                            dynamic_rounds += dynamic_result.rounds
                            dynamic_truncated = (
                                dynamic_truncated or dynamic_result.truncated
                            )
                    context = _aggregate_unit_context(review_plan, unit_runs)
                else:
                    context = self._build_context(review_input)
                    pack = build_review_context_pack(
                        repo_info=review_input.repo_info,
                        config=review_input.config,
                        parsed_diff=review_input.parsed_diff,
                        rule_findings=rule_findings,
                        context=context,
                    )
                    validate_context_pack_dict(pack.to_dict())
                    reflection_provider = create_provider(self.options.provider)
                    reviewer = StructuredReviewer(
                        reflection_provider,
                        snippet_location=self.options.snippet_location,
                    )
                    if self.options.dynamic_context:
                        dynamic_result = run_dynamic_context(
                            context_pack=pack,
                            reviewer=reviewer,
                            executor=ContextToolExecutor(
                                review_input.repo_info.root,
                                review_input.parsed_diff,
                            ),
                            max_rounds=self.options.dynamic_max_rounds,
                            max_tokens=self.options.dynamic_max_tokens,
                            snippet_location=self.options.snippet_location,
                        )
                        llm_result = dynamic_result.review
                        dynamic_trace.extend(
                            _trace_with_unit_id(dynamic_result, None)
                        )
                        _accumulate_dynamic_usage(
                            dynamic_token_usage,
                            dynamic_result,
                        )
                        dynamic_rounds = dynamic_result.rounds
                        dynamic_truncated = dynamic_result.truncated
                    else:
                        llm_result = reviewer.review(pack)
                    llm_results.append(llm_result)
                    llm_context_packs.append(pack)
            except (ConfigError, LLMProviderError) as exc:
                raise ReviewEngineError(f"llm provider error: {exc}") from exc
            except LLMOutputError as exc:
                raise ReviewEngineError(f"llm output error: {exc}") from exc
            except ValueError as exc:
                raise ReviewEngineError(f"context pack error: {exc}") from exc

        tool_collection = ToolCollection(results=[], filter_result=None)
        if self.options.with_tools:
            tool_collection = self._collect_tool_findings(review_input)

        llm_findings = [
            finding
            for result in llm_results
            for finding in result.evidence.findings
        ]
        reflection_decisions: list[ReflectionDecision] = []
        reflection_summary: dict[str, Any] = {
            "enabled": False,
            "eligible": 0,
            "reviewed": 0,
            "keep": 0,
            "downgrade": 0,
            "drop": 0,
            "errors": 0,
            "skipped": 0,
            "token_usage": {"total_tokens": 0},
        }
        if (
            (self.options.reflection or review_input.config.reflection.enabled)
            and reflection_provider is not None
        ):
            reflection_config = replace(review_input.config.reflection, enabled=True)
            filtered_findings: list[Finding] = []
            for result, pack in zip(llm_results, llm_context_packs):
                filtered = ReflectionFilter(
                    reflection_provider,
                    reflection_config,
                ).apply(
                    result.evidence.findings,
                    context_pack=pack,
                    evidence=result.evidence,
                )
                filtered_findings.extend(filtered.findings)
                reflection_decisions.extend(filtered.decisions)
                for key, value in filtered.summary.items():
                    if key in reflection_summary and isinstance(value, int) and not isinstance(value, bool):
                        reflection_summary[key] += value
                reflection_summary["token_usage"]["total_tokens"] += int(
                    (filtered.summary.get("token_usage") or {}).get("total_tokens", 0)
                )
            reflection_summary["enabled"] = True
            llm_findings = filtered_findings

        merge_result = merge_findings(
            rule_findings=rule_findings,
            tool_findings=(
                list(tool_collection.filter_result.included_findings)
                if tool_collection.filter_result is not None
                else []
            ),
            llm_findings=llm_findings,
        )
        report = ReviewReport(
            findings=list(merge_result.findings),
            repo_info=self._build_report_metadata(
                review_input=review_input,
                context=context,
                llm_result=llm_result,
                evidence=llm_result.evidence if llm_result is not None else None,
                tool_collection=tool_collection,
                review_plan=review_plan,
                unit_runs=unit_runs,
                llm_results=llm_results,
                dynamic_trace=dynamic_trace,
                dynamic_token_usage=dynamic_token_usage,
                dynamic_rounds=dynamic_rounds,
                dynamic_truncated=dynamic_truncated,
                reflection_summary=reflection_summary,
                reflection_decisions=reflection_decisions,
            ),
            config_source=review_input.config.source,
            merge_summary=merge_result.summary.to_dict(),
        )
        return ReviewEngineResult(
            report=report,
            debug_payload=self._build_debug_payload(
                rule_findings=rule_findings,
                tool_collection=tool_collection,
                llm_results=llm_results,
                report=report,
                review_plan=review_plan,
                unit_runs=unit_runs,
                dynamic_trace=dynamic_trace,
                llm_findings=llm_findings,
                reflection_summary=reflection_summary,
                reflection_decisions=reflection_decisions,
            ),
        )

    @staticmethod
    def _validate_input(review_input: ReviewInput) -> None:
        if review_input.input_source not in SUPPORTED_INPUT_SOURCES:
            raise ReviewEngineError(
                "unsupported input source: "
                f"{review_input.input_source!r}; expected one of "
                f"{sorted(SUPPORTED_INPUT_SOURCES)}"
            )
        if review_input.parsed_diff.is_empty:
            raise ReviewEngineError("review input diff is empty", exit_code=1)
        if not review_input.repo_info.root:
            raise ReviewEngineError("review input repo_info.root is required")

    def _run_rules(self, review_input: ReviewInput) -> list[Finding]:
        return default_rule_engine(review_input.config).run(
            review_input.parsed_diff,
            repo_info=review_input.repo_info,
        )

    def _build_context(self, review_input: ReviewInput) -> ContextBudgetManifest:
        try:
            index = build_code_index(review_input.repo_info.root, review_input.config)
            candidates = select_context_candidates(review_input.parsed_diff, index)
            return apply_token_budget(
                candidates,
                review_input.parsed_diff,
                review_input.repo_info.root,
                self.options.max_context_tokens,
            )
        except ValueError as exc:
            raise ReviewEngineError(f"context pack error: {exc}") from exc

    def _build_review_plan(self, review_input: ReviewInput) -> ReviewPlan:
        try:
            index = build_code_index(review_input.repo_info.root, review_input.config)
            return build_review_plan(
                review_input.parsed_diff,
                index,
                self.options.max_context_tokens,
            )
        except ValueError as exc:
            raise ReviewEngineError(f"review plan error: {exc}") from exc

    def _build_unit_context(
        self,
        review_input: ReviewInput,
        unit: ReviewUnit,
        unit_diff: ParsedDiff,
        index: Any,
    ) -> ContextBudgetManifest:
        try:
            manifest = select_context_candidates(unit_diff, index)
            candidates = {
                candidate.path: candidate
                for candidate in manifest.candidates
                if candidate.path in set(unit.all_files)
            }
            for path in unit.changed_files:
                if path not in candidates:
                    candidates[path] = ContextCandidate(
                        path=path,
                        reason="changed_file",
                        priority=0,
                        language="unknown",
                        is_changed=True,
                    )
            filtered = ContextCandidateManifest(
                changed_paths=tuple(unit.changed_files),
                candidates=tuple(
                    sorted(
                        candidates.values(),
                        key=lambda candidate: (
                            candidate.priority,
                            candidate.reason,
                            candidate.path,
                        ),
                    )
                ),
                index_file_count=manifest.index_file_count,
            )
            return apply_token_budget(
                filtered,
                unit_diff,
                review_input.repo_info.root,
                unit.budget_tokens,
            )
        except ValueError as exc:
            raise ReviewEngineError(
                f"review unit context error ({unit.unit_id}): {exc}"
            ) from exc

    def _collect_tool_findings(self, review_input: ReviewInput) -> ToolCollection:
        tool_results: list[ToolResult] = []
        tool_filter_result: ToolFilterResult | None = None
        detection = detect_project(review_input.repo_info.root)
        registry = ToolRegistry(detection, review_input.config)
        try:
            semgrep_tool = registry.get(SEMGREP_TOOL_NAME)
        except KeyError:
            semgrep_tool = None
        if semgrep_tool is not None:
            runner = self.options.tool_runner or run_semgrep_tool
            semgrep_result = runner(
                semgrep_tool,
                review_input.repo_info.root,
            )
            tool_results.append(semgrep_result)
            if semgrep_result.status == "success":
                changed_lines = build_changed_line_map(review_input.parsed_diff)
                tool_filter_result = filter_tool_findings(
                    tool_results,
                    changed_lines,
                    include_out_of_diff=self.options.include_out_of_diff,
                )
        return ToolCollection(
            results=tool_results,
            filter_result=tool_filter_result,
        )

    def _build_report_metadata(
        self,
        *,
        review_input: ReviewInput,
        context: ContextBudgetManifest | None,
        llm_result: StructuredReviewResult | None,
        evidence: EvidenceGuardResult | None,
        tool_collection: ToolCollection,
        review_plan: ReviewPlan | None,
        unit_runs: list[dict[str, Any]],
        llm_results: list[StructuredReviewResult],
        dynamic_trace: list[dict[str, Any]],
        dynamic_token_usage: dict[str, int],
        dynamic_rounds: int,
        dynamic_truncated: bool,
        reflection_summary: dict[str, Any],
        reflection_decisions: list[ReflectionDecision],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "root": review_input.repo_info.root,
            "branch": review_input.repo_info.branch,
            "head": review_input.repo_info.head,
            "engine": "review-engine",
            "input_source": review_input.input_source,
            "ai_enabled": self.options.provider is not None,
            "tools_enabled": self.options.with_tools,
            "include_out_of_diff": self.options.include_out_of_diff,
            "review_strategy": self.options.strategy,
            "snippet_location": self.options.snippet_location,
            "reflection": reflection_summary,
            "reflection_decisions": [item.to_dict() for item in reflection_decisions],
            "tool_results": [item.to_dict() for item in tool_collection.results],
            "tool_filter": (
                tool_collection.filter_result.to_dict()
                if tool_collection.filter_result is not None
                else None
            ),
            **dict(review_input.metadata),
        }
        metadata["engine"] = "review-engine"
        metadata["input_source"] = review_input.input_source
        if self.options.dynamic_context:
            metadata["dynamic_context"] = {
                "enabled": True,
                "dynamic_context_trace": dynamic_trace,
                "rounds": dynamic_rounds,
                "tool_calls": len(dynamic_trace),
                "token_usage": dict(dynamic_token_usage),
                "truncated": dynamic_truncated,
                "max_rounds": self.options.dynamic_max_rounds,
                "max_tokens": self.options.dynamic_max_tokens,
            }
        if context is not None:
            metadata["context"] = {
                "used": len(context.context_used),
                "omitted": len(context.context_omitted),
                "used_tokens": context.used_tokens,
                "max_context_tokens": context.max_context_tokens,
            }
        if review_plan is not None:
            metadata["context"] = {
                "used": sum(item["context_used"] for item in unit_runs),
                "omitted": sum(item["omitted"] for item in unit_runs),
                "used_tokens": sum(item["used_tokens"] for item in unit_runs),
                "max_context_tokens": review_plan.max_context_tokens,
            }
            metadata.update(
                {
                    "review_plan": review_plan.to_dict(),
                    "review_unit_summary": unit_runs,
                    "unit_count": len(review_plan.units),
                    "unit_completed_count": sum(
                        item.get("status") == "completed" for item in unit_runs
                    ),
                    "unit_failed_count": sum(
                        item.get("status") == "failed" for item in unit_runs
                    ),
                    "omitted": sum(
                        item["omitted"] for item in unit_runs
                    ),
                }
            )
        if llm_result is not None or llm_results:
            first_result = llm_result or llm_results[0]
            all_evidence = [result.evidence for result in llm_results]
            evidence_summary = (
                evidence.summary
                if evidence is not None
                else _aggregate_evidence_summary(all_evidence)
            )
            metadata.update(
                {
                    "provider": first_result.response.provider,
                    "model": first_result.response.model,
                    "evidence_summary": evidence_summary,
                    "dropped_llm_findings": (
                        [
                            decision.to_dict()
                            for item in all_evidence
                            for decision in item.dropped_findings
                        ]
                    ),
                }
            )
        return metadata

    @staticmethod
    def _build_debug_payload(
        *,
        rule_findings: list[Finding],
        tool_collection: ToolCollection,
        llm_results: list[StructuredReviewResult],
        report: ReviewReport,
        review_plan: ReviewPlan | None,
        unit_runs: list[dict[str, Any]],
        dynamic_trace: list[dict[str, Any]],
        llm_findings: list[Finding],
        reflection_summary: dict[str, Any],
        reflection_decisions: list[ReflectionDecision],
    ) -> dict[str, Any]:
        raw_llm_findings = [
            finding
            for result in llm_results
            for finding in result.evidence.findings
        ]
        tool_findings = (
            list(tool_collection.filter_result.included_findings)
            if tool_collection.filter_result is not None
            else []
        )
        return {
            "rule_findings": [finding.to_dict() for finding in rule_findings],
            "tool_findings": [finding.to_dict() for finding in tool_findings],
            "llm_findings": [finding.to_dict() for finding in llm_findings],
            "raw_llm_findings": [finding.to_dict() for finding in raw_llm_findings],
            "findings": [
                finding.to_dict()
                for finding in [*rule_findings, *tool_findings, *llm_findings]
            ],
            "merge_summary": report.merge_summary,
            "merged_findings": [finding.to_dict() for finding in report.findings],
            "tool_results": [
                result.to_dict()
                for result in tool_collection.results
            ],
            "tool_filter": (
                tool_collection.filter_result.to_dict()
                if tool_collection.filter_result is not None
                else {
                    "total_tool_findings": 0,
                    "included_count": 0,
                    "out_of_diff_count": 0,
                    "out_of_diff_findings": [],
                }
            ),
            "review_plan": review_plan.to_dict() if review_plan is not None else None,
            "review_units": unit_runs,
            "dynamic_context_trace": dynamic_trace,
            "reflection": reflection_summary,
            "reflection_decisions": [item.to_dict() for item in reflection_decisions],
        }


def _diff_for_unit(parsed_diff: ParsedDiff, unit: ReviewUnit) -> ParsedDiff:
    changed = set(unit.changed_files)
    return ParsedDiff(
        files=tuple(
            diff_file
            for diff_file in parsed_diff.files
            if normalize_path(diff_file.path) in changed
        )
    )


def _rule_findings_for_unit(
    findings: list[Finding],
    unit: ReviewUnit,
) -> list[Finding]:
    changed = set(unit.changed_files)
    return [
        finding
        for finding in findings
        if finding.file_path is None
        or normalize_path(finding.file_path) in changed
    ]


def _unit_run_summary(
    unit: ReviewUnit,
    context: ContextBudgetManifest,
    result: StructuredReviewResult,
    *,
    attempts: int,
    elapsed_ms: float,
    dynamic_result: DynamicContextResult | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "unit_id": unit.unit_id,
        "changed_files": list(unit.changed_files),
        "context_files": list(unit.context_files),
        "reasons": {
            path: list(unit.reasons.get(path, ()))
            for path in unit.all_files
        },
        "budget_tokens": unit.budget_tokens,
        "used_tokens": context.used_tokens,
        "context_used": len(context.context_used),
        "omitted": len(context.context_omitted),
        "omitted_context": [item.to_dict() for item in context.context_omitted],
        "finding_count": len(result.evidence.findings),
        "evidence_summary": result.evidence.summary,
        "provider": result.response.provider,
        "model": result.response.model,
        "provider_attempts": attempts,
        "elapsed_ms": round(elapsed_ms, 3),
        "status": "completed",
    }
    if dynamic_result is not None:
        payload["dynamic_context"] = {
            "dynamic_context_trace": [
                item.to_dict() for item in dynamic_result.trace
            ],
            "rounds": dynamic_result.rounds,
            "tool_calls": dynamic_result.tool_calls,
            "token_usage": dict(dynamic_result.token_usage),
            "truncated": dynamic_result.truncated,
        }
    return payload


def _failed_unit_run_summary(
    unit: ReviewUnit,
    context: ContextBudgetManifest,
    *,
    error: str,
    elapsed_ms: float,
) -> dict[str, Any]:
    """Keep one malformed provider response from discarding the whole PR run."""

    return {
        "unit_id": unit.unit_id,
        "changed_files": list(unit.changed_files),
        "context_files": list(unit.context_files),
        "reasons": {
            path: list(unit.reasons.get(path, ()))
            for path in unit.all_files
        },
        "budget_tokens": unit.budget_tokens,
        "used_tokens": context.used_tokens,
        "context_used": len(context.context_used),
        "omitted": len(context.context_omitted),
        "omitted_context": [item.to_dict() for item in context.context_omitted],
        "finding_count": 0,
        "evidence_summary": {
            "total": 0,
            "kept": 0,
            "verified": 0,
            "downgraded": 0,
            "dropped": 0,
        },
        "provider_attempts": 2,
        "elapsed_ms": round(elapsed_ms, 3),
        "status": "failed",
        "error": error,
    }


def _aggregate_unit_context(
    plan: ReviewPlan,
    unit_runs: list[dict[str, Any]],
) -> ContextBudgetManifest:
    used_tokens = sum(int(item["used_tokens"]) for item in unit_runs)
    return ContextBudgetManifest(
        changed_paths=plan.changed_files,
        max_context_tokens=plan.max_context_tokens,
        used_tokens=used_tokens,
        index_file_count=0,
        context_used=(),
        context_omitted=tuple(
            OmittedContext(
                path=item["unit_id"],
                reason="review_unit",
                priority=0,
                language="unknown",
                omitted_reason=omitted["omitted_reason"],
                estimated_tokens=omitted.get("estimated_tokens", 0),
                omitted_lines=omitted.get("omitted_lines", 0),
            )
            for item in unit_runs
            for omitted in item["omitted_context"]
        ),
    )


def _aggregate_evidence_summary(
    results: list[EvidenceGuardResult],
) -> dict[str, int | float]:
    summary: dict[str, int | float] = {
        key: sum(result.summary[key] for result in results)
        for key in ("total", "kept", "verified", "downgraded", "dropped")
    }
    if any("location_total" in result.summary for result in results):
        for key in (
            "location_total",
            "location_matched",
            "location_downgraded",
            "location_dropped",
        ):
            summary[key] = sum(int(result.summary.get(key, 0)) for result in results)
        total = int(summary["location_total"])
        summary["location_failure_rate"] = (
            (int(summary["location_downgraded"]) + int(summary["location_dropped"])) / total
            if total
            else 0.0
        )
    return summary


def _review_unit_with_retry(
    reviewer: StructuredReviewer,
    pack: Any,
) -> tuple[StructuredReviewResult, int]:
    """Retry one real provider request when the model breaks the JSON contract."""

    try:
        return reviewer.review(pack), 1
    except LLMOutputError:
        return reviewer.review(pack), 2


def _dynamic_review_unit_with_retry(
    reviewer: StructuredReviewer,
    pack: Any,
    *,
    root: str,
    parsed_diff: ParsedDiff,
    max_rounds: int,
    max_tokens: int,
) -> tuple[DynamicContextResult, int]:
    try:
        return (
            run_dynamic_context(
                context_pack=pack,
                reviewer=reviewer,
                executor=ContextToolExecutor(root, parsed_diff),
                max_rounds=max_rounds,
                max_tokens=max_tokens,
                snippet_location=reviewer.snippet_location,
            ),
            1,
        )
    except LLMOutputError:
        return (
            run_dynamic_context(
                context_pack=pack,
                reviewer=reviewer,
                executor=ContextToolExecutor(root, parsed_diff),
                max_rounds=max_rounds,
                max_tokens=max_tokens,
                snippet_location=reviewer.snippet_location,
            ),
            2,
        )


def _trace_with_unit_id(
    result: DynamicContextResult,
    unit_id: str | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in result.trace:
        payload = item.to_dict()
        if unit_id is not None:
            payload["unit_id"] = unit_id
        output.append(payload)
    return output


def _accumulate_dynamic_usage(
    total: dict[str, int],
    result: DynamicContextResult,
) -> None:
    for key, value in result.token_usage.items():
        if key in total:
            total[key] += value
