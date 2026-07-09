from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .code_index import build_code_index
from .config import ConfigError, load_project_config
from .context_pack import build_review_context_pack, validate_context_pack_dict
from .context_selector import select_context_candidates
from .diff_line_map import build_changed_line_map
from .diff_parser import DiffParseError
from .diff_reader import DiffReader
from .evidence_guard import EvidenceGuardResult
from .finding_merger import merge_findings
from .git_client import GitClient, GitError, NotGitRepositoryError
from .llm import (
    LLMOutputError,
    LLMProviderError,
    StructuredReviewResult,
    StructuredReviewer,
    create_provider,
)
from .models import ContextBudgetManifest, ParsedDiff, RepoInfo
from .project_detector import detect_project
from .report_models import Finding, ReviewReport
from .report_summary import should_fail_findings
from .report_writer import write_report
from .rule_engine import default_rule_engine
from .token_budget import apply_token_budget
from .tool_filter import ToolFilterResult, filter_tool_findings
from .tool_models import ToolResult
from .tool_registry import ToolRegistry
from .tools.semgrep_tool import SEMGREP_TOOL_NAME, run_semgrep_tool


ReviewProfileName = Literal["manual", "pre-commit", "pre-push"]


class ReviewPipelineError(Exception):
    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class ReviewPipelineOptions:
    staged: bool = True
    no_ai: bool = False
    with_tools: bool = False
    include_out_of_diff: bool = False
    provider: str | None = None
    profile: ReviewProfileName = "manual"
    output_format: Literal["json", "markdown"] = "markdown"
    output: str | Path | None = None
    debug_findings: bool = False
    fail_on: str | None = None
    max_context_tokens: int = 4000


@dataclass(frozen=True)
class ReviewPipelineResult:
    report: ReviewReport
    rendered_output: str
    output_path: Path | None
    exit_code: int
    debug_payload: dict[str, Any] | None = None

    @property
    def message(self) -> str:
        if self.output_path is None:
            return self.rendered_output
        return f"wrote report: {self.output_path}"


@dataclass(frozen=True)
class EffectiveReviewProfile:
    name: ReviewProfileName
    ai_enabled: bool
    tools_enabled: bool
    include_out_of_diff: bool
    provider: str | None
    fail_on: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ai_enabled": self.ai_enabled,
            "tools_enabled": self.tools_enabled,
            "include_out_of_diff": self.include_out_of_diff,
            "provider": self.provider,
            "fail_on": self.fail_on,
        }


@dataclass(frozen=True)
class ToolCollection:
    results: list[ToolResult]
    filter_result: ToolFilterResult | None


class ReviewPipeline:
    def __init__(
        self,
        options: ReviewPipelineOptions,
        *,
        git_client: GitClient | None = None,
    ) -> None:
        self.options = options
        self.git_client = git_client or GitClient.from_cwd()

    def run(self) -> ReviewPipelineResult:
        effective = self._resolve_profile(self.options)
        repo_info, config, parsed_diff = self._load_inputs()
        rule_findings = default_rule_engine(config).run(
            parsed_diff,
            repo_info=repo_info,
        )

        context: ContextBudgetManifest | None = None
        llm_result: StructuredReviewResult | None = None
        if effective.ai_enabled:
            context = self._build_context(parsed_diff, repo_info, config)
            pack = build_review_context_pack(
                repo_info=repo_info,
                config=config,
                parsed_diff=parsed_diff,
                rule_findings=rule_findings,
                context=context,
            )
            try:
                validate_context_pack_dict(pack.to_dict())
                llm_result = StructuredReviewer(
                    create_provider(effective.provider)
                ).review(pack)
            except (ConfigError, LLMProviderError) as exc:
                raise ReviewPipelineError(f"llm provider error: {exc}") from exc
            except LLMOutputError as exc:
                raise ReviewPipelineError(f"llm output error: {exc}") from exc
            except ValueError as exc:
                raise ReviewPipelineError(f"context pack error: {exc}") from exc

        tool_collection = ToolCollection(results=[], filter_result=None)
        if effective.tools_enabled:
            tool_collection = self._collect_tool_findings(
                repo_info,
                config,
                parsed_diff,
                include_out_of_diff=effective.include_out_of_diff,
            )

        merge_result = merge_findings(
            rule_findings=rule_findings,
            tool_findings=(
                list(tool_collection.filter_result.included_findings)
                if tool_collection.filter_result is not None
                else []
            ),
            llm_findings=(
                list(llm_result.evidence.findings)
                if llm_result is not None
                else []
            ),
        )
        report = ReviewReport(
            findings=list(merge_result.findings),
            repo_info=self._build_report_metadata(
                repo_info=repo_info,
                effective=effective,
                context=context,
                llm_result=llm_result,
                evidence=llm_result.evidence if llm_result is not None else None,
                tool_collection=tool_collection,
            ),
            config_source=config.source,
            merge_summary=merge_result.summary.to_dict(),
        )
        rendered_output = self._render_result(report)
        output_path = self._write_output(rendered_output)
        exit_code = 1 if should_fail_findings(report.findings, effective.fail_on) else 0

        return ReviewPipelineResult(
            report=report,
            rendered_output=rendered_output,
            output_path=output_path,
            exit_code=exit_code,
            debug_payload=self._debug_payload(
                rule_findings=rule_findings,
                tool_collection=tool_collection,
                llm_result=llm_result,
                report=report,
            )
            if self.options.debug_findings
            else None,
        )

    @staticmethod
    def _resolve_profile(options: ReviewPipelineOptions) -> EffectiveReviewProfile:
        if options.profile not in {"manual", "pre-commit", "pre-push"}:
            raise ReviewPipelineError(
                "review --profile must be one of: manual, pre-commit, pre-push"
            )
        if options.no_ai and options.provider:
            raise ReviewPipelineError(
                "review accepts either --no-ai or --provider, not both."
            )
        if options.include_out_of_diff and not (
            options.with_tools or options.profile == "pre-push"
        ):
            raise ReviewPipelineError(
                "review --include-out-of-diff requires --with-tools or --profile pre-push."
            )

        tools_enabled = options.with_tools or options.profile == "pre-push"
        provider = None if options.no_ai else options.provider
        return EffectiveReviewProfile(
            name=options.profile,
            ai_enabled=provider is not None,
            tools_enabled=tools_enabled,
            include_out_of_diff=options.include_out_of_diff,
            provider=provider,
            fail_on=options.fail_on,
        )

    def _load_inputs(self) -> tuple[RepoInfo, Any, ParsedDiff]:
        if not self.options.staged:
            raise ReviewPipelineError("review currently supports --staged.")
        try:
            repo_info = self.git_client.repo_info()
            config = load_project_config(repo_info.root)
            parsed_diff = DiffReader(self.git_client).staged_parsed_diff()
        except NotGitRepositoryError as exc:
            raise ReviewPipelineError(f"not a git repository: {exc}") from exc
        except GitError as exc:
            raise ReviewPipelineError(f"git error: {exc}") from exc
        except ConfigError as exc:
            raise ReviewPipelineError(f"config error: {exc}") from exc
        except DiffParseError as exc:
            raise ReviewPipelineError(f"diff parse error: {exc}") from exc

        if parsed_diff.is_empty:
            raise ReviewPipelineError("no staged changes", exit_code=1)
        return repo_info, config, parsed_diff

    def _build_context(self, parsed_diff, repo_info, config) -> ContextBudgetManifest:
        try:
            index = build_code_index(repo_info.root, config)
            candidates = select_context_candidates(parsed_diff, index)
            return apply_token_budget(
                candidates,
                parsed_diff,
                repo_info.root,
                self.options.max_context_tokens,
            )
        except ValueError as exc:
            raise ReviewPipelineError(f"context pack error: {exc}") from exc

    @staticmethod
    def _collect_tool_findings(
        repo_info: RepoInfo,
        config,
        parsed_diff: ParsedDiff,
        *,
        include_out_of_diff: bool,
    ) -> ToolCollection:
        tool_results: list[ToolResult] = []
        tool_filter_result: ToolFilterResult | None = None
        detection = detect_project(repo_info.root)
        registry = ToolRegistry(detection, config)
        try:
            semgrep_tool = registry.get(SEMGREP_TOOL_NAME)
        except KeyError:
            semgrep_tool = None
        if semgrep_tool is not None:
            semgrep_result = run_semgrep_tool(semgrep_tool, repo_info.root)
            tool_results.append(semgrep_result)
            if semgrep_result.status == "success":
                changed_lines = build_changed_line_map(parsed_diff)
                tool_filter_result = filter_tool_findings(
                    tool_results,
                    changed_lines,
                    include_out_of_diff=include_out_of_diff,
                )
        return ToolCollection(
            results=tool_results,
            filter_result=tool_filter_result,
        )

    @staticmethod
    def _build_report_metadata(
        *,
        repo_info: RepoInfo,
        effective: EffectiveReviewProfile,
        context: ContextBudgetManifest | None,
        llm_result: StructuredReviewResult | None,
        evidence: EvidenceGuardResult | None,
        tool_collection: ToolCollection,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "root": repo_info.root,
            "branch": repo_info.branch,
            "head": repo_info.head,
            "profile": effective.name,
            "pipeline": "local-staged",
            "ai_enabled": effective.ai_enabled,
            "tools_enabled": effective.tools_enabled,
            "include_out_of_diff": effective.include_out_of_diff,
            "fail_on": effective.fail_on,
            "tool_results": [item.to_dict() for item in tool_collection.results],
            "tool_filter": (
                tool_collection.filter_result.to_dict()
                if tool_collection.filter_result is not None
                else None
            ),
        }
        if context is not None:
            metadata["context"] = {
                "used": len(context.context_used),
                "omitted": len(context.context_omitted),
                "used_tokens": context.used_tokens,
                "max_context_tokens": context.max_context_tokens,
            }
        if llm_result is not None:
            metadata.update(
                {
                    "provider": llm_result.response.provider,
                    "model": llm_result.response.model,
                    "evidence_summary": (
                        evidence.summary if evidence is not None else None
                    ),
                    "dropped_llm_findings": (
                        [
                            decision.to_dict()
                            for decision in evidence.dropped_findings
                        ]
                        if evidence is not None
                        else []
                    ),
                }
            )
        return metadata

    def _render_result(self, report: ReviewReport) -> str:
        if self.options.debug_findings:
            return ""
        try:
            return write_report(report, self.options.output_format)
        except ValueError as exc:
            raise ReviewPipelineError(f"report error: {exc}") from exc

    def _write_output(self, rendered_output: str) -> Path | None:
        if not self.options.output:
            return None
        output_path = Path(self.options.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered_output + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _debug_payload(
        *,
        rule_findings: list[Finding],
        tool_collection: ToolCollection,
        llm_result: StructuredReviewResult | None,
        report: ReviewReport,
    ) -> dict[str, Any]:
        llm_findings = (
            list(llm_result.evidence.findings)
            if llm_result is not None
            else []
        )
        tool_findings = (
            list(tool_collection.filter_result.included_findings)
            if tool_collection.filter_result is not None
            else []
        )
        return {
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
        }
