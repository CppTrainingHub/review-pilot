from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import ConfigError, load_project_config
from .diff_parser import DiffParseError
from .diff_reader import DiffReader
from .git_client import GitClient, GitError, NotGitRepositoryError
from .models import ParsedDiff, RepoInfo
from .report_models import ReviewReport
from .report_summary import should_fail_findings
from .report_writer import write_report
from .review_engine import (
    ReviewEngine,
    ReviewEngineError,
    ReviewEngineOptions,
    ReviewInput,
)
from .tools.semgrep_tool import run_semgrep_tool


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
    strategy: str = "baseline"
    dynamic_context: bool = False
    snippet_location: bool = False
    reflection: bool = False


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
        try:
            engine_result = ReviewEngine(
                ReviewEngineOptions(
                    provider=effective.provider,
                    with_tools=effective.tools_enabled,
                    include_out_of_diff=effective.include_out_of_diff,
                    max_context_tokens=self.options.max_context_tokens,
                    tool_runner=run_semgrep_tool,
                    strategy=self.options.strategy,
                    dynamic_context=self.options.dynamic_context,
                    snippet_location=self.options.snippet_location,
                    reflection=self.options.reflection,
                )
            ).run(
                ReviewInput(
                    repo_info=repo_info,
                    config=config,
                    parsed_diff=parsed_diff,
                    input_source="local-staged",
                    metadata={
                        "profile": effective.name,
                        "pipeline": "local-staged",
                        "fail_on": effective.fail_on,
                    },
                )
            )
        except ReviewEngineError as exc:
            raise ReviewPipelineError(str(exc), exit_code=exc.exit_code) from exc

        report = engine_result.report
        rendered_output = self._render_result(report)
        output_path = self._write_output(rendered_output)
        exit_code = 1 if should_fail_findings(report.findings, effective.fail_on) else 0

        return ReviewPipelineResult(
            report=report,
            rendered_output=rendered_output,
            output_path=output_path,
            exit_code=exit_code,
            debug_payload=engine_result.debug_payload if self.options.debug_findings else None,
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

    def _render_result(self, report: ReviewReport) -> str:
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
