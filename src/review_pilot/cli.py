from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from . import __version__
from .code_index import build_code_index
from .command_runner import CommandRunnerError, run_registered_tool
from .config import ConfigError, LLMConfig, load_project_config
from .context_pack import build_review_context_pack, validate_context_pack_dict
from .context_selector import select_context_candidates
from .diff_line_map import build_changed_line_map
from .diff_parser import DiffParseError
from .diff_reader import DiffReader
from .doctor import run_checks
from .evidence_guard import guard_llm_findings
from .finding_merger import merge_findings
from .git_providers import GitHubProvider, GitLabProvider, GitProviderError
from .github_action import GitHubActionError, run_github_action
from .gitlab_ci import GitLabCIError, run_gitlab_ci
from .git_client import GitClient, GitError, NotGitRepositoryError
from .hooks import (
    HookError,
    hook_statuses,
    install_hooks,
    selected_hooks,
    uninstall_hooks,
)
from .llm import (
    LLMOutputError,
    LLMProviderError,
    StructuredReviewer,
    build_review_prompt,
    create_provider,
    parse_llm_findings,
    supported_providers,
)
from .naive_llm import NaiveReviewError, run_naive_review
from .notifiers.feishu import FeishuNotifier, FeishuNotifierError, message_from_report
from .project_detector import detect_project
from .report_models import ReviewReport
from .review_pipeline import (
    ReviewPipeline,
    ReviewPipelineError,
    ReviewPipelineOptions,
)
from .report_summary import should_fail_findings
from .report_writer import write_report
from .rule_engine import default_rule_engine
from .token_budget import apply_token_budget
from .tool_filter import filter_tool_findings
from .tool_registry import ToolRegistry
from .tools.semgrep_tool import SEMGREP_TOOL_NAME, run_semgrep_tool
from .workspace import WorkspaceError, build_workspace_plan, prepare_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot",
        description="Code review agent for local diffs, CI, and PR workflows.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "doctor",
        help="Check whether the local environment can run review-pilot.",
    )
    repo_parser = subparsers.add_parser(
        "repo-info",
        help="Print Git repository metadata.",
    )
    repo_parser.add_argument(
        "--json",
        action="store_true",
        required=True,
        help="Print repository metadata as JSON.",
    )
    diff_parser = subparsers.add_parser(
        "diff",
        help="Read Git diffs for later review steps.",
    )
    diff_parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Read staged Git changes.",
    )
    diff_parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw unified diff.",
    )
    diff_parser.add_argument(
        "--json",
        action="store_true",
        help="Print parsed diff as JSON.",
    )
    config_parser = subparsers.add_parser(
        "config",
        help="Inspect and validate project configuration.",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_subparsers.add_parser(
        "show",
        help="Print the effective review-pilot configuration as JSON.",
    )
    config_subparsers.add_parser(
        "validate",
        help="Validate .review-pilot.toml and print the config source.",
    )
    context_parser = subparsers.add_parser(
        "context",
        help="Select repository context candidates for staged changes.",
    )
    context_parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Select context for staged Git changes.",
    )
    context_parser.add_argument(
        "--json",
        action="store_true",
        required=True,
        help="Print context candidates as JSON.",
    )
    context_parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="Apply a token budget and print selected context slices.",
    )
    context_pack_parser = subparsers.add_parser(
        "context-pack",
        help="Generate an auditable review context pack for staged changes.",
    )
    context_pack_parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Build a context pack for staged Git changes.",
    )
    context_pack_parser.add_argument(
        "--output",
        required=True,
        help="Write the context pack JSON to this path.",
    )
    context_pack_parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4000,
        help="Maximum estimated tokens for repository context slices.",
    )
    detect_parser = subparsers.add_parser(
        "detect-project",
        help="Detect project type markers used by the tool registry.",
    )
    detect_parser.add_argument(
        "--json",
        action="store_true",
        help="Print project detection as JSON.",
    )
    tools_parser = subparsers.add_parser(
        "tools",
        help="Inspect registered tools.",
    )
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command")
    tools_list_parser = tools_subparsers.add_parser(
        "list",
        help="Print the tool registry.",
    )
    tools_list_parser.add_argument(
        "--json",
        action="store_true",
        help="Print tools as JSON.",
    )
    run_tool_parser = subparsers.add_parser(
        "run-tool",
        help="Run a whitelisted registered tool.",
    )
    run_tool_parser.add_argument(
        "--name",
        required=True,
        help="Registered tool name.",
    )
    run_tool_parser.add_argument(
        "--json",
        action="store_true",
        help="Print command result as JSON.",
    )
    hooks_parser = subparsers.add_parser(
        "hooks",
        help="Install and inspect local Git hooks.",
    )
    hooks_subparsers = hooks_parser.add_subparsers(dest="hooks_command")
    hooks_install_parser = hooks_subparsers.add_parser(
        "install",
        help="Install review-pilot Git hooks.",
    )
    hooks_install_parser.add_argument(
        "--pre-commit",
        action="store_true",
        help="Install the pre-commit hook.",
    )
    hooks_install_parser.add_argument(
        "--pre-push",
        action="store_true",
        help="Install the pre-push hook.",
    )
    hooks_install_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing non-review-pilot hook.",
    )
    hooks_subparsers.add_parser(
        "status",
        help="Print review-pilot Git hook status.",
    )
    hooks_uninstall_parser = hooks_subparsers.add_parser(
        "uninstall",
        help="Remove review-pilot Git hooks.",
    )
    hooks_uninstall_parser.add_argument(
        "--pre-commit",
        action="store_true",
        help="Remove the pre-commit hook.",
    )
    hooks_uninstall_parser.add_argument(
        "--pre-push",
        action="store_true",
        help="Remove the pre-push hook.",
    )
    llm_parser = subparsers.add_parser(
        "llm",
        help="Inspect LLM provider configuration.",
    )
    llm_subparsers = llm_parser.add_subparsers(dest="llm_command")
    llm_subparsers.add_parser(
        "doctor",
        help="Print provider configuration status without exposing secrets.",
    )
    llm_validate_parser = llm_subparsers.add_parser(
        "validate-output",
        help="Validate a saved LLM findings JSON file.",
    )
    llm_validate_parser.add_argument(
        "--input",
        required=True,
        help="Path to a file containing raw model output.",
    )
    prompt_preview_parser = subparsers.add_parser(
        "prompt-preview",
        help="Preview the structured review prompt for staged changes.",
    )
    prompt_preview_parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Build the prompt from staged Git changes.",
    )
    prompt_preview_parser.add_argument(
        "--provider",
        choices=supported_providers(),
        default="fake",
        help="Provider metadata to show with the prompt preview.",
    )
    evidence_check_parser = subparsers.add_parser(
        "evidence-check",
        help="Validate saved LLM findings against staged review evidence.",
    )
    evidence_check_parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Build evidence from staged Git changes.",
    )
    evidence_check_parser.add_argument(
        "--input",
        required=True,
        help="Path to a file containing structured LLM findings JSON.",
    )
    review_parser = subparsers.add_parser(
        "review",
        help="Run the local code review workflow.",
    )
    review_parser.add_argument(
        "--staged",
        action="store_true",
        help="Review staged Git changes.",
    )
    review_parser.add_argument(
        "--base",
        metavar="REF",
        help="Review changes against a base ref.",
    )
    review_parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Run deterministic checks only, without calling an LLM.",
    )
    review_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Report output format for final reports.",
    )
    review_parser.add_argument(
        "--output",
        help="Write the final merged report to this path.",
    )
    review_parser.add_argument(
        "--debug-findings",
        action="store_true",
        help="Print normalized findings as JSON before report rendering.",
    )
    review_parser.add_argument(
        "--fail-on",
        choices=("P0", "P1", "P2", "P3"),
        help="Return exit code 1 when the highest finding severity meets this threshold.",
    )
    review_parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Run enabled external tools and include their findings.",
    )
    review_parser.add_argument(
        "--include-out-of-diff",
        action="store_true",
        help="Include external tool findings that are outside changed diff lines.",
    )
    review_parser.add_argument(
        "--provider",
        choices=supported_providers(),
        help="Run the formal LLM provider with an auditable context pack.",
    )
    review_parser.add_argument(
        "--profile",
        choices=("manual", "pre-commit", "pre-push"),
        default="manual",
        help="Run the review with a local workflow profile.",
    )
    review_pr_parser = subparsers.add_parser(
        "review-pr",
        help="Read a GitHub pull request and prepare a remote review workspace.",
    )
    review_pr_parser.add_argument(
        "url",
        help="GitHub pull request URL, for example https://github.com/OWNER/REPO/pull/123.",
    )
    review_pr_parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Print the PR metadata and workspace plan without cloning.",
    )
    review_pr_parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Keep this command on deterministic PR input preparation.",
    )
    github_action_parser = subparsers.add_parser(
        "github-action",
        help="Run review-pilot from a GitHub Actions pull_request event.",
    )
    github_action_parser.add_argument(
        "--event-path",
        required=True,
        help="Path to the GitHub pull_request event JSON.",
    )
    github_action_parser.add_argument(
        "--output-dir",
        default="review-pilot-artifacts",
        help="Directory for review-report.md and review-report.json artifacts.",
    )
    github_action_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare PR input and write artifacts without cloning a remote workspace.",
    )
    github_action_parser.add_argument(
        "--fail-on",
        choices=("P0", "P1", "P2", "P3"),
        help="Return exit code 1 when the artifact report reaches this severity.",
    )
    github_action_parser.add_argument(
        "--provider",
        choices=supported_providers(),
        help="Run the formal LLM provider against the pull request workspace.",
    )
    github_action_parser.add_argument(
        "--post-summary-comment",
        action="store_true",
        help="Publish or preview the PR summary comment.",
    )
    github_action_parser.add_argument(
        "--report-url",
        help="URL to the full report artifact shown in comments or notifications.",
    )
    gitlab_ci_parser = subparsers.add_parser(
        "gitlab-ci",
        help="Run review-pilot from a GitLab CI merge request pipeline.",
    )
    gitlab_ci_parser.add_argument(
        "--output-dir",
        default="review-pilot-artifacts",
        help="Directory for review-report.md and review-report.json artifacts.",
    )
    gitlab_ci_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare MR input and write artifacts without posting notes.",
    )
    gitlab_ci_parser.add_argument(
        "--fail-on",
        choices=("P0", "P1", "P2", "P3"),
        help="Return exit code 1 when the artifact report reaches this severity.",
    )
    gitlab_ci_parser.add_argument(
        "--provider",
        choices=supported_providers(),
        help="Run the formal LLM provider against the merge request workspace.",
    )
    gitlab_ci_parser.add_argument(
        "--post-summary-comment",
        action="store_true",
        help="Publish or preview the MR summary note.",
    )
    gitlab_ci_parser.add_argument(
        "--report-url",
        help="URL to the full report artifact shown in comments or notifications.",
    )
    notify_parser = subparsers.add_parser(
        "notify",
        help="Send review report notifications.",
    )
    notify_subparsers = notify_parser.add_subparsers(dest="notify_channel")
    feishu_parser = notify_subparsers.add_parser(
        "feishu",
        help="Send a review summary card to Feishu.",
    )
    feishu_parser.add_argument(
        "--report",
        required=True,
        help="Path to review-report.json.",
    )
    feishu_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the Feishu card payload without sending a webhook request.",
    )
    feishu_parser.add_argument(
        "--report-url",
        help="URL to the full report artifact shown on the Feishu card.",
    )
    naive_parser = subparsers.add_parser(
        "naive-review",
        help="Send staged raw diff directly to a naive LLM reviewer.",
    )
    naive_parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Review staged Git changes.",
    )
    naive_parser.add_argument(
        "--provider",
        choices=("fake", "openai"),
        default="fake",
        help="LLM provider to use. fake is deterministic and safe for tests.",
    )
    return parser


def build_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot review",
        description="Local review workflow entry.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Review staged Git changes.",
    )
    parser.add_argument(
        "--base",
        metavar="REF",
        help="Review changes against a base ref.",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Run deterministic checks only, without calling an LLM.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Report output format for final reports.",
    )
    parser.add_argument(
        "--output",
        help="Write the final merged report to this path.",
    )
    parser.add_argument(
        "--debug-findings",
        action="store_true",
        help="Print normalized findings as JSON before report rendering.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("P0", "P1", "P2", "P3"),
        help="Return exit code 1 when the highest finding severity meets this threshold.",
    )
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Run enabled external tools and include their findings.",
    )
    parser.add_argument(
        "--include-out-of-diff",
        action="store_true",
        help="Include external tool findings that are outside changed diff lines.",
    )
    parser.add_argument(
        "--provider",
        choices=supported_providers(),
        help="Run the formal LLM provider with an auditable context pack.",
    )
    parser.add_argument(
        "--profile",
        choices=("manual", "pre-commit", "pre-push"),
        default="manual",
        help="Run the review with a local workflow profile.",
    )
    return parser


def build_review_pr_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot review-pr",
        description="Read a GitHub pull request and prepare a remote review workspace.",
    )
    parser.add_argument(
        "url",
        help="GitHub pull request URL, for example https://github.com/OWNER/REPO/pull/123.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Print the PR metadata and workspace plan without cloning.",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Keep this command on deterministic PR input preparation.",
    )
    return parser


def build_github_action_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot github-action",
        description="Run review-pilot from a GitHub Actions pull_request event.",
    )
    parser.add_argument(
        "--event-path",
        required=True,
        help="Path to the GitHub pull_request event JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="review-pilot-artifacts",
        help="Directory for review-report.md and review-report.json artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare PR input and write artifacts without cloning a remote workspace.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("P0", "P1", "P2", "P3"),
        help="Return exit code 1 when the artifact report reaches this severity.",
    )
    parser.add_argument(
        "--provider",
        choices=supported_providers(),
        help="Run the formal LLM provider against the pull request workspace.",
    )
    parser.add_argument(
        "--post-summary-comment",
        action="store_true",
        help="Publish or preview the PR summary comment.",
    )
    parser.add_argument(
        "--report-url",
        help="URL to the full report artifact shown in comments or notifications.",
    )
    return parser


def build_gitlab_ci_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot gitlab-ci",
        description="Run review-pilot from a GitLab CI merge request pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default="review-pilot-artifacts",
        help="Directory for review-report.md and review-report.json artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare MR input and write artifacts without posting notes.",
    )
    parser.add_argument(
        "--fail-on",
        choices=("P0", "P1", "P2", "P3"),
        help="Return exit code 1 when the artifact report reaches this severity.",
    )
    parser.add_argument(
        "--provider",
        choices=supported_providers(),
        help="Run the formal LLM provider against the merge request workspace.",
    )
    parser.add_argument(
        "--post-summary-comment",
        action="store_true",
        help="Publish or preview the MR summary note.",
    )
    parser.add_argument(
        "--report-url",
        help="URL to the full report artifact shown in comments or notifications.",
    )
    return parser


def build_notify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot notify",
        description="Send review report notifications.",
    )
    subparsers = parser.add_subparsers(dest="notify_channel")
    feishu_parser = subparsers.add_parser(
        "feishu",
        help="Send a review summary card to Feishu.",
    )
    feishu_parser.add_argument(
        "--report",
        required=True,
        help="Path to review-report.json.",
    )
    feishu_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the Feishu card payload without sending a webhook request.",
    )
    feishu_parser.add_argument(
        "--report-url",
        help="URL to the full report artifact shown on the Feishu card.",
    )
    return parser


def build_llm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot llm",
        description="Inspect LLM provider configuration.",
    )
    subparsers = parser.add_subparsers(dest="llm_command")
    subparsers.add_parser(
        "doctor",
        help="Print provider configuration status without exposing secrets.",
    )
    validate_parser = subparsers.add_parser(
        "validate-output",
        help="Validate a saved LLM findings JSON file.",
    )
    validate_parser.add_argument(
        "--input",
        required=True,
        help="Path to a file containing raw model output.",
    )
    return parser


def build_prompt_preview_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot prompt-preview",
        description="Preview the structured review prompt for staged changes.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Build the prompt from staged Git changes.",
    )
    parser.add_argument(
        "--provider",
        choices=supported_providers(),
        default="fake",
        help="Provider metadata to show with the prompt preview.",
    )
    return parser


def build_evidence_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot evidence-check",
        description="Validate saved LLM findings against staged evidence.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Build evidence from staged Git changes.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a file containing structured LLM findings JSON.",
    )
    return parser


def build_context_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot context",
        description="Select repository context candidates for staged changes.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Select context for staged Git changes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        required=True,
        help="Print context candidates as JSON.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="Apply a token budget and print selected context slices.",
    )
    return parser


def build_context_pack_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot context-pack",
        description="Generate an auditable review context pack for staged changes.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        required=True,
        help="Build a context pack for staged Git changes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Write the context pack JSON to this path.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4000,
        help="Maximum estimated tokens for repository context slices.",
    )
    return parser


def build_hooks_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-pilot hooks",
        description="Install and inspect local Git hooks.",
    )
    subparsers = parser.add_subparsers(dest="hooks_command")
    install_parser = subparsers.add_parser("install", help="Install review-pilot Git hooks.")
    install_parser.add_argument("--pre-commit", action="store_true", help="Install the pre-commit hook.")
    install_parser.add_argument("--pre-push", action="store_true", help="Install the pre-push hook.")
    install_parser.add_argument("--force", action="store_true", help="Replace an existing non-review-pilot hook.")
    subparsers.add_parser("status", help="Print review-pilot Git hook status.")
    uninstall_parser = subparsers.add_parser("uninstall", help="Remove review-pilot Git hooks.")
    uninstall_parser.add_argument("--pre-commit", action="store_true", help="Remove the pre-commit hook.")
    uninstall_parser.add_argument("--pre-push", action="store_true", help="Remove the pre-push hook.")
    return parser


def _run_doctor(stdout: TextIO, stderr: TextIO) -> int:
    results = run_checks()
    for result in results:
        status = "OK" if result.ok else "FAIL"
        stream = stdout if result.ok else stderr
        print(f"[{status}] {result.name}: {result.message}", file=stream)
    return 0 if all(result.ok for result in results) else 1


def _run_repo_info(stdout: TextIO, stderr: TextIO) -> int:
    try:
        info = GitClient.from_cwd().repo_info()
    except NotGitRepositoryError as exc:
        print(f"not a git repository: {exc}", file=stderr)
        return 2
    except GitError as exc:
        print(f"git error: {exc}", file=stderr)
        return 2

    print(
        json.dumps(
            {
                "root": info.root,
                "branch": info.branch,
                "head": info.head,
                "has_staged_changes": info.has_staged_changes,
                "has_unstaged_changes": info.has_unstaged_changes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=stdout,
    )
    return 0


def _run_diff(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        reader = DiffReader(GitClient.from_cwd())
        raw_diff = reader.staged_raw_diff()
    except NotGitRepositoryError as exc:
        print(f"not a git repository: {exc}", file=stderr)
        return 2
    except GitError as exc:
        print(f"git error: {exc}", file=stderr)
        return 2

    if raw_diff.is_empty:
        print("no staged changes", file=stderr)
        return 1

    if args.json:
        try:
            parsed_diff = reader.staged_parsed_diff()
        except DiffParseError as exc:
            print(f"diff parse error: {exc}", file=stderr)
            return 2
        print(json.dumps(parsed_diff.to_dict(), ensure_ascii=False, indent=2), file=stdout)
        return 0

    print(raw_diff.text, end="" if raw_diff.text.endswith("\n") else "\n", file=stdout)
    return 0


def _load_config_from_cwd(stderr: TextIO):
    try:
        repo_info = GitClient.from_cwd().repo_info()
        config = load_project_config(repo_info.root)
    except NotGitRepositoryError as exc:
        print(f"not a git repository: {exc}", file=stderr)
        return None, None, 2
    except GitError as exc:
        print(f"git error: {exc}", file=stderr)
        return None, None, 2
    except ConfigError as exc:
        print(f"config error: {exc}", file=stderr)
        return None, None, 2
    return repo_info, config, 0


def _run_config(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    if args.config_command not in {"show", "validate"}:
        print("config requires a subcommand: show or validate", file=stderr)
        return 2

    _, config, exit_code = _load_config_from_cwd(stderr)
    if config is None:
        return exit_code

    if args.config_command == "show":
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), file=stdout)
        return 0

    print(f"config ok: {config.source}", file=stdout)
    return 0


def _run_review(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    if not (args.staged and not args.base):
        print("review currently supports --staged.", file=stderr)
        return 2

    if args.no_ai and args.provider:
        print("review accepts either --no-ai or --provider, not both.", file=stderr)
        return 2

    if _should_emit_raw_provider_result(args):
        pack, exit_code = _build_context_pack_from_cwd(stderr, max_context_tokens=4000)
        if pack is None:
            return exit_code
        try:
            result = StructuredReviewer(
                create_provider(args.provider)
            ).review(pack)
        except (ConfigError, LLMProviderError) as exc:
            print(f"llm provider error: {exc}", file=stderr)
            return 2
        except LLMOutputError as exc:
            print(f"llm output error: {exc}", file=stderr)
            return 2
        if (
            not args.output
            and not args.with_tools
            and args.format == "markdown"
            and not args.fail_on
        ):
            print(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                file=stdout,
            )
            return 0

    try:
        result = ReviewPipeline(
            ReviewPipelineOptions(
                staged=args.staged,
                no_ai=args.no_ai,
                with_tools=args.with_tools,
                include_out_of_diff=args.include_out_of_diff,
                provider=args.provider,
                profile=args.profile,
                output_format=args.format,
                output=args.output,
                debug_findings=args.debug_findings,
                fail_on=args.fail_on,
            )
        ).run()
    except ReviewPipelineError as exc:
        print(str(exc), file=stderr)
        return exc.exit_code

    if result.debug_payload is not None:
        print(
            json.dumps(result.debug_payload, ensure_ascii=False, indent=2),
            file=stdout,
        )
        return result.exit_code
    print(result.message, file=stdout)
    return result.exit_code


def _run_review_pr(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    if not args.dry_run:
        print("review-pr currently requires --dry-run.", file=stderr)
        return 2

    if not args.no_ai:
        print("review-pr currently prepares PR input; pass --no-ai.", file=stderr)
        return 2

    try:
        pr_info = GitHubProvider().fetch_pull_request(args.url)
        plan = prepare_workspace(
            build_workspace_plan(pr_info, dry_run=True)
        )
        payload = {
            "mode": "github-pr-dry-run",
            "pull_request": pr_info.to_dict(),
            "workspace": plan.to_dict(),
        }
    except GitProviderError as exc:
        print(f"github error: {exc}", file=stderr)
        return 2
    except WorkspaceError as exc:
        print(f"workspace error: {exc}", file=stderr)
        return 2

    print(json.dumps(payload, ensure_ascii=False, indent=2), file=stdout)
    return 0


def _run_github_action(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        result = run_github_action(
            event_path=args.event_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            fail_on=args.fail_on,
            provider=GitHubProvider(),
            llm_provider=args.provider,
            post_summary_comment=args.post_summary_comment,
            report_url=args.report_url,
        )
    except GitHubActionError as exc:
        print(f"github action error: {exc}", file=stderr)
        return 2

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), file=stdout)
    return result.exit_code


def _run_gitlab_ci(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        result = run_gitlab_ci(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            fail_on=args.fail_on,
            provider=GitLabProvider(),
            llm_provider=args.provider,
            post_summary_comment=args.post_summary_comment,
            report_url=args.report_url,
        )
    except GitLabCIError as exc:
        print(f"gitlab ci error: {exc}", file=stderr)
        return 2

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), file=stdout)
    return result.exit_code


def _run_notify(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    if args.notify_channel != "feishu":
        print("notify requires a channel: feishu", file=stderr)
        return 2

    report_path = Path(args.report)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("report JSON must contain an object")
        report = ReviewReport.from_dict(payload)
    except OSError as exc:
        print(f"notify error: cannot read report: {exc}", file=stderr)
        return 2
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"notify error: invalid report JSON: {exc}", file=stderr)
        return 2

    try:
        result = FeishuNotifier().notify(
            message_from_report(report, report_url=args.report_url),
            dry_run=args.dry_run,
        )
    except FeishuNotifierError as exc:
        print(f"notify error: {exc}", file=stderr)
        return 2

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), file=stdout)
    return 0


def _should_emit_raw_provider_result(args: argparse.Namespace) -> bool:
    return (
        bool(args.provider)
        and not args.output
        and not args.with_tools
        and not args.include_out_of_diff
        and args.format == "markdown"
        and not args.fail_on
        and not args.debug_findings
        and args.profile == "manual"
    )


def _collect_tool_findings(
    repo_info,
    config,
    parsed_diff,
    *,
    include_out_of_diff: bool,
):
    tool_results = []
    tool_filter_result = None
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
    return tool_results, tool_filter_result


def _emit_report(
    report: ReviewReport,
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    try:
        output = write_report(report, args.format)
    except ValueError as exc:
        print(f"report error: {exc}", file=stderr)
        return 2

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"wrote report: {output_path}", file=stdout)
        return 0

    print(output, file=stdout)
    return 0


def _run_context(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    repo_info, config, config_exit_code = _load_config_from_cwd(stderr)
    if config is None or repo_info is None:
        return config_exit_code

    try:
        reader = DiffReader(GitClient.from_cwd())
        parsed_diff = reader.staged_parsed_diff()
    except GitError as exc:
        print(f"git error: {exc}", file=stderr)
        return 2
    except DiffParseError as exc:
        print(f"diff parse error: {exc}", file=stderr)
        return 2

    if parsed_diff.is_empty:
        print("no staged changes", file=stderr)
        return 1

    index = build_code_index(repo_info.root, config)
    manifest = select_context_candidates(parsed_diff, index)
    if args.max_context_tokens is not None:
        try:
            manifest_with_budget = apply_token_budget(
                manifest,
                parsed_diff,
                repo_info.root,
                args.max_context_tokens,
            )
        except ValueError as exc:
            print(f"context budget error: {exc}", file=stderr)
            return 2
        print(json.dumps(manifest_with_budget.to_dict(), ensure_ascii=False, indent=2), file=stdout)
        return 0

    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), file=stdout)
    return 0


def _run_context_pack(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    pack, exit_code = _build_context_pack_from_cwd(
        stderr,
        max_context_tokens=args.max_context_tokens,
    )
    if pack is None:
        return exit_code

    payload = pack.to_dict()
    repo_root = pack.repo_info["root"]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(str(repo_root)) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote context pack: {output_path} "
        f"(findings={len(pack.rule_findings)}, context_slices={len(pack.context.context_used)}, "
        f"omitted={len(pack.context.context_omitted)})",
        file=stdout,
    )
    return 0


def _build_context_pack_from_cwd(
    stderr: TextIO,
    *,
    max_context_tokens: int,
):
    repo_info, config, config_exit_code = _load_config_from_cwd(stderr)
    if config is None or repo_info is None:
        return None, config_exit_code

    try:
        reader = DiffReader(GitClient.from_cwd())
        parsed_diff = reader.staged_parsed_diff()
    except GitError as exc:
        print(f"git error: {exc}", file=stderr)
        return None, 2
    except DiffParseError as exc:
        print(f"diff parse error: {exc}", file=stderr)
        return None, 2

    if parsed_diff.is_empty:
        print("no staged changes", file=stderr)
        return None, 1

    try:
        index = build_code_index(repo_info.root, config)
        candidates = select_context_candidates(parsed_diff, index)
        context = apply_token_budget(
            candidates,
            parsed_diff,
            repo_info.root,
            max_context_tokens,
        )
    except ValueError as exc:
        print(f"context pack error: {exc}", file=stderr)
        return None, 2

    findings = default_rule_engine(config).run(parsed_diff, repo_info=repo_info)
    pack = build_review_context_pack(
        repo_info=repo_info,
        config=config,
        parsed_diff=parsed_diff,
        rule_findings=findings,
        context=context,
    )
    payload = pack.to_dict()
    try:
        validate_context_pack_dict(payload)
    except ValueError as exc:
        print(f"context pack error: {exc}", file=stderr)
        return None, 2
    return pack, 0


def _run_llm(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    if args.llm_command not in {"doctor", "validate-output"}:
        print(
            "llm requires a subcommand: doctor or validate-output",
            file=stderr,
        )
        return 2
    if args.llm_command == "validate-output":
        input_path = Path(args.input)
        try:
            content = input_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"llm output error: cannot read {input_path}: {exc}", file=stderr)
            return 2
        try:
            envelope = parse_llm_findings(content)
        except LLMOutputError as exc:
            print(f"llm output error: {exc}", file=stderr)
            return 2
        print(
            f"valid llm output: {input_path} "
            f"(findings={len(envelope.findings)})",
            file=stdout,
        )
        return 0
    try:
        config = LLMConfig.from_env()
        if config.provider not in supported_providers():
            raise ConfigError(
                f"unsupported LLM provider: {config.provider}; "
                f"expected one of {supported_providers()}"
            )
    except ConfigError as exc:
        print(f"llm config error: {exc}", file=stderr)
        return 2
    for key, value in config.status_dict().items():
        print(f"{key}: {value}", file=stdout)
    return 0


def _run_prompt_preview(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    pack, exit_code = _build_context_pack_from_cwd(
        stderr,
        max_context_tokens=4000,
    )
    if pack is None:
        return exit_code
    try:
        provider = create_provider(args.provider)
    except (ConfigError, LLMProviderError) as exc:
        print(f"llm provider error: {exc}", file=stderr)
        return 2
    prompt = build_review_prompt(pack)
    print(f"PROVIDER\n{provider.name} / {provider.model}", file=stdout)
    print(f"\nSYSTEM\n{prompt.system}", file=stdout)
    print(f"\nUSER\n{prompt.user}", file=stdout)
    return 0


def _run_evidence_check(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    pack, exit_code = _build_context_pack_from_cwd(
        stderr,
        max_context_tokens=4000,
    )
    if pack is None:
        return exit_code

    input_path = Path(args.input)
    try:
        content = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"llm output error: cannot read {input_path}: {exc}",
            file=stderr,
        )
        return 2
    try:
        envelope = parse_llm_findings(content)
    except LLMOutputError as exc:
        print(f"llm output error: {exc}", file=stderr)
        return 2

    evidence = guard_llm_findings(envelope.findings, pack)
    print(
        json.dumps(
            {
                "schema_version": envelope.schema_version,
                **evidence.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=stdout,
    )
    return 1 if evidence.dropped_findings else 0


def _registry_from_cwd(stderr: TextIO):
    repo_info, config, exit_code = _load_config_from_cwd(stderr)
    if repo_info is None or config is None:
        return None, None, None, exit_code
    detection = detect_project(repo_info.root)
    return repo_info, detection, ToolRegistry(detection, config), 0


def _run_detect_project(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    repo_info, _, exit_code = _load_config_from_cwd(stderr)
    if repo_info is None:
        return exit_code
    detection = detect_project(repo_info.root)
    if args.json:
        print(json.dumps(detection.to_dict(), ensure_ascii=False, indent=2), file=stdout)
        return 0
    project_types = ", ".join(detection.project_types) if detection.project_types else "unknown"
    markers = ", ".join(detection.markers) if detection.markers else "none"
    print(f"project types: {project_types}", file=stdout)
    print(f"markers: {markers}", file=stdout)
    return 0


def _run_tools(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    if args.tools_command != "list":
        print("tools requires a subcommand: list", file=stderr)
        return 2

    _, detection, registry, exit_code = _registry_from_cwd(stderr)
    if registry is None or detection is None:
        return exit_code

    tools = registry.list_tools()
    if args.json:
        print(
            json.dumps(
                {
                    "project": detection.to_dict(),
                    "tools": [tool.to_dict() for tool in tools],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=stdout,
        )
        return 0

    for tool in tools:
        status = "enabled" if tool.enabled else "disabled"
        print(f"{tool.spec.name}\t{status}\t{tool.reason}", file=stdout)
    return 0


def _run_tool(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    repo_info, _, registry, exit_code = _registry_from_cwd(stderr)
    if registry is None or repo_info is None:
        return exit_code

    try:
        tool = registry.get(args.name)
    except KeyError as exc:
        print(str(exc).strip("'"), file=stderr)
        return 2

    if args.name == SEMGREP_TOOL_NAME:
        semgrep_result = run_semgrep_tool(tool, repo_info.root)
        if args.json:
            print(json.dumps(semgrep_result.to_dict(), ensure_ascii=False, indent=2), file=stdout)
        else:
            print(f"tool: {semgrep_result.tool_name}", file=stdout)
            print(f"status: {semgrep_result.status}", file=stdout)
            print(f"findings: {len(semgrep_result.findings)}", file=stdout)
            if semgrep_result.error:
                print(f"error: {semgrep_result.error}", file=stdout)
        return 0

    try:
        result = run_registered_tool(tool, repo_info.root)
    except (CommandRunnerError, ValueError) as exc:
        print(f"tool run error: {exc}", file=stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), file=stdout)
    else:
        print(f"tool: {result.tool_name}", file=stdout)
        print(f"exit_code: {result.exit_code}", file=stdout)
        print(f"timed_out: {str(result.timed_out).lower()}", file=stdout)
        print(f"stdout: {result.stdout_path}", file=stdout)
        print(f"stderr: {result.stderr_path}", file=stdout)
    return result.exit_code


def _run_hooks(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    if args.hooks_command not in {"install", "status", "uninstall"}:
        print("hooks requires a subcommand: install, status, or uninstall", file=stderr)
        return 2

    try:
        repo_info = GitClient.from_cwd().repo_info()
    except NotGitRepositoryError as exc:
        print(f"not a git repository: {exc}", file=stderr)
        return 2
    except GitError as exc:
        print(f"git error: {exc}", file=stderr)
        return 2

    try:
        if args.hooks_command == "status":
            for status in hook_statuses(repo_info.root):
                print(status.format_line(), file=stdout)
            return 0

        hooks = selected_hooks(
            pre_commit=args.pre_commit,
            pre_push=args.pre_push,
        )
        if args.hooks_command == "install":
            changes = install_hooks(repo_info.root, hooks, force=args.force)
        else:
            changes = uninstall_hooks(repo_info.root, hooks)
    except HookError as exc:
        print(f"hook error: {exc}", file=stderr)
        return 2

    for change in changes:
        print(change.format_line(), file=stdout)
    return 0


def _read_staged_raw_diff(stderr: TextIO):
    try:
        raw_diff = DiffReader(GitClient.from_cwd()).staged_raw_diff()
    except NotGitRepositoryError as exc:
        print(f"not a git repository: {exc}", file=stderr)
        return None, 2
    except GitError as exc:
        print(f"git error: {exc}", file=stderr)
        return None, 2

    if raw_diff.is_empty:
        print("no staged changes", file=stderr)
        return None, 1
    return raw_diff, 0


def _run_naive_review(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    raw_diff, exit_code = _read_staged_raw_diff(stderr)
    if raw_diff is None:
        return exit_code
    try:
        review_text = run_naive_review(raw_diff, provider=args.provider)
    except NaiveReviewError as exc:
        print(f"naive review failed: {exc}", file=stderr)
        return 2
    print(review_text, file=stdout)
    return 0


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = build_parser()
    command_args = list(argv) if argv is not None else sys.argv[1:]

    if not command_args or command_args == ["--help"] or command_args == ["-h"]:
        parser.print_help(out)
        return 0

    if command_args == ["--version"]:
        print(__version__, file=out)
        return 0

    if command_args in (["review"], ["review", "--help"], ["review", "-h"]):
        build_review_parser().print_help(out)
        return 0

    if command_args in (["review-pr"], ["review-pr", "--help"], ["review-pr", "-h"]):
        build_review_pr_parser().print_help(out)
        return 0

    if command_args in (
        ["github-action"],
        ["github-action", "--help"],
        ["github-action", "-h"],
    ):
        build_github_action_parser().print_help(out)
        return 0

    if command_args in (
        ["gitlab-ci"],
        ["gitlab-ci", "--help"],
        ["gitlab-ci", "-h"],
    ):
        build_gitlab_ci_parser().print_help(out)
        return 0

    if command_args in (["notify"], ["notify", "--help"], ["notify", "-h"]):
        build_notify_parser().print_help(out)
        return 0

    if command_args in (["context"], ["context", "--help"], ["context", "-h"]):
        build_context_parser().print_help(out)
        return 0

    if command_args in (["context-pack"], ["context-pack", "--help"], ["context-pack", "-h"]):
        build_context_pack_parser().print_help(out)
        return 0

    if command_args in (["hooks"], ["hooks", "--help"], ["hooks", "-h"]):
        build_hooks_parser().print_help(out)
        return 0

    if command_args in (["llm"], ["llm", "--help"], ["llm", "-h"]):
        build_llm_parser().print_help(out)
        return 0

    if command_args in (
        ["prompt-preview"],
        ["prompt-preview", "--help"],
        ["prompt-preview", "-h"],
    ):
        build_prompt_preview_parser().print_help(out)
        return 0

    if command_args in (
        ["evidence-check"],
        ["evidence-check", "--help"],
        ["evidence-check", "-h"],
    ):
        build_evidence_check_parser().print_help(out)
        return 0

    args = parser.parse_args(command_args)

    if args.command == "doctor":
        return _run_doctor(out, err)

    if args.command == "repo-info":
        return _run_repo_info(out, err)

    if args.command == "diff":
        if args.raw == args.json:
            print("diff requires exactly one output format: --raw or --json", file=err)
            return 2
        return _run_diff(args, out, err)

    if args.command == "config":
        return _run_config(args, out, err)

    if args.command == "context":
        return _run_context(args, out, err)

    if args.command == "context-pack":
        return _run_context_pack(args, out, err)

    if args.command == "detect-project":
        return _run_detect_project(args, out, err)

    if args.command == "tools":
        return _run_tools(args, out, err)

    if args.command == "run-tool":
        return _run_tool(args, out, err)

    if args.command == "hooks":
        return _run_hooks(args, out, err)

    if args.command == "llm":
        return _run_llm(args, out, err)

    if args.command == "prompt-preview":
        return _run_prompt_preview(args, out, err)

    if args.command == "evidence-check":
        return _run_evidence_check(args, out, err)

    if args.command == "review":
        return _run_review(args, out, err)

    if args.command == "review-pr":
        return _run_review_pr(args, out, err)

    if args.command == "github-action":
        return _run_github_action(args, out, err)

    if args.command == "gitlab-ci":
        return _run_gitlab_ci(args, out, err)

    if args.command == "notify":
        return _run_notify(args, out, err)

    if args.command == "naive-review":
        return _run_naive_review(args, out, err)

    parser.print_help(out)
    return 0
