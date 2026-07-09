from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from review_pilot.config import ReviewPilotConfig
from review_pilot.models import ContextBudgetManifest, ParsedDiff, RepoInfo
from review_pilot.report_models import Finding


CONTEXT_PACK_SCHEMA_VERSION = "review-pilot.context-pack.v1"


@dataclass(frozen=True)
class ReviewContextPack:
    schema_version: str
    repo_info: dict[str, Any]
    config: dict[str, Any]
    diff: dict[str, Any]
    rule_findings: tuple[Finding, ...]
    context: ContextBudgetManifest
    generated_by: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_info": self.repo_info,
            "config": self.config,
            "diff": self.diff,
            "rule_findings": [finding.to_dict() for finding in self.rule_findings],
            "context": self.context.to_dict(),
            "generated_by": self.generated_by,
        }


def build_review_context_pack(
    *,
    repo_info: RepoInfo,
    config: ReviewPilotConfig,
    parsed_diff: ParsedDiff,
    rule_findings: list[Finding],
    context: ContextBudgetManifest,
) -> ReviewContextPack:
    return ReviewContextPack(
        schema_version=CONTEXT_PACK_SCHEMA_VERSION,
        repo_info={
            "root": repo_info.root,
            "branch": repo_info.branch,
            "head": repo_info.head,
            "has_staged_changes": repo_info.has_staged_changes,
            "has_unstaged_changes": repo_info.has_unstaged_changes,
        },
        config=config.to_dict(),
        diff=parsed_diff.to_dict(),
        rule_findings=tuple(rule_findings),
        context=context,
        generated_by={
            "tool": "review-pilot",
            "command": "context-pack",
        },
    )


def validate_context_pack_dict(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "repo_info",
        "config",
        "diff",
        "rule_findings",
        "context",
        "generated_by",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"missing context pack fields: {missing}")
    if payload["schema_version"] != CONTEXT_PACK_SCHEMA_VERSION:
        raise ValueError(f"invalid context pack schema_version: {payload['schema_version']!r}")
    _require_dict(payload["repo_info"], "repo_info")
    _require_dict(payload["config"], "config")
    diff = _require_dict(payload["diff"], "diff")
    if not isinstance(diff.get("files"), list):
        raise ValueError("diff.files must be a list")
    if not isinstance(payload["rule_findings"], list):
        raise ValueError("rule_findings must be a list")
    context = _require_dict(payload["context"], "context")
    for field in ("max_context_tokens", "used_tokens", "remaining_tokens", "context_used", "context_omitted"):
        if field not in context:
            raise ValueError(f"context.{field} is required")
    if not isinstance(context["context_used"], list):
        raise ValueError("context.context_used must be a list")
    if not isinstance(context["context_omitted"], list):
        raise ValueError("context.context_omitted must be a list")
    _require_dict(payload["generated_by"], "generated_by")


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value
