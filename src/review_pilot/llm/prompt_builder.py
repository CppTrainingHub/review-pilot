from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from review_pilot.context_pack import ReviewContextPack


@dataclass(frozen=True)
class ReviewPrompt:
    system: str
    user: str

    def to_dict(self) -> dict[str, str]:
        return {
            "system": self.system,
            "user": self.user,
        }


def build_review_prompt(
    context_pack: ReviewContextPack,
    *,
    snippet_location: bool = False,
) -> ReviewPrompt:
    payload = context_pack.to_dict()
    repository = {
        "branch": payload["repo_info"].get("branch"),
        "head": payload["repo_info"].get("head"),
        "has_staged_changes": payload["repo_info"].get("has_staged_changes"),
    }
    context = payload["context"]
    sections = [
        _section("REPOSITORY", repository),
        _section("DIFF", payload["diff"]),
        _section("DETERMINISTIC_FINDINGS", payload["rule_findings"]),
        _section("TOOL_FINDINGS", payload.get("tool_findings", [])),
        _section("CONTEXT_USED", context["context_used"]),
        _section("CONTEXT_OMITTED", context["context_omitted"]),
        _section("OUTPUT_CONTRACT", llm_output_contract(snippet_location=snippet_location)),
    ]
    system = (
        "You are a code review model inside review-pilot. "
        "Use only evidence present in the supplied sections. "
        "Do not invent files, line numbers, code, or repository behavior. "
        "Return exactly one plain JSON object that matches OUTPUT_CONTRACT. "
        "Do not use markdown fences, prose before or after JSON, or a Markdown report. "
        "Every finding source must be exactly 'llm'. "
        + (
            "existing_code must copy the exact source snippet you are commenting on; "
            "prefer a snippet that includes the changed added line; "
            "the program will calculate the final line_no. "
            if snippet_location
            else
            "line_no must be a JSON integer copied from a supplied diff new_line_no; "
            "never output a decimal, token offset, character offset, or prompt position. "
        )
        + "Write every user-facing finding message, suggestion, and evidence.reason "
        "in natural Simplified Chinese. Keep JSON keys, enum values, schema_version, "
        "file paths, and code tokens unchanged."
    )
    return ReviewPrompt(system=system, user="\n\n".join(sections))


def build_dynamic_review_prompt(
    context_pack: ReviewContextPack,
    *,
    snippet_location: bool = False,
) -> ReviewPrompt:
    """Build the initial message for a bounded tool-call conversation."""

    prompt = build_review_prompt(
        context_pack,
        snippet_location=snippet_location,
    )
    system = (
        prompt.system
        + " You may request more evidence with one of the supplied read-only tools: "
        "file_read, code_search, file_read_diff, or file_find. "
        "Before final findings, make at least one small read-only lookup when a supplied "
        "tool can verify a changed file or its diff; then use the returned evidence. "
        "After the tool result, either request another bounded lookup or return the "
        "same final findings JSON contract. Never ask for shell commands, file writes, "
        "test execution, network access, or code changes."
    )
    return ReviewPrompt(system=system, user=prompt.user)


def llm_output_contract(*, snippet_location: bool = False) -> dict[str, Any]:
    item_fields = [
        "message",
        "file_path",
        "line_no",
        "severity",
        "category",
        "source",
        "confidence",
        "evidence",
        "suggestion",
    ]
    if snippet_location:
        item_fields.append("existing_code")
    return {
        "schema_version": "review-pilot.llm-findings.v1",
        "root_fields": ["schema_version", "findings"],
        "findings": {
            "type": "non-empty array",
            "item_fields": item_fields,
            "line_no": (
                "nullable legacy field; ignored when snippet_location is enabled"
                if snippet_location
                else "positive integer copied from diff new_line_no"
            ),
            "severity": ["P0", "P1", "P2", "P3"],
            "category": [
                "size",
                "test",
                "security",
                "style",
                "bug",
                "maintainability",
                "other",
            ],
            "source": "llm",
            "confidence": ["high", "medium", "low"],
            "evidence_fields": ["reason"],
        },
    }


def _section(name: str, value: Any) -> str:
    return f"## {name}\n{json.dumps(value, ensure_ascii=False, indent=2)}"
