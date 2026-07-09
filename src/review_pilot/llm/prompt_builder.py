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


def build_review_prompt(context_pack: ReviewContextPack) -> ReviewPrompt:
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
        _section("OUTPUT_CONTRACT", llm_output_contract()),
    ]
    system = (
        "You are a code review model inside review-pilot. "
        "Use only evidence present in the supplied sections. "
        "Do not invent files, line numbers, code, or repository behavior. "
        "Return exactly one plain JSON object that matches OUTPUT_CONTRACT. "
        "Do not use markdown fences, prose before or after JSON, or a Markdown report. "
        "Every finding source must be exactly 'llm'. "
        "Write user-facing finding text in Simplified Chinese: message, "
        "suggestion, and evidence.reason must be Chinese. Keep JSON field names, "
        "enum values, file paths, code identifiers, and schema_version unchanged. "
        "Only report issues that can be tied to a concrete added line in DIFF. "
        "Every line_no must be a positive integer from an added line; if you "
        "cannot identify such a line, do not emit that finding."
    )
    return ReviewPrompt(system=system, user="\n\n".join(sections))


def llm_output_contract() -> dict[str, Any]:
    return {
        "schema_version": "review-pilot.llm-findings.v1",
        "root_fields": ["schema_version", "findings"],
        "findings": {
            "type": "non-empty array",
            "item_fields": [
                "message",
                "file_path",
                "line_no",
                "severity",
                "category",
                "source",
                "confidence",
                "evidence",
                "suggestion",
            ],
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
            "line_no": "positive integer from an added line in DIFF",
            "evidence_fields": ["reason"],
        },
    }


def _section(name: str, value: Any) -> str:
    return f"## {name}\n{json.dumps(value, ensure_ascii=False, indent=2)}"
