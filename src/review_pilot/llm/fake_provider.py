from __future__ import annotations

import json
from dataclasses import dataclass

from review_pilot.context_pack import ReviewContextPack

from .base import LLMResponse


@dataclass(frozen=True)
class FakeProvider:
    name: str = "fake"
    model: str = "fake-review-model"

    def review(self, context_pack: ReviewContextPack) -> LLMResponse:
        file_path, line_no = _first_changed_location(context_pack)
        rule_message = _matching_rule_message(
            context_pack,
            file_path,
            line_no,
        ) or (
            "Fake provider found a deterministic review issue."
        )
        content = json.dumps(
            {
                "schema_version": "review-pilot.llm-findings.v1",
                "findings": [
                    {
                        "message": rule_message,
                        "file_path": file_path,
                        "line_no": line_no,
                        "severity": "P2",
                        "category": "maintainability",
                        "source": "llm",
                        "confidence": "medium",
                        "evidence": {
                            "reason": (
                                "Deterministic FakeProvider evidence derived "
                                "from the supplied Context Pack."
                            )
                        },
                        "suggestion": (
                            "Review the changed line and keep the final "
                            "implementation covered by tests."
                        ),
                    }
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return LLMResponse(
            provider=self.name,
            model=self.model,
            content=content,
        )


def _first_changed_location(
    context_pack: ReviewContextPack,
) -> tuple[str, int]:
    for diff_file in context_pack.diff.get("files", []):
        path = diff_file.get("path")
        if not path:
            continue
        for hunk in diff_file.get("hunks", []):
            for line in hunk.get("lines", []):
                line_no = line.get("new_line_no")
                if line.get("kind") == "added" and isinstance(line_no, int):
                    return str(path), line_no
        return str(path), 1
    return "unknown", 1


def _matching_rule_message(
    context_pack: ReviewContextPack,
    file_path: str,
    line_no: int,
) -> str | None:
    for finding in context_pack.rule_findings:
        if finding.file_path == file_path and finding.line_no == line_no:
            return finding.message
    return None
