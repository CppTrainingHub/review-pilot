from __future__ import annotations

import json
from dataclasses import dataclass

from review_pilot.context_pack import ReviewContextPack
from review_pilot.evidence_guard import EvidenceGuardResult
from review_pilot.report_models import Finding

from .base import LLMResponse, LLMToolCall


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

    def review_with_snippet_location(
        self,
        context_pack: ReviewContextPack,
    ) -> LLMResponse:
        file_path, line_no = _first_changed_location(context_pack)
        existing_code = _changed_line_content(context_pack, file_path, line_no)
        payload = json.loads(self.review(context_pack).content)
        payload["findings"][0]["existing_code"] = existing_code
        return LLMResponse(
            provider=self.name,
            model=self.model,
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def review_with_tools(
        self,
        context_pack: ReviewContextPack,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Provide a deterministic tool-call fixture for offline unit tests."""

        del max_tokens, tools
        if any(message.get("role") == "tool" for message in messages):
            return self.review(context_pack)
        file_path, _ = _first_changed_location(context_pack)
        return LLMResponse(
            provider=self.name,
            model=self.model,
            content="",
            tool_calls=(
                LLMToolCall(
                    call_id="fake-call-1",
                    name="file_read",
                    arguments=json.dumps(
                        {"path": file_path, "start_line": 1, "end_line": 20},
                        separators=(",", ":"),
                    ),
                ),
            ),
            finish_reason="tool_calls",
        )

    def reflect_finding(
        self,
        *,
        finding: Finding,
        context_pack: ReviewContextPack,
        evidence: EvidenceGuardResult,
    ) -> LLMResponse:
        del context_pack, evidence
        if finding.confidence == "low":
            decision = "drop"
            severity = None
            reason = "离线复核夹具将低置信度 finding 标记为噪声。"
        elif finding.severity == "P0":
            decision = "downgrade"
            severity = "P1"
            reason = "离线复核夹具将过高的严重级别下调一级。"
        else:
            decision = "keep"
            severity = None
            reason = "离线复核夹具确认 finding 的证据和范围足够。"
        return LLMResponse(
            provider=self.name,
            model=self.model,
            content=json.dumps(
                {
                    "schema_version": "review-pilot.reflection.v1",
                    "decision": decision,
                    "reason": reason,
                    "severity": severity,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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


def _changed_line_content(
    context_pack: ReviewContextPack,
    file_path: str,
    line_no: int,
) -> str:
    for diff_file in context_pack.diff.get("files", []):
        if diff_file.get("path") != file_path:
            continue
        for hunk in diff_file.get("hunks", []):
            for line in hunk.get("lines", []):
                if (
                    line.get("kind") == "added"
                    and line.get("new_line_no") == line_no
                ):
                    return str(line.get("content", ""))
    return ""
