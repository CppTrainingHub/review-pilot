from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from review_pilot.code_index import build_code_index
from review_pilot.config import ReviewPilotConfig
from review_pilot.context_pack import build_review_context_pack
from review_pilot.context_selector import select_context_candidates
from review_pilot.context_tools import ContextToolExecutor
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.dynamic_context import (
    DynamicContextError,
    _strip_json_fence,
    run_dynamic_context,
)
from review_pilot.llm import LLMResponse, LLMToolCall, StructuredReviewer
from review_pilot.models import RawDiff, RepoInfo
from review_pilot.token_budget import apply_token_budget


def test_dynamic_context_runs_tool_then_final_findings(tmp_path: Path) -> None:
    pack = _context_pack(tmp_path)
    result = run_dynamic_context(
        context_pack=pack,
        reviewer=StructuredReviewer(ScriptedProvider(["tool", "final"])),
        executor=ContextToolExecutor(tmp_path, pack_diff(pack)),
    )

    assert result.rounds == 2
    assert result.tool_calls == 1
    assert result.trace[0].tool_name == "file_read"
    assert result.trace[0].status == "completed"
    assert result.review.evidence.summary["verified"] == 1


def test_dynamic_context_records_tool_error_and_continues(tmp_path: Path) -> None:
    pack = _context_pack(tmp_path)
    provider = ScriptedProvider(["bad-tool", "final"])

    result = run_dynamic_context(
        context_pack=pack,
        reviewer=StructuredReviewer(provider),
        executor=ContextToolExecutor(tmp_path, pack_diff(pack)),
    )

    assert result.trace[0].status == "error"
    assert result.trace[0].error_code == "path_out_of_workspace"
    assert result.review.evidence.summary["verified"] == 1


def test_dynamic_context_enforces_round_limit(tmp_path: Path) -> None:
    pack = _context_pack(tmp_path)

    result = run_dynamic_context(
        context_pack=pack,
        reviewer=StructuredReviewer(ScriptedProvider(["tool"] * 5)),
        executor=ContextToolExecutor(tmp_path, pack_diff(pack)),
        max_rounds=2,
    )

    assert result.truncated is True
    assert result.rounds == 2
    assert result.review.evidence.summary["verified"] == 1


def test_dynamic_context_enforces_token_limit(tmp_path: Path) -> None:
    pack = _context_pack(tmp_path)

    result = run_dynamic_context(
        context_pack=pack,
        reviewer=StructuredReviewer(
            ScriptedProvider(["tool"], usage_total=20)
        ),
        executor=ContextToolExecutor(tmp_path, pack_diff(pack)),
        max_tokens=10,
    )

    assert result.truncated is True
    assert result.review.evidence.summary["verified"] == 1


def test_dynamic_context_accepts_provider_json_fence_with_leading_text() -> None:
    payload = '{"schema_version":"review-pilot.llm-findings.v1","findings":[]}'

    assert _strip_json_fence(f"Here is the result:\n```json\n{payload}\n```") == payload


@dataclass
class ScriptedProvider:
    steps: list[str]
    name: str = "scripted"
    model: str = "scripted-model"
    usage_total: int = 0

    def review(self, context_pack):
        return self._final_response(context_pack)

    def review_with_tools(self, context_pack, messages, tools, *, max_tokens=None):
        del max_tokens
        if not tools:
            return self._final_response(context_pack)
        step = self.steps.pop(0) if self.steps else "final"
        if step == "tool":
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                tool_calls=(
                    LLMToolCall(
                        call_id=f"call-{len(messages)}",
                        name="file_read",
                        arguments=json.dumps(
                            {"path": "src/service.py", "start_line": 1, "end_line": 20}
                        ),
                    ),
                ),
                usage=(
                    {"prompt_tokens": self.usage_total, "completion_tokens": 0, "total_tokens": self.usage_total}
                    if self.usage_total
                    else None
                ),
            )
        if step == "bad-tool":
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                tool_calls=(
                    LLMToolCall(
                        call_id="bad-call",
                        name="file_read",
                        arguments=json.dumps({"path": "../outside.py"}),
                    ),
                ),
            )
        return self._final_response(context_pack)

    def _final_response(self, context_pack):
        return LLMResponse(
            provider=self.name,
            model=self.model,
            content=json.dumps(
                {
                    "schema_version": "review-pilot.llm-findings.v1",
                    "findings": [
                        {
                            "message": "The changed line needs a focused test.",
                            "file_path": "src/service.py",
                            "line_no": 1,
                            "severity": "P2",
                            "category": "maintainability",
                            "source": "llm",
                            "confidence": "medium",
                            "evidence": {"reason": "The file was read from the workspace."},
                            "suggestion": "Add a focused test.",
                        }
                    ],
                }
            ),
        )


def _context_pack(root: Path):
    (root / "src").mkdir()
    (root / "src" / "service.py").write_text("return 2\n", encoding="utf-8")
    parsed_diff = pack_diff_from_text()
    index = build_code_index(root)
    context = apply_token_budget(
        select_context_candidates(parsed_diff, index),
        parsed_diff,
        root,
        100,
    )
    return build_review_context_pack(
        repo_info=RepoInfo(
            root=str(root),
            branch="main",
            head="0" * 40,
            has_staged_changes=True,
            has_unstaged_changes=False,
        ),
        config=ReviewPilotConfig.default(),
        parsed_diff=parsed_diff,
        rule_findings=[],
        context=context,
    )


def pack_diff(pack):
    return parse_unified_diff(RawDiff(_raw_diff_text()))


def pack_diff_from_text():
    return pack_diff(None)


def _raw_diff_text() -> str:
    return "\n".join(
        [
            "diff --git a/src/service.py b/src/service.py",
            "--- a/src/service.py",
            "+++ b/src/service.py",
            "@@ -1 +1 @@",
            "-return 1",
            "+return 2",
        ]
    )
