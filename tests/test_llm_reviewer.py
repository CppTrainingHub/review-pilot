from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from review_pilot.code_index import build_code_index
from review_pilot.config import ReviewPilotConfig
from review_pilot.context_pack import build_review_context_pack
from review_pilot.context_selector import select_context_candidates
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.llm import (
    LLMOutputError,
    LLMRequestError,
    LLMResponse,
    StructuredReviewer,
)
from review_pilot.models import RawDiff, RepoInfo
from review_pilot.token_budget import apply_token_budget


def test_structured_reviewer_parses_provider_response(
    tmp_path: Path,
) -> None:
    reviewer = StructuredReviewer(
        StaticProvider(json.dumps(_valid_payload()))
    )

    result = reviewer.review(_context_pack(tmp_path))

    assert result.response.provider == "static"
    assert result.envelope.findings[0].source == "llm"
    assert result.evidence.summary["verified"] == 1
    assert result.evidence.summary["dropped"] == 0
    assert result.to_dict()["schema_version"] == (
        "review-pilot.llm-findings.v1"
    )
    assert (
        result.to_dict()["findings"][0]["evidence"]["verification"]["source"]
        == "diff_added_line"
    )


def test_structured_reviewer_preserves_provider_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(LLMRequestError, match="offline"):
        StructuredReviewer(FailingProvider()).review(
            _context_pack(tmp_path)
        )


def test_structured_reviewer_surfaces_output_error(
    tmp_path: Path,
) -> None:
    reviewer = StructuredReviewer(StaticProvider("not-json"))

    with pytest.raises(LLMOutputError, match="not valid JSON"):
        reviewer.review(_context_pack(tmp_path))


def test_structured_reviewer_drops_hallucinated_reference(
    tmp_path: Path,
) -> None:
    payload = _valid_payload()
    payload["findings"][0]["file_path"] = "missing.py"
    payload["findings"][0]["line_no"] = 99
    reviewer = StructuredReviewer(
        StaticProvider(json.dumps(payload))
    )

    result = reviewer.review(_context_pack(tmp_path))

    assert result.evidence.summary["dropped"] == 1
    assert result.to_dict()["findings"] == []
    assert (
        result.to_dict()["dropped_findings"][0]["finding"]["file_path"]
        == "missing.py"
    )


@dataclass(frozen=True)
class StaticProvider:
    content: str
    name: str = "static"
    model: str = "static-model"

    def review(self, context_pack):
        return LLMResponse(
            provider=self.name,
            model=self.model,
            content=self.content,
        )


@dataclass(frozen=True)
class FailingProvider:
    name: str = "failing"
    model: str = "failing-model"

    def review(self, context_pack):
        raise LLMRequestError("offline")


def _context_pack(root: Path):
    (root / "app.py").write_text("value = 1\n", encoding="utf-8")
    parsed_diff = parse_unified_diff(
        RawDiff(
            "\n".join(
                [
                    "diff --git a/app.py b/app.py",
                    "--- /dev/null",
                    "+++ b/app.py",
                    "@@ -0,0 +1 @@",
                    "+value = 1",
                ]
            )
        )
    )
    index = build_code_index(root)
    candidates = select_context_candidates(parsed_diff, index)
    context = apply_token_budget(
        candidates,
        parsed_diff,
        root,
        max_context_tokens=100,
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


def _valid_payload() -> dict:
    return {
        "schema_version": "review-pilot.llm-findings.v1",
        "findings": [
            {
                "message": "Review the changed assignment.",
                "file_path": "app.py",
                "line_no": 1,
                "severity": "P3",
                "category": "maintainability",
                "source": "llm",
                "confidence": "low",
                "evidence": {
                    "reason": "The assignment is the changed line.",
                },
                "suggestion": "Keep the assignment covered by tests.",
            }
        ],
    }
