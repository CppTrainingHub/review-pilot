from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_pilot.config import ReflectionConfig, ReviewPilotConfig
from review_pilot.context_pack import build_review_context_pack
from review_pilot.evidence_guard import guard_llm_findings
from review_pilot.llm import LLMResponse
from review_pilot.models import DiffFile, DiffHunk, DiffLine, ParsedDiff, RepoInfo
from review_pilot.reflection import (
    ReflectionFilter,
    ReflectionOutputError,
    build_reflection_prompt,
    parse_reflection_decision,
)
from review_pilot.report_models import Finding
from review_pilot.review_engine import ReviewEngine, ReviewEngineOptions, ReviewInput
from review_pilot.token_budget import apply_token_budget


class DecisionProvider:
    name = "test-provider"
    model = "test-model"

    def __init__(self, decisions: list[dict[str, object]], *, usage: int = 0) -> None:
        self.decisions = list(decisions)
        self.usage = usage
        self.calls = 0

    def reflect_finding(self, *, finding, context_pack, evidence):
        del finding, context_pack, evidence
        self.calls += 1
        return LLMResponse(
            provider=self.name,
            model=self.model,
            content=json.dumps(self.decisions.pop(0), ensure_ascii=False),
            usage={
                "prompt_tokens": self.usage,
                "completion_tokens": self.usage,
                "total_tokens": self.usage,
            },
        )


def test_parse_reflection_decision_requires_strict_schema() -> None:
    decision = parse_reflection_decision(
        json.dumps(
            {
                "schema_version": "review-pilot.reflection.v1",
                "decision": "downgrade",
                "reason": "证据支持问题存在，但影响范围被高估。",
                "severity": "P2",
            },
            ensure_ascii=False,
        )
    )

    assert decision.action == "downgrade"
    assert decision.severity == "P2"
    with pytest.raises(ReflectionOutputError, match="exactly"):
        parse_reflection_decision(
            '{"schema_version":"review-pilot.reflection.v1",'
            '"decision":"keep","reason":"ok","severity":null,"extra":1}'
        )


def test_reflection_filter_maps_keep_downgrade_and_drop(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    findings = [
        make_finding("P1", "medium"),
        make_finding("P1", "medium", message="范围过大的建议。"),
        make_finding("P2", "low", message="没有证据的泛化建议。"),
    ]
    provider = DecisionProvider(
        [
            decision("keep", "证据和影响范围一致。"),
            decision("downgrade", "问题存在，但严重级别需要下调。", severity="P2"),
            decision("drop", "这条建议没有足够的代码证据。"),
        ]
    )
    result = ReflectionFilter(
        provider,
        ReflectionConfig(review_all=True),
    ).apply(
        findings,
        context_pack=pack,
        evidence=guard_llm_findings(tuple(findings), pack),
    )

    assert provider.calls == 3
    assert [item.action for item in result.decisions] == ["keep", "downgrade", "drop"]
    assert [item.severity for item in result.findings] == ["P1", "P2"]
    assert result.summary["drop"] == 1
    assert result.summary["downgrade"] == 1


def test_reflection_failure_keeps_original_finding(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    finding = make_finding("P2", "low")
    provider = DecisionProvider([{"bad": "payload"}])

    result = ReflectionFilter(
        provider,
        ReflectionConfig(review_all=True),
    ).apply(
        [finding],
        context_pack=pack,
        evidence=guard_llm_findings((finding,), pack),
    )

    assert result.findings == (finding,)
    assert result.decisions[0].action == "error"
    assert result.decisions[0].kept_original is True
    assert result.decisions[0].to_dict()["reflection_error"]
    assert result.summary["errors"] == 1


def test_reflection_respects_token_limit_and_source_policy(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    eligible = make_finding("P2", "low")
    rule_finding = Finding(
        message="deterministic rule",
        file_path="src/app.py",
        line_no=1,
        severity="P2",
        category="bug",
        source="rule",
    )
    provider = DecisionProvider(
        [decision("keep", "证据足够。")],
        usage=10,
    )
    result = ReflectionFilter(
        provider,
        ReflectionConfig(review_all=False, max_tokens=10),
    ).apply(
        [eligible, make_finding("P2", "low", message="第二条"), rule_finding],
        context_pack=pack,
        evidence=guard_llm_findings((eligible,), pack),
    )

    assert provider.calls == 1
    assert [item.action for item in result.decisions] == ["keep", "skip", "skip"]
    assert len(result.findings) == 3


def test_reflection_prompt_contains_only_the_target_file_and_contract(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    finding = make_finding("P2", "low")
    system, user = build_reflection_prompt(
        finding,
        pack,
        guard_llm_findings((finding,), pack),
    )

    assert "Never invent a new finding" in system
    assert "REFLECTION_INPUT" in user
    assert "OUTPUT_CONTRACT" in user
    assert "src/app.py" in user


def test_review_engine_records_reflection_metadata(tmp_path: Path) -> None:
    review_input = make_input(tmp_path)
    review_input = ReviewInput(
        repo_info=review_input.repo_info,
        config=ReviewPilotConfig(
            reflection=ReflectionConfig(enabled=True, review_all=True)
        ),
        parsed_diff=review_input.parsed_diff,
        input_source=review_input.input_source,
    )

    result = ReviewEngine(ReviewEngineOptions(provider="fake")).run(review_input)

    assert result.report.repo_info["reflection"]["enabled"] is True
    assert result.report.repo_info["reflection"]["reviewed"] == 1
    assert result.debug_payload["reflection_decisions"][0]["action"] == "keep"


def test_reflection_disabled_does_not_review_findings(tmp_path: Path) -> None:
    result = ReviewEngine(ReviewEngineOptions(provider="fake")).run(make_input(tmp_path))

    assert result.report.repo_info["reflection"]["enabled"] is False
    assert result.report.repo_info["reflection"]["reviewed"] == 0
    assert result.report.repo_info["reflection"]["token_usage"]["total_tokens"] == 0


def decision(action: str, reason: str, *, severity: str | None = None) -> dict[str, object]:
    return {
        "schema_version": "review-pilot.reflection.v1",
        "decision": action,
        "reason": reason,
        "severity": severity,
    }


def make_finding(
    severity: str,
    confidence: str,
    *,
    message: str = "变更行上的行为可能不符合预期。",
) -> Finding:
    return Finding(
        message=message,
        file_path="src/app.py",
        line_no=1,
        severity=severity,
        category="bug",
        source="llm",
        confidence=confidence,
        evidence={"reason": "变更行直接展示了这个行为。"},
        suggestion="补充边界测试并确认行为。",
    )


def make_pack(tmp_path: Path):
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    parsed_diff = ParsedDiff(
        files=(
            DiffFile(
                old_path="src/app.py",
                new_path="src/app.py",
                change_type="modified",
                hunks=(
                    DiffHunk(
                        old_start=1,
                        old_count=1,
                        new_start=1,
                        new_count=1,
                        section="",
                        lines=(
                            DiffLine(
                                kind="added",
                                content="print('debug')",
                                old_line_no=None,
                                new_line_no=1,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    from review_pilot.code_index import build_code_index
    from review_pilot.context_selector import select_context_candidates

    context = apply_token_budget(
        select_context_candidates(parsed_diff, build_code_index(tmp_path)),
        parsed_diff,
        tmp_path,
        400,
    )
    return build_review_context_pack(
        repo_info=RepoInfo(
            root=str(tmp_path),
            branch="main",
            head="abc123",
            has_staged_changes=True,
            has_unstaged_changes=False,
        ),
        config=ReviewPilotConfig.default(),
        parsed_diff=parsed_diff,
        rule_findings=[],
        context=context,
    )


def make_input(tmp_path: Path) -> ReviewInput:
    pack = make_pack(tmp_path)
    return ReviewInput(
        repo_info=RepoInfo(**pack.repo_info),
        config=ReviewPilotConfig.default(),
        parsed_diff=ParsedDiff(
            files=(
                DiffFile(
                    old_path="src/app.py",
                    new_path="src/app.py",
                    change_type="modified",
                    hunks=(
                        DiffHunk(
                            old_start=1,
                            old_count=1,
                            new_start=1,
                            new_count=1,
                            section="",
                            lines=(
                                DiffLine(
                                    kind="added",
                                    content="print('debug')",
                                    old_line_no=None,
                                    new_line_no=1,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        input_source="local-staged",
    )
