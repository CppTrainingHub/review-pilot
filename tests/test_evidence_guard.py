from __future__ import annotations

import pytest

from review_pilot.context_pack import ReviewContextPack
from review_pilot.evidence_guard import (
    DOWNGRADED,
    DROPPED,
    VERIFIED,
    build_evidence_index,
    guard_llm_findings,
)
from review_pilot.models import (
    ContextBudgetManifest,
    ContextSlice,
    OmittedContext,
)
from review_pilot.report_models import Finding


def test_build_evidence_index_uses_added_lines_and_supplied_context() -> None:
    index = build_evidence_index(_context_pack())

    assert index.added_lines["src/app.py"][2] == "    print('debug')"
    assert index.context_lines["src/helper.py"][10] == "def helper():"
    assert "src/omitted.py" not in index.context_lines


def test_guard_keeps_finding_on_added_line() -> None:
    result = guard_llm_findings(
        (_finding(file_path="src/app.py", line_no=2),),
        _context_pack(),
    )

    decision = result.decisions[0]
    assert decision.status == VERIFIED
    assert decision.reference is not None
    assert decision.reference.source == "diff_added_line"
    assert decision.finding.confidence == "high"
    assert decision.finding.evidence["verification"] == {
        "status": "verified",
        "source": "diff_added_line",
        "line_content": "    print('debug')",
    }


def test_guard_downgrades_finding_on_supplied_context_line() -> None:
    result = guard_llm_findings(
        (_finding(file_path="src/helper.py", line_no=11),),
        _context_pack(),
    )

    decision = result.decisions[0]
    assert decision.status == DOWNGRADED
    assert decision.reference is not None
    assert decision.reference.source == "context_used"
    assert decision.finding.confidence == "low"
    assert (
        decision.finding.evidence["verification"]["line_content"]
        == "    return 1"
    )


def test_guard_can_reference_trailing_blank_context_line() -> None:
    result = guard_llm_findings(
        (_finding(file_path="src/helper.py", line_no=12),),
        _context_pack(),
    )

    decision = result.decisions[0]
    assert decision.status == DOWNGRADED
    assert decision.reference is not None
    assert decision.reference.line_content == ""


def test_guard_drops_unknown_file() -> None:
    result = guard_llm_findings(
        (_finding(file_path="src/missing.py", line_no=1),),
        _context_pack(),
    )

    decision = result.decisions[0]
    assert decision.status == DROPPED
    assert "file_path is not present" in decision.reason
    assert result.findings == ()


def test_guard_drops_line_outside_supplied_evidence() -> None:
    result = guard_llm_findings(
        (_finding(file_path="src/app.py", line_no=99),),
        _context_pack(),
    )

    assert result.decisions[0].status == DROPPED
    assert "line_no is not present" in result.decisions[0].reason


def test_guard_does_not_treat_omitted_context_as_evidence() -> None:
    result = guard_llm_findings(
        (_finding(file_path="src/omitted.py", line_no=1),),
        _context_pack(),
    )

    assert result.decisions[0].status == DROPPED
    assert "file_path is not present" in result.decisions[0].reason


@pytest.mark.parametrize(
    "file_path",
    [
        "/etc/passwd",
        "../secret.py",
        "src/../../secret.py",
        r"src\app.py",
    ],
)
def test_guard_drops_unsafe_or_non_posix_paths(file_path: str) -> None:
    result = guard_llm_findings(
        (_finding(file_path=file_path, line_no=2),),
        _context_pack(),
    )

    assert result.decisions[0].status == DROPPED
    assert "repository-relative POSIX path" in result.decisions[0].reason


def test_guard_normalizes_leading_dot_path() -> None:
    result = guard_llm_findings(
        (_finding(file_path="./src/app.py", line_no=2),),
        _context_pack(),
    )

    assert result.decisions[0].status == VERIFIED
    assert result.findings[0].file_path == "src/app.py"


def test_guard_reports_mixed_summary_and_preserves_dropped_finding() -> None:
    findings = (
        _finding(file_path="src/app.py", line_no=2),
        _finding(file_path="src/helper.py", line_no=10),
        _finding(file_path="src/missing.py", line_no=9),
    )

    result = guard_llm_findings(findings, _context_pack())
    payload = result.to_dict()

    assert result.summary == {
        "total": 3,
        "kept": 2,
        "verified": 1,
        "downgraded": 1,
        "dropped": 1,
    }
    assert len(payload["findings"]) == 2
    assert payload["dropped_findings"][0]["finding"]["file_path"] == (
        "src/missing.py"
    )
    assert payload["dropped_findings"][0]["status"] == "dropped"


def _finding(*, file_path: str, line_no: int) -> Finding:
    return Finding(
        message="Review this reference.",
        file_path=file_path,
        line_no=line_no,
        severity="P2",
        category="bug",
        source="llm",
        confidence="high",
        evidence={"reason": "The referenced line may be incorrect."},
        suggestion="Update the referenced line.",
    )


def _context_pack() -> ReviewContextPack:
    context = ContextBudgetManifest(
        changed_paths=("src/app.py",),
        max_context_tokens=100,
        used_tokens=20,
        index_file_count=3,
        context_used=(
            ContextSlice(
                path="src/helper.py",
                reason="related import",
                priority=80,
                language="python",
                start_line=10,
                end_line=12,
                estimated_tokens=10,
                content="def helper():\n    return 1\n",
            ),
        ),
        context_omitted=(
            OmittedContext(
                path="src/omitted.py",
                reason="related symbol",
                priority=10,
                language="python",
                omitted_reason="token_budget",
                omitted_lines=2,
            ),
        ),
    )
    return ReviewContextPack(
        schema_version="review-pilot.context-pack.v1",
        repo_info={
            "root": "/tmp/repo",
            "branch": "main",
            "head": "0" * 40,
            "has_staged_changes": True,
            "has_unstaged_changes": False,
        },
        config={},
        diff={
            "files": [
                {
                    "old_path": "src/app.py",
                    "new_path": "src/app.py",
                    "path": "src/app.py",
                    "change_type": "modified",
                    "hunks": [
                        {
                            "old_start": 1,
                            "old_count": 1,
                            "new_start": 1,
                            "new_count": 2,
                            "section": "",
                            "lines": [
                                {
                                    "kind": "context",
                                    "content": "def run():",
                                    "old_line_no": 1,
                                    "new_line_no": 1,
                                    "no_newline_at_eof": False,
                                },
                                {
                                    "kind": "added",
                                    "content": "    print('debug')",
                                    "old_line_no": None,
                                    "new_line_no": 2,
                                    "no_newline_at_eof": False,
                                },
                            ],
                        }
                    ],
                }
            ]
        },
        rule_findings=(),
        context=context,
        generated_by={
            "tool": "review-pilot",
            "command": "context-pack",
        },
    )
