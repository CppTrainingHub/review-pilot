from __future__ import annotations

from pathlib import Path

from review_pilot.code_index import build_code_index
from review_pilot.config import ReviewPilotConfig
from review_pilot.context_pack import build_review_context_pack
from review_pilot.context_selector import select_context_candidates
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.llm.prompt_builder import (
    build_review_prompt,
    llm_output_contract,
)
from review_pilot.models import RawDiff, RepoInfo
from review_pilot.report_models import Finding
from review_pilot.token_budget import apply_token_budget


def test_build_review_prompt_uses_fixed_sections(tmp_path: Path) -> None:
    prompt = build_review_prompt(_context_pack(tmp_path))

    assert "Use only evidence present" in prompt.system
    assert "plain JSON object" in prompt.system
    assert "markdown fences" in prompt.system
    assert "source must be exactly 'llm'" in prompt.system
    assert "Simplified Chinese" in prompt.system
    assert "message, suggestion, and evidence.reason must be Chinese" in prompt.system
    assert "positive integer from an added line" in prompt.system
    for section in (
        "## REPOSITORY",
        "## DIFF",
        "## DETERMINISTIC_FINDINGS",
        "## TOOL_FINDINGS",
        "## CONTEXT_USED",
        "## CONTEXT_OMITTED",
        "## OUTPUT_CONTRACT",
    ):
        assert section in prompt.user


def test_prompt_contains_pack_data_without_absolute_repo_root(
    tmp_path: Path,
) -> None:
    prompt = build_review_prompt(_context_pack(tmp_path))

    assert "src/service.py" in prompt.user
    assert "rule.debug-output" in prompt.user
    assert "review-pilot.llm-findings.v1" in prompt.user
    assert str(tmp_path) not in prompt.user


def test_output_contract_lists_strict_fields() -> None:
    contract = llm_output_contract()

    assert contract["schema_version"] == "review-pilot.llm-findings.v1"
    assert contract["root_fields"] == ["schema_version", "findings"]
    assert contract["findings"]["type"] == "non-empty array"
    assert contract["findings"]["source"] == "llm"
    assert contract["findings"]["line_no"] == "positive integer from an added line in DIFF"
    assert contract["findings"]["evidence_fields"] == ["reason"]


def _context_pack(root: Path):
    (root / "src").mkdir()
    (root / "src" / "service.py").write_text(
        "def run():\n    print('debug')\n",
        encoding="utf-8",
    )
    parsed_diff = parse_unified_diff(
        RawDiff(
            "\n".join(
                [
                    "diff --git a/src/service.py b/src/service.py",
                    "--- /dev/null",
                    "+++ b/src/service.py",
                    "@@ -0,0 +1,2 @@",
                    "+def run():",
                    "+    print('debug')",
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
        max_context_tokens=200,
    )
    finding = Finding(
        message="debug output left in changed code",
        file_path="src/service.py",
        line_no=2,
        severity="P2",
        category="style",
        source="rule",
        rule_id="rule.debug-output",
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
        rule_findings=[finding],
        context=context,
    )
