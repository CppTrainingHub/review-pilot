from __future__ import annotations

from pathlib import Path

import pytest

from review_pilot.code_index import build_code_index
from review_pilot.config import ReviewPilotConfig
from review_pilot.context_pack import (
    CONTEXT_PACK_SCHEMA_VERSION,
    build_review_context_pack,
    validate_context_pack_dict,
)
from review_pilot.context_selector import select_context_candidates
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.models import RawDiff, RepoInfo
from review_pilot.report_models import Finding
from review_pilot.token_budget import apply_token_budget


def test_build_review_context_pack_contains_auditable_inputs(tmp_path: Path) -> None:
    _write_project(tmp_path)
    parsed_diff = parse_unified_diff(_added_file_diff("src/service.py"))
    index = build_code_index(tmp_path)
    candidates = select_context_candidates(parsed_diff, index)
    context = apply_token_budget(candidates, parsed_diff, tmp_path, max_context_tokens=400)
    finding = Finding(
        message="debug output left in changed code",
        file_path="src/service.py",
        line_no=2,
        severity="P2",
        category="style",
        source="rule",
        rule_id="rule.debug-output",
    )

    pack = build_review_context_pack(
        repo_info=_repo_info(tmp_path),
        config=ReviewPilotConfig.default(),
        parsed_diff=parsed_diff,
        rule_findings=[finding],
        context=context,
    )
    payload = pack.to_dict()

    assert payload["schema_version"] == CONTEXT_PACK_SCHEMA_VERSION
    assert payload["repo_info"]["root"] == str(tmp_path)
    assert payload["config"]["source"] == "default"
    assert payload["diff"]["files"][0]["path"] == "src/service.py"
    assert payload["rule_findings"][0]["rule_id"] == "rule.debug-output"
    assert payload["context"]["context_used"][0]["path"] == "src/service.py"
    assert payload["generated_by"]["command"] == "context-pack"
    validate_context_pack_dict(payload)


def test_context_pack_preserves_omitted_context(tmp_path: Path) -> None:
    _write_project(tmp_path)
    parsed_diff = parse_unified_diff(_added_file_diff("src/service.py"))
    index = build_code_index(tmp_path)
    candidates = select_context_candidates(parsed_diff, index)
    context = apply_token_budget(candidates, parsed_diff, tmp_path, max_context_tokens=12)

    pack = build_review_context_pack(
        repo_info=_repo_info(tmp_path),
        config=ReviewPilotConfig.default(),
        parsed_diff=parsed_diff,
        rule_findings=[],
        context=context,
    )
    payload = pack.to_dict()

    omitted_reasons = {item["omitted_reason"] for item in payload["context"]["context_omitted"]}
    assert "budget_exhausted" in omitted_reasons
    validate_context_pack_dict(payload)


def test_validate_context_pack_dict_rejects_missing_fields(tmp_path: Path) -> None:
    payload = {
        "schema_version": CONTEXT_PACK_SCHEMA_VERSION,
        "repo_info": {},
    }

    with pytest.raises(ValueError, match="missing context pack fields"):
        validate_context_pack_dict(payload)


def test_validate_context_pack_dict_rejects_wrong_schema_version() -> None:
    payload = {
        "schema_version": "wrong",
        "repo_info": {},
        "config": {},
        "diff": {"files": []},
        "rule_findings": [],
        "context": {
            "max_context_tokens": 4000,
            "used_tokens": 0,
            "remaining_tokens": 4000,
            "context_used": [],
            "context_omitted": [],
        },
        "generated_by": {},
    }

    with pytest.raises(ValueError, match="schema_version"):
        validate_context_pack_dict(payload)


def _write_project(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "helpers.py").write_text("def load():\n    return 1\n", encoding="utf-8")
    (root / "src" / "service.py").write_text(
        "from .helpers import load\n\n"
        "def run():\n"
        "    print('debug')\n"
        "    return load()\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_service.py").write_text(
        "from src.service import run\n\n"
        "def test_run():\n"
        "    assert run() == 1\n",
        encoding="utf-8",
    )


def _repo_info(root: Path) -> RepoInfo:
    return RepoInfo(
        root=str(root),
        branch="main",
        head="0" * 40,
        has_staged_changes=True,
        has_unstaged_changes=False,
    )


def _added_file_diff(path: str) -> RawDiff:
    return RawDiff(
        "\n".join(
            [
                f"diff --git a/{path} b/{path}",
                "--- /dev/null",
                f"+++ b/{path}",
                "@@ -0,0 +1,5 @@",
                "+from .helpers import load",
                "+",
                "+def run():",
                "+    print('debug')",
                "+    return load()",
            ]
        )
    )
