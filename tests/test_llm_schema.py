from __future__ import annotations

import json

import pytest

from review_pilot.llm.schema import (
    LLM_FINDINGS_SCHEMA_VERSION,
    LLMOutputError,
    parse_llm_findings,
)


def test_parse_valid_llm_findings() -> None:
    envelope = parse_llm_findings(json.dumps(_valid_payload()))

    assert envelope.schema_version == LLM_FINDINGS_SCHEMA_VERSION
    assert len(envelope.findings) == 1
    finding = envelope.findings[0]
    assert finding.file_path == "src/app.py"
    assert finding.line_no == 2
    assert finding.source == "llm"
    assert finding.evidence == {"reason": "Debug output is executed."}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "non-empty JSON"),
        ("not-json", "not valid JSON"),
        ('[{"findings":[]}]', "root must be an object"),
        (
            "```json\n"
            '{"schema_version":"review-pilot.llm-findings.v1",'
            '"findings":[]}\n```',
            "without markdown fences",
        ),
    ],
)
def test_parse_rejects_non_protocol_content(
    content: str,
    message: str,
) -> None:
    with pytest.raises(LLMOutputError, match=message):
        parse_llm_findings(content)


def test_parse_rejects_unexpected_root_fields() -> None:
    payload = _valid_payload()
    payload["summary"] = "extra"

    with pytest.raises(LLMOutputError, match="unexpected fields"):
        parse_llm_findings(json.dumps(payload))


def test_parse_rejects_wrong_schema_version() -> None:
    payload = _valid_payload()
    payload["schema_version"] = "wrong"

    with pytest.raises(LLMOutputError, match="schema_version"):
        parse_llm_findings(json.dumps(payload))


def test_parse_rejects_empty_findings() -> None:
    payload = _valid_payload()
    payload["findings"] = []

    with pytest.raises(LLMOutputError, match="at least one"):
        parse_llm_findings(json.dumps(payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("message", "", "message must be a non-empty string"),
        ("file_path", "", "file_path must be a non-empty string"),
        ("line_no", 0, "line_no must be a positive integer"),
        ("line_no", True, "line_no must be a positive integer"),
        ("severity", "P9", "severity must be one of"),
        ("category", "logic", "category must be one of"),
        ("source", "rule", "source must be 'llm'"),
        ("confidence", "certain", "confidence must be one of"),
        ("suggestion", "", "suggestion must be a non-empty string"),
    ],
)
def test_parse_rejects_invalid_finding_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _valid_payload()
    payload["findings"][0][field] = value

    with pytest.raises(LLMOutputError, match=message):
        parse_llm_findings(json.dumps(payload))


def test_parse_rejects_missing_or_extra_finding_fields() -> None:
    missing = _valid_payload()
    del missing["findings"][0]["suggestion"]
    with pytest.raises(LLMOutputError, match="missing fields"):
        parse_llm_findings(json.dumps(missing))

    extra = _valid_payload()
    extra["findings"][0]["rule_id"] = "made-up"
    with pytest.raises(LLMOutputError, match="unexpected fields"):
        parse_llm_findings(json.dumps(extra))


def test_parse_rejects_invalid_evidence() -> None:
    not_object = _valid_payload()
    not_object["findings"][0]["evidence"] = "reason"
    with pytest.raises(LLMOutputError, match="evidence must be an object"):
        parse_llm_findings(json.dumps(not_object))

    extra = _valid_payload()
    extra["findings"][0]["evidence"]["line"] = "invented"
    with pytest.raises(LLMOutputError, match="unexpected fields"):
        parse_llm_findings(json.dumps(extra))

    empty_reason = _valid_payload()
    empty_reason["findings"][0]["evidence"]["reason"] = ""
    with pytest.raises(LLMOutputError, match="reason must be a non-empty string"):
        parse_llm_findings(json.dumps(empty_reason))


def _valid_payload() -> dict:
    return {
        "schema_version": LLM_FINDINGS_SCHEMA_VERSION,
        "findings": [
            {
                "message": "Debug output remains in changed code.",
                "file_path": "src/app.py",
                "line_no": 2,
                "severity": "P2",
                "category": "maintainability",
                "source": "llm",
                "confidence": "medium",
                "evidence": {
                    "reason": "Debug output is executed.",
                },
                "suggestion": "Remove the debug print.",
            }
        ],
    }
