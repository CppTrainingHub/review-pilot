from __future__ import annotations

import json
from http.client import IncompleteRead
import urllib.error
from pathlib import Path

import pytest

from review_pilot.code_index import build_code_index
from review_pilot.config import ConfigError, LLMConfig, ReviewPilotConfig
from review_pilot.context_pack import build_review_context_pack
from review_pilot.context_selector import select_context_candidates
from review_pilot.diff_parser import parse_unified_diff
from review_pilot.llm import (
    FakeProvider,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponse,
    OpenAICompatibleProvider,
    create_provider,
)
from review_pilot.models import RawDiff, RepoInfo
from review_pilot.token_budget import apply_token_budget


def test_llm_response_requires_non_empty_fields() -> None:
    with pytest.raises(ValueError, match="content"):
        LLMResponse(provider="fake", model="fake-review-model", content="")


def test_fake_provider_returns_deterministic_context_summary(tmp_path: Path) -> None:
    pack = _context_pack(tmp_path)
    provider = FakeProvider()

    first = provider.review(pack)
    second = provider.review(pack)

    assert first == second
    assert first.provider == "fake"
    assert first.model == "fake-review-model"
    payload = json.loads(first.content)
    assert payload["schema_version"] == "review-pilot.llm-findings.v1"
    assert payload["findings"][0]["file_path"] == "src/service.py"
    assert payload["findings"][0]["source"] == "llm"


def test_provider_factory_rejects_unknown_provider() -> None:
    with pytest.raises(LLMConfigurationError, match="unsupported provider"):
        create_provider("unknown")


def test_llm_config_reads_openai_compatible_environment(monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_PILOT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("REVIEW_PILOT_LLM_MODEL", "review-model")
    monkeypatch.setenv("REVIEW_PILOT_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("REVIEW_PILOT_LLM_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("REVIEW_PILOT_API_KEY", "secret-value")

    config = LLMConfig.from_env()

    assert config.provider == "openai-compatible"
    assert config.model == "review-model"
    assert config.base_url == "https://llm.example/v1"
    assert config.timeout_seconds == 17
    assert config.api_key == "secret-value"
    assert config.status_dict()["api_key"] == "configured"
    assert "secret-value" not in json.dumps(config.status_dict())


def test_llm_config_uses_fake_provider_defaults(monkeypatch) -> None:
    monkeypatch.delenv("REVIEW_PILOT_LLM_MODEL", raising=False)
    monkeypatch.delenv("REVIEW_PILOT_LLM_BASE_URL", raising=False)

    config = LLMConfig.from_env("fake")

    assert config.model == "fake-review-model"
    assert config.base_url == "not-used"


def test_llm_config_rejects_invalid_timeout(monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_PILOT_LLM_TIMEOUT_SECONDS", "zero")

    with pytest.raises(ConfigError, match="positive integer"):
        LLMConfig.from_env()


def test_openai_compatible_provider_requires_api_key(tmp_path: Path) -> None:
    provider = OpenAICompatibleProvider(
        LLMConfig(
            provider="openai-compatible",
            model="review-model",
            base_url="https://llm.example/v1",
            api_key=None,
        )
    )

    with pytest.raises(LLMConfigurationError, match="missing API key"):
        provider.review(_context_pack(tmp_path))


def test_openai_compatible_provider_builds_request_and_parses_response(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Review completed.",
                        }
                    }
                ]
            }
        )

    provider = OpenAICompatibleProvider(
        LLMConfig(
            provider="openai-compatible",
            model="review-model",
            base_url="https://llm.example/v1/",
            api_key="secret-value",
            timeout_seconds=17,
        ),
        urlopen=fake_urlopen,
    )

    response = provider.review(_context_pack(tmp_path))

    assert response == LLMResponse(
        provider="openai-compatible",
        model="review-model",
        content="Review completed.",
    )
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret-value"
    assert captured["timeout"] == 17
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "review-model"
    assert "Use only evidence present" in payload["messages"][0]["content"]
    assert "## REPOSITORY" in payload["messages"][1]["content"]
    assert "## OUTPUT_CONTRACT" in payload["messages"][1]["content"]
    assert "review-pilot.llm-findings.v1" in payload["messages"][1]["content"]


def test_openai_compatible_reflection_requests_json_object(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "schema_version": "review-pilot.reflection.v1",
                                    "decision": "keep",
                                    "reason": "证据足够。",
                                    "severity": None,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 9},
            }
        )

    provider = OpenAICompatibleProvider(
        LLMConfig(
            provider="openai-compatible",
            model="review-model",
            base_url="https://llm.example/v1",
            api_key="secret-value",
        ),
        urlopen=fake_urlopen,
    )
    pack = _context_pack(tmp_path)
    from review_pilot.evidence_guard import guard_llm_findings
    from review_pilot.report_models import Finding

    finding = Finding(
        message="变更行为需要边界测试。",
        file_path="src/service.py",
        line_no=2,
        severity="P2",
        category="bug",
        source="llm",
        confidence="low",
    )
    provider.reflect_finding(
        finding=finding,
        context_pack=pack,
        evidence=guard_llm_findings((finding,), pack),
    )

    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_final_tool_request_limits_json_output(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"schema_version":"review-pilot.llm-findings.v1","findings":[]}',
                        }
                    }
                ]
            }
        )

    provider = OpenAICompatibleProvider(
        LLMConfig(
            provider="openai-compatible",
            model="review-model",
            base_url="https://llm.example/v1",
            api_key="secret-value",
        ),
        urlopen=fake_urlopen,
    )

    provider.review_with_tools(
        _context_pack(tmp_path),
        [{"role": "user", "content": "Return the final JSON."}],
        [],
        max_tokens=4096,
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["max_tokens"] == 4096
    assert payload["response_format"] == {"type": "json_object"}


def test_openai_compatible_provider_normalizes_network_failure(
    tmp_path: Path,
) -> None:
    def failing_urlopen(request, timeout):
        raise urllib.error.URLError("offline")

    provider = OpenAICompatibleProvider(
        LLMConfig(
            provider="openai-compatible",
            model="review-model",
            base_url="https://llm.example/v1",
            api_key="secret-value",
        ),
        urlopen=failing_urlopen,
    )

    with pytest.raises(LLMRequestError, match="offline") as error:
        provider.review(_context_pack(tmp_path))
    assert "secret-value" not in str(error.value)


def test_openai_compatible_provider_normalizes_incomplete_http_response(
    tmp_path: Path,
) -> None:
    def interrupted_urlopen(request, timeout):
        raise IncompleteRead(b"partial")

    provider = OpenAICompatibleProvider(
        LLMConfig(
            provider="openai-compatible",
            model="review-model",
            base_url="https://llm.example/v1",
            api_key="secret-value",
        ),
        urlopen=interrupted_urlopen,
    )

    with pytest.raises(LLMRequestError, match="llm request failed"):
        provider.review(_context_pack(tmp_path))


def test_openai_compatible_provider_rejects_invalid_json(tmp_path: Path) -> None:
    provider = OpenAICompatibleProvider(
        LLMConfig(
            provider="openai-compatible",
            model="review-model",
            base_url="https://llm.example/v1",
            api_key="secret-value",
        ),
        urlopen=lambda request, timeout: FakeHTTPResponse("not-json", raw=True),
    )

    with pytest.raises(LLMRequestError, match="not valid JSON"):
        provider.review(_context_pack(tmp_path))


def test_openai_compatible_provider_rejects_missing_content(
    tmp_path: Path,
) -> None:
    provider = OpenAICompatibleProvider(
        LLMConfig(
            provider="openai-compatible",
            model="review-model",
            base_url="https://llm.example/v1",
            api_key="secret-value",
        ),
        urlopen=lambda request, timeout: FakeHTTPResponse({"choices": []}),
    )

    with pytest.raises(LLMRequestError, match="message content"):
        provider.review(_context_pack(tmp_path))


class FakeHTTPResponse:
    def __init__(self, payload: object, *, raw: bool = False) -> None:
        self._body = (
            str(payload).encode("utf-8")
            if raw
            else json.dumps(payload).encode("utf-8")
        )

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _context_pack(root: Path):
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "service.py").write_text(
        "def run():\n    return 1\n",
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
                    "+    return 1",
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
