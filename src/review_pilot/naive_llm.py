from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .config import LLMConfig
from .models import RawDiff


class NaiveReviewError(RuntimeError):
    pass


class NaiveLLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        pass


@dataclass(frozen=True)
class FakeNaiveLLMClient:
    def complete(self, prompt: str) -> str:
        changed_files = _guess_changed_files(prompt)
        file_line = ", ".join(changed_files) if changed_files else "未知文件"
        return (
            "Naive Review Result\n"
            f"- 这次变更涉及：{file_line}\n"
            "- 可能风险：需要人工确认新增逻辑是否有测试覆盖。\n"
            "- 不稳定点：这是自由文本输出，没有结构化 severity、行号证据或去重规则。\n"
        )


@dataclass(frozen=True)
class OpenAICompatibleNaiveLLMClient:
    config: LLMConfig

    def complete(self, prompt: str) -> str:
        if not self.config.api_key:
            raise NaiveReviewError(
                "missing API key: set REVIEW_PILOT_API_KEY or OPENAI_API_KEY"
            )

        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a pragmatic code reviewer. Return a concise plain-text review.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise NaiveReviewError(f"llm request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise NaiveReviewError("llm response was not valid JSON") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise NaiveReviewError("llm response did not contain message content") from exc
        return str(content).strip()


def build_naive_review_prompt(raw_diff: RawDiff) -> str:
    return (
        "请直接 review 下面这段 Git staged raw diff。\n"
        "输出一段自然语言审查意见，指出你认为可能有风险的地方。\n"
        "这是一个 naive 版本，先不要求 JSON schema、证据校验或固定 severity。\n\n"
        "DIFF:\n"
        f"{raw_diff.text}"
    )


def create_naive_client(provider: str) -> NaiveLLMClient:
    if provider == "fake":
        return FakeNaiveLLMClient()
    if provider == "openai":
        return OpenAICompatibleNaiveLLMClient(LLMConfig.from_env(provider))
    raise NaiveReviewError(f"unsupported provider: {provider}")


def run_naive_review(raw_diff: RawDiff, provider: str) -> str:
    if raw_diff.is_empty:
        raise NaiveReviewError("no staged changes")
    prompt = build_naive_review_prompt(raw_diff)
    client = create_naive_client(provider)
    return client.complete(prompt)


def _guess_changed_files(prompt: str) -> list[str]:
    files: list[str] = []
    for line in prompt.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            files.append(parts[3].removeprefix("b/"))
    return files
