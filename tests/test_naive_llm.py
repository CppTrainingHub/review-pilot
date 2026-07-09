from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from review_pilot.models import RawDiff
from review_pilot.naive_llm import (
    NaiveReviewError,
    build_naive_review_prompt,
    run_naive_review,
)


RAW_DIFF = """diff --git a/app.py b/app.py
new file mode 100644
index 0000000..b376c99
--- /dev/null
+++ b/app.py
@@ -0,0 +1 @@
+print('hello')
"""


def test_build_naive_review_prompt_contains_raw_diff_and_plain_text_goal() -> None:
    prompt = build_naive_review_prompt(RawDiff(RAW_DIFF))

    assert "Git staged raw diff" in prompt
    assert "自然语言审查意见" in prompt
    assert "diff --git a/app.py b/app.py" in prompt
    assert "JSON schema" in prompt


def test_fake_provider_returns_unstructured_review_text() -> None:
    review = run_naive_review(RawDiff(RAW_DIFF), provider="fake")

    assert "Naive Review Result" in review
    assert "app.py" in review
    assert "自由文本输出" in review
    assert "severity" in review


def test_empty_raw_diff_fails_before_provider_call() -> None:
    with pytest.raises(NaiveReviewError, match="no staged changes"):
        run_naive_review(RawDiff(""), provider="fake")


def test_unsupported_provider_is_rejected() -> None:
    with pytest.raises(NaiveReviewError, match="unsupported provider"):
        run_naive_review(RawDiff(RAW_DIFF), provider="unknown")


def test_openai_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("REVIEW_PILOT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(NaiveReviewError, match="missing API key"):
        run_naive_review(RawDiff(RAW_DIFF), provider="openai")
