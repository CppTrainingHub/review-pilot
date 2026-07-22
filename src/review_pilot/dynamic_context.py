from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Any

from .context_tools import (
    ContextToolError,
    ContextToolExecutor,
    ContextToolRequest,
    ContextToolResult,
    DynamicContextTrace,
)
from .context_pack import ReviewContextPack
from .llm.base import LLMResponse
from .llm.prompt_builder import build_dynamic_review_prompt
from .llm.reviewer import StructuredReviewResult, StructuredReviewer
from .llm.schema import LLMOutputError


MAX_DYNAMIC_ROUNDS = 5
DEFAULT_DYNAMIC_TOKEN_BUDGET = 12_000
FINAL_FINDINGS_MAX_OUTPUT_TOKENS = 4_096


class DynamicContextError(LLMOutputError):
    """Raised when a bounded dynamic review cannot reach a final response."""

    def __init__(self, message: str, *, trace: tuple[DynamicContextTrace, ...] = ()) -> None:
        super().__init__(message)
        self.trace = trace


@dataclass(frozen=True)
class DynamicContextResult:
    review: StructuredReviewResult
    trace: tuple[DynamicContextTrace, ...]
    rounds: int
    token_usage: dict[str, int]
    truncated: bool

    @property
    def tool_calls(self) -> int:
        return len(self.trace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dynamic_context_trace": [item.to_dict() for item in self.trace],
            "rounds": self.rounds,
            "tool_calls": self.tool_calls,
            "token_usage": dict(self.token_usage),
            "truncated": self.truncated,
            "review": self.review.to_dict(),
        }


def run_dynamic_context(
    *,
    context_pack: ReviewContextPack,
    reviewer: StructuredReviewer,
    executor: ContextToolExecutor,
    max_rounds: int = MAX_DYNAMIC_ROUNDS,
    max_tokens: int = DEFAULT_DYNAMIC_TOKEN_BUDGET,
    snippet_location: bool = False,
) -> DynamicContextResult:
    if max_rounds < 1 or max_rounds > MAX_DYNAMIC_ROUNDS:
        raise ValueError(f"max_rounds must be between 1 and {MAX_DYNAMIC_ROUNDS}")
    if max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")

    prompt = build_dynamic_review_prompt(
        context_pack,
        snippet_location=snippet_location,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    trace: list[DynamicContextTrace] = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    truncated = False

    for round_index in range(1, max_rounds + 1):
        response = reviewer.review_with_tools(
            context_pack,
            messages,
            executor.tool_schemas(),
        )
        _accumulate_usage(token_usage, response)
        if response.tool_calls:
            if round_index == max_rounds:
                return _finalize_after_bound(
                    context_pack=context_pack,
                    reviewer=reviewer,
                    messages=messages,
                    trace=trace,
                    token_usage=token_usage,
                    truncated=True,
                    snippet_location=snippet_location,
                    rounds=round_index,
                    reason=f"dynamic context reached the {max_rounds}-round limit",
                )
            messages.append(_assistant_tool_message(response))
            for call in response.tool_calls:
                result, item = _execute_call(
                    executor=executor,
                    call_id=call.call_id,
                    tool_name=call.name,
                    raw_arguments=call.arguments,
                    round_index=round_index,
                )
                trace.append(item)
                truncated = truncated or result.truncated
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": result.content,
                    }
                )
            if token_usage["total_tokens"] >= max_tokens:
                return _finalize_after_bound(
                    context_pack=context_pack,
                    reviewer=reviewer,
                    messages=messages,
                    trace=trace,
                    token_usage=token_usage,
                    truncated=True,
                    snippet_location=snippet_location,
                    rounds=round_index,
                    reason=f"dynamic context reached the {max_tokens}-token limit",
                )
            continue

        return _parse_final_response(
            response=response,
            context_pack=context_pack,
            trace=trace,
            rounds=round_index,
            token_usage=token_usage,
            truncated=truncated,
            snippet_location=snippet_location,
        )

    raise DynamicContextError(
        "dynamic context ended without final findings",
        trace=tuple(trace),
    )


def _finalize_after_bound(
    *,
    context_pack: ReviewContextPack,
    reviewer: StructuredReviewer,
    messages: list[dict[str, Any]],
    trace: list[DynamicContextTrace],
    token_usage: dict[str, int],
    truncated: bool,
    snippet_location: bool,
    rounds: int,
    reason: str,
) -> DynamicContextResult:
    """Ask for the final contract after the tool budget has been exhausted."""

    messages.append(
        {
            "role": "user",
            "content": (
                f"{reason}. Do not call any more tools. Return the final findings "
                "JSON now, using only the diff, supplied context, and evidence "
                "already returned by the tools."
            ),
        }
    )
    response = reviewer.review_with_tools(
        context_pack,
        messages,
        [],
        max_tokens=FINAL_FINDINGS_MAX_OUTPUT_TOKENS,
    )
    _accumulate_usage(token_usage, response)
    if response.tool_calls:
        raise DynamicContextError(
            f"{reason}, but provider requested another tool call",
            trace=tuple(trace),
        )
    return _parse_final_response(
        response=response,
        context_pack=context_pack,
        trace=trace,
        rounds=rounds,
        token_usage=token_usage,
        truncated=truncated,
        snippet_location=snippet_location,
    )


def _parse_final_response(
    *,
    response: LLMResponse,
    context_pack: ReviewContextPack,
    trace: list[DynamicContextTrace],
    rounds: int,
    token_usage: dict[str, int],
    truncated: bool,
    snippet_location: bool,
) -> DynamicContextResult:
    try:
        parsed = StructuredReviewer.parse_response(
            response,
            context_pack,
            snippet_location=snippet_location,
        )
    except (ValueError, LLMOutputError) as exc:
        repaired_content = _strip_json_fence(response.content)
        if repaired_content == response.content:
            raise DynamicContextError(str(exc), trace=tuple(trace)) from exc
        try:
            parsed = StructuredReviewer.parse_response(
                replace(response, content=repaired_content),
                context_pack,
                snippet_location=snippet_location,
            )
        except (ValueError, LLMOutputError) as repaired_exc:
            raise DynamicContextError(
                str(repaired_exc),
                trace=tuple(trace),
            ) from repaired_exc
    return DynamicContextResult(
        review=parsed,
        trace=tuple(trace),
        rounds=rounds,
        token_usage=token_usage,
        truncated=truncated,
    )


def _execute_call(
    *,
    executor: ContextToolExecutor,
    call_id: str,
    tool_name: str,
    raw_arguments: str,
    round_index: int,
) -> tuple[ContextToolResult, DynamicContextTrace]:
    started = time.perf_counter()
    result: ContextToolResult
    try:
        arguments = json.loads(raw_arguments)
        request = ContextToolRequest(
            tool_name=tool_name,
            arguments=arguments,
            call_id=call_id,
        )
        result = executor.execute(request)
    except json.JSONDecodeError as exc:
        result = ContextToolResult(
            tool_name=tool_name,
            call_id=call_id,
            content=json.dumps(
                {"error": {"code": "invalid_arguments", "message": "arguments must be valid JSON"}},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            error_code="invalid_arguments",
            error_message=f"arguments must be valid JSON: {exc.msg}",
        )
    except ContextToolError as exc:
        result = ContextToolResult(
            tool_name=tool_name,
            call_id=call_id,
            content=json.dumps(
                {"error": {"code": exc.code, "message": str(exc)}},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            error_code=exc.code,
            error_message=str(exc),
        )
    try:
        arguments_for_trace = json.loads(raw_arguments)
        if not isinstance(arguments_for_trace, dict):
            arguments_for_trace = {"raw": raw_arguments}
    except json.JSONDecodeError:
        arguments_for_trace = {"raw": raw_arguments}
    trace = DynamicContextTrace(
        round=round_index,
        tool_name=tool_name,
        arguments=arguments_for_trace,
        call_id=call_id,
        status="completed" if result.ok else "error",
        returned_count=result.returned_count,
        truncated=result.truncated,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        error_code=result.error_code,
        error_message=result.error_message,
    )
    return result, trace


def _assistant_tool_message(response: LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content or None,
        "tool_calls": [call.to_dict() for call in response.tool_calls],
    }


def _accumulate_usage(total: dict[str, int], response: LLMResponse) -> None:
    if not response.usage:
        return
    for key in total:
        value = response.usage.get(key)
        if isinstance(value, int) and value >= 0:
            total[key] += value


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    start = stripped.find("```")
    if start < 0:
        return content
    end = stripped.find("```", start + 3)
    if end < 0:
        return content
    fenced = stripped[start + 3 : end]
    lines = fenced.splitlines()
    if lines and lines[0].strip().lower() in {"json", "javascript"}:
        lines = lines[1:]
    if not lines:
        return content
    return "\n".join(lines).strip()
