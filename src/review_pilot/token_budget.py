from __future__ import annotations

from math import ceil
from pathlib import Path

from review_pilot.language_detection import normalize_path
from review_pilot.models import (
    ContextBudgetManifest,
    ContextCandidate,
    ContextCandidateManifest,
    ContextSlice,
    DiffFile,
    OmittedContext,
    ParsedDiff,
)


DEFAULT_CHANGED_LINE_RADIUS = 3
MIN_SLICE_TOKENS = 8


def estimate_tokens(text: str) -> int:
    normalized = " ".join(text.split())
    if not normalized:
        return 1
    return max(1, ceil(len(normalized) / 4))


def apply_token_budget(
    manifest: ContextCandidateManifest,
    parsed_diff: ParsedDiff,
    repo_root: str | Path,
    max_context_tokens: int,
) -> ContextBudgetManifest:
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")

    root = Path(repo_root)
    changed_lines = _changed_line_map(parsed_diff)
    used: list[ContextSlice] = []
    omitted: list[OmittedContext] = []
    remaining = max_context_tokens

    for candidate in manifest.candidates:
        lines = _read_candidate_lines(root, candidate)
        if lines is None:
            omitted.append(
                OmittedContext(
                    path=candidate.path,
                    reason=candidate.reason,
                    priority=candidate.priority,
                    language=candidate.language,
                    omitted_reason="file_unreadable",
                )
            )
            continue

        ranges = _candidate_ranges(candidate, lines, changed_lines.get(candidate.path, ()))
        candidate_used = False
        consumed_lines: set[int] = set()

        for start, end in ranges:
            if remaining < MIN_SLICE_TOKENS:
                break
            slice_lines = lines[start - 1 : end]
            fitted_lines = _fit_lines(slice_lines, remaining)
            if not fitted_lines:
                break
            content = "\n".join(fitted_lines)
            tokens = estimate_tokens(content)
            actual_end = start + len(fitted_lines) - 1
            used.append(
                ContextSlice(
                    path=candidate.path,
                    reason=candidate.reason,
                    priority=candidate.priority,
                    language=candidate.language,
                    start_line=start,
                    end_line=actual_end,
                    estimated_tokens=tokens,
                    content=content,
                    is_changed=candidate.is_changed,
                    is_test=candidate.is_test,
                )
            )
            remaining -= tokens
            candidate_used = True
            consumed_lines.update(range(start, actual_end + 1))

            if actual_end < end:
                break

        omitted_lines = max(len(lines) - len(consumed_lines), 0)
        if not candidate_used:
            omitted.append(
                OmittedContext(
                    path=candidate.path,
                    reason=candidate.reason,
                    priority=candidate.priority,
                    language=candidate.language,
                    omitted_reason="budget_exhausted",
                    estimated_tokens=estimate_tokens("\n".join(lines)),
                    omitted_lines=len(lines),
                )
            )
        elif omitted_lines:
            omitted.append(
                OmittedContext(
                    path=candidate.path,
                    reason=candidate.reason,
                    priority=candidate.priority,
                    language=candidate.language,
                    omitted_reason="truncated",
                    estimated_tokens=estimate_tokens("\n".join(lines)),
                    omitted_lines=omitted_lines,
                )
            )

    used_tokens = sum(item.estimated_tokens for item in used)
    return ContextBudgetManifest(
        changed_paths=manifest.changed_paths,
        max_context_tokens=max_context_tokens,
        used_tokens=used_tokens,
        index_file_count=manifest.index_file_count,
        context_used=tuple(used),
        context_omitted=tuple(omitted),
    )


def _changed_line_map(parsed_diff: ParsedDiff) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for file in parsed_diff.files:
        path = normalize_path(file.path)
        if not path:
            continue
        line_numbers = _changed_lines_for_file(file)
        if line_numbers:
            result[path] = line_numbers
    return result


def _changed_lines_for_file(file: DiffFile) -> tuple[int, ...]:
    lines: list[int] = []
    for hunk in file.hunks:
        for line in hunk.lines:
            if line.kind == "added" and line.new_line_no is not None:
                lines.append(line.new_line_no)
            elif line.kind == "deleted":
                lines.append(hunk.new_start)
    return tuple(dict.fromkeys(lines))


def _read_candidate_lines(root: Path, candidate: ContextCandidate) -> list[str] | None:
    path = root / candidate.path
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def _candidate_ranges(
    candidate: ContextCandidate,
    lines: list[str],
    changed_lines: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    if not lines:
        return ((1, 1),)

    if candidate.is_changed and changed_lines:
        ranges = [
            (
                max(line_no - DEFAULT_CHANGED_LINE_RADIUS, 1),
                min(line_no + DEFAULT_CHANGED_LINE_RADIUS, len(lines)),
            )
            for line_no in changed_lines
        ]
        return _merge_ranges(tuple(ranges))

    return ((1, len(lines)),)


def _merge_ranges(ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not ranges:
        return ()
    ordered = sorted(ranges)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _fit_lines(lines: list[str], token_budget: int) -> list[str]:
    fitted: list[str] = []
    for line in lines:
        candidate = [*fitted, line]
        if estimate_tokens("\n".join(candidate)) > token_budget:
            break
        fitted = candidate
    return fitted
