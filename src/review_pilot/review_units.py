"""Deterministic grouping of a diff into auditable review units.

The grouping rules stay deterministic.  The model receives
the units after they have been built; it never decides which files belong
together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

from .code_index import build_code_index
from .config import ReviewPilotConfig
from .context_selector import select_context_candidates
from .language_detection import normalize_path
from .models import (
    CodeIndex,
    ContextCandidate,
    DiffFile,
    ParsedDiff,
)


REASON_LABELS = {
    "changed_file": "变更文件",
    "related_test": "相关测试",
    "local_import": "直接 import 的本地文件",
    "same_directory_related": "同目录接口或实现",
    "project_doc": "项目说明文档",
    "project_config": "项目配置文件",
}
GLOBAL_REASONS = {"project_doc", "project_config"}
MAX_UNIT_DIFF_CHARS = 60_000
MAX_UNIT_CHANGED_FILES = 24


@dataclass(frozen=True)
class ReviewUnit:
    """A stable, auditable group of changed files and supporting context."""

    unit_id: str
    changed_files: tuple[str, ...]
    context_files: tuple[str, ...]
    reasons: Mapping[str, tuple[str, ...]]
    budget_tokens: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError("unit_id must be a non-empty string")
        if not self.changed_files:
            raise ValueError("review unit must contain at least one changed file")
        if len(set(self.changed_files)) != len(self.changed_files):
            raise ValueError(f"review unit {self.unit_id} has duplicate changed files")
        if len(set(self.context_files)) != len(self.context_files):
            raise ValueError(f"review unit {self.unit_id} has duplicate context files")
        overlap = set(self.changed_files) & set(self.context_files)
        if overlap:
            raise ValueError(
                f"review unit {self.unit_id} puts changed files in context_files: "
                f"{sorted(overlap)}"
            )
        if self.budget_tokens < 1:
            raise ValueError("review unit budget_tokens must be positive")
        expected_paths = set(self.changed_files) | set(self.context_files)
        unknown_reason_paths = set(self.reasons) - expected_paths
        if unknown_reason_paths:
            raise ValueError(
                f"review unit {self.unit_id} has reasons for unknown files: "
                f"{sorted(unknown_reason_paths)}"
            )

    @property
    def all_files(self) -> tuple[str, ...]:
        return (*self.changed_files, *self.context_files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "changed_files": list(self.changed_files),
            "context_files": list(self.context_files),
            "reasons": {
                path: list(self.reasons.get(path, ()))
                for path in self.all_files
            },
            "budget_tokens": self.budget_tokens,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReviewPlan:
    """The complete deterministic plan for one parsed diff."""

    units: tuple[ReviewUnit, ...]
    changed_files: tuple[str, ...]
    max_context_tokens: int

    def __post_init__(self) -> None:
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if not self.units:
            raise ValueError("review plan must contain at least one unit")
        validate_review_plan(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_files": list(self.changed_files),
            "max_context_tokens": self.max_context_tokens,
            "unit_count": len(self.units),
            "review_units": [unit.to_dict() for unit in self.units],
        }


def build_review_plan(
    parsed_diff: ParsedDiff,
    index: CodeIndex,
    max_context_tokens: int,
) -> ReviewPlan:
    """Build a stable plan from changed files and indexed relationships."""

    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive")

    diff_by_path: dict[str, DiffFile] = {}
    for diff_file in parsed_diff.files:
        path = normalize_path(diff_file.path)
        if not path:
            raise ValueError("changed file path is required")
        if path in diff_by_path:
            raise ValueError(f"duplicate changed file path: {path}")
        diff_by_path[path] = diff_file

    changed_files = tuple(sorted(diff_by_path))
    if not changed_files:
        raise ValueError("review plan requires at least one changed file")

    root_candidates: dict[str, dict[str, ContextCandidate]] = {}
    for path in changed_files:
        single_file_diff = ParsedDiff(files=(diff_by_path[path],))
        manifest = select_context_candidates(single_file_diff, index)
        candidates = {candidate.path: candidate for candidate in manifest.candidates}
        if path not in candidates:
            candidates[path] = ContextCandidate(
                path=path,
                reason="changed_file",
                priority=0,
                language="unknown",
                is_changed=True,
            )
        root_candidates[path] = candidates

    global_candidates = _global_candidates(index, parsed_diff)
    components = _connected_components(
        changed_files,
        root_candidates,
        global_candidates,
    )
    components = _rebalance_components(components, diff_by_path)
    budgets = _allocate_budgets(max_context_tokens, len(components))

    units: list[ReviewUnit] = []
    for index_number, roots in enumerate(components, start=1):
        component_candidates: dict[str, ContextCandidate] = {}
        for root in roots:
            for candidate in root_candidates[root].values():
                _merge_candidate(component_candidates, candidate)
        for candidate in global_candidates.values():
            _merge_candidate(component_candidates, candidate)

        component_changed = tuple(sorted(roots))
        context_candidates = {
            path: candidate
            for path, candidate in component_candidates.items()
            if path not in component_changed
        }
        context_files = tuple(
            candidate.path
            for candidate in sorted(
                context_candidates.values(),
                key=lambda item: (item.priority, item.reason, item.path),
            )
        )
        all_candidates = {
            path: candidate
            for path, candidate in component_candidates.items()
        }
        reasons = {
            path: _human_reasons(all_candidates[path].reason)
            for path in (*component_changed, *context_files)
        }
        units.append(
            ReviewUnit(
                unit_id=f"unit-{index_number:03d}",
                changed_files=component_changed,
                context_files=context_files,
                reasons=reasons,
                budget_tokens=budgets[index_number - 1],
                metadata={
                    "root_files": list(component_changed),
                    "shared_context_files": [
                        path
                        for path in context_files
                        if _is_global_reason(all_candidates[path].reason)
                    ],
                },
            )
        )

    plan = ReviewPlan(
        units=tuple(units),
        changed_files=changed_files,
        max_context_tokens=max_context_tokens,
    )
    return plan


def build_review_units(
    parsed_diff: ParsedDiff,
    index: CodeIndex,
    max_context_tokens: int,
) -> ReviewPlan:
    """Compatibility alias with an explicit plural name."""

    return build_review_plan(parsed_diff, index, max_context_tokens)


def plan_review_units(
    parsed_diff: ParsedDiff,
    index: CodeIndex,
    max_context_tokens: int,
) -> ReviewPlan:
    """Compatibility alias used by callers that treat planning as an action."""

    return build_review_plan(parsed_diff, index, max_context_tokens)


def validate_review_plan(plan: ReviewPlan) -> None:
    """Check that every changed file has exactly one owner."""

    plan_changed = tuple(plan.changed_files)
    if len(set(plan_changed)) != len(plan_changed):
        raise ValueError("review plan has duplicate changed files")
    owned: list[str] = []
    for unit in plan.units:
        owned.extend(unit.changed_files)
    if sorted(owned) != sorted(plan_changed):
        missing = sorted(set(plan_changed) - set(owned))
        duplicate = sorted(path for path in set(owned) if owned.count(path) > 1)
        raise ValueError(
            f"review plan changed file ownership mismatch; missing={missing}, "
            f"duplicate={duplicate}"
        )
    for unit in plan.units:
        for path in unit.all_files:
            if not normalize_path(path):
                raise ValueError(f"review unit {unit.unit_id} contains an empty path")


def _connected_components(
    changed_files: tuple[str, ...],
    root_candidates: Mapping[str, Mapping[str, ContextCandidate]],
    global_candidates: Mapping[str, ContextCandidate],
) -> list[tuple[str, ...]]:
    parent = {path: path for path in changed_files}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        parent[right_root] = left_root

    owners: dict[str, str] = {}
    global_paths = set(global_candidates)
    for root in changed_files:
        for candidate_path, candidate in root_candidates[root].items():
            if candidate_path in global_paths:
                continue
            if not _is_grouping_relation(candidate.reason):
                continue
            previous = owners.get(candidate_path)
            if previous is None:
                owners[candidate_path] = root
            else:
                union(root, previous)

    grouped: dict[str, list[str]] = {}
    for path in changed_files:
        grouped.setdefault(find(path), []).append(path)
    return [tuple(sorted(paths)) for paths in sorted(grouped.values(), key=lambda items: items[0])]


def _rebalance_components(
    components: list[tuple[str, ...]],
    diff_by_path: Mapping[str, DiffFile],
) -> list[tuple[str, ...]]:
    """Keep related roots together while bounding one provider request.

    A large PR can contain many unrelated files in one directory.  Putting
    all of them into one unit recreates the oversized whole-diff prompt that
    this policy is meant to avoid.  We therefore pack deterministic
    directory-local components up to a small file and serialized-diff limit.
    """

    by_directory: dict[str, list[tuple[str, ...]]] = {}
    for component in components:
        directory = str(PurePosixPath(component[0]).parent)
        by_directory.setdefault(directory, []).append(component)

    balanced: list[tuple[str, ...]] = []
    for directory in sorted(by_directory):
        directory_components = by_directory[directory]
        directory_chars = sum(
            _serialized_diff_size(diff_by_path[path])
            for component in directory_components
            for path in component
        )
        directory_files = sum(len(component) for component in directory_components)
        if (
            directory_files <= MAX_UNIT_CHANGED_FILES
            and directory_chars <= MAX_UNIT_DIFF_CHARS
        ):
            balanced.extend(directory_components)
            continue
        bucket: list[str] = []
        bucket_chars = 0
        for component in directory_components:
            component_chars = sum(
                _serialized_diff_size(diff_by_path[path])
                for path in component
            )
            too_large = (
                bucket
                and (
                    len(bucket) + len(component) > MAX_UNIT_CHANGED_FILES
                    or bucket_chars + component_chars > MAX_UNIT_DIFF_CHARS
                )
            )
            if too_large:
                balanced.append(tuple(bucket))
                bucket = []
                bucket_chars = 0
            bucket.extend(component)
            bucket_chars += component_chars
            if len(bucket) >= MAX_UNIT_CHANGED_FILES:
                balanced.append(tuple(bucket))
                bucket = []
                bucket_chars = 0
        if bucket:
            balanced.append(tuple(bucket))
    return sorted(
        (tuple(sorted(component)) for component in balanced),
        key=lambda component: component[0],
    )


def _serialized_diff_size(diff_file: DiffFile) -> int:
    return len(json.dumps(diff_file.to_dict(), ensure_ascii=False))


def _global_candidates(
    index: CodeIndex,
    parsed_diff: ParsedDiff,
) -> dict[str, ContextCandidate]:
    manifest = select_context_candidates(parsed_diff, index)
    return {
        candidate.path: candidate
        for candidate in manifest.candidates
        if _is_global_reason(candidate.reason)
    }


def _is_global_reason(reason: str) -> bool:
    return bool(set(reason.split(",")) & GLOBAL_REASONS)


def _is_grouping_relation(reason: str) -> bool:
    return bool(
        set(reason.split(","))
        & {"related_test", "same_directory_related"}
    )


def _merge_candidate(
    candidates: dict[str, ContextCandidate],
    candidate: ContextCandidate,
) -> None:
    current = candidates.get(candidate.path)
    if current is None or candidate.priority < current.priority:
        candidates[candidate.path] = candidate
        return
    if current.priority == candidate.priority and current.reason != candidate.reason:
        reasons = tuple(dict.fromkeys((*current.reason.split(","), *candidate.reason.split(","))))
        candidates[candidate.path] = ContextCandidate(
            path=current.path,
            reason=",".join(reasons),
            priority=current.priority,
            language=current.language,
            is_changed=current.is_changed or candidate.is_changed,
            is_test=current.is_test or candidate.is_test,
            matched_symbols=tuple(dict.fromkeys((*current.matched_symbols, *candidate.matched_symbols))),
            matched_imports=tuple(dict.fromkeys((*current.matched_imports, *candidate.matched_imports))),
        )


def _human_reasons(reason: str) -> tuple[str, ...]:
    return tuple(
        REASON_LABELS.get(item, item)
        for item in reason.split(",")
        if item
    )


def _allocate_budgets(total: int, count: int) -> tuple[int, ...]:
    base, remainder = divmod(total, count)
    return tuple(base + (1 if index < remainder else 0) for index in range(count))
