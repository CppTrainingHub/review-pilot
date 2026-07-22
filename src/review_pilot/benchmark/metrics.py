from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dataset import BenchmarkComment


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CATEGORY_ALIASES = {
    "security": "security",
    "defect": "bug",
    "bug": "bug",
    "maintainability": "maintainability",
    "performance": "performance",
    "style": "style",
    "test": "test",
}


@dataclass(frozen=True)
class FindingMatch:
    predicted_index: int
    expected_index: int
    similarity: float
    line_match: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_index": self.predicted_index,
            "expected_index": self.expected_index,
            "similarity": round(self.similarity, 4),
            "line_match": self.line_match,
        }


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    language: str
    context_level: str
    expected_comments: tuple[BenchmarkComment, ...]
    predicted_findings: tuple[dict[str, Any], ...]
    matches: tuple[FindingMatch, ...]

    @property
    def expected_count(self) -> int:
        return len(self.expected_comments)

    @property
    def generated_count(self) -> int:
        return len(self.predicted_findings)

    @property
    def matched_count(self) -> int:
        return len(self.matches)

    @property
    def line_match_count(self) -> int:
        return sum(match.line_match for match in self.matches)

    def metric_dict(self, *, latency_ms: float = 0.0, token_cost_usd: float | None = None) -> dict[str, Any]:
        return metrics_from_counts(
            expected_count=self.expected_count,
            generated_count=self.generated_count,
            matched_count=self.matched_count,
            line_match_count=self.line_match_count,
            latency_ms=latency_ms,
            token_cost_usd=token_cost_usd,
        )

    def to_dict(self, *, latency_ms: float = 0.0, token_cost_usd: float | None = None) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "language": self.language,
            "context_level": self.context_level,
            "expected_comments": [comment.to_dict() for comment in self.expected_comments],
            "predicted_findings": list(self.predicted_findings),
            "matches": [match.to_dict() for match in self.matches],
            "metrics": self.metric_dict(
                latency_ms=latency_ms,
                token_cost_usd=token_cost_usd,
            ),
        }


def case_evaluation_from_dict(payload: Mapping[str, Any]) -> CaseEvaluation:
    expected_comments = tuple(
        BenchmarkComment(
            note=str(item["note"]),
            path=str(item["path"]),
            side=str(item.get("side", "right")),
            from_line=item.get("from_line"),
            to_line=item.get("to_line"),
            category=str(item.get("category", "unknown")),
            context=str(item.get("context", "unknown")),
            source_model=item.get("source_model"),
            is_ai_comment=item.get("is_ai_comment"),
        )
        for item in payload.get("expected_comments", [])
    )
    matches = tuple(
        FindingMatch(
            predicted_index=int(item["predicted_index"]),
            expected_index=int(item["expected_index"]),
            similarity=float(item["similarity"]),
            line_match=bool(item["line_match"]),
        )
        for item in payload.get("matches", [])
    )
    return CaseEvaluation(
        case_id=str(payload["case_id"]),
        language=str(payload["language"]),
        context_level=str(payload["context_level"]),
        expected_comments=expected_comments,
        predicted_findings=tuple(
            dict(item) for item in payload.get("predicted_findings", [])
        ),
        matches=matches,
    )


def evaluate_case(
    *,
    case_id: str,
    language: str,
    context_level: str,
    expected_comments: Sequence[BenchmarkComment],
    predicted_findings: Sequence[Mapping[str, Any]],
) -> CaseEvaluation:
    expected = tuple(expected_comments)
    predicted = tuple(dict(item) for item in predicted_findings)
    candidates: list[tuple[float, int, int, bool]] = []
    for predicted_index, finding in enumerate(predicted):
        for expected_index, comment in enumerate(expected):
            similarity = _match_similarity(finding, comment)
            if similarity is None:
                continue
            line_match = _line_overlaps(finding, comment)
            candidates.append((similarity, predicted_index, expected_index, line_match))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_expected: set[int] = set()
    matches: list[FindingMatch] = []
    for similarity, predicted_index, expected_index, line_match in candidates:
        if predicted_index in used_predictions or expected_index in used_expected:
            continue
        used_predictions.add(predicted_index)
        used_expected.add(expected_index)
        matches.append(
            FindingMatch(
                predicted_index=predicted_index,
                expected_index=expected_index,
                similarity=similarity,
                line_match=line_match,
            )
        )

    return CaseEvaluation(
        case_id=case_id,
        language=language,
        context_level=context_level,
        expected_comments=expected,
        predicted_findings=predicted,
        matches=tuple(matches),
    )


def normalize_finding(
    finding: Mapping[str, Any],
    *,
    context: str = "diff",
) -> dict[str, Any]:
    """Convert a review-pilot finding to the AACR-style comment shape."""

    raw_path = finding.get("file_path")
    path = _normalized_finding_path(raw_path) if isinstance(raw_path, str) else ""
    raw_line = finding.get("line_no")
    line = raw_line if isinstance(raw_line, int) and raw_line > 0 else None
    return {
        "path": path,
        "side": "right",
        "from_line": line,
        "to_line": line,
        "note": str(finding.get("message", "")),
        "category": _prediction_category(finding),
        "context": context,
    }


def metrics_from_counts(
    *,
    expected_count: int,
    generated_count: int,
    matched_count: int,
    line_match_count: int,
    latency_ms: float = 0.0,
    token_cost_usd: float | None = None,
) -> dict[str, Any]:
    precision = _ratio(matched_count, generated_count)
    recall = _ratio(matched_count, expected_count)
    f1 = _f1(precision, recall)
    line_precision = _ratio(line_match_count, generated_count)
    line_recall = _ratio(line_match_count, expected_count)
    line_accuracy = _ratio(line_match_count, matched_count)
    noise_rate = _ratio(generated_count - matched_count, generated_count)
    return {
        "expected_comments": expected_count,
        "generated_comments": generated_count,
        "positive_matches": matched_count,
        "line_matches": line_match_count,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "line_precision": round(line_precision, 6),
        "line_recall": round(line_recall, 6),
        "line_accuracy": round(line_accuracy, 6),
        "noise_rate": round(noise_rate, 6),
        "latency_ms": round(latency_ms, 3),
        "token_cost_usd": (
            round(token_cost_usd, 8) if token_cost_usd is not None else None
        ),
    }


def aggregate_evaluations(
    evaluations: Sequence[CaseEvaluation],
    *,
    latency_by_case: Mapping[str, float] | None = None,
    token_cost_by_case: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    latency_by_case = latency_by_case or {}
    token_cost_by_case = token_cost_by_case or {}
    total_latency = sum(latency_by_case.get(item.case_id, 0.0) for item in evaluations)
    token_costs = [
        token_cost_by_case[item.case_id]
        for item in evaluations
        if token_cost_by_case.get(item.case_id) is not None
    ]
    metrics = metrics_from_counts(
        expected_count=sum(item.expected_count for item in evaluations),
        generated_count=sum(item.generated_count for item in evaluations),
        matched_count=sum(item.matched_count for item in evaluations),
        line_match_count=sum(item.line_match_count for item in evaluations),
        latency_ms=(total_latency / len(evaluations)) if evaluations else 0.0,
        token_cost_usd=sum(token_costs) if token_costs else None,
    )
    metrics["cases"] = len(evaluations)
    metrics["average_comments"] = round(
        _ratio(metrics["generated_comments"], len(evaluations)), 6
    )
    metrics["average_token_cost_usd"] = (
        round(_ratio(sum(token_costs), len(token_costs)), 8)
        if token_costs
        else None
    )
    return metrics


def grouped_metrics(
    evaluations: Sequence[CaseEvaluation],
    dimension: str,
    *,
    latency_by_case: Mapping[str, float] | None = None,
    token_cost_by_case: Mapping[str, float | None] | None = None,
) -> dict[str, dict[str, Any]]:
    if dimension not in {"language", "context_level", "category"}:
        raise ValueError(f"unsupported benchmark dimension: {dimension}")
    if dimension != "category":
        groups: dict[str, list[CaseEvaluation]] = {}
        for evaluation in evaluations:
            key = getattr(evaluation, dimension)
            groups.setdefault(key, []).append(evaluation)
        return {
            key: aggregate_evaluations(
                value,
                latency_by_case=latency_by_case,
                token_cost_by_case=token_cost_by_case,
            )
            for key, value in sorted(groups.items())
        }

    category_counts: dict[str, list[int]] = {}
    for evaluation in evaluations:
        for comment in evaluation.expected_comments:
            bucket = category_counts.setdefault(comment.category, [0, 0, 0, 0])
            bucket[0] += 1
        for finding in evaluation.predicted_findings:
            category = _prediction_category(finding)
            bucket = category_counts.setdefault(category, [0, 0, 0, 0])
            bucket[1] += 1
        for match in evaluation.matches:
            category = evaluation.expected_comments[match.expected_index].category
            bucket = category_counts.setdefault(category, [0, 0, 0, 0])
            bucket[2] += 1
            bucket[3] += int(match.line_match)

    output: dict[str, dict[str, Any]] = {}
    for category, (expected, generated, matched, line_matches) in sorted(category_counts.items()):
        category_cases = [
            evaluation
            for evaluation in evaluations
            if any(comment.category == category for comment in evaluation.expected_comments)
            or any(_prediction_category(item) == category for item in evaluation.predicted_findings)
        ]
        latency = sum(
            (latency_by_case or {}).get(item.case_id, 0.0)
            for item in category_cases
        )
        costs = [
            (token_cost_by_case or {}).get(item.case_id)
            for item in category_cases
            if (token_cost_by_case or {}).get(item.case_id) is not None
        ]
        output[category] = metrics_from_counts(
            expected_count=expected,
            generated_count=generated,
            matched_count=matched,
            line_match_count=line_matches,
            latency_ms=(latency / len(category_cases)) if category_cases else 0.0,
            token_cost_usd=sum(costs) if costs else None,
        ) | {"cases": len(category_cases)}
    return output


def _match_similarity(
    finding: Mapping[str, Any],
    comment: BenchmarkComment,
) -> float | None:
    path = finding.get("file_path")
    if not isinstance(path, str) or _normalized_finding_path(path) != comment.path:
        return None
    message = finding.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    message_tokens = _tokens(message)
    note_tokens = _tokens(comment.note)
    if not message_tokens or not note_tokens:
        return None
    intersection = len(message_tokens & note_tokens)
    union = len(message_tokens | note_tokens)
    jaccard = intersection / union if union else 0.0
    containment = intersection / min(len(message_tokens), len(note_tokens))
    if _normalize_text(message) == _normalize_text(comment.note):
        return 1.0
    category_bonus = 0.1 if _categories_compatible(finding.get("category"), comment.category) else 0.0
    similarity = min(1.0, max(jaccard, containment * 0.75) + category_bonus)
    if similarity < 0.2:
        return None
    return similarity


def _line_overlaps(finding: Mapping[str, Any], comment: BenchmarkComment) -> bool:
    line = finding.get("line_no")
    if not isinstance(line, int) or line < 1:
        return False
    line_range = comment.line_range
    return line_range is not None and line_range[0] <= line <= line_range[1]


def _normalized_finding_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith(("a/", "b/")):
        normalized = normalized[2:]
    return normalized.strip("/")


def _categories_compatible(predicted: Any, expected: str) -> bool:
    if not isinstance(predicted, str):
        return False
    return _CATEGORY_ALIASES.get(predicted.strip().lower(), predicted.strip().lower()) == expected


def _prediction_category(finding: Mapping[str, Any]) -> str:
    category = finding.get("category")
    return str(category).strip().lower() if category else "unknown"


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
