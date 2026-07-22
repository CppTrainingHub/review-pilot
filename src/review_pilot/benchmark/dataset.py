from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


AACR_DATASET_SCHEMA_VERSION = "aacr-bench.v1"
AACR_DATASET_URL = "https://github.com/alibaba/aacr-bench"
_GITHUB_PR_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)/?$",
    re.IGNORECASE,
)


class DatasetError(ValueError):
    """Raised when an AACR-Bench dataset cannot be read as a JSON array."""


@dataclass(frozen=True)
class BenchmarkComment:
    note: str
    path: str
    side: str
    from_line: int | None
    to_line: int | None
    category: str
    context: str
    source_model: str | None = None
    is_ai_comment: bool | None = None

    @property
    def line_range(self) -> tuple[int, int] | None:
        if self.from_line is None and self.to_line is None:
            return None
        start = self.from_line or self.to_line
        end = self.to_line or self.from_line
        if start is None or end is None:
            return None
        return min(start, end), max(start, end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "note": self.note,
            "path": self.path,
            "side": self.side,
            "from_line": self.from_line,
            "to_line": self.to_line,
            "category": self.category,
            "context": self.context,
            "source_model": self.source_model,
            "is_ai_comment": self.is_ai_comment,
        }


@dataclass(frozen=True)
class AACRCase:
    case_id: str
    dataset_index: int
    pr_url: str
    repo_url: str
    source_commit: str
    target_commit: str
    project_main_language: str
    change_line_count: int | None
    comments: tuple[BenchmarkComment, ...]
    context_level: str
    category: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], dataset_index: int) -> "AACRCase":
        pr_url = _required_string(raw, "githubPrUrl")
        source_commit = _required_string(raw, "source_commit")
        target_commit = _required_string(raw, "target_commit")
        repo_url = _repo_url_from_raw(raw, pr_url)
        language = _required_string(raw, "project_main_language")
        raw_comments = raw.get("comments")
        if not isinstance(raw_comments, list):
            raise DatasetError("comments must be an array")

        comments: list[BenchmarkComment] = []
        for comment_index, value in enumerate(raw_comments):
            if not isinstance(value, dict):
                raise DatasetError(f"comments[{comment_index}] must be an object")
            comments.append(_parse_comment(value, comment_index))

        explicit_id = raw.get("case_id", raw.get("id"))
        case_id = (
            _required_string_value(explicit_id, "case_id")
            if explicit_id is not None
            else case_id_for(pr_url, source_commit, target_commit)
        )
        change_line_count = _optional_positive_int(
            raw.get("change_line_count"), "change_line_count"
        )
        raw_context_level = raw.get("context_level")
        context_level = _context_level(raw_context_level, comments)
        raw_category = raw.get("category")
        category = (
            _normalize_label(_required_string_value(raw_category, "category"))
            if raw_category is not None
            else None
        )
        return cls(
            case_id=case_id,
            dataset_index=dataset_index,
            pr_url=pr_url,
            repo_url=repo_url,
            source_commit=source_commit,
            target_commit=target_commit,
            project_main_language=language,
            change_line_count=change_line_count,
            comments=tuple(comments),
            context_level=context_level,
            category=category,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset_index": self.dataset_index,
            "githubPrUrl": self.pr_url,
            "repo_url": self.repo_url,
            "source_commit": self.source_commit,
            "target_commit": self.target_commit,
            "project_main_language": self.project_main_language,
            "change_line_count": self.change_line_count,
            "category": self.category,
            "context_level": self.context_level,
            "comments": [comment.to_dict() for comment in self.comments],
        }


@dataclass(frozen=True)
class DatasetIssue:
    dataset_index: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_index": self.dataset_index, "message": self.message}


@dataclass(frozen=True)
class DatasetLoadResult:
    path: str
    sha256: str
    cases: tuple[AACRCase, ...]
    issues: tuple[DatasetIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "case_count": len(self.cases),
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def case_id_for(pr_url: str, source_commit: str, target_commit: str) -> str:
    seed = "|".join((pr_url.strip(), source_commit.strip(), target_commit.strip()))
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def load_aacr_dataset(path: str | Path) -> DatasetLoadResult:
    dataset_path = Path(path)
    try:
        raw_bytes = dataset_path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"cannot read dataset {dataset_path}: {exc}") from exc

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"dataset must be UTF-8 JSON: {dataset_path}") from exc
    if not isinstance(payload, list):
        raise DatasetError("AACR-Bench dataset root must be an array")

    cases: list[AACRCase] = []
    issues: list[DatasetIssue] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            issues.append(DatasetIssue(index, "case must be an object"))
            continue
        try:
            cases.append(AACRCase.from_dict(raw, index))
        except (DatasetError, TypeError, ValueError) as exc:
            issues.append(DatasetIssue(index, str(exc)))

    return DatasetLoadResult(
        path=str(dataset_path),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        cases=tuple(cases),
        issues=tuple(issues),
    )


def normalize_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise DatasetError("comment path must be a non-empty string")
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith(("a/", "b/")):
        normalized = normalized[2:]
    if normalized.startswith("/"):
        raise DatasetError(f"comment path must be relative: {path!r}")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise DatasetError(f"comment path cannot escape repository: {path!r}")
    return "/".join(part for part in parts if part not in ("", "."))


def _parse_comment(raw: dict[str, Any], index: int) -> BenchmarkComment:
    note = _required_string(raw, "note", prefix=f"comments[{index}].")
    path = normalize_path(_required_string(raw, "path", prefix=f"comments[{index}]."))
    side = raw.get("side", "right")
    if not isinstance(side, str) or side.strip().lower() not in {"left", "right"}:
        raise DatasetError(f"comments[{index}].side must be 'left' or 'right'")
    from_line = _optional_positive_int(
        raw.get("from_line"), f"comments[{index}].from_line"
    )
    to_line = _optional_positive_int(
        raw.get("to_line"), f"comments[{index}].to_line"
    )
    if from_line is None and to_line is not None:
        from_line = to_line
    if to_line is None and from_line is not None:
        to_line = from_line
    if from_line is not None and to_line is not None and from_line > to_line:
        from_line, to_line = to_line, from_line
    category = _normalize_label(
        _required_string(raw, "category", prefix=f"comments[{index}].")
    )
    context = _normalize_label(
        _required_string(raw, "context", prefix=f"comments[{index}].")
    )
    source_model = raw.get("source_model")
    if source_model is not None and not isinstance(source_model, str):
        raise DatasetError(f"comments[{index}].source_model must be a string")
    is_ai_comment = raw.get("is_ai_comment")
    if is_ai_comment is not None and not isinstance(is_ai_comment, bool):
        raise DatasetError(f"comments[{index}].is_ai_comment must be boolean")
    return BenchmarkComment(
        note=note,
        path=path,
        side=side.strip().lower(),
        from_line=from_line,
        to_line=to_line,
        category=category,
        context=context,
        source_model=source_model,
        is_ai_comment=is_ai_comment,
    )


def _repo_url_from_raw(raw: dict[str, Any], pr_url: str) -> str:
    explicit = raw.get("repo_url") or raw.get("repository_url")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise DatasetError("repo_url must be a non-empty string")
        return explicit.strip()
    match = _GITHUB_PR_RE.match(pr_url.strip())
    if match is not None:
        return f"https://github.com/{match.group('owner')}/{match.group('repo')}.git"
    parsed = urlparse(pr_url)
    if parsed.scheme and parsed.netloc:
        raise DatasetError(f"cannot derive repository URL from PR URL: {pr_url!r}")
    raise DatasetError(f"githubPrUrl must be a GitHub PR URL: {pr_url!r}")


def _context_level(raw_context: Any, comments: list[BenchmarkComment]) -> str:
    if raw_context is not None:
        return _normalize_label(_required_string_value(raw_context, "context_level"))
    contexts = sorted({comment.context for comment in comments if comment.context})
    if len(contexts) == 1:
        return contexts[0]
    return "mixed" if contexts else "unknown"


def _normalize_label(value: str) -> str:
    return " ".join(value.strip().replace("_", " ").split()).lower()


def _required_string(raw: dict[str, Any], key: str, *, prefix: str = "") -> str:
    if key not in raw:
        raise DatasetError(f"{prefix}{key} is required")
    return _required_string_value(raw[key], f"{prefix}{key}")


def _required_string_value(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_positive_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DatasetError(f"{key} must be a positive integer or null")
    return value
