from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SEVERITY_RANK = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
}

VALID_SEVERITIES = set(SEVERITY_RANK)
VALID_CATEGORIES = {
    "size",
    "test",
    "security",
    "style",
    "bug",
    "maintainability",
    "other",
}
VALID_SOURCES = {"rule", "semgrep", "llm", "simple-check"}
VALID_CONFIDENCES = {"high", "medium", "low"}


def _validate_enum(name: str, value: str, valid: set[str]) -> None:
    if value not in valid:
        raise ValueError(f"invalid {name}: {value!r}; expected one of {sorted(valid)}")


@dataclass(frozen=True)
class Finding:
    """A single review finding that can be rendered in reports or sent to CI."""

    message: str
    file_path: str | None = None
    line_no: int | None = None
    severity: str = "P2"
    category: str = "other"
    source: str = "simple-check"
    confidence: str = "high"
    rule_id: str | None = None
    evidence: dict[str, Any] | None = None
    suggestion: str | None = None

    def __post_init__(self) -> None:
        if self.file_path is not None and not self.file_path:
            raise ValueError("file_path must be a non-empty string or None")
        if self.line_no is not None and self.line_no < 1:
            raise ValueError("line_no must be a positive integer or None")
        _validate_enum("severity", self.severity, VALID_SEVERITIES)
        _validate_enum("category", self.category, VALID_CATEGORIES)
        _validate_enum("source", self.source, VALID_SOURCES)
        _validate_enum("confidence", self.confidence, VALID_CONFIDENCES)

    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANK[self.severity]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Finding":
        allowed = {f for f in cls.__dataclass_fields__}
        extra = set(data) - allowed
        if extra:
            raise ValueError(f"unexpected finding fields: {sorted(extra)}")
        return cls(**data)


@dataclass(frozen=True)
class ContextRecord:
    """A snippet of repository context included in the review input."""

    file_path: str
    content: str
    start_line: int = 1
    end_line: int | None = None
    relevance: str = "medium"
    omitted_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.file_path:
            raise ValueError("file_path must be a non-empty string")
        if self.start_line < 1:
            raise ValueError("start_line must be a positive integer")
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewReport:
    """Structured container for all review outputs."""

    findings: list[Finding] = field(default_factory=list)
    context_records: list[ContextRecord] = field(default_factory=list)
    repo_info: dict[str, Any] | None = None
    config_source: str | None = None
    merge_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "merge_summary": self.merge_summary,
            "findings": [f.to_dict() for f in self.findings],
            "context_records": [c.to_dict() for c in self.context_records],
            "repo_info": self.repo_info,
            "config_source": self.config_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewReport":
        findings = [Finding.from_dict(f) for f in data.get("findings", [])]
        context_records = [
            ContextRecord(**c) for c in data.get("context_records", [])
        ]
        return cls(
            findings=findings,
            context_records=context_records,
            repo_info=data.get("repo_info"),
            config_source=data.get("config_source"),
            merge_summary=data.get("merge_summary"),
        )

    @property
    def summary(self) -> dict[str, Any]:
        from .report_summary import build_report_summary

        return build_report_summary(self.findings).to_dict()

    def sort_findings(self) -> None:
        """Sort findings in place by severity, then stable location fields."""
        from .finding_normalizer import sort_key

        self.findings.sort(key=sort_key)
