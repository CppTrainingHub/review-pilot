from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from review_pilot.config import ReviewPilotConfig
from review_pilot.models import DiffFile, ParsedDiff, RepoInfo
from review_pilot.report_models import Finding


@dataclass(frozen=True)
class RuleMetadata:
    rule_id: str
    name: str
    description: str
    category: str
    default_severity: str


@dataclass(frozen=True)
class RuleContext:
    parsed_diff: ParsedDiff
    repo_info: RepoInfo | None = None
    config: ReviewPilotConfig | None = None

    @property
    def review_files(self) -> tuple[DiffFile, ...]:
        if self.config is None:
            return self.parsed_diff.files
        return tuple(
            diff_file
            for diff_file in self.parsed_diff.files
            if diff_file.path and not self.config.is_ignored(diff_file.path)
        )

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(file.path for file in self.review_files if file.path)


class Rule(Protocol):
    metadata: RuleMetadata

    def run(self, context: RuleContext) -> list[Finding]:
        ...
