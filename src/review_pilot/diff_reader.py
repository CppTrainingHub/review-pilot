from __future__ import annotations

from dataclasses import dataclass

from .diff_parser import parse_unified_diff
from .git_client import GitClient
from .models import ParsedDiff, RawDiff


@dataclass(frozen=True)
class DiffReader:
    git: GitClient

    def staged_raw_diff(self) -> RawDiff:
        return RawDiff(self.git.staged_raw_diff())

    def staged_parsed_diff(self) -> ParsedDiff:
        return parse_unified_diff(self.staged_raw_diff())
