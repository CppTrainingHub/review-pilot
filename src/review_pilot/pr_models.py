from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .diff_parser import parse_unified_diff
from .models import ParsedDiff, RawDiff


@dataclass(frozen=True)
class PullRequestRef:
    label: str
    ref: str
    sha: str
    repo_full_name: str
    repo_clone_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ref": self.ref,
            "sha": self.sha,
            "repo_full_name": self.repo_full_name,
            "repo_clone_url": self.repo_clone_url,
        }


@dataclass(frozen=True)
class PullRequestFile:
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str | None = None
    previous_filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "filename": self.filename,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "changes": self.changes,
            "has_patch": self.patch is not None,
        }
        if self.previous_filename:
            payload["previous_filename"] = self.previous_filename
        return payload


@dataclass(frozen=True)
class PullRequestInfo:
    provider: str
    url: str
    owner: str
    repo: str
    number: int
    title: str
    state: str
    base: PullRequestRef
    head: PullRequestRef
    files: tuple[PullRequestFile, ...]

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def raw_diff(self) -> RawDiff:
        return build_pull_request_raw_diff(self.files)

    @property
    def parsed_diff(self) -> ParsedDiff:
        return parse_unified_diff(self.raw_diff)

    def to_dict(self) -> dict[str, Any]:
        parsed = self.parsed_diff
        return {
            "provider": self.provider,
            "url": self.url,
            "owner": self.owner,
            "repo": self.repo,
            "full_name": self.full_name,
            "number": self.number,
            "title": self.title,
            "state": self.state,
            "base": self.base.to_dict(),
            "head": self.head.to_dict(),
            "files": [item.to_dict() for item in self.files],
            "diff": {
                "file_count": len(parsed.files),
                "changed_paths": [file.path for file in parsed.files],
            },
        }


def build_pull_request_raw_diff(files: tuple[PullRequestFile, ...]) -> RawDiff:
    chunks: list[str] = []
    for item in files:
        if not item.patch:
            continue

        old_path = item.previous_filename or item.filename
        new_path = item.filename
        chunks.append(f"diff --git a/{old_path} b/{new_path}")

        if item.status == "added":
            chunks.append("new file mode 100644")
            chunks.append("--- /dev/null")
            chunks.append(f"+++ b/{new_path}")
        elif item.status == "removed":
            chunks.append("deleted file mode 100644")
            chunks.append(f"--- a/{old_path}")
            chunks.append("+++ /dev/null")
        elif item.status == "renamed":
            chunks.append("similarity index 100%")
            chunks.append(f"rename from {old_path}")
            chunks.append(f"rename to {new_path}")
            chunks.append(f"--- a/{old_path}")
            chunks.append(f"+++ b/{new_path}")
        else:
            chunks.append(f"--- a/{old_path}")
            chunks.append(f"+++ b/{new_path}")
        chunks.append(item.patch.rstrip("\n"))

    text = "\n".join(chunks)
    if text:
        text += "\n"
    return RawDiff(text=text)
