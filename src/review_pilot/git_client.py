from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .models import RepoInfo


class GitError(RuntimeError):
    pass


class NotGitRepositoryError(GitError):
    pass


@dataclass(frozen=True)
class GitClient:
    cwd: Path

    @classmethod
    def from_cwd(cls, cwd: str | Path | None = None) -> GitClient:
        return cls(Path(cwd or ".").resolve())

    def repo_info(self) -> RepoInfo:
        root = self.repo_root()
        branch = self.current_branch()
        head = self.head_sha()
        return RepoInfo(
            root=str(root),
            branch=branch,
            head=head,
            has_staged_changes=self.has_staged_changes(),
            has_unstaged_changes=self.has_unstaged_changes(),
        )

    def repo_root(self) -> Path:
        result = self._run(["rev-parse", "--show-toplevel"], allow_not_git=True)
        if result.returncode != 0:
            raise NotGitRepositoryError(_clean_error(result.stderr) or "not a git repository")
        return Path(result.stdout.strip()).resolve()

    def current_branch(self) -> str:
        result = self._run(["branch", "--show-current"])
        branch = result.stdout.strip()
        if branch:
            return branch
        detached = self._run(["rev-parse", "--short", "HEAD"]).stdout.strip()
        return f"HEAD detached at {detached}"

    def head_sha(self) -> str:
        return self._run(["rev-parse", "HEAD"]).stdout.strip()

    def has_staged_changes(self) -> bool:
        result = self._run(["diff", "--staged", "--quiet"], check=False)
        return result.returncode == 1

    def has_unstaged_changes(self) -> bool:
        result = self._run(["diff", "--quiet"], check=False)
        return result.returncode == 1

    def staged_raw_diff(self) -> str:
        return self._run(["diff", "--staged", "--no-ext-diff", "--binary"]).stdout

    def _run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        allow_not_git: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=self.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and result.returncode != 0:
            if allow_not_git:
                return result
            raise GitError(_clean_error(result.stderr) or f"git {' '.join(args)} failed")
        return result


def _clean_error(stderr: str) -> str:
    return stderr.strip().splitlines()[-1] if stderr.strip() else ""
