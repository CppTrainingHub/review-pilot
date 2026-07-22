from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..diff_parser import parse_unified_diff
from ..models import ParsedDiff, RepoInfo
from .dataset import AACRCase


class RepositoryError(RuntimeError):
    """Raised when a benchmark repository cannot be prepared."""


@dataclass(frozen=True)
class PreparedCase:
    repo_info: RepoInfo
    parsed_diff: ParsedDiff
    workspace_path: str
    repository_source: str


@dataclass
class RepositoryCache:
    cache_root: Path
    workspace_root: Path
    _failed_repositories: set[str] = field(default_factory=set, init=False, repr=False)
    _git_unavailable_repositories: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def prepare(self, case: AACRCase) -> PreparedCase:
        if case.repo_url in self._failed_repositories:
            raise RepositoryError(
                f"repository preparation already failed in this run: {case.repo_url}"
            )
        if (
            case.repo_url in self._git_unavailable_repositories
            and _github_pr_coordinates(case.pr_url) is not None
        ):
            return self._prepare_github_patch(case)
        try:
            return self._prepare_git(case)
        except RepositoryError as git_error:
            if _github_pr_coordinates(case.pr_url) is not None:
                self._git_unavailable_repositories.add(case.repo_url)
                try:
                    return self._prepare_github_patch(case)
                except RepositoryError as fallback_error:
                    self._failed_repositories.add(case.repo_url)
                    raise RepositoryError(
                        f"git preparation failed: {git_error}; "
                        f"GitHub patch fallback failed: {fallback_error}"
                    ) from fallback_error
            self._failed_repositories.add(case.repo_url)
            raise

    def _prepare_git(self, case: AACRCase) -> PreparedCase:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        cache_repo = self._ensure_clone(case.repo_url)
        self._ensure_commit(cache_repo, case.target_commit)
        self._ensure_commit(cache_repo, case.source_commit)
        workspace = self._ensure_worktree(
            cache_repo,
            case.case_id,
            case.source_commit,
        )
        merge_base = self._run(
            ["git", "merge-base", case.target_commit, case.source_commit],
            cwd=workspace,
        ).stdout.strip()
        if not merge_base:
            raise RepositoryError(
                f"cannot find merge base between {case.target_commit} and {case.source_commit}"
            )
        diff = self._run(
            [
                "git",
                "-c",
                "core.quotePath=false",
                "diff",
                "--no-ext-diff",
                merge_base,
                case.source_commit,
            ],
            cwd=workspace,
        ).stdout
        parsed_diff = parse_unified_diff(diff)
        if parsed_diff.is_empty:
            raise RepositoryError(
                f"empty diff between {case.target_commit} and {case.source_commit}"
            )
        repo_info = RepoInfo(
            root=str(workspace),
            branch=f"aacr/{case.case_id}",
            head=case.source_commit,
            has_staged_changes=False,
            has_unstaged_changes=False,
        )
        return PreparedCase(
            repo_info=repo_info,
            parsed_diff=parsed_diff,
            workspace_path=str(workspace),
            repository_source="git-cache",
        )

    def _prepare_github_patch(self, case: AACRCase) -> PreparedCase:
        coordinates = _github_pr_coordinates(case.pr_url)
        if coordinates is None:
            raise RepositoryError("GitHub patch fallback requires a GitHub pull request URL")
        owner, repository, pull_number = coordinates
        patch_url = (
            f"https://patch-diff.githubusercontent.com/raw/{owner}/{repository}/"
            f"pull/{pull_number}.diff"
        )
        patch = self._fetch_url(patch_url)
        parsed_diff = parse_unified_diff(patch)
        if parsed_diff.is_empty:
            raise RepositoryError(f"GitHub pull request patch is empty: {case.pr_url}")

        workspace = self.workspace_root / case.case_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        raw_base = f"https://raw.githubusercontent.com/{owner}/{repository}/{case.source_commit}/"
        for diff_file in parsed_diff.files:
            if diff_file.change_type == "deleted":
                continue
            relative_path = _safe_relative_path(diff_file.path)
            if relative_path is None:
                continue
            try:
                content = self._fetch_url(raw_base + quote(relative_path, safe="/"))
            except RepositoryError:
                continue
            destination = workspace / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

        repo_info = RepoInfo(
            root=str(workspace),
            branch=f"aacr/{case.case_id}",
            head=case.source_commit,
            has_staged_changes=False,
            has_unstaged_changes=False,
        )
        return PreparedCase(
            repo_info=repo_info,
            parsed_diff=parsed_diff,
            workspace_path=str(workspace),
            repository_source="github-patch-fallback",
        )

    def _fetch_url(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "review-pilot/aacr-bench"})
        try:
            with urlopen(request, timeout=self._git_timeout_seconds()) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            raise RepositoryError(f"HTTP {exc.code} while reading {url}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RepositoryError(f"network error while reading {url}: {exc}") from exc

    def _ensure_clone(self, repo_url: str) -> Path:
        key = hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:16]
        repo_path = self.cache_root / key
        if not repo_path.exists():
            try:
                self._run(
                    [
                        "git",
                        "clone",
                        "--filter=blob:none",
                        "--depth=1",
                        "--no-tags",
                        "--no-checkout",
                        repo_url,
                        str(repo_path),
                    ],
                    cwd=self.cache_root,
                )
            except RepositoryError:
                if repo_path.is_symlink():
                    repo_path.unlink()
                elif repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                raise
            return repo_path
        if not (repo_path / ".git").exists():
            raise RepositoryError(f"benchmark cache path is not a Git repository: {repo_path}")
        return repo_path

    def _ensure_commit(self, repo_path: Path, commit: str) -> None:
        if self._commit_exists(repo_path, commit):
            return
        self._run(
            ["git", "fetch", "--filter=blob:none", "--depth=1", "--no-tags", "origin", commit],
            cwd=repo_path,
        )
        if not self._commit_exists(repo_path, commit):
            raise RepositoryError(f"commit does not exist in repository: {commit}")

    def _commit_exists(self, repo_path: Path, commit: str) -> bool:
        timeout = self._git_timeout_seconds()
        environment = os.environ.copy()
        environment["GIT_NO_LAZY_FETCH"] = "1"
        try:
            found = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=repo_path,
                text=True,
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RepositoryError(
                f"git cat-file timed out after {timeout:g}s while checking commit {commit}"
            ) from exc
        return found.returncode == 0

    def _ensure_worktree(
        self,
        repo_path: Path,
        case_id: str,
        commit: str,
    ) -> Path:
        workspace = self.workspace_root / case_id
        if workspace.exists():
            current = self._run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace,
                check=False,
            )
            if current.returncode == 0 and current.stdout.strip() == commit:
                return workspace
            self._run(
                ["git", "worktree", "remove", "--force", str(workspace)],
                cwd=repo_path,
                check=False,
            )
            if workspace.exists():
                shutil.rmtree(workspace)
        self._run(["git", "worktree", "prune"], cwd=repo_path, check=False)
        self._run(
            ["git", "worktree", "add", "--detach", str(workspace), commit],
            cwd=repo_path,
        )
        return workspace

    @staticmethod
    def _run(
        args: Sequence[str],
        *,
        cwd: Path,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [str(arg) for arg in args]
        timeout = RepositoryCache._git_timeout_seconds()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RepositoryError(
                f"command timed out after {timeout:g}s: {' '.join(command)}"
            ) from exc
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RepositoryError(
                f"command failed ({' '.join(command)}): {detail or 'unknown error'}"
            )
        return result

    @staticmethod
    def _git_timeout_seconds() -> float:
        raw = os.environ.get("REVIEW_PILOT_GIT_TIMEOUT_SECONDS", "120")
        try:
            timeout = float(raw)
        except ValueError as exc:
            raise RepositoryError(
                "REVIEW_PILOT_GIT_TIMEOUT_SECONDS must be a positive number"
            ) from exc
        if timeout <= 0:
            raise RepositoryError(
                "REVIEW_PILOT_GIT_TIMEOUT_SECONDS must be a positive number"
            )
        return timeout


_GITHUB_PR_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)/?$",
    re.IGNORECASE,
)


def _github_pr_coordinates(pr_url: str) -> tuple[str, str, str] | None:
    match = _GITHUB_PR_RE.match(pr_url.strip())
    if match is None:
        return None
    return match.group("owner"), match.group("repo"), match.group("number")


def _safe_relative_path(path: str) -> str | None:
    relative = path.replace("\\", "/").lstrip("./")
    parts = Path(relative).parts
    if not relative or relative.startswith("/") or ".." in parts:
        return None
    return relative
