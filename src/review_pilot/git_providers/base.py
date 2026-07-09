from __future__ import annotations

from typing import Protocol

from review_pilot.pr_models import PullRequestInfo


class GitProviderError(RuntimeError):
    pass


class GitProvider(Protocol):
    def fetch_pull_request(self, url: str) -> PullRequestInfo:
        raise NotImplementedError
