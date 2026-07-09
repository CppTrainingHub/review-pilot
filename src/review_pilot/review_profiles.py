from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


HookName = Literal["pre-commit", "pre-push"]
ProfileName = Literal["manual", "pre_commit", "pre_push"]


@dataclass(frozen=True)
class ReviewProfile:
    name: ProfileName
    description: str
    review_args: tuple[str, ...]

    def command(self) -> tuple[str, ...]:
        return ("review-pilot", *self.review_args)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "command": list(self.command()),
        }


MANUAL_PROFILE = ReviewProfile(
    name="manual",
    description="Manual local review profile.",
    review_args=("review", "--staged", "--no-ai"),
)

PRE_COMMIT_PROFILE = ReviewProfile(
    name="pre_commit",
    description="Fast staged diff review for git pre-commit.",
    review_args=("review", "--staged", "--no-ai", "--fail-on", "P1"),
)

PRE_PUSH_PROFILE = ReviewProfile(
    name="pre_push",
    description="Fuller staged diff review for git pre-push.",
    review_args=(
        "review",
        "--staged",
        "--no-ai",
        "--with-tools",
        "--fail-on",
        "P2",
    ),
)

PROFILES = {
    "manual": MANUAL_PROFILE,
    "pre_commit": PRE_COMMIT_PROFILE,
    "pre_push": PRE_PUSH_PROFILE,
}

HOOK_PROFILES: dict[HookName, ReviewProfile] = {
    "pre-commit": PRE_COMMIT_PROFILE,
    "pre-push": PRE_PUSH_PROFILE,
}


def profile_for_hook(hook_name: HookName) -> ReviewProfile:
    return HOOK_PROFILES[hook_name]
