from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass


MIN_PYTHON = (3, 11)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str


def check_python_version(version_info: tuple[int, int, int] | None = None) -> CheckResult:
    version = version_info or sys.version_info[:3]
    current = ".".join(str(part) for part in version[:3])
    required = ".".join(str(part) for part in MIN_PYTHON)
    ok = version >= (*MIN_PYTHON, 0)
    if ok:
        return CheckResult("python", True, f"Python {current} >= {required}")
    return CheckResult("python", False, f"Python {current} < {required}")


def check_git_available(git_path: str | None = None) -> CheckResult:
    path = git_path if git_path is not None else shutil.which("git")
    if path:
        return CheckResult("git", True, f"git found at {path}")
    return CheckResult("git", False, "git executable was not found in PATH")


def run_checks(git_path: str | None = None) -> list[CheckResult]:
    return [
        check_python_version(),
        check_git_available(git_path=git_path),
    ]
