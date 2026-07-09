from __future__ import annotations

from pathlib import PurePosixPath


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".md": "markdown",
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}

TEST_PATH_MARKERS = (
    "tests/",
    "test/",
    "__tests__/",
    "/tests/",
    "/test/",
    "/__tests__/",
    ".spec.",
    ".test.",
    "_test.",
)


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def detect_language(path: str) -> str:
    normalized = normalize_path(path).lower()
    name = PurePosixPath(normalized).name
    if name in {"makefile", "dockerfile"}:
        return name
    return LANGUAGE_BY_EXTENSION.get(PurePosixPath(normalized).suffix, "unknown")


def is_test_path(path: str) -> bool:
    normalized = normalize_path(path).lower()
    name = PurePosixPath(normalized).name
    return (
        any(marker in normalized for marker in TEST_PATH_MARKERS)
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith("_spec.py")
    )


def stem_for_matching(path: str) -> str:
    name = PurePosixPath(normalize_path(path)).name
    suffix = PurePosixPath(name).suffix
    stem = name.removesuffix(suffix) if suffix else name
    for prefix in ("test_",):
        if stem.startswith(prefix):
            stem = stem.removeprefix(prefix)
    for suffix_marker in ("_test", ".test", ".spec", "_spec"):
        if stem.endswith(suffix_marker):
            stem = stem.removesuffix(suffix_marker)
    return stem
