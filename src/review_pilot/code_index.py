from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath

from review_pilot.config import ReviewPilotConfig
from review_pilot.language_detection import detect_language, is_test_path, normalize_path
from review_pilot.models import CodeIndex, IndexedFile


SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
}
MAX_INDEX_FILE_BYTES = 200_000
TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hpp",
    ".hh",
    ".md",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
}

_JS_IMPORT_RE = re.compile(
    r"(?:import\s+.*?\s+from\s+|import\s*\(\s*|require\s*\()\s*['\"](?P<target>[^'\"]+)['\"]"
)
_JS_SYMBOL_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(?P<function>[A-Za-z_$][\w$]*)|"
    r"(?:export\s+)?class\s+(?P<class>[A-Za-z_$][\w$]*)|"
    r"export\s+const\s+(?P<const>[A-Za-z_$][\w$]*)"
)
_INCLUDE_RE = re.compile(r"^\s*#include\s+\"(?P<target>[^\"]+)\"", re.MULTILINE)
_CPP_SYMBOL_RE = re.compile(
    r"^\s*(?:class|struct)\s+(?P<class>[A-Za-z_]\w*)|"
    r"^\s*(?:[A-Za-z_][\w:<>,~*&\s]+)\s+(?P<function>[A-Za-z_]\w*)\s*\([^;]*\)\s*\{?",
    re.MULTILINE,
)


def build_code_index(
    root: str | Path,
    config: ReviewPilotConfig | None = None,
) -> CodeIndex:
    repo_root = Path(root).resolve()
    active_config = config or ReviewPilotConfig.default()
    indexed: list[IndexedFile] = []

    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        relative = normalize_path(path.relative_to(repo_root).as_posix())
        if _should_skip_path(path, relative, active_config):
            continue
        language = detect_language(relative)
        size_bytes = path.stat().st_size
        imports: tuple[str, ...] = ()
        symbols: tuple[str, ...] = ()
        if size_bytes <= MAX_INDEX_FILE_BYTES:
            text = _read_text(path)
            if text is not None:
                imports = extract_imports(relative, text)
                symbols = extract_symbols(relative, text)
        indexed.append(
            IndexedFile(
                path=relative,
                language=language,
                size_bytes=size_bytes,
                is_test=is_test_path(relative),
                imports=imports,
                symbols=symbols,
            )
        )
    return CodeIndex(root=str(repo_root), files=tuple(indexed))


def extract_imports(path: str, text: str) -> tuple[str, ...]:
    language = detect_language(path)
    if language == "python":
        return _extract_python_imports(text)
    if language in {"javascript", "typescript"}:
        return tuple(dict.fromkeys(match.group("target") for match in _JS_IMPORT_RE.finditer(text)))
    if language in {"c", "cpp"}:
        return tuple(dict.fromkeys(match.group("target") for match in _INCLUDE_RE.finditer(text)))
    return ()


def extract_symbols(path: str, text: str) -> tuple[str, ...]:
    language = detect_language(path)
    if language == "python":
        return _extract_python_symbols(text)
    if language in {"javascript", "typescript"}:
        symbols: list[str] = []
        for match in _JS_SYMBOL_RE.finditer(text):
            symbols.extend(value for value in match.groupdict().values() if value)
        return tuple(dict.fromkeys(symbols))
    if language in {"c", "cpp"}:
        symbols = []
        for match in _CPP_SYMBOL_RE.finditer(text):
            symbols.extend(value for value in match.groupdict().values() if value)
        return tuple(dict.fromkeys(symbols))
    return ()


def resolve_local_imports(index: CodeIndex, file: IndexedFile) -> tuple[str, ...]:
    resolved: list[str] = []
    available = {item.path for item in index.files}
    for target in file.imports:
        for candidate in _candidate_import_paths(file.path, target):
            if candidate in available:
                resolved.append(candidate)
                break
    return tuple(dict.fromkeys(resolved))


def find_by_path(index: CodeIndex, path: str) -> IndexedFile | None:
    normalized = normalize_path(path)
    for file in index.files:
        if file.path == normalized:
            return file
    return None


def _should_skip_path(path: Path, relative: str, config: ReviewPilotConfig) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if config.is_ignored(relative):
        return True
    if path.suffix.lower() not in TEXT_EXTENSIONS and path.name.lower() not in {"readme", "makefile"}:
        return True
    return False


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _extract_python_imports(text: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ()
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            module = node.module or ""
            imports.append(f"{prefix}{module}" if module else prefix)
    return tuple(dict.fromkeys(imports))


def _extract_python_symbols(text: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ()
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
    return tuple(dict.fromkeys(symbols))


def _candidate_import_paths(source_path: str, target: str) -> tuple[str, ...]:
    source_dir = PurePosixPath(source_path).parent
    candidates: list[str] = []

    if target.startswith("./") or target.startswith("../"):
        base = normalize_path(str(source_dir / target))
        return tuple(normalize_path(candidate) for candidate in _with_source_extensions(base))

    if target.startswith("."):
        dots = len(target) - len(target.lstrip("."))
        remainder = target[dots:]
        base = source_dir
        for _ in range(max(dots - 1, 0)):
            base = base.parent
        if remainder:
            candidates.extend(_with_source_extensions(str(base / remainder.replace(".", "/"))))
        candidates.extend(_with_source_extensions(str(base / "__init__")))
        return tuple(normalize_path(candidate) for candidate in candidates)

    if "/" in target:
        return tuple(normalize_path(candidate) for candidate in _with_source_extensions(target))

    candidates.extend(_with_source_extensions(str(source_dir / target)))
    candidates.extend(_with_source_extensions(target.replace(".", "/")))
    return tuple(normalize_path(candidate) for candidate in candidates)


def _with_source_extensions(base: str) -> tuple[str, ...]:
    suffix = PurePosixPath(base).suffix
    if suffix:
        return (base,)
    return tuple(
        f"{base}{extension}"
        for extension in (
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".c",
            ".h",
            ".cc",
            ".cpp",
            ".hpp",
        )
    )
