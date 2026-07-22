from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath

from review_pilot.code_index import find_by_path, resolve_local_imports
from review_pilot.language_detection import normalize_path, stem_for_matching
from review_pilot.models import (
    CodeIndex,
    ContextCandidate,
    ContextCandidateManifest,
    IndexedFile,
    ParsedDiff,
)


PRIORITY_CHANGED = 0
PRIORITY_RELATED_TEST = 10
PRIORITY_LOCAL_IMPORT = 20
PRIORITY_SAME_DIRECTORY = 25
PRIORITY_PROJECT_DOC = 40
PRIORITY_PROJECT_CONFIG = 50

PROJECT_DOC_NAMES = {"readme.md", "readme"}
PROJECT_CONFIG_NAMES = {
    ".review-pilot.toml",
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
    "requirements.txt",
}


def select_context_candidates(
    parsed_diff: ParsedDiff,
    index: CodeIndex,
) -> ContextCandidateManifest:
    changed_paths = tuple(
        normalize_path(file.path)
        for file in parsed_diff.files
        if file.path and file.change_type != "deleted"
    )
    candidates: dict[str, ContextCandidate] = {}

    for path in changed_paths:
        indexed_file = find_by_path(index, path)
        if indexed_file is None:
            candidates[path] = ContextCandidate(
                path=path,
                reason="changed_file",
                priority=PRIORITY_CHANGED,
                language="unknown",
                is_changed=True,
            )
            continue
        _add_candidate(
            candidates,
            _candidate_from_file(
                indexed_file,
                reason="changed_file",
                priority=PRIORITY_CHANGED,
                is_changed=True,
            ),
        )
        for related_test in related_test_files(index, indexed_file):
            _add_candidate(
                candidates,
                _candidate_from_file(
                    related_test,
                    reason="related_test",
                    priority=PRIORITY_RELATED_TEST,
                    matched_symbols=tuple(
                        symbol
                        for symbol in indexed_file.symbols
                        if symbol in related_test.symbols or stem_for_matching(symbol) in related_test.path
                    ),
                ),
            )
        for imported_path in resolve_local_imports(index, indexed_file):
            imported_file = find_by_path(index, imported_path)
            if imported_file is None:
                continue
            _add_candidate(
                candidates,
                _candidate_from_file(
                    imported_file,
                    reason="local_import",
                    priority=PRIORITY_LOCAL_IMPORT,
                    matched_imports=tuple(
                        item
                        for item in indexed_file.imports
                        if imported_file.path.endswith(_import_tail(item))
                        or _import_tail(item) in imported_file.path
                    ),
                ),
            )
        for sibling in related_sibling_files(index, indexed_file):
            _add_candidate(
                candidates,
                _candidate_from_file(
                    sibling,
                    reason="same_directory_related",
                    priority=PRIORITY_SAME_DIRECTORY,
                ),
            )

    for file in index.files:
        name = PurePosixPath(file.path).name.lower()
        if name in PROJECT_DOC_NAMES:
            _add_candidate(
                candidates,
                _candidate_from_file(file, reason="project_doc", priority=PRIORITY_PROJECT_DOC),
            )
        elif name in PROJECT_CONFIG_NAMES:
            _add_candidate(
                candidates,
                _candidate_from_file(
                    file,
                    reason="project_config",
                    priority=PRIORITY_PROJECT_CONFIG,
                ),
            )

    return ContextCandidateManifest(
        changed_paths=changed_paths,
        candidates=tuple(sorted(candidates.values(), key=_candidate_sort_key)),
        index_file_count=len(index.files),
    )


def related_test_files(index: CodeIndex, changed_file: IndexedFile) -> tuple[IndexedFile, ...]:
    if changed_file.is_test:
        return ()
    changed_stem = stem_for_matching(changed_file.path)
    changed_dir = PurePosixPath(changed_file.path).parent
    matches: list[IndexedFile] = []
    for file in index.files:
        if not file.is_test:
            continue
        test_stem = stem_for_matching(file.path)
        test_dir = PurePosixPath(file.path).parent
        if test_stem == changed_stem:
            matches.append(file)
            continue
        if changed_stem and changed_stem in test_stem:
            matches.append(file)
            continue
        if test_dir == changed_dir:
            matches.append(file)
    return tuple(sorted(dict.fromkeys(matches), key=lambda file: file.path))


def related_sibling_files(
    index: CodeIndex,
    changed_file: IndexedFile,
) -> tuple[IndexedFile, ...]:
    """Find same-directory interface/implementation companions.

    The rule is intentionally narrow: it accepts the same stem or a common
    stem family such as ``parser``/``parser_impl``.  It does not pull an
    entire directory into one review unit.
    """

    changed_stem = stem_for_matching(changed_file.path).lower()
    changed_dir = PurePosixPath(changed_file.path).parent
    matches: list[IndexedFile] = []
    for file in index.files:
        if file.path == changed_file.path or file.is_test:
            continue
        if PurePosixPath(file.path).parent != changed_dir:
            continue
        sibling_stem = stem_for_matching(file.path).lower()
        if _same_stem_family(changed_stem, sibling_stem):
            matches.append(file)
    return tuple(sorted(dict.fromkeys(matches), key=lambda file: file.path))


def _same_stem_family(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    suffixes = (
        "_impl",
        "_implementation",
        "_interface",
        "_iface",
        "_base",
        "_detail",
        "_model",
        "_controller",
    )
    left_base = _remove_known_sibling_suffix(left, suffixes)
    right_base = _remove_known_sibling_suffix(right, suffixes)
    return left_base == right_base and (
        left_base != left or right_base != right
    )


def _remove_known_sibling_suffix(value: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]
    return value


def _candidate_from_file(
    file: IndexedFile,
    *,
    reason: str,
    priority: int,
    is_changed: bool = False,
    matched_symbols: tuple[str, ...] = (),
    matched_imports: tuple[str, ...] = (),
) -> ContextCandidate:
    return ContextCandidate(
        path=file.path,
        reason=reason,
        priority=priority,
        language=file.language,
        is_changed=is_changed,
        is_test=file.is_test,
        matched_symbols=matched_symbols,
        matched_imports=matched_imports,
    )


def _add_candidate(
    candidates: dict[str, ContextCandidate],
    candidate: ContextCandidate,
) -> None:
    current = candidates.get(candidate.path)
    if current is None or candidate.priority < current.priority:
        candidates[candidate.path] = candidate
        return
    if current.priority == candidate.priority and current.reason != candidate.reason:
        candidates[candidate.path] = replace(
            current,
            reason=f"{current.reason},{candidate.reason}",
            matched_symbols=tuple(dict.fromkeys((*current.matched_symbols, *candidate.matched_symbols))),
            matched_imports=tuple(dict.fromkeys((*current.matched_imports, *candidate.matched_imports))),
        )


def _candidate_sort_key(candidate: ContextCandidate) -> tuple[int, str, str]:
    return (candidate.priority, candidate.reason, candidate.path)


def _import_tail(import_name: str) -> str:
    return normalize_path(import_name).replace(".", "/").strip("/")
