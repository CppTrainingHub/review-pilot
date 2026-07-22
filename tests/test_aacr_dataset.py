from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_pilot.benchmark.dataset import (
    DatasetError,
    case_id_for,
    load_aacr_dataset,
    normalize_path,
)


FIXTURE = Path(__file__).parent / "fixtures" / "aacr" / "positive_samples.json"


def test_load_aacr_dataset_normalizes_case_and_comment_fields() -> None:
    result = load_aacr_dataset(FIXTURE)

    assert len(result.cases) == 1
    assert result.issues == ()
    case = result.cases[0]
    assert case.case_id == case_id_for(
        "https://github.com/example/review-fixture/pull/1",
        "a" * 40,
        "b" * 40,
    )
    assert case.repo_url == "https://github.com/example/review-fixture.git"
    assert case.context_level == "file level"
    assert case.comments[0].path == "src/app.py"
    assert case.comments[0].from_line == 12
    assert case.comments[0].to_line == 13
    assert case.comments[0].category == "maintainability"


def test_dataset_loader_records_bad_cases_without_dropping_valid_cases(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            [
                {
                    "githubPrUrl": "https://github.com/example/repo/pull/1",
                    "source_commit": "a" * 40,
                    "target_commit": "b" * 40,
                    "project_main_language": "Python",
                    "comments": [],
                },
                {"githubPrUrl": "https://github.com/example/repo/pull/2"},
            ]
        ),
        encoding="utf-8",
    )

    result = load_aacr_dataset(path)

    assert len(result.cases) == 1
    assert len(result.issues) == 1
    assert result.issues[0].dataset_index == 1
    assert "source_commit is required" in result.issues[0].message


def test_normalize_path_rejects_repository_escape() -> None:
    with pytest.raises(DatasetError, match="escape repository"):
        normalize_path("../outside.py")


def test_dataset_root_must_be_an_array(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"items": []}), encoding="utf-8")

    with pytest.raises(DatasetError, match="root must be an array"):
        load_aacr_dataset(path)
