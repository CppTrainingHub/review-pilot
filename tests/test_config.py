from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_pilot.config import ConfigError, load_project_config, path_matches


def write_config(repo: Path, content: str) -> None:
    (repo / ".review-pilot.toml").write_text(content, encoding="utf-8")


def test_missing_config_returns_defaults(tmp_path: Path) -> None:
    config = load_project_config(tmp_path)

    assert config.source == "default"
    assert config.ignore_paths == ()
    assert config.rules == {}
    assert config.tools == {}


def test_reads_ignore_paths_rules_and_tools(tmp_path: Path) -> None:
    write_config(
        tmp_path,
        """
ignore_paths = ["generated/**", "vendor/"]

[rules."rule.file-too-large"]
enabled = true
max_added_lines = 20

[rules."rule.debug-output"]
enabled = false
patterns = ["print("]

[tools.semgrep]
enabled = true
timeout_seconds = 12
severity_threshold = "P1"
ignore_paths = ["third_party/**"]
""",
    )

    config = load_project_config(tmp_path)

    assert config.source == str(tmp_path / ".review-pilot.toml")
    assert config.ignore_paths == ("generated/**", "vendor/")
    assert config.rule("rule.file-too-large").max_added_lines == 20
    assert config.rule("rule.debug-output").enabled is False
    assert config.rule("rule.debug-output").patterns == ("print(",)
    assert config.tools["semgrep"].enabled is True
    assert config.tools["semgrep"].timeout_seconds == 12
    assert config.tools["semgrep"].severity_threshold == "P1"
    assert config.tools["semgrep"].ignore_paths == ("third_party/**",)


def test_config_to_dict_is_json_serializable(tmp_path: Path) -> None:
    write_config(
        tmp_path,
        """
ignore_paths = ["generated/**"]

[tools.semgrep]
enabled = false
timeout_seconds = 30
severity_threshold = "P2"
""",
    )

    payload = load_project_config(tmp_path).to_dict()

    assert json.loads(json.dumps(payload, ensure_ascii=False))["ignore_paths"] == [
        "generated/**"
    ]


def test_unknown_top_level_key_raises_config_error(tmp_path: Path) -> None:
    write_config(tmp_path, "unexpected = true\n")

    with pytest.raises(ConfigError, match="unknown config key"):
        load_project_config(tmp_path)


def test_rule_positive_int_validation(tmp_path: Path) -> None:
    write_config(
        tmp_path,
        """
[rules."rule.file-too-large"]
max_added_lines = 0
""",
    )

    with pytest.raises(ConfigError, match="positive integer"):
        load_project_config(tmp_path)


def test_tool_severity_threshold_validation(tmp_path: Path) -> None:
    write_config(
        tmp_path,
        """
[tools.semgrep]
severity_threshold = "P9"
""",
    )

    with pytest.raises(ConfigError, match="severity_threshold"):
        load_project_config(tmp_path)


def test_string_list_validation(tmp_path: Path) -> None:
    write_config(tmp_path, "ignore_paths = [\"\"]\n")

    with pytest.raises(ConfigError, match="non-empty string"):
        load_project_config(tmp_path)


def test_path_matches_supports_glob_prefix_and_path_component() -> None:
    assert path_matches("generated/**", "generated/client.py")
    assert path_matches("vendor/", "vendor/lib/a.py")
    assert path_matches("vendor", "src/vendor/lib.py")
    assert not path_matches("vendor/", "src/vendor/lib.py")
    assert not path_matches("generated/**", "src/generated/client.py")
