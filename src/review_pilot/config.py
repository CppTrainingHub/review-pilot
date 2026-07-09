from __future__ import annotations

import fnmatch
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


VALID_CONFIG_TOP_LEVEL_KEYS = {"ignore_paths", "rules", "tools"}
VALID_RULE_KEYS = {
    "enabled",
    "max_added_lines",
    "max_total_added_lines",
    "patterns",
    "markers",
}
VALID_TOOL_KEYS = {"enabled", "timeout_seconds", "severity_threshold", "ignore_paths"}
VALID_SEVERITY_THRESHOLDS = {"P0", "P1", "P2", "P3"}


class ConfigError(ValueError):
    """Raised when .review-pilot.toml cannot be parsed or validated."""


def _require_table(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a TOML table")
    return value


def _require_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")
    return value


def _require_positive_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def _require_string_list(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a list of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{path}[{index}] must be a non-empty string")
        items.append(item)
    return tuple(items)


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"unknown config key at {path}: {', '.join(unknown)}")


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls, provider: str | None = None) -> LLMConfig:
        configured_provider = provider or os.getenv("REVIEW_PILOT_LLM_PROVIDER", "fake")
        default_model = (
            "fake-review-model"
            if configured_provider == "fake"
            else "gpt-4o-mini"
        )
        default_base_url = (
            "not-used"
            if configured_provider == "fake"
            else "https://api.openai.com/v1"
        )
        timeout_value = os.getenv("REVIEW_PILOT_LLM_TIMEOUT_SECONDS", "30")
        try:
            timeout_seconds = int(timeout_value)
        except ValueError as exc:
            raise ConfigError(
                "REVIEW_PILOT_LLM_TIMEOUT_SECONDS must be a positive integer"
            ) from exc
        if timeout_seconds < 1:
            raise ConfigError(
                "REVIEW_PILOT_LLM_TIMEOUT_SECONDS must be a positive integer"
            )
        return cls(
            provider=configured_provider,
            model=os.getenv("REVIEW_PILOT_LLM_MODEL", default_model),
            base_url=os.getenv("REVIEW_PILOT_LLM_BASE_URL", default_base_url),
            api_key=os.getenv("REVIEW_PILOT_API_KEY") or os.getenv("OPENAI_API_KEY"),
            timeout_seconds=timeout_seconds,
        )

    def status_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "api_key": "configured" if self.api_key else "missing",
        }


@dataclass(frozen=True)
class RuleConfig:
    enabled: bool = True
    max_added_lines: int | None = None
    max_total_added_lines: int | None = None
    patterns: tuple[str, ...] | None = None
    markers: tuple[str, ...] | None = None

    @classmethod
    def from_dict(cls, rule_id: str, data: dict[str, Any]) -> "RuleConfig":
        _reject_unknown_keys(data, VALID_RULE_KEYS, f"rules.{rule_id}")
        return cls(
            enabled=_require_bool(data.get("enabled", True), f"rules.{rule_id}.enabled"),
            max_added_lines=(
                _require_positive_int(data["max_added_lines"], f"rules.{rule_id}.max_added_lines")
                if "max_added_lines" in data
                else None
            ),
            max_total_added_lines=(
                _require_positive_int(
                    data["max_total_added_lines"],
                    f"rules.{rule_id}.max_total_added_lines",
                )
                if "max_total_added_lines" in data
                else None
            ),
            patterns=(
                _require_string_list(data["patterns"], f"rules.{rule_id}.patterns")
                if "patterns" in data
                else None
            ),
            markers=(
                _require_string_list(data["markers"], f"rules.{rule_id}.markers")
                if "markers" in data
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.patterns is not None:
            payload["patterns"] = list(self.patterns)
        if self.markers is not None:
            payload["markers"] = list(self.markers)
        return payload


@dataclass(frozen=True)
class ToolConfig:
    enabled: bool = False
    timeout_seconds: int = 30
    severity_threshold: str = "P2"
    ignore_paths: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, tool_name: str, data: dict[str, Any]) -> "ToolConfig":
        _reject_unknown_keys(data, VALID_TOOL_KEYS, f"tools.{tool_name}")
        severity_threshold = data.get("severity_threshold", "P2")
        if not isinstance(severity_threshold, str) or severity_threshold not in VALID_SEVERITY_THRESHOLDS:
            raise ConfigError(
                f"tools.{tool_name}.severity_threshold must be one of "
                f"{sorted(VALID_SEVERITY_THRESHOLDS)}"
            )
        return cls(
            enabled=_require_bool(data.get("enabled", False), f"tools.{tool_name}.enabled"),
            timeout_seconds=_require_positive_int(
                data.get("timeout_seconds", 30),
                f"tools.{tool_name}.timeout_seconds",
            ),
            severity_threshold=severity_threshold,
            ignore_paths=(
                _require_string_list(data["ignore_paths"], f"tools.{tool_name}.ignore_paths")
                if "ignore_paths" in data
                else ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "severity_threshold": self.severity_threshold,
            "ignore_paths": list(self.ignore_paths),
        }


@dataclass(frozen=True)
class ReviewPilotConfig:
    source: str = "default"
    ignore_paths: tuple[str, ...] = field(default_factory=tuple)
    rules: dict[str, RuleConfig] = field(default_factory=dict)
    tools: dict[str, ToolConfig] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "ReviewPilotConfig":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str) -> "ReviewPilotConfig":
        _reject_unknown_keys(data, VALID_CONFIG_TOP_LEVEL_KEYS, "root")
        ignore_paths = (
            _require_string_list(data["ignore_paths"], "ignore_paths")
            if "ignore_paths" in data
            else ()
        )
        rules: dict[str, RuleConfig] = {}
        for rule_id, rule_data in _require_table(data.get("rules", {}), "rules").items():
            rules[rule_id] = RuleConfig.from_dict(rule_id, _require_table(rule_data, f"rules.{rule_id}"))
        tools: dict[str, ToolConfig] = {}
        for tool_name, tool_data in _require_table(data.get("tools", {}), "tools").items():
            tools[tool_name] = ToolConfig.from_dict(
                tool_name,
                _require_table(tool_data, f"tools.{tool_name}"),
            )
        return cls(
            source=source,
            ignore_paths=ignore_paths,
            rules=rules,
            tools=tools,
        )

    def rule(self, rule_id: str) -> RuleConfig:
        return self.rules.get(rule_id, RuleConfig())

    def is_ignored(self, path: str) -> bool:
        return any(path_matches(pattern, path) for pattern in self.ignore_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ignore_paths": list(self.ignore_paths),
            "rules": {rule_id: config.to_dict() for rule_id, config in sorted(self.rules.items())},
            "tools": {name: config.to_dict() for name, config in sorted(self.tools.items())},
        }


def path_matches(pattern: str, path: str) -> bool:
    normalized_pattern = pattern.strip().replace("\\", "/").strip("/")
    normalized_path = path.replace("\\", "/").strip("/")
    if not normalized_pattern or not normalized_path:
        return False
    if pattern.strip().endswith("/"):
        return normalized_path == normalized_pattern or normalized_path.startswith(f"{normalized_pattern}/")
    if fnmatch.fnmatch(normalized_path, normalized_pattern):
        return True
    if "/" not in normalized_pattern and "*" not in normalized_pattern and "?" not in normalized_pattern:
        return normalized_pattern in normalized_path.split("/")
    return False


def load_project_config(repo_root: str | Path | None = None) -> ReviewPilotConfig:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    path = root / ".review-pilot.toml"
    if not path.exists():
        return ReviewPilotConfig.default()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return ReviewPilotConfig.from_dict(data, source=str(path))
