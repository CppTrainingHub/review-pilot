from __future__ import annotations

from review_pilot.config import ReviewPilotConfig, ToolConfig
from review_pilot.project_detector import detect_project
from review_pilot.tool_registry import ToolRegistry


def test_registry_enables_python_tools_for_python_project(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")

    registry = ToolRegistry(detect_project(tmp_path), ReviewPilotConfig.default())
    by_name = {tool.spec.name: tool for tool in registry.list_tools()}

    assert by_name["python-version"].enabled is True
    assert by_name["python-tests"].enabled is True
    assert by_name["semgrep"].enabled is True
    assert by_name["npm-test"].enabled is False
    assert by_name["python-tests"].timeout_seconds == 60


def test_registry_applies_tool_config_timeout_and_enabled_flag(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    config = ReviewPilotConfig(
        tools={
            "python-tests": ToolConfig(
                enabled=True,
                timeout_seconds=7,
            )
        }
    )

    registry = ToolRegistry(detect_project(tmp_path), config)
    tool = registry.get("python-tests")

    assert tool.enabled is True
    assert tool.timeout_seconds == 7
    assert tool.reason == "enabled_by_config"


def test_registry_disables_configured_tool_when_project_type_does_not_match(tmp_path) -> None:
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")
    config = ReviewPilotConfig(
        tools={
            "python-tests": ToolConfig(
                enabled=True,
                timeout_seconds=7,
            )
        }
    )

    registry = ToolRegistry(detect_project(tmp_path), config)
    tool = registry.get("python-tests")

    assert tool.enabled is False
    assert tool.reason == "project_type_not_detected"


def test_registry_rejects_unknown_tool(tmp_path) -> None:
    registry = ToolRegistry(detect_project(tmp_path), ReviewPilotConfig.default())

    try:
        registry.get("shell")
    except KeyError as exc:
        assert "unknown tool: shell" in str(exc)
    else:
        raise AssertionError("expected unknown tool to fail")
