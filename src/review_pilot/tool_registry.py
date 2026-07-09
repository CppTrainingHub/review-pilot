from __future__ import annotations

from .config import ReviewPilotConfig
from .tool_models import CommandSpec, ProjectDetection, RegisteredTool, ToolSpec


DEFAULT_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="python-version",
        description="Print the active Python version.",
        command=CommandSpec(("python", "--version")),
        project_types=("python",),
        default_timeout_seconds=10,
    ),
    ToolSpec(
        name="python-tests",
        description="Run the project's pytest suite.",
        command=CommandSpec(("python", "-m", "pytest", "-q")),
        project_types=("python",),
        default_timeout_seconds=60,
    ),
    ToolSpec(
        name="npm-test",
        description="Run npm test for Node projects.",
        command=CommandSpec(("npm", "test")),
        project_types=("node",),
        default_timeout_seconds=60,
    ),
    ToolSpec(
        name="go-test",
        description="Run go test for Go projects.",
        command=CommandSpec(("go", "test", "./...")),
        project_types=("go",),
        default_timeout_seconds=60,
    ),
    ToolSpec(
        name="semgrep",
        description="Run Semgrep workspace scan and print JSON findings.",
        command=CommandSpec(("semgrep", "--config", "auto", "--json", ".")),
        project_types=("python", "node", "go"),
        default_timeout_seconds=120,
    ),
)


class ToolRegistry:
    def __init__(
        self,
        detection: ProjectDetection,
        config: ReviewPilotConfig,
        specs: tuple[ToolSpec, ...] = DEFAULT_TOOL_SPECS,
    ) -> None:
        self._detection = detection
        self._config = config
        self._tools = {
            tool.spec.name: tool
            for tool in self._build_registered_tools(detection, config, specs)
        }

    def list_tools(self) -> tuple[RegisteredTool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    @staticmethod
    def _build_registered_tools(
        detection: ProjectDetection,
        config: ReviewPilotConfig,
        specs: tuple[ToolSpec, ...],
    ) -> tuple[RegisteredTool, ...]:
        detected_types = set(detection.project_types)
        tools: list[RegisteredTool] = []
        for spec in specs:
            config_item = config.tools.get(spec.name)
            matches_project = bool(detected_types.intersection(spec.project_types))
            if config_item is None:
                enabled = spec.enabled and matches_project
                timeout_seconds = spec.default_timeout_seconds
                reason = "project_detected" if matches_project else "project_type_not_detected"
            else:
                enabled = config_item.enabled and matches_project
                timeout_seconds = config_item.timeout_seconds
                if not matches_project:
                    reason = "project_type_not_detected"
                elif config_item.enabled:
                    reason = "enabled_by_config"
                else:
                    reason = "disabled_by_config"
            tools.append(
                RegisteredTool(
                    spec=spec,
                    enabled=enabled,
                    timeout_seconds=timeout_seconds,
                    reason=reason,
                )
            )
        return tuple(tools)
