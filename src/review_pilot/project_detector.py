from __future__ import annotations

from pathlib import Path

from .tool_models import ProjectDetection


PROJECT_MARKERS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("requirements.txt", "python"),
    ("package.json", "node"),
    ("go.mod", "go"),
)


def detect_project(repo_root: str | Path) -> ProjectDetection:
    root = Path(repo_root).resolve()
    project_types: list[str] = []
    markers: list[str] = []

    for marker, project_type in PROJECT_MARKERS:
        if (root / marker).exists():
            markers.append(marker)
            if project_type not in project_types:
                project_types.append(project_type)

    if (root / "tests").is_dir() and "python" not in project_types:
        markers.append("tests/")
        project_types.append("python")

    return ProjectDetection(
        root=str(root),
        project_types=tuple(project_types),
        markers=tuple(markers),
    )
