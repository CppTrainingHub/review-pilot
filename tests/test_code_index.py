from __future__ import annotations

from pathlib import Path

from review_pilot.code_index import build_code_index, find_by_path, resolve_local_imports
from review_pilot.config import ReviewPilotConfig
from review_pilot.language_detection import detect_language, is_test_path


def test_detect_language_and_test_paths() -> None:
    assert detect_language("src/app.py") == "python"
    assert detect_language("web/Button.tsx") == "typescript"
    assert detect_language("include/service.hpp") == "cpp"
    assert detect_language("README.md") == "markdown"
    assert is_test_path("tests/test_service.py") is True
    assert is_test_path("src/service.py") is False


def test_build_code_index_extracts_python_imports_and_symbols(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "import os\nfrom .helpers import load\n\nclass Service:\n    pass\n\ndef run():\n    return load()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "helpers.py").write_text(
        "def load():\n    return 1\n",
        encoding="utf-8",
    )

    index = build_code_index(tmp_path)
    service = find_by_path(index, "src/service.py")

    assert service is not None
    assert service.language == "python"
    assert service.imports == ("os", ".helpers")
    assert service.symbols == ("Service", "run")
    assert resolve_local_imports(index, service) == ("src/helpers.py",)


def test_build_code_index_extracts_js_imports_and_symbols(tmp_path: Path) -> None:
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "app.ts").write_text(
        "import { helper } from './helper'\nexport function render() { return helper() }\n",
        encoding="utf-8",
    )
    (tmp_path / "web" / "helper.ts").write_text(
        "export const helper = () => 1\n",
        encoding="utf-8",
    )

    index = build_code_index(tmp_path)
    app = find_by_path(index, "web/app.ts")

    assert app is not None
    assert app.imports == ("./helper",)
    assert app.symbols == ("render",)
    assert resolve_local_imports(index, app) == ("web/helper.ts",)


def test_build_code_index_extracts_local_cpp_include(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.cpp").write_text(
        '#include "service.hpp"\nint run() { return 1; }\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "service.hpp").write_text(
        "class Service {};\n",
        encoding="utf-8",
    )

    index = build_code_index(tmp_path)
    service = find_by_path(index, "src/service.cpp")

    assert service is not None
    assert service.imports == ("service.hpp",)
    assert "run" in service.symbols
    assert resolve_local_imports(index, service) == ("src/service.hpp",)


def test_build_code_index_respects_ignore_paths(tmp_path: Path) -> None:
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated" / "client.py").write_text("def generated():\n    pass\n", encoding="utf-8")
    (tmp_path / "src.py").write_text("def run():\n    pass\n", encoding="utf-8")

    index = build_code_index(
        tmp_path,
        ReviewPilotConfig(ignore_paths=("generated/**",)),
    )

    assert find_by_path(index, "src.py") is not None
    assert find_by_path(index, "generated/client.py") is None
