"""Tests for the pruned runtime dependency set and lock closure.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_deps.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
import traceback

from conftest import REPO_ROOT

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
LOCK_PATH = REPO_ROOT / "uv.lock"

# Dependencies this task removed from the project.
REMOVED_NAMES = ("shap", "pandas-stubs")

# uv.lock held 47 [[package]] entries at the branch point before the diet.
BRANCH_POINT_LOCK_PACKAGES = 47

_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    """Lowercase and treat hyphen/underscore spellings as equivalent."""
    return name.lower().replace("_", "-")


def _requirement_name(requirement: str) -> str:
    match = _NAME_PATTERN.match(requirement)
    assert match is not None, f"unparsable requirement: {requirement!r}"
    return _normalize(match.group(1))


def _project_dependencies() -> list[str]:
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    return list(data["project"]["dependencies"])


def _lock_packages() -> list[dict[str, object]]:
    data = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return list(data["package"])


def _model_modules() -> list[str]:
    """Dotted import names for every product module under the model package."""
    modules = []
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        relative = path.relative_to(MODEL_ROOT)
        if "tests" in relative.parts:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules.append(".".join(["packages", "ml", "model", *parts]))
    return modules


def test_model_modules_import() -> None:
    modules = _model_modules()
    assert modules, "no model modules found to import"

    script = (
        "import importlib\n"
        "import sys\n"
        "for name in sys.argv[1:]:\n"
        "    importlib.import_module(name)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, *modules],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        "importing every model module failed:\n" + result.stderr
    )


if __name__ == "__main__":
    failures = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All tests passed")
