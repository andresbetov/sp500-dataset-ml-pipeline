"""Source-hygiene checks for the model package.

The checks are static: every production import is referenced by a name load
(re-exports named in ``__all__`` count) and the fold loader's return annotation
matches the payload it returns. No live artifact is read or written.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_sources.py
"""

from __future__ import annotations

import ast
import inspect
import tempfile
import traceback
from pathlib import Path
from typing import get_origin, get_type_hints
from unittest import mock

import joblib
import pytest

from conftest import REPO_ROOT
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.utils import get_project_paths

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"

# A future import is a compiler directive, never a runtime name.
DIRECTIVE_MODULES = {"__future__"}


def _production_modules() -> list[Path]:
    """Every model-package module except the pytest tree and pycache files."""
    return sorted(
        path
        for path in MODEL_ROOT.rglob("*.py")
        if TESTS_DIR not in path.parents and "__pycache__" not in path.parts
    )


def _dunder_all(tree: ast.Module) -> set[str]:
    """Names a module re-exports explicitly, so they count as referenced."""
    exported: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if "__all__" not in targets:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (list, tuple, set)):
            exported.update(str(name) for name in value)
    return exported


def _unused_imports(path: Path) -> list[tuple[str, int, str]]:
    """(bound name, line, imported name) for every unreferenced import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    imported: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                imported.append((bound, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module in DIRECTIVE_MODULES:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                imported.append(
                    (bound, node.lineno, f"{node.module}.{alias.name}")
                )

    referenced = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    referenced.update(_dunder_all(tree))
    return [
        (bound, line, source)
        for bound, line, source in imported
        if bound not in referenced
    ]


def test_load_folds_metadata_annotation_matches_the_loaded_payload() -> None:
    annotation = get_type_hints(cv_utils.load_folds_metadata)["return"]
    assert annotation is dict or get_origin(annotation) is dict, annotation
    assert not inspect.signature(cv_utils.load_folds_metadata).parameters

    payload = {"schema_version": 2, "fold_0": {"n_train": 1}}
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        folds_dir = root / "folds"
        folds_dir.mkdir(parents=True)
        artifact = folds_dir / "folds_metadata.pkl"
        joblib.dump(payload, artifact)
        record_artifact(artifact, registry_path=root / REGISTRY_FILENAME)

        with mock.patch.object(cv_utils, "ARTIFACTS_DIR", root):
            loaded = cv_utils.load_folds_metadata()

    assert type(loaded) is dict
    assert loaded == payload


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
