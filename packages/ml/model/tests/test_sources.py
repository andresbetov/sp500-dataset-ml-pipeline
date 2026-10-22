"""Source-hygiene checks for the model package.

The checks are static: every production import is referenced by a name load
(re-exports named in ``__all__`` count), the fold loader's return annotation
matches the payload it returns, and the package README documents both output
roots and the stale-directory dispositions. No live artifact is read or
written.

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
README_PATH = MODEL_ROOT / "README.md"
ARTIFACT_GUIDE_PATH = REPO_ROOT / "docs" / "model-artifacts.md"

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


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split()).lower()


def test_production_imports_are_all_referenced() -> None:
    findings = []
    for path in _production_modules():
        for bound, line, source in _unused_imports(path):
            relative = path.relative_to(REPO_ROOT).as_posix()
            findings.append(f"{relative}:{line}: {bound} ({source})")

    assert not findings, "unused imports: " + "; ".join(findings)


def test_unused_import_scan_detects_a_seeded_import() -> None:
    source = (
        "from __future__ import annotations\n"
        "import json\n"
        "import matplotlib.patches as mpatches\n"
        "from pathlib import Path\n"
        "\n"
        "VALUE: Path | None = None\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        seeded = Path(tmpdir) / "seeded.py"
        seeded.write_text(source, encoding="utf-8")
        unused = {bound for bound, _line, _source in _unused_imports(seeded)}

    assert unused == {"json", "mpatches"}, unused


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


def test_load_fold_data_stays_deleted() -> None:
    assert not hasattr(cv_utils, "load_fold_data")

    with pytest.raises(ImportError):
        from packages.ml.model.cross_validation.cv_utils import (  # noqa: F401
            load_fold_data,
        )


def test_model_readme_documents_roots_phases_and_entry_points() -> None:
    assert README_PATH.is_file(), f"missing model README: {README_PATH}"
    text = _normalized(README_PATH)

    required = (
        "artifacts/",
        "data/model_outputs/",
        "docs/model-artifacts.md",
        "phase 2",
        "phase 3",
        "phase 4",
        "phase 5",
        "python -m packages.ml.model.main",
        "--dry-run",
        "--resume",
        "--folds",
    )
    missing = [token for token in required if token not in text]
    assert not missing, f"README is missing: {missing}"


def test_model_readme_attributes_the_artifacts_root_writers() -> None:
    rows = [
        " ".join(line.split()).lower()
        for line in README_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `packages/ml/model/artifacts/`")
    ]
    assert len(rows) == 1, rows
    row = rows[0]

    for token in (
        "phase 1",
        "regime_characterization.json",
        "phases 2-4",
        "model entry point",
        "run state",
        "after phase 5",
    ):
        assert token in row, f"artifacts-root row is missing {token!r}: {row}"


def test_model_readme_records_the_stale_directory_dispositions() -> None:
    text = _normalized(README_PATH)

    required = (
        "artifacts/legacy/",
        "empty",
        "artifacts/baseline_v1/",
        "provenance",
        "explicit",
        "override",
        "runs.jsonl",
    )
    missing = [token for token in required if token not in text]
    assert not missing, f"README is missing the disposition notes: {missing}"


def test_artifact_guide_names_both_stale_directories() -> None:
    text = _normalized(ARTIFACT_GUIDE_PATH)

    assert "artifacts/legacy/" in text
    assert "artifacts/baseline_v1/" in text


def test_output_roots_are_unchanged() -> None:
    paths = get_project_paths()

    assert paths["project_root"] == REPO_ROOT
    assert paths["model_dir"] == MODEL_ROOT
    assert paths["artifacts_dir"] == MODEL_ROOT / "artifacts"
    assert paths["outputs_dir"] == REPO_ROOT / "data" / "model_outputs"
    assert paths["dataset_path"] == REPO_ROOT / "data" / "processed" / "dataset.parquet"


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
