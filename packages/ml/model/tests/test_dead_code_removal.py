"""Tests for the removed dead evaluation paths and their doctrine bullets.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_dead_code_removal.py
"""

from __future__ import annotations

import importlib
import re
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import pytest

from conftest import ARTIFACTS_DIR, REPO_ROOT, requires_live_artifacts
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"
AGENTS_PATH = REPO_ROOT / "AGENTS.md"

DEAD_SYMBOLS = (
    "training.inference",
    "generate_predictions",
    "_validate_predictions",
    "load_fold_data",
)

REMOVED_BULLETS = (
    "- `model/training/inference.py` is never imported; the live path inlines "
    "`model.predict()` in `training.py`. Editing it changes nothing.",
    "- `cross_validation/cv_utils.py:load_fold_data` is never called.",
    "- `metadata.pkl` carries no `regime` key, so per-fold regime distributions "
    "are always empty.",
    "- The early-stopping branch in `model_trainer.py` is unreachable: "
    "`training.py` passes `X_val=None, y_val=None`.",
)

RETAINED_BULLETS = (
    "- `.idea/workspace.xml` still carries a `RunManager` entry for the deleted "
    "`run_pipeline.py`. It is untracked and stale.",
)


def test_inference_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("packages.ml.model.training.inference")

    assert excinfo.value.name == "packages.ml.model.training.inference"


def test_load_fold_data_is_removed() -> None:
    cv_utils = importlib.import_module(
        "packages.ml.model.cross_validation.cv_utils"
    )
    assert not hasattr(cv_utils, "load_fold_data")

    with pytest.raises(ImportError):
        from packages.ml.model.cross_validation.cv_utils import load_fold_data  # noqa: F401


def test_live_import_surface() -> None:
    from packages.ml.model.cross_validation.cv_utils import load_folds_metadata
    from packages.ml.model.training import training

    assert callable(load_folds_metadata)
    assert callable(training.phase_4_training)

    for path in sorted((MODEL_ROOT / "training").glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_live_fold_metadata(artifacts_dir: Path) -> None:
    requires_live_artifacts(artifacts_dir)

    from packages.ml.model.cross_validation import cv_utils
    from packages.ml.model.cross_validation.cv_utils import load_folds_metadata

    # The loader now verifies a recorded size+sha256 before loading. The live
    # tree has no registry yet, so record the live artifact's hash into a
    # temporary registry and point the loader's artifact root at a symlinked
    # copy: the live artifact is only read, never rewritten.
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        (temp_root / "folds").mkdir()
        (temp_root / "folds" / "folds_metadata.pkl").symlink_to(
            (artifacts_dir / "folds" / "folds_metadata.pkl").resolve()
        )
        record_artifact(
            temp_root / "folds" / "folds_metadata.pkl",
            registry_path=temp_root / REGISTRY_FILENAME,
        )
        with mock.patch.object(cv_utils, "ARTIFACTS_DIR", temp_root):
            folds_metadata = load_folds_metadata()

    # Only fold_* keys are folds; versioned schema metadata keys may coexist.
    fold_keys = {key for key in folds_metadata if key.startswith("fold_")}
    assert fold_keys == {f"fold_{fold_id}" for fold_id in range(5)}


def test_no_live_references() -> None:
    patterns = {
        symbol: re.compile(rf"\b{re.escape(symbol)}\b")
        for symbol in DEAD_SYMBOLS
    }
    hits = []
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        if TESTS_DIR in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for symbol, pattern in patterns.items():
            if pattern.search(text):
                hits.append(f"{path.relative_to(REPO_ROOT)}: {symbol}")

    assert not hits, "live references remain: " + ", ".join(hits)


def test_doctrine_bullets_removed() -> None:
    text = AGENTS_PATH.read_text(encoding="utf-8")

    for bullet in REMOVED_BULLETS:
        assert bullet not in text, f"stale bullet remains: {bullet}"

    for bullet in RETAINED_BULLETS:
        assert bullet in text, f"unrelated bullet was removed: {bullet}"


if __name__ == "__main__":
    import inspect

    from conftest import DATASET_PATH, OUTPUTS_DIR

    fixtures = {
        "repo_root": REPO_ROOT,
        "artifacts_dir": ARTIFACTS_DIR,
        "outputs_dir": OUTPUTS_DIR,
        "dataset_path": DATASET_PATH,
    }

    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            kwargs = {
                parameter: fixtures[parameter]
                for parameter in inspect.signature(_fn).parameters
            }
            try:
                _fn(**kwargs)
                print(f"PASS {_name}")
            except pytest.skip.Exception as exc:
                skipped += 1
                print(f"SKIP {_name}: {exc}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    suffix = f" ({skipped} skipped)" if skipped else ""
    print(f"All tests passed{suffix}")
