"""Shared pytest fixtures for the model package tests.

Importing this module puts the repository root on ``sys.path`` before any test
imports ``packages.ml.model``, so the suite works regardless of the working
directory pytest is invoked from. The fixtures centralize the generated-data
paths, and the skip guards keep artifact-dependent checks honest on a fresh
clone or in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS_DIR = REPO_ROOT / "packages" / "ml" / "model" / "artifacts"
OUTPUTS_DIR = REPO_ROOT / "data" / "model_outputs"
DATASET_PATH = REPO_ROOT / "data" / "processed" / "dataset.parquet"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root, resolved from this file rather than the CWD."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def artifacts_dir(repo_root: Path) -> Path:
    """Generated model artifacts directory."""
    return repo_root / "packages" / "ml" / "model" / "artifacts"


@pytest.fixture(scope="session")
def outputs_dir(repo_root: Path) -> Path:
    """Generated evaluation-chart directory."""
    return repo_root / "data" / "model_outputs"


@pytest.fixture(scope="session")
def dataset_path(repo_root: Path) -> Path:
    """Phase 1 dataset parquet."""
    return repo_root / "data" / "processed" / "dataset.parquet"


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def requires_live_artifacts(artifacts_dir: Path) -> None:
    """Skip when the generated model inputs or fold indices are absent."""
    missing = [
        path
        for path in (artifacts_dir / "inputs", artifacts_dir / "folds")
        if not path.is_dir()
    ]
    if missing:
        pytest.skip(
            "requires generated model artifacts absent from this checkout: "
            + ", ".join(_display(path) for path in missing)
        )


def requires_generated_data(
    artifacts_dir: Path, dataset_path: Path, outputs_dir: Path
) -> None:
    """Skip when a path asserted by the resolution checks is absent."""
    missing = [
        path
        for path in (artifacts_dir, dataset_path, outputs_dir)
        if not path.exists()
    ]
    if missing:
        pytest.skip(
            "requires generated data absent from this checkout: "
            + ", ".join(_display(path) for path in missing)
        )
