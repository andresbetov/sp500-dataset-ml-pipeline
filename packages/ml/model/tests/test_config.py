"""Tests for the model package's config single source and its consumers.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_config.py
"""

from __future__ import annotations

import importlib
import os
import tempfile
import traceback
from pathlib import Path

import joblib
import numpy as np
import yaml

from conftest import requires_generated_data, requires_live_artifacts
from packages.ml.model import config
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.training import model_trainer, training
from packages.ml.model.utils import get_project_paths

# Fold floor-row counts frozen from the holdout-aware split recomputed from the
# live inputs (Y <= rv_floor, training rows and test rows respectively).
EXPECTED_FLOOR_TRAIN = [683, 690, 1267, 1997, 2561]
EXPECTED_FLOOR_TEST = [7, 581, 733, 557, 44]


def _reload_config_consumers() -> None:
    """Reload the config module and every module that reads constants from it."""
    importlib.reload(config)
    importlib.reload(cross_validator)
    importlib.reload(model_trainer)
    importlib.reload(training)


def test_defaults() -> None:
    cfg = config.load_config()

    assert cfg["rv_floor"] == 1e-12
    assert cfg["folds"] == {
        "n_folds": 5,
        "purge_horizon_days": 5,
        "label_horizon_days": 5,
        "separation_min": 6,
    }
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["n_jobs"] == -1
    assert cfg["tuning"]["select_on"] == "validation_qlike"
    assert cfg["holdout"]["n_dates"] == 500
    assert cfg["paths"]["dataset"] == "data/processed/dataset.parquet"
    assert cfg["paths"]["artifacts"] == "packages/ml/model/artifacts"
    assert cfg["paths"]["outputs"] == "data/model_outputs"

    # The default path is base.yaml next to the loader, not the CWD.
    assert config.get_config_path() == Path(config.__file__).resolve().parent / "base.yaml"
    assert config.get_config() == cfg

    # Every load is a fresh dict: mutating one result must not leak into the next.
    fresh = config.load_config()
    fresh["folds"]["n_folds"] = 999
    assert config.load_config()["folds"]["n_folds"] == 5


def test_propagation() -> None:
    changed = config.load_config(config.get_config_path())
    changed["rv_floor"] = 1e-9
    changed["folds"]["n_folds"] = 7
    changed["training"]["seed"] = 7
    changed["training"]["n_jobs"] = 2

    previous_env = os.environ.get("MODEL_CONFIG_PATH")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "base.yaml"
        temp_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        os.environ["MODEL_CONFIG_PATH"] = str(temp_path)
        try:
            assert config.get_config_path() == temp_path
            _reload_config_consumers()

            assert training.RV_FLOOR == 1e-9
            assert cross_validator.N_FOLDS == 7
            assert cross_validator.PURGE_HORIZON_D == 5
            assert cross_validator.LABEL_HORIZON_DAYS == 5
            assert cross_validator.SEPARATION_MIN == 6
            assert model_trainer.RANDOM_STATE == 7
            assert model_trainer.N_JOBS == 2
        finally:
            if previous_env is None:
                os.environ.pop("MODEL_CONFIG_PATH", None)
            else:
                os.environ["MODEL_CONFIG_PATH"] = previous_env
            _reload_config_consumers()

    assert training.RV_FLOOR == 1e-12
    assert cross_validator.N_FOLDS == 5


def test_no_behavior_drift(artifacts_dir: Path) -> None:
    requires_live_artifacts(artifacts_dir)

    folds_metadata = joblib.load(artifacts_dir / "folds" / "folds_metadata.pkl")
    Y = np.load(artifacts_dir / "inputs" / "Y.npy")
    metadata = joblib.load(artifacts_dir / "inputs" / "metadata.pkl")

    dates = metadata["date"]
    unique_dates = np.sort(np.unique(dates))
    holdout_n_dates = cross_validator.HOLDOUT_N_DATES
    fold_dates = unique_dates[:-holdout_n_dates]
    tpos = np.searchsorted(fold_dates, dates)

    rv_floor = config.get_config()["rv_floor"]
    boundaries = cross_validator._fold_boundaries(len(fold_dates))

    # The regenerated stored payload is schema-versioned; only its fold_* keys
    # are folds.
    stored_fold_keys = {key for key in folds_metadata if key.startswith("fold_")}
    assert stored_fold_keys == {
        f"fold_{fold_idx}" for fold_idx in range(cross_validator.N_FOLDS)
    }

    # The holdout-aware membership is recomputed from the live inputs and
    # compared index-for-index against the stored artifact on both sides; the
    # exact membership contract is also asserted in tests/test_purge.py.
    train_sets: list[set[int]] = []
    test_sets: list[set[int]] = []
    for fold_idx in range(cross_validator.N_FOLDS):
        fold = folds_metadata[f"fold_{fold_idx}"]
        block_start = int(boundaries[fold_idx + 1])
        block_end = int(boundaries[fold_idx + 2])

        train_indices, test_indices, purge_train_count = (
            cross_validator._split_date_blocks(tpos, block_start, block_end)
        )

        # Structure of the recomputed split: the test block is kept whole, the
        # train side stays strictly before it, and the train purge is non-empty.
        assert purge_train_count > 0, fold_idx
        assert np.array_equal(
            test_indices,
            np.flatnonzero((tpos >= block_start) & (tpos < block_end)),
        ), fold_idx
        assert (tpos[train_indices] < block_start).all(), fold_idx
        assert (dates[train_indices] < unique_dates[len(fold_dates)]).all(), fold_idx
        assert (dates[test_indices] < unique_dates[len(fold_dates)]).all(), fold_idx
        train_sets.append(set(train_indices.tolist()))
        test_sets.append(set(test_indices.tolist()))

        # Stored-vs-fresh equality on both membership sides, plus the stored
        # purge count and the stored block-edge indices.
        assert np.array_equal(
            train_indices, np.asarray(fold["train_indices"])
        ), fold_idx
        assert np.array_equal(test_indices, np.asarray(fold["test_indices"])), fold_idx
        assert purge_train_count == fold["purge_train_count"], fold_idx
        assert int(tpos[train_indices].max()) == fold["last_train_idx"], fold_idx
        assert int(tpos[test_indices].min()) == fold["first_test_idx"], fold_idx

        # Floor census: Y is untouched, so the frozen counts still describe the
        # fold population.
        floor_train = int(np.count_nonzero(Y[train_indices] <= rv_floor))
        floor_test = int(np.count_nonzero(Y[test_indices] <= rv_floor))
        assert floor_train == EXPECTED_FLOOR_TRAIN[fold_idx]
        assert floor_test == EXPECTED_FLOOR_TEST[fold_idx]

    # Training windows expand (kept train sets grow strictly) and the test
    # blocks stay pairwise disjoint.
    for earlier, later in zip(train_sets, train_sets[1:]):
        assert earlier < later
    seen: set[int] = set()
    for test_set in test_sets:
        assert not (seen & test_set)
        seen |= test_set


def test_path_resolution(
    repo_root: Path,
    artifacts_dir: Path,
    outputs_dir: Path,
    dataset_path: Path,
) -> None:
    requires_generated_data(artifacts_dir, dataset_path, outputs_dir)

    paths = config.get_config()["paths"]
    project_paths = get_project_paths()

    resolved_dataset = (repo_root / paths["dataset"]).resolve()
    assert resolved_dataset == project_paths["dataset_path"]
    assert resolved_dataset.exists()

    resolved_artifacts = (repo_root / paths["artifacts"]).resolve()
    assert resolved_artifacts == project_paths["artifacts_dir"]
    assert resolved_artifacts.is_dir()

    resolved_outputs = (repo_root / paths["outputs"]).resolve()
    assert resolved_outputs == project_paths["outputs_dir"]
    assert resolved_outputs.is_dir()


if __name__ == "__main__":
    import inspect

    import pytest

    from conftest import ARTIFACTS_DIR, DATASET_PATH, OUTPUTS_DIR, REPO_ROOT

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
