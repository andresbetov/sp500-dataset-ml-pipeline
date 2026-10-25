"""Tests for the model package's config single source and its consumers.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_config.py
"""

from __future__ import annotations

import importlib
import os
import re
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO_ROOT, requires_generated_data, requires_live_artifacts
from packages.ml.model import config
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cross_validator, cv_utils
from packages.ml.model.training import model_trainer, training
from packages.ml.model.utils import get_project_paths
from packages.ml.model.visualization import training_diagnostics

# Fold floor-row counts frozen from the holdout-aware split recomputed from the
# live inputs (Y <= rv_floor, training rows and test rows respectively).
EXPECTED_FLOOR_TRAIN = [683, 690, 1267, 1997, 2561]
EXPECTED_FLOOR_TEST = [7, 581, 733, 557, 44]

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"

# Hardcoded fold-count idioms that must not appear in product code: a literal
# range over folds (loops and the fold-name list), the combined-subplot slot,
# the fold-count guard comparison, and the guard message.
FOLD_COUNT_LITERAL = re.compile(
    r"range\(5\)|axes_flat\[5\]|(?:==|!=)\s*5\b|Expected 5 folds"
)

# The propagation fixture calendar: the 500-date holdout plus 60 fold dates =
# 12 dates in each of the N_FOLDS + 1 = 5 blocks, so a 4-fold run keeps exact
# block edges and every fold (through the inner-validation tail) retains fit
# rows.
PROPAGATION_N_DATES = 560
PROPAGATION_N_FOLDS = 4
PROPAGATION_TICKERS = ("AAA", "BBB", "CCC")


def _reload_config_consumers() -> None:
    """Reload the config module and every module that reads constants from it."""
    importlib.reload(config)
    importlib.reload(cross_validator)
    importlib.reload(model_trainer)
    importlib.reload(training)
    importlib.reload(training_diagnostics)


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
    assert cfg["training"]["validation_tail_fraction"] == 0.15
    # The trainer's frozen default recipe (the selected tuning configuration);
    # its values are pinned against the campaign's selection artifact in the
    # tuning tests when that live artifact is present.
    assert set(cfg["training"]["hyperparameters"]) == {
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "gamma",
        "n_estimators",
        "random_state",
        "n_jobs",
        "reg_alpha",
        "reg_lambda",
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "tree_method",
        "verbosity",
    }
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
    changed["training"]["hyperparameters"]["random_state"] = 7
    changed["training"]["n_jobs"] = 2
    changed["training"]["hyperparameters"]["n_jobs"] = 2
    changed["training"]["validation_tail_fraction"] = 0.2

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
            assert model_trainer.default_params()["random_state"] == 7
            assert model_trainer.default_params()["n_jobs"] == 2
            assert training.VALIDATION_TAIL_FRACTION == 0.2
        finally:
            if previous_env is None:
                os.environ.pop("MODEL_CONFIG_PATH", None)
            else:
                os.environ["MODEL_CONFIG_PATH"] = previous_env
            _reload_config_consumers()

    assert training.RV_FLOOR == 1e-12
    assert cross_validator.N_FOLDS == 5


@contextmanager
def _injected_fold_count(n_folds: int):
    """Point MODEL_CONFIG_PATH at a config with ``n_folds`` and reload consumers."""
    changed = config.load_config(config.get_config_path())
    changed["folds"]["n_folds"] = n_folds

    previous_env = os.environ.get("MODEL_CONFIG_PATH")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "base.yaml"
        temp_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        os.environ["MODEL_CONFIG_PATH"] = str(temp_path)
        try:
            _reload_config_consumers()
            yield Path(tmpdir)
        finally:
            if previous_env is None:
                os.environ.pop("MODEL_CONFIG_PATH", None)
            else:
                os.environ["MODEL_CONFIG_PATH"] = previous_env
            _reload_config_consumers()


def _write_synthetic_inputs(root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Write a minimal Phase 2 input set to a temporary artifact root."""
    dates = np.repeat(
        pd.bdate_range("2000-01-03", periods=PROPAGATION_N_DATES).values,
        len(PROPAGATION_TICKERS),
    )
    n_rows = len(dates)
    X = np.zeros((n_rows, 4), dtype=np.float64)
    y = np.full(n_rows, 0.25, dtype=np.float64)
    metadata = {
        "date": dates,
        "ticker": np.tile(np.array(PROPAGATION_TICKERS), PROPAGATION_N_DATES),
        "regime": (
            np.arange(n_rows) % len(cross_validator.REGIME_NAMES)
        ).astype(np.int8),
        "n_indicators": 4,
        "n_ticker_features": 0,
    }
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "results").mkdir()
    np.save(inputs / "X.npy", X)
    np.save(inputs / "Y.npy", y)
    joblib.dump(metadata, inputs / "metadata.pkl")
    # The live readers verify recorded hashes before loading; a synthetic input
    # set must carry its registry entry like the real writer's.
    record_artifact(inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME)
    return X, y, metadata


class _StubModel:
    """Picklable stand-in for a trained fold model."""

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), -2.0, dtype=np.float64)


def _stub_trainer(trained: list):
    """Trainer double recording one entry per fold fit."""

    def trainer(X_train, y_train, X_val=None, y_val=None):
        trained.append(len(X_train))
        return _StubModel(), {}

    return trainer


def _stub_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _stub_har_benchmark(**kwargs) -> dict:
    """HAR double: the phase only needs the fold's scored-row contract."""
    n_test = len(kwargs["test_indices"])
    return {
        "fold_key": kwargs["fold_key"],
        "n_train": len(kwargs["train_indices"]),
        "n_test": n_test,
        "n_warmup_excluded": 0,
        "n_warmup_test_rows": 0,
        "har_r2": 0.25,
        "clip_count": 0,
        "clip_rate": 0.0,
        "coefficient_interpretation": kwargs["coefficient_interpretation"],
        "forecast_log": np.full(n_test, -2.0, dtype=np.float64),
    }


def test_no_literal_fold_count_in_product_code() -> None:
    """The configured count is the single source; no fold-count literal remains."""
    offenders = []
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        if TESTS_DIR in path.parents:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if FOLD_COUNT_LITERAL.search(line):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}"
                )
    assert not offenders, "hardcoded fold-count literals: " + "; ".join(offenders)


def test_fold_count_propagates_through_the_config_seam() -> None:
    """n_folds=4 reaches the generator, the guard, the loops, and the names."""
    with _injected_fold_count(PROPAGATION_N_FOLDS) as tmpdir:
        assert cross_validator.N_FOLDS == PROPAGATION_N_FOLDS
        assert training_diagnostics.FOLD_NAMES == [
            f"fold_{fold_idx}" for fold_idx in range(PROPAGATION_N_FOLDS)
        ]
        assert len(training_diagnostics.FOLD_COLORS) == PROPAGATION_N_FOLDS
        assert training_diagnostics._fold_panel_grid(PROPAGATION_N_FOLDS) == (2, 3)
        assert training_diagnostics._fold_panel_grid(7) == (3, 3)

        artifacts = tmpdir / "artifacts"
        X, y, metadata = _write_synthetic_inputs(artifacts)

        with mock.patch.object(cross_validator, "ARTIFACTS_DIR", artifacts):
            folds = cross_validator.phase_3_cross_validation()
        assert set(folds) == {
            f"fold_{fold_idx}" for fold_idx in range(PROPAGATION_N_FOLDS)
        }

        # The guard accepts the persisted 4-fold payload (its schema metadata
        # keys are not folds)...
        training._validate_training_prerequisites(X, y, metadata, folds)

        # ...and the phase-4 loop trains exactly those four folds.
        trained: list[int] = []
        with mock.patch.object(training, "ARTIFACTS_DIR", artifacts), mock.patch.object(
            cv_utils, "ARTIFACTS_DIR", artifacts
        ), mock.patch.object(
            training, "load_har_regressors", _stub_har_regressors
        ), mock.patch.object(
            training, "train_xgboost_model", _stub_trainer(trained)
        ), mock.patch.object(
            training, "fit_har_benchmark", _stub_har_benchmark
        ):
            validation, summary, model_paths = training.phase_4_training()

        fold_keys = {f"fold_{fold_idx}" for fold_idx in range(PROPAGATION_N_FOLDS)}
        assert set(validation) == fold_keys
        assert set(summary) == fold_keys
        assert len(trained) == PROPAGATION_N_FOLDS
        assert len(model_paths) == PROPAGATION_N_FOLDS

        # The visualization loops iterate the same four folds (the combined
        # panel is the fifth slot of the 2x3 grid, one past the last fold).
        scatter_path = artifacts / "four_fold_scatter.png"
        training_diagnostics.plot_actual_vs_predicted(
            validation, output_path=scatter_path
        )
        assert scatter_path.is_file()

    # The default is restored for every other check.
    assert cross_validator.N_FOLDS == 5
    assert training_diagnostics.FOLD_NAMES == [
        f"fold_{fold_idx}" for fold_idx in range(5)
    ]
    assert training_diagnostics._fold_panel_grid(5) == (2, 3)


def _five_fold_payload() -> tuple:
    """A valid guard fixture carrying five fold keys."""
    rng = np.random.default_rng(11)
    n_rows = 60
    X = rng.normal(size=(n_rows, 4))
    y = rng.uniform(0.05, 0.6, size=n_rows)
    metadata = {
        "ticker": np.array([f"T{row % 3}" for row in range(n_rows)]),
        "date": np.datetime64("2000-01-03") + np.arange(n_rows),
        "regime": (np.arange(n_rows) % 7).astype(np.int8),
        "n_indicators": 4,
        "n_ticker_features": 0,
    }
    folds_metadata = {}
    for fold_idx in range(5):
        train_stop = 10 * (fold_idx + 1)
        test_stop = 10 * (fold_idx + 2)
        folds_metadata[f"fold_{fold_idx}"] = {
            "train_indices": list(range(train_stop)),
            "test_indices": list(range(train_stop, test_stop)),
        }
    return X, y, metadata, folds_metadata


def test_fold_count_mismatch_raises_named_error() -> None:
    """A fold_* count that disagrees with the configured count fails by name."""
    X, y, metadata, folds_metadata = _five_fold_payload()

    with _injected_fold_count(PROPAGATION_N_FOLDS):
        with pytest.raises(ValueError, match="Expected 4 folds, got 5"):
            training._validate_training_prerequisites(X, y, metadata, folds_metadata)


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
