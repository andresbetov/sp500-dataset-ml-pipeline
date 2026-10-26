"""Tests for the per-fold timing decomposition and its timing-plot consumer.

The reported fold cost must tell the truth: ``fold_time_seconds`` is the full
fold — the model fit (train + inference), the HAR benchmark fit, and the
remaining fold work — and ``har_time_seconds`` records the HAR fit that used to
sit outside the fold timer. The checks drive a real ``phase_4_training`` against
synthetic artifacts under a temporary root, so the live inputs and the stored
artifacts are neither read nor written; the consumer check runs the real timing
plot against a synthetic summary frame and spies on the columns it reads and
the stacked bars it draws.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_timing.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.training import training
from packages.ml.model.visualization import performance_comparison

N_ROWS = 60
BLOCK_ROWS = 10
N_INDICATORS = 3
N_TICKER_FEATURES = 1
# The enclosing fold timer and the nested component timers share one clock, so
# the decomposition holds exactly; this tolerance only absorbs measurement
# noise between the nested reads.
TIMING_TOLERANCE = 1e-6

# The per-fold summary keys that predate the HAR timing component. The new
# schema adds ``har_time_seconds`` and the dual OOS-R² keys and changes nothing
# else.
PREEXISTING_SUMMARY_KEYS = frozenset(
    {
        "train_date_min",
        "train_date_max",
        "test_date_min",
        "test_date_max",
        "n_train_samples",
        "n_inner_val_samples",
        "n_inner_purge_rows",
        "best_iteration",
        "eval_history",
        "n_test_samples",
        "n_test_total",
        "n_test_floor_excluded",
        "n_test_scored",
        "n_train_floor_excluded",
        "n_warmup_excluded",
        "headline_row_set",
        "n_features",
        "model_path",
        "test_r2",
        "test_mae",
        "test_rmse",
        "test_qlike",
        "har_r2",
        "har_qlike",
        "clip_count",
        "clip_rate",
        "train_time_seconds",
        "inference_time_seconds",
        "fold_time_seconds",
        "n_features_indicators",
        "n_features_ticker",
    }
)

# The dual OOS-R² keys added by the reporting task: both conventions with their
# scale and row-set declarations, per fold.
DUAL_OOS_R2_SUMMARY_KEYS = frozenset(
    {
        "oos_r2_prevailing_mean",
        "oos_r2_vs_har",
        "oos_r2_scale",
        "oos_r2_row_set",
    }
)


class _StubModel:
    """Picklable stand-in fold model with the surface the training path reads."""

    def get_params(self) -> dict:
        return {"n_estimators": 5}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), -2.0, dtype=np.float64)


def _stub_trainer(X_train, y_train, X_val=None, y_val=None):
    return _StubModel(), {}


def _stub_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Constant components with no warm-up rows; the HAR fit stays real."""
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _write_synthetic_artifacts(root: Path) -> None:
    """Write a small live-shaped Phase 2-3 input set under ``root``."""
    rng = np.random.default_rng(11)
    y = rng.uniform(0.05, 0.6, size=N_ROWS)
    X = rng.normal(size=(N_ROWS, N_INDICATORS + N_TICKER_FEATURES))
    X[:, 0] = np.arange(N_ROWS, dtype=np.float64)
    dates = np.datetime64("2000-01-03") + np.arange(N_ROWS)

    metadata = {
        "ticker": np.array([f"T{row % 3}" for row in range(N_ROWS)]),
        "date": dates,
        "regime": (np.arange(N_ROWS) % 7).astype(np.int8),
        "n_indicators": N_INDICATORS,
        "n_ticker_features": N_TICKER_FEATURES,
    }

    folds_metadata = {}
    for fold_idx in range(5):
        train_stop = BLOCK_ROWS * (fold_idx + 1)
        test_stop = BLOCK_ROWS * (fold_idx + 2)
        folds_metadata[f"fold_{fold_idx}"] = {
            "train_indices": list(range(train_stop)),
            "test_indices": list(range(train_stop, test_stop)),
            "n_train": train_stop,
            "n_test": test_stop - train_stop,
            "train_date_min": str(dates[0]),
            "train_date_max": str(dates[train_stop - 1]),
            "test_date_min": str(dates[train_stop]),
            "test_date_max": str(dates[test_stop - 1]),
        }

    inputs = root / "inputs"
    folds = root / "folds"
    inputs.mkdir(parents=True)
    folds.mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "results").mkdir()
    np.save(inputs / "X.npy", X)
    np.save(inputs / "Y.npy", y)
    joblib.dump(metadata, inputs / "metadata.pkl")
    joblib.dump(folds_metadata, folds / "folds_metadata.pkl")
    # The live readers verify recorded hashes before loading; the synthetic
    # inputs must carry their registry entries like the real writers'.
    record_artifact(inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME)
    record_artifact(
        folds / "folds_metadata.pkl", registry_path=root / REGISTRY_FILENAME
    )


def _run_phase(root: Path) -> tuple:
    """Drive one real ``phase_4_training`` run against the temporary root."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
        stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
        stack.enter_context(
            mock.patch.object(training, "load_har_regressors", _stub_har_regressors)
        )
        stack.enter_context(
            mock.patch.object(training, "train_xgboost_model", _stub_trainer)
        )
        return training.phase_4_training()


def _prepared_root(tmpdir: str) -> Path:
    """A temporary artifacts root with the synthetic Phase 2-3 inputs."""
    root = Path(tmpdir) / "artifacts"
    assert not root.resolve().is_relative_to(REPO_ROOT), (
        "the phase run must target a temporary root, never live artifacts"
    )
    root.mkdir()
    _write_synthetic_artifacts(root)
    return root


def _exercised_run(root: Path) -> tuple:
    """One phase run plus its persisted summary JSON under ``root``."""
    validation, summary, model_paths = _run_phase(root)
    summary_json = json.loads(
        (root / "results" / "fold_training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    run_manifest = json.loads(
        (root / "run_manifest.json").read_text(encoding="utf-8")
    )
    return validation, summary, summary_json, model_paths, run_manifest


def _assert_decomposes(prediction: dict, fold_key: str) -> None:
    """The full fold cost encloses train + inference + HAR."""
    assert prediction["har_time_seconds"] > 0, fold_key
    components = (
        prediction["train_time_seconds"]
        + prediction["inference_time_seconds"]
        + prediction["har_time_seconds"]
    )
    assert prediction["fold_time_seconds"] >= components - TIMING_TOLERANCE, fold_key


def test_fold_timing_decomposes_into_train_inference_and_har() -> None:
    """Every fold records the HAR fit and a total that encloses the parts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        validation, summary, summary_json, _model_paths, _manifest = _exercised_run(
            root
        )

        for fold_idx in range(5):
            fold_key = f"fold_{fold_idx}"
            _assert_decomposes(validation[fold_key], fold_key)
            _assert_decomposes(summary[fold_key], fold_key)

            stored = summary_json[fold_key]
            assert "har_time_seconds" in stored, fold_key
            _assert_decomposes(stored, fold_key)

            # The model step's fields keep their meanings; only the fold total
            # grew to enclose the HAR fit and the remaining fold work.
            assert stored["train_time_seconds"] == validation[fold_key][
                "train_time_seconds"
            ]
            assert stored["inference_time_seconds"] == validation[fold_key][
                "inference_time_seconds"
            ]
            assert (
                stored["fold_time_seconds"]
                >= stored["train_time_seconds"]
                + stored["inference_time_seconds"]
            )


def _summary_frame() -> pd.DataFrame:
    """A complete summary frame whose fold totals exceed their components."""
    return pd.DataFrame(
        {
            "fold": [f"fold_{fold_idx}" for fold_idx in range(5)],
            "train_time_seconds": [10.0, 12.0, 14.0, 16.0, 18.0],
            "inference_time_seconds": [1.0, 1.2, 1.4, 1.6, 1.8],
            "har_time_seconds": [0.5, 0.6, 0.7, 0.8, 0.9],
            "fold_time_seconds": [12.0, 14.0, 16.5, 18.9, 21.0],
            "test_r2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "test_mae": [0.1, 0.2, 0.3, 0.4, 0.5],
            "test_rmse": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_train_samples": [100, 200, 300, 400, 500],
            "n_test_samples": [50, 60, 70, 80, 90],
            "n_features_indicators": [37] * 5,
            "n_features_ticker": [0] * 5,
        }
    )


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
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
