"""Tests for the floor-degenerate training-fold guard at the fold-loop call site.

An all-floored fold — every raw training target at or below ``RV_FLOOR`` — leaves
the floor filter with zero training rows. The fold loop must raise a named
``ValueError`` naming the fold, the raw fold-train size, the floor, and the cause
before any per-fold log line or trainer entry-point call. The same setup is
exercised through ``phase_4_training`` against a temporary live-shaped artifact
set, so no real pipeline run and no real artifacts are involved.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_floor.py
"""

from __future__ import annotations

import contextlib
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pytest

from conftest import ARTIFACTS_DIR, requires_live_artifacts
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.training import training

N_ROWS = 60
BLOCK_ROWS = 10
N_INDICATORS = 3
N_TICKER_FEATURES = 1
ALL_FLOORED_TRAIN_SIZE = BLOCK_ROWS
LIVE_FOLD_0_KEPT_ROWS = 366_518


def _synthetic_har_regressors(dates: np.ndarray) -> np.ndarray:
    return np.full((len(dates), 3), 0.04, dtype=np.float64)


def _write_synthetic_artifacts(root: Path) -> None:
    """Write a live-shaped Phase 2-3 input set under ``root``.

    Fold 0's training rows are all exactly zero (the flat-window artifact that
    the floor exists for); its test rows are normal. Later folds inherit those
    degenerate rows in their expanding training window while keeping normal
    rows, mirroring the live fold geometry.
    """
    rng = np.random.default_rng(23)
    y = rng.uniform(0.05, 0.6, size=N_ROWS)
    y[:ALL_FLOORED_TRAIN_SIZE] = 0.0
    X = rng.normal(size=(N_ROWS, N_INDICATORS + N_TICKER_FEATURES))
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
    np.save(inputs / "X.npy", X)
    np.save(inputs / "Y.npy", y)
    joblib.dump(metadata, inputs / "metadata.pkl")
    joblib.dump(folds_metadata, folds / "folds_metadata.pkl")


def _run_fold_loop(*, trainer=None, logger=None) -> tuple:
    """Run phase_4_training against synthetic artifacts under a temporary root.

    Only the artifact locations and the HAR loader (which reads the real dataset
    parquet) are redirected; the prerequisite pass, floor filter, and fold loop
    run as in the pipeline. ``trainer`` and ``logger`` replace the module-level
    trainer entry point and module logger when provided.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_synthetic_artifacts(root)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
            stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(
                    training, "load_har_regressors", _synthetic_har_regressors
                )
            )
            if trainer is not None:
                stack.enter_context(
                    mock.patch.object(training, "train_xgboost_model", trainer)
                )
            if logger is not None:
                stack.enter_context(mock.patch.object(training, "logger", logger))
            return training.phase_4_training()


def test_all_floored_fold_raises_named_value_error() -> None:
    fit_calls = []

    def forbidden_fit(*args, **kwargs):
        fit_calls.append((args, kwargs))
        raise AssertionError(
            "train_xgboost_model was reached for the all-floored fold before the guard"
        )

    with pytest.raises(ValueError) as excinfo:
        _run_fold_loop(trainer=forbidden_fit)

    message = str(excinfo.value)
    assert "fold_0" in message
    assert f"{ALL_FLOORED_TRAIN_SIZE} fold-train rows" in message
    assert f"{training.RV_FLOOR:g}" in message
    assert "at or below the floor" in message
    assert fit_calls == []


def test_guard_fires_before_fold_log_line_and_fit_call() -> None:
    fit_calls = []
    logger = mock.Mock()

    def forbidden_fit(*args, **kwargs):
        fit_calls.append((args, kwargs))
        raise AssertionError(
            "train_xgboost_model was reached for the all-floored fold before the guard"
        )

    with pytest.raises(ValueError):
        _run_fold_loop(trainer=forbidden_fit, logger=logger)

    messages = [call.args[0] for call in logger.info.call_args_list]
    # The mock is wired to the module: pre-loop phase logs were captured.
    assert any("Log target" in message for message in messages)
    assert fit_calls == []
    excluded_fold_lines = [
        message
        for message in messages
        if "degenerate at-or-below-floor rows" in message
    ]
    assert excluded_fold_lines == []
    assert not [message for message in messages if "Training model on" in message]


def test_branch_point_failure_is_the_named_error_not_a_zero_size_reduction() -> None:
    # Without the guard, this setup reaches the trainer's target-range logging
    # line with an empty training target and dies with NumPy's zero-size
    # reduction error; with the guard, the named fold error fires first.
    with pytest.raises(ValueError) as excinfo:
        _run_fold_loop()

    message = str(excinfo.value)
    assert "zero-size array" not in message
    assert "fold_0" in message
    assert "at or below the floor" in message


def test_strict_floor_filter_drops_at_floor_rows_and_keeps_smallest_positive() -> None:
    floor = training.RV_FLOOR
    smallest_positive = np.nextafter(floor, np.inf)
    y = np.array([0.0, floor, smallest_positive, floor * 2.0, 0.25])

    kept = training._model_training_indices(
        {"train_indices": [0, 1, 2, 3, 4]}, y
    )

    assert kept.tolist() == [2, 3, 4]
    assert smallest_positive > floor


def test_live_fold_0_keeps_exactly_the_above_floor_rows() -> None:
    requires_live_artifacts(ARTIFACTS_DIR)

    folds_metadata = joblib.load(ARTIFACTS_DIR / "folds" / "folds_metadata.pkl")
    y = np.load(ARTIFACTS_DIR / "inputs" / "Y.npy", mmap_mode="r")
    fold_0 = folds_metadata["fold_0"]

    raw_train = np.asarray(fold_0["train_indices"])
    kept = training._model_training_indices(fold_0, y)
    at_or_below_floor = int(
        np.count_nonzero(np.asarray(y[raw_train]) <= training.RV_FLOOR)
    )

    assert len(kept) == LIVE_FOLD_0_KEPT_ROWS
    assert len(kept) == len(raw_train) - at_or_below_floor
    assert at_or_below_floor > 0


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
