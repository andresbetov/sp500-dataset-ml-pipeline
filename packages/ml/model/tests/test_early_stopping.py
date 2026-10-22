"""Tests for the purged inner-validation tail and the wired early stopping.

Each fold holds out the temporally last 15% (config-pinned) of its
floor-retained training rows as an inner-validation tail: the tail is ordered
by trading date (the stored rows are ticker-major), purged from the fit side
by the same 5-day label horizon the outer train/test split uses, floor-filtered
on both sides, and disjoint from the fold's test block and the reserved
holdout. The trainer receives the tail as ``X_val``/``y_val`` (``eval_set``)
and stops on validation QLIKE; every fold persists ``best_iteration`` and the
per-round QLIKE history in its predictions and in the fold summary. All checks
run against synthetic folds under temporary artifact roots, so the live inputs
and the stored artifacts are neither read nor written.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_early_stopping.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from packages.ml.model import config as model_config
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cross_validator, cv_utils
from packages.ml.model.training import model_trainer, training
from packages.ml.model.training.qlike_objective import qlike_eval_metric

# One row per trading date for the phase-level fixture: five expanding 10-row
# folds over the first 60 dates plus a 10-date reserved remainder beyond the
# last test block, mirroring the live holdout reservation.
N_ROWS = 70
BLOCK_ROWS = 10
N_FOLD_ROWS = 60
N_INDICATORS = 3
N_TICKER_FEATURES = 1
TICKERS = ("AAA", "BBB", "CCC")
EXCLUDED_DATES = 5
CONFIGURED_ROUNDS = 500
SCALER_TOKENS = ("scaler", "standardscaler", "minmaxscaler", "robustscaler")


def _date_pos(dates: np.ndarray) -> np.ndarray:
    """Per-row position in the sorted unique trading dates."""
    return np.unique(dates, return_inverse=True)[1]


def _expected_tail_size(n_retained: int) -> int:
    """The nearest integer to the configured fraction, at least one row."""
    return max(
        1, int(np.floor(n_retained * training.VALIDATION_TAIL_FRACTION + 0.5))
    )


def _time_ordered_inputs() -> tuple:
    """X/y over one row per trading date; five expanding 10-row folds.

    The first feature is the row index, letting a recording model identify
    exactly which rows it received. The final 10 dates sit beyond the last
    test block and belong to no fold, mirroring the reserved holdout window.
    """
    rng = np.random.default_rng(23)
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
    folds_metadata["holdout"] = {
        "bounds": {
            "start": str(dates[N_FOLD_ROWS]),
            "end": str(dates[-1]),
        },
        "n_dates": N_ROWS - N_FOLD_ROWS,
    }
    return X, y, metadata, folds_metadata, dates


def _write_synthetic_artifacts(root: Path) -> tuple:
    """Write the synthetic Phase 2-3 input set under ``root``."""
    X, y, metadata, folds_metadata, dates = _time_ordered_inputs()
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
    return y, folds_metadata, dates


def _synthetic_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Constant components with no warm-up rows: this suite pins the split."""
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _run_fold_loop(*, trainer=None, persisted: bool = False) -> dict:
    """Run ``phase_4_training`` against synthetic artifacts under a temp root.

    Only the artifact locations and the HAR regressor loader (which reads the
    real dataset parquet) are redirected; the prerequisite pass, floor policy,
    inner split, and fold loop run as in the pipeline. With ``trainer`` unset
    the real ``train_xgboost_model`` runs, so early stopping is exercised
    end-to-end. ``persisted`` appends the on-disk summary JSON and predictions
    pickle to the result.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        y, folds_metadata, dates = _write_synthetic_artifacts(root)
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
            validation, summary, model_paths = training.phase_4_training()
            result = {
                "validation": validation,
                "summary": summary,
                "model_paths": model_paths,
                "y": y,
                "folds_metadata": folds_metadata,
                "dates": dates,
                "X": np.load(root / "inputs" / "X.npy"),
            }
            if persisted:
                result["summary_json"] = json.loads(
                    (root / "results" / "fold_training_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                result["stored_predictions"] = joblib.load(
                    root / "results" / "validation_predictions.pkl"
                )
            return result


class _RecordingModel:
    """Picklable stand-in model that records the row indices it scores."""

    def __init__(self, scored_rows: list) -> None:
        self._scored_rows = scored_rows

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._scored_rows.append(np.asarray(X)[:, 0].astype(np.int64))
        return np.full(len(X), -2.5, dtype=np.float64)


def _recording_trainer(record: dict):
    """Trainer double: records the fit and validation rows it receives."""

    def trainer(X_train, y_train, X_val=None, y_val=None):
        record["train_rows"].append(np.asarray(X_train)[:, 0].astype(np.int64))
        record["val_rows"].append(
            None if X_val is None else np.asarray(X_val)[:, 0].astype(np.int64)
        )
        record["val_targets"].append(None if y_val is None else np.asarray(y_val))
        return _RecordingModel(record["scored_rows"]), {}

    return trainer


def test_tail_is_the_temporal_last_fifteen_percent_purged_by_the_horizon() -> None:
    """Fit, tail, and purge partition fold-train with a label-horizon gap."""
    _, y, _, folds_metadata, dates = _time_ordered_inputs()
    date_pos = _date_pos(dates)

    for fold_idx in range(5):
        fold = folds_metadata[f"fold_{fold_idx}"]
        retained = training._model_training_indices(fold, y)
        fit, tail, purge_count = training.inner_validation_split(
            fold_idx, retained, date_pos, y
        )

        # The tail is the temporally last 15% of the floor-retained fold-train
        # rows (nearest-integer rounding, at least one row). The stored row
        # order is temporal here, so the suffix check is order-for-order.
        n_tail = _expected_tail_size(len(retained))
        assert len(tail) == n_tail, fold_idx
        assert abs(len(tail) - 0.15 * len(retained)) <= 1, fold_idx
        order = np.lexsort((retained, date_pos[retained]))
        assert np.array_equal(tail, np.sort(retained[order][-n_tail:])), fold_idx

        # Fit, tail, and purge partition the floor-retained fold-train.
        purged = np.setdiff1d(retained, np.union1d(fit, tail))
        assert len(purged) == purge_count, fold_idx
        assert np.intersect1d(fit, tail).size == 0, fold_idx

        # Purged by at least the label horizon: the last kept fit date sits
        # EXCLUDED_DATES + 1 date positions before the first tail date, so no
        # fit row's 5-day label window reaches the tail.
        assert (
            date_pos[tail].min() - date_pos[fit].max() >= EXCLUDED_DATES + 1
        ), fold_idx
        assert (
            date_pos[fit] + training.LABEL_HORIZON_DAYS < date_pos[tail].min()
        ).all(), fold_idx

        # No test row enters the tail (or the fit), every test date is
        # strictly after every tail date, and the tail stays before the
        # reserved holdout window.
        test = np.asarray(fold["test_indices"])
        holdout_start = np.datetime64(folds_metadata["holdout"]["bounds"]["start"])
        assert np.intersect1d(tail, test).size == 0, fold_idx
        assert np.intersect1d(fit, test).size == 0, fold_idx
        assert dates[tail].max() < dates[test].min(), fold_idx
        assert dates[tail].max() < holdout_start, fold_idx

        # Both sides are floor-filtered.
        assert (y[fit] > training.RV_FLOOR).all(), fold_idx
        assert (y[tail] > training.RV_FLOOR).all(), fold_idx


def test_tail_orders_by_trading_date_not_by_stored_row_order() -> None:
    """Input rows are ticker-major: the tail is temporal, not the stored suffix."""
    n_dates = 60
    dates = np.tile(np.datetime64("2020-01-02") + np.arange(n_dates), len(TICKERS))
    y = np.full(dates.size, 0.25)
    train_indices = np.flatnonzero(dates < np.datetime64("2020-01-02") + 50)
    test_indices = np.flatnonzero(dates >= np.datetime64("2020-01-02") + 50)
    date_pos = _date_pos(dates)

    fit, tail, _purge = training.inner_validation_split(
        4, train_indices, date_pos, y
    )

    n_tail = _expected_tail_size(len(train_indices))
    assert len(tail) == n_tail

    # The stored suffix is the tail of the last ticker's block, not of the
    # calendar; a row-order split would pick it.
    assert set(tail) != set(train_indices[-n_tail:])

    # The tail is the temporal suffix: its dates are the fold-train calendar's
    # last dates, and the fit side ends at least the label horizon before it.
    tail_dates = np.unique(date_pos[tail])
    assert tail_dates[-1] == date_pos[train_indices].max()
    assert len(tail_dates) == int(np.ceil(n_tail / len(TICKERS)))
    assert date_pos[tail].min() > date_pos[fit].max()
    assert date_pos[tail].min() - date_pos[fit].max() >= EXCLUDED_DATES + 1

    # No test row enters the tail; every test date is after every tail date.
    assert np.intersect1d(tail, test_indices).size == 0
    assert np.intersect1d(fit, test_indices).size == 0
    assert dates[tail].max() < dates[test_indices].min()


def test_tail_and_fit_drop_at_or_below_floor_rows() -> None:
    """The tail size comes from floor-retained rows and neither side carries one."""
    n_rows = 100
    dates = np.datetime64("2000-01-03") + np.arange(n_rows)
    y = np.full(n_rows, 0.25)
    y[[72, 88, 89]] = 0.0  # flat windows inside fold-train
    y[85] = training.RV_FLOOR
    fold = {"train_indices": list(range(90)), "test_indices": list(range(90, 100))}
    retained = training._model_training_indices(fold, y)
    assert not np.isin([72, 85, 88, 89], retained).any()

    fit, tail, _purge = training.inner_validation_split(
        0, retained, np.arange(n_rows), y
    )

    assert len(tail) == _expected_tail_size(len(retained))
    assert not np.isin([72, 85, 88, 89], fit).any()
    assert not np.isin([72, 85, 88, 89], tail).any()
    assert (y[fit] > training.RV_FLOOR).all()
    assert (y[tail] > training.RV_FLOOR).all()

    # Raw (non-floor-retained) input rows are refused, not silently split.
    with pytest.raises(ValueError, match="floor-retained"):
        training.inner_validation_split(0, fold["train_indices"], np.arange(n_rows), y)


def test_degenerate_folds_raise_named_errors() -> None:
    """A tail needs retained rows on both sides of the label-horizon purge."""
    with pytest.raises(ValueError, match="Fold 3") as excinfo:
        training.inner_validation_split(
            3, np.array([0]), np.array([0]), np.array([0.25])
        )
    assert "inner-validation tail" in str(excinfo.value)

    # Six rows over six dates leave a one-row tail and no fit date outside
    # the 5-day horizon.
    with pytest.raises(ValueError, match="no fit rows") as excinfo:
        training.inner_validation_split(
            2, np.arange(6), np.arange(6), np.full(6, 0.25)
        )
    assert "Fold 2" in str(excinfo.value)


def test_phase_wires_the_purged_tail_as_the_trainer_validation_set() -> None:
    """The trainer receives non-None X_val/y_val equal to the tail rows."""
    record = {"train_rows": [], "val_rows": [], "val_targets": [], "scored_rows": []}
    result = _run_fold_loop(trainer=_recording_trainer(record))
    y = result["y"]
    date_pos = _date_pos(result["dates"])

    for fold_idx in range(5):
        fold_key = f"fold_{fold_idx}"
        fold = result["folds_metadata"][fold_key]
        retained = training._model_training_indices(fold, y)
        expected_fit, expected_tail, expected_purge = training.inner_validation_split(
            fold_idx, retained, date_pos, y
        )

        assert record["val_rows"][fold_idx] is not None, fold_key
        assert record["val_rows"][fold_idx].tolist() == expected_tail.tolist()
        assert record["train_rows"][fold_idx].tolist() == expected_fit.tolist()
        assert np.array_equal(
            record["val_targets"][fold_idx],
            np.log(np.maximum(y[expected_tail], training.RV_FLOOR)),
        )
        assert (
            np.intersect1d(expected_tail, np.asarray(fold["test_indices"])).size
            == 0
        ), fold_key

        # The persisted prediction record carries the split geometry.
        prediction = result["validation"][fold_key]
        assert prediction["n_inner_val_samples"] == len(expected_tail), fold_key
        assert prediction["n_inner_purge_rows"] == expected_purge, fold_key


def test_early_stopping_fires_on_rising_validation_qlike() -> None:
    """A tail whose QLIKE rises stops the fit below the 500-round upper bound."""
    # The fixture must give the frozen recipe room to grow trees: constant
    # labels carry zero split gain (gamma > 0 blocks them outright), and the
    # row count must clear the configured min_child_weight Hessian floor. The
    # noisy train labels overfit a far validation tail, so the recorded QLIKE
    # rises round over round and early stopping fires below the bound.
    rng = np.random.default_rng(5)
    X_train = rng.normal(size=(200, 3))
    y_train = rng.normal(loc=-1.0, scale=1.0, size=200)
    X_val = rng.normal(size=(20, 3))
    y_val = np.full(20, 10.0)

    model, _label_mapping = model_trainer.train_xgboost_model(
        X_train, y_train, X_val=X_val, y_val=y_val
    )

    assert model.n_estimators == CONFIGURED_ROUNDS
    assert model.best_iteration is not None
    assert model.best_iteration < model.n_estimators
    assert (
        model.get_params()["early_stopping_rounds"]
        == model_trainer.EARLY_STOPPING_ROUNDS
    )

    scores = model.evals_result()["validation_0"][qlike_eval_metric.__name__]
    assert len(scores) < model.n_estimators
    assert len(scores) == model.best_iteration + 1 + model_trainer.EARLY_STOPPING_ROUNDS
    assert scores[model.best_iteration] == min(scores)
    assert scores[-1] > scores[model.best_iteration]


def test_best_iteration_and_eval_history_persist_per_fold() -> None:
    """Predictions and summary carry best_iteration and the per-round history."""
    result = _run_fold_loop(persisted=True)

    for fold_idx in range(5):
        fold_key = f"fold_{fold_idx}"
        prediction = result["validation"][fold_key]
        stored = result["stored_predictions"][fold_key]
        summary = result["summary"][fold_key]
        summary_json = result["summary_json"][fold_key]

        # Early stopping ran through the real trainer: the eval_set history
        # exists, is finite, and is bounded by the configured upper bound.
        best_iteration = prediction["best_iteration"]
        assert isinstance(best_iteration, int), fold_key
        assert 0 <= best_iteration < CONFIGURED_ROUNDS, fold_key
        history = prediction["eval_history"]
        assert isinstance(history, dict), fold_key
        assert set(history) == {"qlike"}, fold_key
        scores = history["qlike"]
        assert len(scores) >= best_iteration + 1, fold_key
        assert len(scores) <= CONFIGURED_ROUNDS, fold_key
        assert all(isinstance(score, float) for score in scores), fold_key
        assert np.isfinite(scores).all(), fold_key

        # The predictions pickle, the in-memory summary, and the summary JSON
        # all carry the same per-fold record.
        for carrier in (stored, summary, summary_json):
            assert carrier["best_iteration"] == best_iteration, fold_key
            assert carrier["eval_history"] == history, fold_key

        # The split geometry reconciles with the fold-train census:
        # fit + tail + purge + floor-excluded = raw fold-train.
        raw_n_train = result["folds_metadata"][fold_key]["n_train"]
        assert (
            prediction["n_train_samples"]
            + prediction["n_inner_val_samples"]
            + prediction["n_inner_purge_rows"]
            + prediction["n_train_floor_excluded"]
            == raw_n_train
        ), fold_key


def test_non_regression_outer_purge_floor_policy_and_config() -> None:
    """The outer purge, the floor policy, and the config pin are unchanged."""
    # The outer train-only purge keeps the branch-point geometry: five
    # excluded trading dates between the kept train and test sides.
    tickers = np.array(TICKERS)
    n_dates = 536
    dates = np.repeat(
        pd.bdate_range("2000-01-03", periods=n_dates).values, len(tickers)
    )
    fold_dates = np.sort(np.unique(dates))[:36]
    tpos = np.searchsorted(fold_dates, dates)
    boundaries = cross_validator._fold_boundaries(len(fold_dates))
    purge_counts = []
    for fold_idx in range(cross_validator.N_FOLDS):
        _, _, purge_train_count = cross_validator._split_date_blocks(
            tpos, int(boundaries[fold_idx + 1]), int(boundaries[fold_idx + 2])
        )
        purge_counts.append(purge_train_count)
    assert purge_counts == [EXCLUDED_DATES * len(tickers)] * cross_validator.N_FOLDS

    # The floor policy is still the single strict predicate, and the inner
    # split consumes only its output.
    values = np.array(
        [0.0, training.RV_FLOOR, np.nextafter(training.RV_FLOOR, np.inf), 0.25]
    )
    assert training.floor_retained_mask(values).tolist() == [False, False, True, True]
    assert training._model_training_indices(
        {"train_indices": [0, 1, 2, 3]}, values
    ).tolist() == [2, 3]

    # The tail fraction is config-pinned and read from the config file.
    loaded = model_config.load_config()
    assert (
        training.VALIDATION_TAIL_FRACTION
        == loaded["training"]["validation_tail_fraction"]
        == 0.15
    )

    # No scaler vocabulary is introduced on the touched path.
    for relative_path in (
        "training/training.py",
        "training/model_trainer.py",
        "training/qlike_objective.py",
    ):
        source = (REPO_ROOT / "packages" / "ml" / "model" / relative_path).read_text(
            encoding="utf-8"
        ).lower()
        introduced = [token for token in SCALER_TOKENS if token in source]
        assert not introduced, f"{relative_path}: {introduced}"


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
