"""Tests for the single floor policy, row masks, and the empty-fold guard.

At-or-below-floor target rows are flat-window artifacts: the same policy excludes
them from model fitting, model scoring, and the HAR benchmark, and the fold
artifacts record the original test indices, the aligned retained mask, the scored
indices, and reconciled counts. An all-floored fold — every raw training target at
or below ``RV_FLOOR`` — leaves the policy with zero training rows and must raise a
named ``ValueError`` before any per-fold log line or trainer entry-point call. The
same setups run through ``phase_4_training`` against temporary live-shaped
artifact sets, so no real pipeline run and no real artifacts are involved.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_floor.py
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
import pytest

from conftest import ARTIFACTS_DIR, DATASET_PATH, requires_live_artifacts
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.training import training

N_ROWS = 60
BLOCK_ROWS = 10
N_INDICATORS = 3
N_TICKER_FEATURES = 1
ALL_FLOORED_TRAIN_SIZE = BLOCK_ROWS
# Fold 0 keeps 10 rows over 10 dates; two floored rows leave 8 retained rows
# whose fit side survives the 5-date label-horizon purge before the tail.
POLICY_FLOORED_TRAIN_SIZE = 2
# One at-or-below-floor test row per fold test block, plus a smallest-positive
# row in fold 0's test block so the retained side is exercised too.
TEST_FLOOR_ROWS = (12, 25, 38, 42, 57)
TEST_SMALLEST_POSITIVE_ROW = 13
LIVE_FOLD_0_KEPT_ROWS = 337_067
# The fold-0 floor-retained model-vs-HAR QLIKE edge recomputed from the
# regenerated stored predictions and HAR coefficients.
LIVE_FOLD_0_FLOOR_RETAINED_EDGE = 0.068318


def _synthetic_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Constant components with no warm-up rows: this suite pins the floor policy."""
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _write_synthetic_artifacts(
    root: Path,
    *,
    floored_train_size: int = ALL_FLOORED_TRAIN_SIZE,
    floor_test_rows: tuple[int, ...] = TEST_FLOOR_ROWS,
) -> None:
    """Write a live-shaped Phase 2-3 input set under ``root``.

    The first ``floored_train_size`` rows are exactly zero (the flat-window
    artifact that the floor exists for). Each fold's test block carries one
    at-or-below-floor row, and fold 0's test block also carries the smallest
    positive value, so excluded and retained test rows are both exercised. The
    first feature is the row index, letting a recording model identify exactly
    which rows it scored.
    """
    rng = np.random.default_rng(23)
    y = rng.uniform(0.05, 0.6, size=N_ROWS)
    y[:floored_train_size] = 0.0
    for row in floor_test_rows:
        y[row] = training.RV_FLOOR
    y[TEST_SMALLEST_POSITIVE_ROW] = np.nextafter(training.RV_FLOOR, np.inf)
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


def _run_fold_loop(
    *,
    trainer=None,
    logger=None,
    har_fit=None,
    floored_train_size: int = ALL_FLOORED_TRAIN_SIZE,
    return_inputs: bool = False,
) -> tuple:
    """Run phase_4_training against synthetic artifacts under a temporary root.

    Only the artifact locations, the HAR loader (which reads the real dataset
    parquet), and optionally the trainer/HAR entry points are redirected; the
    prerequisite pass, floor policy, and fold loop run as in the pipeline.
    ``return_inputs`` appends the synthetic raw targets and fold metadata to the
    returned tuple so checks can reconstruct masks outside the temporary root.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_synthetic_artifacts(root, floored_train_size=floored_train_size)
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
            if har_fit is not None:
                stack.enter_context(
                    mock.patch.object(training, "fit_har_benchmark", har_fit)
                )
            if logger is not None:
                stack.enter_context(mock.patch.object(training, "logger", logger))
            result = training.phase_4_training()
            if return_inputs:
                y = np.load(root / "inputs" / "Y.npy")
                folds_metadata = joblib.load(root / "folds" / "folds_metadata.pkl")
                return (*result, y, folds_metadata)
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
    """Trainer double: records fit and validation rows, returns a recording model."""

    def trainer(X_train, y_train, X_val=None, y_val=None):
        record["train_rows"].append(np.asarray(X_train)[:, 0].astype(np.int64))
        record["val_rows"].append(
            None if X_val is None else np.asarray(X_val)[:, 0].astype(np.int64)
        )
        return _RecordingModel(record["scored_rows"]), {}

    return trainer


def _har_spy(real_fit, calls: list, results: list):
    """Wrap the HAR entry point, capturing arguments and the returned dict."""

    def spy(**kwargs):
        calls.append(
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in kwargs.items()
            }
        )
        result = real_fit(**kwargs)
        results.append(result)
        return result

    return spy


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


def test_floor_policy_semantics_are_strict_at_the_boundary() -> None:
    floor = training.RV_FLOOR
    values = np.array(
        [
            0.0,
            floor,
            np.nextafter(floor, -np.inf),
            np.nextafter(floor, np.inf),
            floor * 2.0,
            0.25,
        ]
    )

    mask = training.floor_retained_mask(values)

    assert mask.dtype == bool
    assert mask.tolist() == [False, False, False, True, True, True]
    assert training.floor_retained_mask([]).tolist() == []


def test_floor_policy_and_headline_are_single_source() -> None:
    source = Path(training.__file__).read_text(encoding="utf-8")

    # Exactly one comparison implements the policy; every consumer calls it.
    assert source.count("> RV_FLOOR") == 1
    assert "<= RV_FLOOR" not in source
    # The declared headline row set is a module constant reused by name.
    assert source.count(f'"{training.HEADLINE_ROW_SET}"') == 1


def test_floor_policy_symmetry() -> None:
    record = {"train_rows": [], "val_rows": [], "scored_rows": []}
    har_calls: list = []
    har_results: list = []
    real_fit = training.fit_har_benchmark
    har_spy = _har_spy(real_fit, har_calls, har_results)

    validation, summary, _, y, folds_metadata = _run_fold_loop(
        trainer=_recording_trainer(record),
        har_fit=har_spy,
        floored_train_size=POLICY_FLOORED_TRAIN_SIZE,
        return_inputs=True,
    )

    assert len(record["scored_rows"]) == 5
    assert len(har_calls) == 5
    assert len(har_results) == 5

    # The fixture stores one row per trading date, so date order is row order.
    dates = np.datetime64("2000-01-03") + np.arange(N_ROWS)
    date_pos = np.unique(dates, return_inverse=True)[1]

    for fold_idx in range(5):
        fold_key = f"fold_{fold_idx}"
        fold_data = folds_metadata[fold_key]
        original_test = np.asarray(fold_data["test_indices"])
        expected_mask = np.asarray(y[original_test]) > training.RV_FLOOR
        retained_test = original_test[expected_mask]
        expected_train = training._model_training_indices(fold_data, y)
        expected_fit, expected_tail, expected_purge = training.inner_validation_split(
            fold_idx, expected_train, date_pos, y
        )

        # Both sides of the fold hold excluded rows, so the checks cannot pass
        # vacuously on a fold with nothing to drop.
        assert expected_mask.any()
        assert not expected_mask.all()
        assert len(expected_train) < len(fold_data["train_indices"])

        # The model fits the inner-fit rows and early-stops on the tail; both
        # are floor-retained and test-disjoint. The HAR benchmark keeps the
        # full floor-retained fold-train, so no at-or-below-floor row reaches
        # fitting, scoring, or the benchmark.
        assert np.array_equal(record["train_rows"][fold_idx], expected_fit)
        assert np.array_equal(record["val_rows"][fold_idx], expected_tail)
        assert np.array_equal(record["scored_rows"][fold_idx], retained_test)
        assert np.array_equal(har_calls[fold_idx]["train_indices"], expected_train)
        assert np.array_equal(har_calls[fold_idx]["test_indices"], retained_test)
        assert np.array_equal(
            har_calls[fold_idx]["target_log_test"],
            np.log(np.maximum(y[retained_test], training.RV_FLOOR)),
        )
        assert np.all(y[record["train_rows"][fold_idx]] > training.RV_FLOOR)
        assert np.all(y[record["val_rows"][fold_idx]] > training.RV_FLOOR)
        assert np.all(y[record["scored_rows"][fold_idx]] > training.RV_FLOOR)
        assert np.all(y[har_calls[fold_idx]["test_indices"]] > training.RV_FLOOR)
        assert not set(retained_test.tolist()) & set(expected_fit.tolist())
        assert not set(retained_test.tolist()) & set(expected_tail.tolist())

        # The stored fold record exposes the same retained set and the inner
        # split geometry.
        assert np.array_equal(
            validation[fold_key]["test_floor_mask"], expected_mask
        )
        assert np.array_equal(
            validation[fold_key]["test_indices_scored"], retained_test
        )
        assert validation[fold_key]["n_inner_val_samples"] == len(expected_tail)
        assert validation[fold_key]["n_inner_purge_rows"] == expected_purge


def test_floor_row_masks_emitted() -> None:
    record = {"train_rows": [], "val_rows": [], "scored_rows": []}
    har_calls: list = []
    har_results: list = []
    real_fit = training.fit_har_benchmark
    har_spy = _har_spy(real_fit, har_calls, har_results)

    validation, summary, _, y, folds_metadata = _run_fold_loop(
        trainer=_recording_trainer(record),
        har_fit=har_spy,
        floored_train_size=POLICY_FLOORED_TRAIN_SIZE,
        return_inputs=True,
    )

    for fold_idx in range(5):
        fold_key = f"fold_{fold_idx}"
        fold_data = folds_metadata[fold_key]
        original_test = np.asarray(fold_data["test_indices"])
        expected_mask = np.asarray(y[original_test]) > training.RV_FLOOR
        retained_test = original_test[expected_mask]
        n_train_floor_excluded = len(fold_data["train_indices"]) - len(
            training._model_training_indices(fold_data, y)
        )

        prediction = validation[fold_key]
        assert np.array_equal(prediction["test_indices"], original_test)
        assert prediction["test_floor_mask"].dtype == bool
        assert np.array_equal(prediction["test_floor_mask"], expected_mask)
        assert np.array_equal(prediction["test_indices_scored"], retained_test)

        expected_counts = {
            "n_test_total": len(original_test),
            "n_test_floor_excluded": int(len(original_test) - expected_mask.sum()),
            "n_test_scored": int(expected_mask.sum()),
            "n_train_floor_excluded": n_train_floor_excluded,
        }
        for key, expected in expected_counts.items():
            assert prediction[key] == expected, f"{fold_key}.{key}"
            assert summary[fold_key][key] == expected, f"summary.{fold_key}.{key}"

        assert (
            prediction["n_test_scored"] + prediction["n_test_floor_excluded"]
            == prediction["n_test_total"]
        )
        assert prediction["n_test_samples"] == prediction["n_test_scored"]
        assert len(prediction["test_pred"]) == prediction["n_test_scored"]
        assert len(prediction["test_true"]) == prediction["n_test_scored"]
        assert prediction["headline_row_set"] == training.HEADLINE_ROW_SET
        assert summary[fold_key]["headline_row_set"] == training.HEADLINE_ROW_SET

        # The HAR result carries the matching counts and the declaration.
        har_result = har_results[fold_idx]
        assert har_result["n_test"] == prediction["n_test_scored"]
        assert har_result["n_test_total"] == prediction["n_test_total"]
        assert (
            har_result["n_test_floor_excluded"]
            == prediction["n_test_floor_excluded"]
        )
        assert har_result["n_test_scored"] == prediction["n_test_scored"]
        assert (
            har_result["n_train_floor_excluded"]
            == prediction["n_train_floor_excluded"]
        )
        assert har_result["headline_row_set"] == training.HEADLINE_ROW_SET


def test_floor_policy_non_regression() -> None:
    # Training membership is unchanged: the live fold-0 set is pinned by the
    # live check below, and the policy still keeps strict `>` semantics.
    low = training.qlike_loss(np.array([-23.025851]), np.array([0.0]))
    high = training.qlike_loss(np.array([23.025851]), np.array([0.0]))
    assert low == pytest.approx(22.025851, rel=1e-9)
    assert high == pytest.approx(1e10, rel=1e-6)


def test_live_floor_retained_edge_reference() -> None:
    requires_live_artifacts(ARTIFACTS_DIR)
    predictions_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"
    har_report_path = ARTIFACTS_DIR / "results" / "har_benchmark.json"
    metadata_path = ARTIFACTS_DIR / "inputs" / "metadata.pkl"
    if not (
        predictions_path.is_file()
        and har_report_path.is_file()
        and metadata_path.is_file()
        and DATASET_PATH.is_file()
    ):
        pytest.skip(
            "requires stored live predictions, HAR report, metadata, and dataset parquet"
        )

    from packages.ml.model.training.har_benchmark import load_har_regressors

    folds_metadata = joblib.load(ARTIFACTS_DIR / "folds" / "folds_metadata.pkl")
    y = np.asarray(np.load(ARTIFACTS_DIR / "inputs" / "Y.npy", mmap_mode="r"))
    metadata = joblib.load(metadata_path)
    prediction = joblib.load(predictions_path)["fold_0"]
    test_indices = np.asarray(folds_metadata["fold_0"]["test_indices"])
    retained_mask = np.asarray(y[test_indices]) > training.RV_FLOOR
    assert retained_mask.any() and not retained_mask.all()

    # Predictions store the floor-retained arrays: the stored scored indices,
    # targets, and forecasts all align with the retained side of the fold-0
    # test block.
    scored_indices = test_indices[retained_mask]
    assert np.array_equal(prediction["test_indices_scored"], scored_indices)
    assert len(prediction["test_true"]) == len(prediction["test_pred"]) == len(
        scored_indices
    )
    assert np.array_equal(
        prediction["test_true"],
        np.log(np.maximum(y[scored_indices], training.RV_FLOOR)),
    )

    coefficients = np.asarray(
        json.loads(har_report_path.read_text())["folds"]["fold_0"]["coefficients"]
    )
    regressors, _ = load_har_regressors(metadata["date"], metadata["ticker"])
    design = np.column_stack(
        [np.ones(len(test_indices)), regressors[test_indices]]
    )
    forecast_variance = np.maximum(
        design @ coefficients, training.RV_FLOOR**2
    )
    har_forecast_log = 0.5 * np.log(forecast_variance)

    model_qlike = training.qlike_loss(prediction["test_true"], prediction["test_pred"])
    har_qlike = training.qlike_loss(
        prediction["test_true"], har_forecast_log[retained_mask]
    )
    edge = abs(model_qlike - har_qlike)

    assert edge == pytest.approx(
        LIVE_FOLD_0_FLOOR_RETAINED_EDGE, abs=1e-3
    ), f"floor-retained fold-0 model-vs-HAR edge {edge:.8f}"


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
