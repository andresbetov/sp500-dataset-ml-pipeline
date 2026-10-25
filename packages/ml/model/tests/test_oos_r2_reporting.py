"""Tests for the dual OOS-R² reporting conventions.

The fold summary used to carry a single R²: sklearn's fold-own-mean value
(``test_r2``), whose denominator is each fold's own test mean. That single
number is not comparable to a pooled non-demeaned OOS-R² or to the HAR
benchmark, so the training phase now emits both honest conventions beside it,
per fold and pooled, each with its scale and row set stated:

- ``oos_r2_prevailing_mean`` = ``1 - SSE_model/SSE_mean`` on the log target
  over the floor-retained scored rows; the mean is the fold's own test mean per
  fold and the pooled test mean at the pooled level;
- ``oos_r2_vs_har`` = ``1 - SSE_model/SSE_HAR`` on the same rows, against the
  HAR benchmark's persisted per-row log forecasts.

These checks pin:

- a synthetic 2-fold phase run whose three conventions differ, with every
  emitted value (fold summary JSON, validation predictions, pooled HAR report
  block) matching explicit SSE hand computations to 1e-12;
- the fold-own-mean predictor identity: per-fold exactly 0, pooled ≠ 0
  (the pooled prevailing mean is the pooled test mean, not the fold means);
- the persisted per-row HAR forecast vector reconciling with the persisted
  HAR coefficients;
- the diagnostics pooled label naming each convention instead of a bare R²;
- a read-only live recompute of the current pooled log R² (and the pooled
  model-vs-HAR value) from the stored predictions and persisted HAR
  coefficients; the pinned live values are interim and recomputed at the
  branch point.

All phase checks run against synthetic artifacts under a temporary root, so
the live inputs and the stored artifacts are neither read nor written; the
live check reads only.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_oos_r2_reporting.py
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from sklearn.metrics import r2_score

from conftest import ARTIFACTS_DIR, DATASET_PATH, requires_live_artifacts
from packages.ml.model import config
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cross_validator, cv_utils
from packages.ml.model.training import model_trainer, training
from packages.ml.model.training.har_benchmark import load_har_regressors
from packages.ml.model.visualization import training_diagnostics

# The synthetic 2-fold phase fixture: one row per trading date, two expanding
# folds over the first 70 dates, and a 30-date remainder beyond the last test
# block. The fold test blocks differ in size (20 and 30 rows) so the pooled
# (pooled-mean) value cannot coincide with the mean of the fold values.
N_ROWS = 100
N_INDICATORS = 3
N_TICKER_FEATURES = 1
FOLD_0_TRAIN_STOP = 20
FOLD_0_TEST_STOP = 40
FOLD_1_TEST_STOP = 70
STUB_COMPONENT = 0.04

# The pooled live values recomputed read-only at this branch point from the
# current stored predictions and the persisted HAR coefficients; they are
# interim and regenerate with the artifacts.


class _RecordingModel:
    """Picklable fold-model double with deterministic row-indexed forecasts."""

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        rows = np.asarray(X)[:, 0].astype(np.float64)
        return -1.4 + 0.05 * np.sin(rows)


def _stub_trainer(X_train, y_train, X_val=None, y_val=None):
    return _RecordingModel(), {}


def _stub_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Constant components with no warm-up rows; the HAR fit stays real."""
    return (
        np.full((len(dates), 3), STUB_COMPONENT, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _sse(values: np.ndarray, predictions: np.ndarray) -> float:
    """Explicit sum of squared errors, the hand-computation form."""
    return float(np.sum((np.asarray(values) - np.asarray(predictions)) ** 2))


def _reload_config_consumers() -> None:
    """Reload the config module and every module that reads constants from it."""
    importlib.reload(config)
    importlib.reload(cross_validator)
    importlib.reload(model_trainer)
    importlib.reload(training)
    importlib.reload(training_diagnostics)


@contextmanager
def _injected_fold_count(n_folds: int):
    """Run consumers with a temporary config whose ``folds.n_folds`` differs."""
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


def _write_two_fold_artifacts(root: Path) -> None:
    """Write the synthetic 2-fold Phase 2-3 input set under ``root``."""
    rng = np.random.default_rng(29)
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
    folds_metadata = {
        "fold_0": {
            "train_indices": list(range(FOLD_0_TRAIN_STOP)),
            "test_indices": list(range(FOLD_0_TRAIN_STOP, FOLD_0_TEST_STOP)),
            "n_train": FOLD_0_TRAIN_STOP,
            "n_test": FOLD_0_TEST_STOP - FOLD_0_TRAIN_STOP,
            "train_date_min": str(dates[0]),
            "train_date_max": str(dates[FOLD_0_TRAIN_STOP - 1]),
            "test_date_min": str(dates[FOLD_0_TRAIN_STOP]),
            "test_date_max": str(dates[FOLD_0_TEST_STOP - 1]),
        },
        "fold_1": {
            "train_indices": list(range(FOLD_0_TEST_STOP)),
            "test_indices": list(range(FOLD_0_TEST_STOP, FOLD_1_TEST_STOP)),
            "n_train": FOLD_0_TEST_STOP,
            "n_test": FOLD_1_TEST_STOP - FOLD_0_TEST_STOP,
            "train_date_min": str(dates[0]),
            "train_date_max": str(dates[FOLD_0_TEST_STOP - 1]),
            "test_date_min": str(dates[FOLD_0_TEST_STOP]),
            "test_date_max": str(dates[FOLD_1_TEST_STOP - 1]),
        },
        "holdout": {
            "bounds": {
                "start": str(dates[FOLD_1_TEST_STOP]),
                "end": str(dates[-1]),
            },
            "n_dates": N_ROWS - FOLD_1_TEST_STOP,
        },
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


def _run_two_fold_phase() -> dict:
    """Drive a real 2-fold ``phase_4_training`` under a temporary root.

    Only the artifact locations, the fold count, and the HAR regressor loader
    are redirected; the prerequisite pass, floor policy, inner split, fold
    loop, HAR fit, and every writer run as in the pipeline. All payloads are
    read back inside the temporary root before it is removed.
    """
    with _injected_fold_count(2) as tmpdir:
        root = Path(tmpdir) / "artifacts"
        root.mkdir()
        _write_two_fold_artifacts(root)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
            stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(training, "load_har_regressors", _stub_har_regressors)
            )
            stack.enter_context(
                mock.patch.object(training, "train_xgboost_model", _stub_trainer)
            )
            validation, summary, _model_paths = training.phase_4_training()

        return {
            "validation": validation,
            "summary": summary,
            "summary_json": json.loads(
                (root / "results" / "fold_training_summary.json").read_text(
                    encoding="utf-8"
                )
            ),
            "har_report": json.loads(
                (root / "results" / "har_benchmark.json").read_text(encoding="utf-8")
            ),
            "stored": joblib.load(root / "results" / "validation_predictions.pkl"),
        }


def _capture_combined_label(predictions: dict, output_path: Path) -> list[str]:
    """Render the scatter figure and return the combined panel's text labels."""
    texts: list[str] = []
    real_savefig = plt.savefig

    def spy_savefig(*args, **kwargs):
        figure = plt.gcf()
        texts.extend(text.get_text() for text in figure.axes[-1].texts)
        return real_savefig(*args, **kwargs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(training_diagnostics, "FOLD_NAMES", ["fold_0", "fold_1"])
        )
        stack.enter_context(
            mock.patch.object(
                training_diagnostics, "FOLD_COLORS", ["#1f77b4", "#ff7f0e"]
            )
        )
        stack.enter_context(
            mock.patch.object(training_diagnostics.plt, "savefig", spy_savefig)
        )
        training_diagnostics.plot_actual_vs_predicted(
            predictions, output_path=output_path
        )
    return texts


def test_phase_4_emits_dual_conventions_matching_hand_computations() -> None:
    """Emitted per-fold and pooled values equal explicit SSE hand computations."""
    evidence = _run_two_fold_phase()
    summary_json = evidence["summary_json"]
    har_report = evidence["har_report"]
    stored = evidence["stored"]

    fold_keys = ["fold_0", "fold_1"]
    assert set(stored) == set(evidence["validation"]) == set(evidence["summary"])
    assert set(stored) == set(fold_keys)

    fold_prevailing = []
    for fold_key in fold_keys:
        emitted = summary_json[fold_key]
        assert "oos_r2_prevailing_mean" in emitted, (
            f"{fold_key} summary is missing oos_r2_prevailing_mean; keys: "
            f"{sorted(emitted)}"
        )
        assert "oos_r2_vs_har" in emitted, (
            f"{fold_key} summary is missing oos_r2_vs_har; keys: {sorted(emitted)}"
        )

        stored_fold = stored[fold_key]
        assert "har_forecast_log" in stored_fold, (
            f"{fold_key} predictions are missing har_forecast_log; keys: "
            f"{sorted(stored_fold)}"
        )
        y = np.asarray(stored_fold["test_true"])
        p = np.asarray(stored_fold["test_pred"])
        h = np.asarray(stored_fold["har_forecast_log"])
        assert len(h) == len(y)

        expected_prevailing = 1.0 - _sse(y, p) / _sse(y, np.full_like(y, y.mean()))
        expected_vs_har = 1.0 - _sse(y, p) / _sse(y, h)
        assert abs(expected_prevailing - expected_vs_har) > 1e-6, (
            "the fixture must keep the prevailing-mean and vs-HAR conventions "
            f"distinct ({fold_key})"
        )
        fold_prevailing.append(expected_prevailing)

        assert emitted["oos_r2_prevailing_mean"] == pytest.approx(
            expected_prevailing, rel=1e-12, abs=1e-12
        )
        assert emitted["oos_r2_vs_har"] == pytest.approx(
            expected_vs_har, rel=1e-12, abs=1e-12
        )
        assert emitted["oos_r2_scale"] == "log target"
        assert emitted["oos_r2_row_set"] == training.HEADLINE_ROW_SET
        # The pre-existing key keeps the same fold-own-mean convention.
        assert emitted["test_r2"] == pytest.approx(
            expected_prevailing, rel=1e-12, abs=1e-12
        )
        # The prediction pickle carries the same emitted values and the
        # per-row HAR forecast.
        assert stored_fold["oos_r2_prevailing_mean"] == pytest.approx(
            expected_prevailing, rel=1e-12, abs=1e-12
        )
        assert stored_fold["oos_r2_vs_har"] == pytest.approx(
            expected_vs_har, rel=1e-12, abs=1e-12
        )
        assert stored_fold["oos_r2_scale"] == "log target"
        assert stored_fold["oos_r2_row_set"] == training.HEADLINE_ROW_SET

        # The persisted HAR forecast vector reconciles with the persisted
        # coefficients on the stub components (no refit at report time).
        coefficients = np.asarray(har_report["folds"][fold_key]["coefficients"])
        scored = np.asarray(stored_fold["test_indices_scored"])
        design = np.column_stack(
            [np.ones(scored.size), np.full((scored.size, 3), STUB_COMPONENT)]
        )
        expected_forecast = 0.5 * np.log(
            np.maximum(design @ coefficients, training.RV_FLOOR**2)
        )
        assert np.allclose(h, expected_forecast, rtol=0.0, atol=1e-12)

    y_all = np.concatenate([stored[key]["test_true"] for key in fold_keys])
    p_all = np.concatenate([stored[key]["test_pred"] for key in fold_keys])
    h_all = np.concatenate([stored[key]["har_forecast_log"] for key in fold_keys])
    expected_pooled_prevailing = 1.0 - _sse(y_all, p_all) / _sse(
        y_all, np.full_like(y_all, y_all.mean())
    )
    expected_pooled_vs_har = 1.0 - _sse(y_all, p_all) / _sse(y_all, h_all)

    block = har_report["pooled_oos_r2"]
    assert block["oos_r2_prevailing_mean"] == pytest.approx(
        expected_pooled_prevailing, rel=1e-12, abs=1e-12
    )
    assert block["oos_r2_vs_har"] == pytest.approx(
        expected_pooled_vs_har, rel=1e-12, abs=1e-12
    )
    assert block["scale"] == "log target"
    assert block["row_set"] == training.HEADLINE_ROW_SET
    assert block["n_scored_rows"] == y_all.size
    assert "1 - SSE_model" in block["prevailing_mean_convention"]
    assert "1 - SSE_model" in block["vs_har_convention"]

    # The pooled (pooled-mean) value is not the mean of the fold values, and
    # the two pooled conventions are distinct: the pooled values carry
    # information the per-fold numbers cannot.
    assert abs(expected_pooled_prevailing - np.mean(fold_prevailing)) > 1e-6
    assert abs(expected_pooled_prevailing - expected_pooled_vs_har) > 1e-6


def test_fold_own_mean_predictor_is_zero_per_fold_and_nonzero_pooled() -> None:
    """The fold-mean predictor scores exactly 0 per fold; pooled ≠ 0."""
    folds = {
        "fold_0": np.array([0.10, 0.30, 0.80]),
        "fold_1": np.array([0.40, 0.90, 1.60, 2.00]),
    }
    records = {}
    for fold_key, y in folds.items():
        fold_mean = np.full_like(y, y.mean())
        assert training.oos_r2_prevailing_mean(y, fold_mean) == 0.0
        records[fold_key] = {
            "test_true": y,
            "test_pred": fold_mean,
            "har_forecast_log": fold_mean,
        }

    y_all = np.concatenate([folds[key] for key in sorted(folds)])
    p_all = np.concatenate(
        [np.full_like(folds[key], folds[key].mean()) for key in sorted(folds)]
    )
    expected_pooled = 1.0 - _sse(y_all, p_all) / _sse(
        y_all, np.full_like(y_all, y_all.mean())
    )

    block = training.pooled_oos_r2(records)
    assert block["oos_r2_prevailing_mean"] == pytest.approx(
        expected_pooled, rel=1e-12, abs=1e-12
    )
    assert block["oos_r2_prevailing_mean"] != pytest.approx(0.0, abs=1e-12)


def test_prevailing_mean_matches_sklearn_and_constant_target_convention() -> None:
    """The explicit SSE form equals sklearn's r2_score, degenerate case included."""
    y = np.array([0.10, 0.20, 0.50, 0.80])
    p = np.array([0.15, 0.25, 0.45, 0.70])
    assert training.oos_r2_prevailing_mean(y, p) == pytest.approx(
        r2_score(y, p), rel=1e-12, abs=1e-12
    )

    constant = np.full(4, 0.25)
    assert training.oos_r2_prevailing_mean(constant, constant) == 1.0
    assert training.oos_r2_prevailing_mean(constant, np.full(4, 0.5)) == 0.0


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
