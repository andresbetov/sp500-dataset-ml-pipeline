"""Tests for the documented HAR forecast constraint and its clip counters.

The HAR benchmark applies one declared forecast constraint: every scored
fold's test forecast variance is clipped at ``forecast_floor**2``, equivalently
``forecast_log >= ln(forecast_floor)``, with the floor coming from the shared
``rv_floor`` config. These checks pin:

- a deterministic fixture whose fitted forecast falls below the floor reports
  ``clip_count > 0``, ``clip_rate == clip_count / n_test``, and logs a warning
  naming the fold and the rate;
- a live-shaped fixture with comfortable margins reports zero clips and stays
  silent;
- the module text documents the variance-floor clip, the
  ``forecast_log >= ln(forecast_floor)`` equivalence with the ``-27.631021``
  numeric bound, and the shared-config floor source;
- the counted clip equals the applied clip exactly at the counted rows, and
  ``floor**2 == exp(2 * ln(floor))`` within ``1e-12``;
- the per-fold counters reach the fold summary, and the global HAR report
  carries summed counts over summed scored rows with an alert when any fold
  clipped;
- live data (read-only, skipped without the generated artifacts) keeps a wide
  margin above the floor, so the live clip rate stays zero.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_har_clip.py
"""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pytest

from conftest import ARTIFACTS_DIR, DATASET_PATH, requires_live_artifacts
from packages.ml.model.training import har_benchmark, training

# The configured floor the whole check set is written against; read from the
# shared config through the model module rather than hardcoded here.
FLOOR = float(training.RV_FLOOR)

# The variance floor the fit clips at, and its log-space equivalent
# (ln(1e-12) = -27.631021115928547, quoted as -27.631021 in the docs).
FLOOR_SQUARED = FLOOR**2
LOG_FLOOR = float(np.log(FLOOR))

CLIP_TRAIN_ROWS = 6
CLIP_TEST_ROWS = 4
# Test-row regressor values chosen so two of the four fitted forecasts land
# below the floor and two stay above it (raw variance = 1e-26 * (1 + value)):
#   7e-26 and 8e-26 < 1e-24 (clipped); 2.001e-23 and 3.001e-23 > 1e-24.
CLIP_TEST_REGIME = (6.0, 7.0, 2000.0, 3000.0)

# Live-shaped magnitude: the live daily/weekly/monthly components average
# ~5.8e-4 and the target variance averages ~4e-2, so a fixture in that range
# sits ~20 orders of magnitude above the floor.
LIVE_STYLE_COMPONENT = 6e-4


def _fit_fixture(
    regressors: np.ndarray,
    target_log: np.ndarray,
    *,
    n_train: int,
    n_test: int,
    fold_key: str | None = None,
) -> dict:
    """Fit the real benchmark on the first ``n_train`` / next ``n_test`` rows."""
    indices = np.arange(len(regressors))
    train_indices = indices[:n_train]
    test_indices = indices[n_train : n_train + n_test]
    kwargs = {"fold_key": fold_key} if fold_key is not None else {}
    return har_benchmark.fit_har_benchmark(
        train_indices=train_indices,
        test_indices=test_indices,
        target_log_train=target_log[train_indices],
        target_log_test=target_log[test_indices],
        regressors=regressors,
        forecast_floor=FLOOR,
        warmup_mask=np.zeros(len(regressors), dtype=bool),
        **kwargs,
    )


def _clip_fixture() -> tuple[np.ndarray, np.ndarray]:
    """A frame whose fitted test forecasts straddle the floor.

    Only the daily component varies, and the training target variance is an
    exact linear function of it (``1e-26 * (1 + daily)``), so the OLS fit
    recovers ``intercept = daily coefficient = 1e-26`` and the test forecasts
    are exactly ``1e-26 * (1 + test value)``.
    """
    daily = np.concatenate(
        [np.arange(CLIP_TRAIN_ROWS, dtype=np.float64), np.asarray(CLIP_TEST_REGIME)]
    )
    regressors = np.zeros((daily.size, 3), dtype=np.float64)
    regressors[:, 0] = daily
    target_log = 0.5 * np.log(1e-26 * (1.0 + daily))
    return regressors, target_log


def _live_style_fixture() -> tuple[np.ndarray, np.ndarray]:
    """A deterministic live-shaped frame with comfortable floor margins."""
    rows = 40
    index = np.arange(rows, dtype=np.float64)
    regressors = np.column_stack(
        [
            LIVE_STYLE_COMPONENT * (1.0 + 0.05 * np.sin(0.3 * index)),
            LIVE_STYLE_COMPONENT * (1.0 + 0.05 * np.sin(0.3 * index + 1.0)),
            LIVE_STYLE_COMPONENT * (1.0 + 0.05 * np.sin(0.3 * index + 2.0)),
        ]
    )
    target_log = 0.5 * np.log(0.04 * (1.0 + 0.1 * np.cos(0.2 * index)))
    return regressors, target_log


def _fold_prediction(clip_count: int, n_test: int) -> dict:
    """A complete fold-prediction record with the given clip counters."""
    return {
        "train_date_range": ("2000-01-03", "2004-12-31"),
        "test_date_range": ("2005-01-03", "2005-12-30"),
        "n_train_samples": 100,
        "n_inner_val_samples": 15,
        "n_inner_purge_rows": 5,
        "best_iteration": 0,
        "eval_history": {"qlike": [0.3]},
        "n_test_samples": n_test,
        "n_test_total": n_test,
        "n_test_floor_excluded": 0,
        "n_test_scored": n_test,
        "n_train_floor_excluded": 0,
        "n_warmup_excluded": 0,
        "headline_row_set": training.HEADLINE_ROW_SET,
        "n_features": 38,
        "fold_model_path": "model.json",
        "test_r2": 0.5,
        "test_mae": 0.1,
        "test_rmse": 0.2,
        "test_qlike": 0.3,
        "oos_r2_prevailing_mean": 0.5,
        "oos_r2_vs_har": 0.3,
        "oos_r2_scale": training.OOS_R2_SCALE,
        "oos_r2_row_set": training.HEADLINE_ROW_SET,
        "har_r2": 0.1,
        "har_qlike": 0.4,
        "clip_count": clip_count,
        "clip_rate": clip_count / n_test,
        "train_time_seconds": 1.0,
        "inference_time_seconds": 0.1,
        "har_time_seconds": 0.2,
        "fold_time_seconds": 1.3,
    }


def test_clipped_forecast_counted_and_warned() -> None:
    """Below-floor fitted forecasts are counted, rated, and warned about."""
    regressors, target_log = _clip_fixture()
    logger = mock.Mock()

    with mock.patch.object(har_benchmark, "logger", logger):
        result = _fit_fixture(
            regressors,
            target_log,
            n_train=CLIP_TRAIN_ROWS,
            n_test=CLIP_TEST_ROWS,
            fold_key="fold_0",
        )

    assert result["clip_count"] > 0
    assert result["clip_count"] == 2
    assert result["clip_rate"] == result["clip_count"] / CLIP_TEST_ROWS
    assert result["clip_rate"] == pytest.approx(0.5)

    logger.warning.assert_called_once()
    warning_args = logger.warning.call_args.args
    message = warning_args[0] % warning_args[1:]
    assert "fold_0" in message
    assert f"{result['clip_rate']:.6f}" in message


def test_live_style_forecast_not_clipped_and_silent() -> None:
    """A comfortable live-shaped frame reports zero clips and no warning."""
    regressors, target_log = _live_style_fixture()
    n_train = 30
    n_test = 10
    logger = mock.Mock()

    with mock.patch.object(har_benchmark, "logger", logger):
        result = _fit_fixture(
            regressors,
            target_log,
            n_train=n_train,
            n_test=n_test,
            fold_key="fold_0",
        )

    assert result["clip_count"] == 0
    assert result["clip_rate"] == 0.0

    test_indices = np.arange(n_train, n_train + n_test)
    design = np.column_stack([np.ones(n_test), regressors[test_indices]])
    raw = design @ np.asarray(result["coefficients"])
    margin = float(raw.min() / FLOOR_SQUARED)
    assert margin > 1e15, (
        f"live-shaped fixture margin {margin:.3e} is not comfortably above "
        "the floor"
    )

    logger.warning.assert_not_called()


def test_constraint_documented_in_module_text() -> None:
    """The module text declares the clip, its log equivalence, and its source."""
    source = Path(har_benchmark.__file__).read_text(encoding="utf-8")

    assert "forecast_floor**2" in source
    assert "forecast_log >= ln(forecast_floor)" in source
    assert "-27.631021" in source
    assert "shared config" in source
    assert "must equal" in source
    assert "exp(2 * ln(floor))" in source


def test_counted_clip_equals_applied_clip_exactly() -> None:
    """The counter marks exactly the rows the clip changed, bit-for-bit."""
    regressors, target_log = _clip_fixture()
    result = _fit_fixture(
        regressors,
        target_log,
        n_train=CLIP_TRAIN_ROWS,
        n_test=CLIP_TEST_ROWS,
        fold_key="fold_0",
    )

    test_indices = np.arange(CLIP_TRAIN_ROWS, CLIP_TRAIN_ROWS + CLIP_TEST_ROWS)
    design = np.column_stack([np.ones(CLIP_TEST_ROWS), regressors[test_indices]])
    raw = design @ np.asarray(result["coefficients"])
    expected = np.maximum(raw, FLOOR_SQUARED)
    below_floor = raw < FLOOR_SQUARED

    assert int(np.count_nonzero(below_floor)) == result["clip_count"]
    assert np.array_equal(result["forecast_log"], 0.5 * np.log(expected))
    assert np.array_equal(
        result["forecast_log"][below_floor],
        np.full(np.count_nonzero(below_floor), 0.5 * np.log(FLOOR_SQUARED)),
    )
    assert np.array_equal(
        result["forecast_log"][~below_floor], 0.5 * np.log(raw[~below_floor])
    )

    # The applied clip is bit-exact in log space above; this cross-check is the
    # exp(2*log) round-trip band (observed ~3e-15 relative, ~4 ulp).
    returned_variance = np.exp(2.0 * result["forecast_log"])
    assert np.allclose(returned_variance, expected, rtol=1e-12, atol=0.0)
    assert np.allclose(
        returned_variance[below_floor], FLOOR_SQUARED, rtol=1e-12, atol=0.0
    )

    # The variance-space floor is the exact exp(2*log) of the configured floor.
    assert FLOOR_SQUARED == pytest.approx(np.exp(2.0 * np.log(FLOOR)), rel=1e-12)
    assert LOG_FLOOR == pytest.approx(-27.631021, rel=1e-7)


def test_fold_summary_and_global_report_carry_clip_counters() -> None:
    """Summary and artifact carry per-fold and summed-over-scored counters."""
    clip_counts = [0, 3, 0, 7, 0]
    n_tests = [10, 20, 30, 40, 50]
    predictions = [
        _fold_prediction(clip_counts[fold], n_tests[fold]) for fold in range(5)
    ]

    validation, summary = training._aggregate_fold_results(predictions)

    for fold in range(5):
        fold_key = f"fold_{fold}"
        assert summary[fold_key]["clip_count"] == clip_counts[fold]
        assert summary[fold_key]["clip_rate"] == pytest.approx(
            clip_counts[fold] / n_tests[fold]
        )
        assert validation[fold_key]["clip_count"] == clip_counts[fold]

    har_results = {
        f"fold_{fold}": {
            "coefficients": [1.0, 0.0, 0.0, 0.0],
            "har_r2": 0.1,
            "n_train": 100,
            "n_test": n_tests[fold],
            "clip_count": clip_counts[fold],
            "clip_rate": clip_counts[fold] / n_tests[fold],
        }
        for fold in range(5)
    }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "results").mkdir()
        logger = mock.Mock()
        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            with mock.patch.object(training, "logger", logger):
                training._save_training_results(validation, summary, har_results)
        report = json.loads(
            (root / "results" / "har_benchmark.json").read_text(encoding="utf-8")
        )

    total_count = sum(clip_counts)
    total_scored = sum(n_tests)
    assert report["clip_count"] == total_count
    assert report["clip_rate"] == pytest.approx(total_count / total_scored)
    for fold in range(5):
        assert report["folds"][f"fold_{fold}"]["clip_count"] == clip_counts[fold]

    logger.warning.assert_called_once()
    global_message = logger.warning.call_args.args[0]
    assert "fold_1" in global_message and "fold_3" in global_message
    assert f"{total_count / total_scored:.6f}" in global_message


def test_live_forecasts_keep_a_wide_margin_above_the_floor() -> None:
    """Read-only live fold 0: zero clips with a wide, measured margin."""
    requires_live_artifacts(ARTIFACTS_DIR)
    folds_path = ARTIFACTS_DIR / "folds" / "folds_metadata.pkl"
    y_path = ARTIFACTS_DIR / "inputs" / "Y.npy"
    metadata_path = ARTIFACTS_DIR / "inputs" / "metadata.pkl"
    if not (
        folds_path.is_file()
        and y_path.is_file()
        and metadata_path.is_file()
        and DATASET_PATH.is_file()
    ):
        pytest.skip(
            "requires stored fold metadata, targets, metadata, and the dataset"
        )

    folds = joblib.load(folds_path)
    y = np.asarray(np.load(y_path, mmap_mode="r"))
    metadata = joblib.load(metadata_path)
    regressors, warmup_mask = har_benchmark.load_har_regressors(
        metadata["date"], metadata["ticker"]
    )

    retained = training.floor_retained_mask(y)
    train_indices = np.asarray(folds["fold_0"]["train_indices"])
    test_indices = np.asarray(folds["fold_0"]["test_indices"])
    train_indices = train_indices[retained[train_indices]]
    test_indices = test_indices[retained[test_indices]]
    y_log = np.log(np.maximum(y, training.RV_FLOOR))

    result = har_benchmark.fit_har_benchmark(
        train_indices=train_indices,
        test_indices=test_indices,
        target_log_train=y_log[train_indices],
        target_log_test=y_log[test_indices],
        regressors=regressors,
        forecast_floor=training.RV_FLOOR,
        warmup_mask=warmup_mask,
        fold_key="fold_0",
    )

    design = np.column_stack([np.ones(test_indices.size), regressors[test_indices]])
    raw = design @ np.asarray(result["coefficients"])
    margin = float(raw.min() / training.RV_FLOOR**2)
    assert result["clip_count"] == 0
    assert result["clip_rate"] == 0.0
    assert margin > 1e6, (
        f"live fold-0 forecast margin {margin:.3e} is not comfortably above the "
        "floor"
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
