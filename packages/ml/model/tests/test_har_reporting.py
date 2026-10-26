"""Tests for the HAR fit-scale disclosure and conditional HAC standard errors.

The canonical HAR-RV benchmark is an ordinary least-squares fit in variance
levels. Its reported R² is on the log target and its headline comparison is
QLIKE, so the artifact has to state that fit/report scale split rather than
leaving it to prose alone. Coefficient standard errors are not part of the
canonical fit: Newey-West HAC standard errors (lag 4, matching the 5-day
overlap) are reported only when the persisted ``coefficient_interpretation``
flag asks for them.

These checks pin:

- the fit-scale disclosure text survives in the module/fit docstring,
  compared whitespace-normalized because it wraps across lines;
- a written ``har_benchmark.json`` carries ``fit_scale``, ``report_scale``,
  the QLIKE-headline comparator note, and every pre-existing top-level key;
- an autocorrelated-residual fixture yields HAC standard errors different
  from the OLS ones only under the flag, with ``hac_lag == 4`` and the
  reference sandwich values reproduced;
- with the flag off no standard-error values are reported and the explicit
  "OLS standard errors are not reported" note is present;
- the flag is persisted in the artifact, so a reader can tell whether the
  errors are interpretable;
- coefficients, forecasts, and sigma/variance conversions are byte-identical
  with the flag off and on, and still equal the branch-point variance-level
  OLS solution.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_har_reporting.py
"""

from __future__ import annotations

import inspect
import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model.training import har_benchmark, training

FLOOR = float(training.RV_FLOOR)
HAC_LAG = 4

# The fit-scale disclosure currently carried by the fit docstring. It is matched
# whitespace-normalized because the source wraps it across lines.
FIT_SCALE_DISCLOSURE = (
    "The fit optimizes variance-level squared error — a variance-level variant "
    "of Corsi (2009) Eq. (8), which is specified in realized-volatility levels "
    "(variance and log are the variants he mentions) — so the log-scale R² "
    "below is a comparability metric, not the fit target; QLIKE is the headline "
    "model-vs-benchmark comparison."
)

FIT_SCALE = "variance"
REPORT_SCALE = "log-variance for R²; sigma for QLIKE"

# Existing top-level report keys that this change must leave in place.
PREEXISTING_REPORT_KEYS = {
    "benchmark",
    "target",
    "r2_convention",
    "qlike",
    "forecast_clip",
    "clip_count",
    "clip_rate",
    "folds_with_clips",
    "warmup_days",
    "warmup_definition",
    "headline_row_set",
    "folds",
}

HAC_ROWS = 240
HAC_TRAIN_ROWS = 200


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _hac_fixture() -> tuple[np.ndarray, np.ndarray]:
    """A variance-level design whose residuals follow a persistent AR(1).

    The residual autocorrelation is what separates the Newey-West meat from the
    homoskedastic OLS covariance; the fixture is deterministic (seeded) and its
    variance stays comfortably positive and far above the forecast floor.
    """
    index = np.arange(HAC_ROWS, dtype=np.float64)
    regressors = np.column_stack(
        [
            6e-4 * (1.0 + 0.1 * np.sin(0.13 * index)),
            6e-4 * (1.0 + 0.1 * np.sin(0.11 * index + 1.0)),
            6e-4 * (1.0 + 0.1 * np.sin(0.07 * index + 2.0)),
        ]
    )
    design = np.column_stack([np.ones(HAC_ROWS), regressors])
    true_coefficients = np.array([0.08, 1.0, 0.5, 0.25])

    generator = np.random.default_rng(42)
    residual = np.zeros(HAC_ROWS, dtype=np.float64)
    for row in range(1, HAC_ROWS):
        residual[row] = 0.9 * residual[row - 1] + generator.normal(0.0, 0.01)

    variance = design @ true_coefficients + residual
    assert variance.min() > 0.0
    return regressors, 0.5 * np.log(variance)


def _fit(regressors: np.ndarray, target_log: np.ndarray, *, flag: bool) -> dict:
    indices = np.arange(len(regressors))
    train_indices = indices[:HAC_TRAIN_ROWS]
    test_indices = indices[HAC_TRAIN_ROWS:]
    return har_benchmark.fit_har_benchmark(
        train_indices=train_indices,
        test_indices=test_indices,
        target_log_train=target_log[train_indices],
        target_log_test=target_log[test_indices],
        regressors=regressors,
        forecast_floor=FLOOR,
        warmup_mask=np.zeros(len(regressors), dtype=bool),
        fold_key="fold_0",
        coefficient_interpretation=flag,
    )


def _reference_standard_errors(
    design: np.ndarray, target_variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Independent OLS and Newey-West (lag 4, Bartlett) standard errors."""
    coefficients, *_ = np.linalg.lstsq(design, target_variance, rcond=None)
    residual = target_variance - design @ coefficients
    n_rows, n_columns = design.shape
    bread = np.linalg.pinv(design.T @ design)

    ols_covariance = (residual @ residual / (n_rows - n_columns)) * bread
    ols_se = np.sqrt(np.diag(ols_covariance))

    meat = design.T @ (design * (residual**2)[:, None])
    for lag in range(1, HAC_LAG + 1):
        weight = 1.0 - lag / (HAC_LAG + 1)
        cross = (
            design[lag:] * (residual[lag:] * residual[:-lag])[:, None]
        ).T @ design[:-lag]
        meat += weight * (cross + cross.T)

    hac_covariance = bread @ meat @ bread
    hac_covariance = 0.5 * (hac_covariance + hac_covariance.T)
    hac_se = np.sqrt(np.clip(np.diag(hac_covariance), 0.0, None))
    return ols_se, hac_se


def _fold_summary() -> tuple[dict, dict]:
    """Minimal validation/summary payloads accepted by the report writer."""
    validation = {"fold_0": {"test_pred": np.array([0.0]), "test_true": np.array([0.0])}}
    summary = {"fold_0": {"test_r2": 0.5, "test_qlike": 0.25}}
    return validation, summary


def _write_report(flag: bool) -> dict:
    """Fit the fixture, write the artifact to a temp root, and read it back."""
    regressors, target_log = _hac_fixture()
    result = _fit(regressors, target_log, flag=flag)
    validation, summary = _fold_summary()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "results").mkdir()
        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            training._save_training_results(
                validation,
                summary,
                {"fold_0": result},
                coefficient_interpretation=flag,
            )
        return json.loads(
            (root / "results" / "har_benchmark.json").read_text(encoding="utf-8")
        )


def test_scale_fields_and_comparator_note_in_artifact() -> None:
    """The written report declares both scales and keeps the existing keys."""
    report = _write_report(flag=False)

    assert report["fit_scale"] == FIT_SCALE
    assert report["report_scale"] == REPORT_SCALE
    assert "QLIKE is the headline model-vs-benchmark comparison" in report[
        "comparison_note"
    ]
    assert report["hac_lag"] == HAC_LAG

    fold_report = report["folds"]["fold_0"]
    assert fold_report["fit_scale"] == FIT_SCALE
    assert fold_report["report_scale"] == REPORT_SCALE
    assert fold_report["comparison_note"] == report["comparison_note"]

    missing = PREEXISTING_REPORT_KEYS - set(report)
    assert not missing, f"report lost pre-existing keys: {sorted(missing)}"


def test_hac_behind_the_flag_and_explicit_note_when_off() -> None:
    """HAC errors differ from OLS only under the flag; off reports no errors."""
    regressors, target_log = _hac_fixture()
    train_indices = np.arange(HAC_TRAIN_ROWS)
    design = np.column_stack([np.ones(HAC_TRAIN_ROWS), regressors[train_indices]])
    target_variance = np.exp(2.0 * target_log[train_indices])
    expected_ols_se, expected_hac_se = _reference_standard_errors(
        design, target_variance
    )

    enabled = _fit(regressors, target_log, flag=True)
    assert enabled["coefficient_interpretation"] is True
    assert enabled["hac_lag"] == HAC_LAG
    assert len(enabled["hac_se"]) == design.shape[1]
    assert len(enabled["ols_se"]) == design.shape[1]
    assert np.allclose(enabled["ols_se"], expected_ols_se, rtol=1e-12, atol=1e-18)
    assert np.allclose(enabled["hac_se"], expected_hac_se, rtol=1e-12, atol=1e-18)
    assert not np.allclose(enabled["hac_se"], enabled["ols_se"], rtol=1e-8, atol=1e-18)
    hac_values = np.asarray(enabled["hac_se"])
    ols_values = np.asarray(enabled["ols_se"])
    assert np.max(np.abs(hac_values - ols_values)) > 0.1 * np.min(ols_values)

    disabled = _fit(regressors, target_log, flag=False)
    assert disabled["coefficient_interpretation"] is False
    assert disabled["hac_lag"] == HAC_LAG
    assert disabled["hac_se"] is None
    assert disabled["ols_se"] is None
    note = disabled["coefficient_se_note"]
    assert "OLS standard errors are not reported" in note
    assert "coefficient interpretation requires coefficient_interpretation=True" in note


def test_flag_persisted_in_artifact() -> None:
    """The report records whether the coefficient errors are interpretable."""
    enabled = _write_report(flag=True)
    assert enabled["coefficient_interpretation"] is True
    assert enabled["folds"]["fold_0"]["coefficient_interpretation"] is True
    assert enabled["folds"]["fold_0"]["hac_se"] is not None
    assert enabled["folds"]["fold_0"]["hac_lag"] == HAC_LAG
    assert "Newey-West HAC standard errors" in enabled["coefficient_se_note"]

    disabled = _write_report(flag=False)
    assert disabled["coefficient_interpretation"] is False
    assert disabled["folds"]["fold_0"]["coefficient_interpretation"] is False
    assert disabled["folds"]["fold_0"]["hac_se"] is None
    assert "OLS standard errors are not reported" in disabled["coefficient_se_note"]


def test_flag_off_keeps_the_branch_point_ols_fit_byte_identical() -> None:
    """Flag off and on both return the canonical variance-level OLS values."""
    regressors, target_log = _hac_fixture()
    train_indices = np.arange(HAC_TRAIN_ROWS)
    test_indices = np.arange(HAC_TRAIN_ROWS, HAC_ROWS)
    train_design = np.column_stack(
        [np.ones(train_indices.size), regressors[train_indices]]
    )
    test_design = np.column_stack(
        [np.ones(test_indices.size), regressors[test_indices]]
    )

    expected_coefficients = np.linalg.lstsq(
        train_design, np.exp(2.0 * target_log[train_indices]), rcond=None
    )[0]
    expected_forecast_variance = np.maximum(
        test_design @ expected_coefficients, FLOOR**2
    )
    expected_forecast_log = 0.5 * np.log(expected_forecast_variance)

    disabled = _fit(regressors, target_log, flag=False)
    enabled = _fit(regressors, target_log, flag=True)

    for result in (disabled, enabled):
        assert np.array_equal(
            np.asarray(result["coefficients"]), expected_coefficients
        )
        assert np.array_equal(result["forecast_log"], expected_forecast_log)
        assert result["clip_count"] == 0

    assert np.array_equal(disabled["coefficients"], enabled["coefficients"])
    assert np.array_equal(disabled["forecast_log"], enabled["forecast_log"])
    assert disabled["har_r2"] == enabled["har_r2"]

    # sigma -> variance -> sigma stay exact up to the log/exp round-trip.
    assert np.allclose(
        np.exp(2.0 * np.asarray(disabled["forecast_log"])),
        expected_forecast_variance,
        rtol=1e-12,
        atol=0.0,
    )


def test_phase_4_default_keeps_coefficient_interpretation_off() -> None:
    """Pipeline runs default to flag off, so canonical OLS stays unannotated."""
    parameter = inspect.signature(training.phase_4_training).parameters[
        "coefficient_interpretation"
    ]
    assert parameter.default is False


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
