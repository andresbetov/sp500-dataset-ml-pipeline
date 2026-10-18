"""Canonical HAR-RV benchmark (Corsi 2008) for the 5-day-ahead volatility target.

The benchmark uses the canonical 1/5/22 trailing realized-variance components of the
daily log returns — daily (day t), weekly mean (t-4..t), monthly mean (t-21..t) — and is
fitted by OLS in variance levels per fold on training rows only, then evaluated on the
fold's test rows. Test rows never enter the fit.

The components are read from the dataset parquet and verified row-aligned against the
model input dates, so the benchmark cannot silently drift from the training rows.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from packages.ml.model.utils import get_project_paths, normalize_dates

logger = logging.getLogger(__name__)

DATASET_PATH = get_project_paths()["dataset_path"]

# Canonical Corsi lags: daily, weekly, monthly.
HAR_WINDOWS = (1, 5, 22)


def load_har_regressors(dates: np.ndarray) -> np.ndarray:
    """Trailing mean squared log returns over the canonical 1/5/22 windows.

    Every component ends on the row's own date, the close at which the prediction is
    decided. Rows near the frame start carry shorter windows because the pre-cut
    history is not shipped; later rows carry the full 1/5/22 windows.

    Args:
        dates: The model input date column, used to verify row alignment.

    Returns:
        (n, 3) float64 array of [daily, weekly, monthly] variance components.
    """
    frame = pd.read_parquet(DATASET_PATH, columns=["ticker", "date", "log_return"])
    frame_dates = normalize_dates(frame["date"].to_numpy())
    dates = normalize_dates(dates)
    if len(frame) != len(dates) or not np.array_equal(frame_dates, dates):
        raise ValueError(
            "HAR regressors are misaligned with the model input rows: the dataset parquet "
            "and the Phase 2 metadata disagree on the row order"
        )

    squared = frame["log_return"].to_numpy(dtype=np.float64) ** 2
    grouped = pd.Series(squared).groupby(frame["ticker"], sort=False)
    components = np.empty((len(frame), len(HAR_WINDOWS)), dtype=np.float64)
    for column, window in enumerate(HAR_WINDOWS):
        components[:, column] = grouped.transform(
            lambda values, window=window: values.rolling(window, min_periods=1).mean()
        ).to_numpy()

    if not np.isfinite(components).all():
        raise ValueError("HAR regressors contain non-finite values")
    logger.info(
        "HAR regressors: %d rows, windows %s, mean daily/weekly/monthly variance %.6g/%.6g/%.6g",
        len(frame),
        HAR_WINDOWS,
        float(components[:, 0].mean()),
        float(components[:, 1].mean()),
        float(components[:, 2].mean()),
    )
    return components


def fit_har_benchmark(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    target_log_train: np.ndarray,
    target_log_test: np.ndarray,
    regressors: np.ndarray,
    forecast_floor: float,
) -> dict:
    """Fit the canonical HAR-RV regression on one fold's training rows.

    OLS of the squared target (the realized variance) on the three trailing variance
    components, using training rows only. The fit optimizes variance-level squared
    error — Corsi's objective — so the log-scale R² below is a comparability metric,
    not the fit target; QLIKE is the headline model-vs-benchmark comparison. Test-row
    forecasts are clipped at ``forecast_floor`` so the implied volatility stays
    positive for the loss, and R² is reported on the log target against the fold's own
    test mean — the same convention the fold models use.

    Returns:
        dict with coefficients, har_r2, forecast_log (test rows), n_train, n_test.
    """
    train_design = np.column_stack([np.ones(len(train_indices)), regressors[train_indices]])
    test_design = np.column_stack([np.ones(len(test_indices)), regressors[test_indices]])

    target_variance = np.exp(2.0 * np.asarray(target_log_train, dtype=np.float64))
    coefficients, *_ = np.linalg.lstsq(train_design, target_variance, rcond=None)

    forecast_variance = np.maximum(test_design @ coefficients, forecast_floor**2)
    forecast_log = 0.5 * np.log(forecast_variance)
    har_r2 = float(r2_score(target_log_test, forecast_log))

    return {
        "coefficients": [float(value) for value in coefficients],
        "har_r2": har_r2,
        "forecast_log": forecast_log,
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
    }
