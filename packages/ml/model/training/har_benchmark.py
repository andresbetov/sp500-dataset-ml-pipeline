"""Canonical HAR-RV benchmark (Corsi 2008) for the 5-day-ahead volatility target.

The benchmark uses the canonical 1/5/22 trailing realized-variance components of the
daily log returns — daily (day t), weekly mean (t-4..t), monthly mean (t-21..t) — and is
fitted by OLS in variance levels per fold on training rows only, then evaluated on the
fold's test rows. Test rows never enter the fit.

The components are read from the dataset parquet and verified row-aligned against the
model input (ticker, date) pairs, so the benchmark cannot silently drift from the
training rows. The first ``HAR_WARMUP_DAYS`` dates of each ticker's series carry
incomplete trailing windows; the returned warm-up mask flags those rows so the HAR fit
excludes them.
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

# The longest trailing window spans HAR_WINDOWS[-1] observations, so the first
# HAR_WINDOWS[-1] - 1 dates of each ticker's series carry incomplete windows. Those
# rows are flagged by the warm-up mask and excluded from the fit.
HAR_WARMUP_DAYS = 21


def load_har_regressors(
    dates: np.ndarray, tickers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean squared log returns over the canonical 1/5/22 windows.

    Every component ends on the row's own date, the close at which the prediction is
    decided. Rows near a ticker's series start carry shorter windows because the
    pre-cut history is not shipped; later rows carry the full 1/5/22 windows. The
    first ``HAR_WARMUP_DAYS`` dates of each ticker's series (in frame order) are
    flagged by the returned mask and excluded from the fit; the component values keep
    the trailing-only ``min_periods=1`` math on every row.

    The parquet frame is verified against the model metadata by (ticker, date) pair
    equality in row order, so a same-date ticker swap cannot silently attach another
    ticker's returns, and each ticker's dates must be strictly increasing. The checks
    are plain numpy/pandas comparisons with no shared thread state.

    Args:
        dates: The model input date column, used to verify row alignment.
        tickers: The model input ticker column, used to verify row alignment.

    Returns:
        (components, warmup_mask): an (n, 3) float64 array of [daily, weekly,
        monthly] variance components and an (n,) bool mask that is True for the
        first ``HAR_WARMUP_DAYS`` dates of each ticker's series.
    """
    frame = pd.read_parquet(DATASET_PATH, columns=["ticker", "date", "log_return"])
    frame_tickers = frame["ticker"].to_numpy()
    frame_dates = normalize_dates(frame["date"].to_numpy())
    tickers = np.asarray(tickers)
    dates = normalize_dates(dates)

    if len(frame) != len(dates) or len(tickers) != len(dates):
        raise ValueError(
            "HAR regressors are misaligned with the model input rows: the dataset "
            f"parquet holds {len(frame)} rows but the Phase 2 metadata holds "
            f"{len(dates)} dates and {len(tickers)} tickers"
        )

    mismatch = np.flatnonzero((frame_tickers != tickers) | (frame_dates != dates))
    if mismatch.size:
        row = int(mismatch[0])
        raise ValueError(
            "HAR regressors are misaligned with the model input rows: the dataset "
            "parquet and the Phase 2 metadata disagree on the (ticker, date) pairs "
            f"(first mismatch at row {row}: parquet {frame_tickers[row]!r}/"
            f"{frame_dates[row]}, metadata {tickers[row]!r}/{dates[row]})"
        )

    ticker_position = (
        pd.Series(np.arange(len(frame))).groupby(frame_tickers, sort=False).cumcount()
    )
    previous_dates = (
        pd.Series(frame_dates).groupby(frame_tickers, sort=False).shift(1)
    )
    out_of_order = previous_dates.notna().to_numpy() & (
        frame_dates <= previous_dates.to_numpy()
    )
    if out_of_order.any():
        row = int(np.argmax(out_of_order))
        raise ValueError(
            f"HAR regressors: ticker {frame_tickers[row]!r} dates are not strictly "
            f"increasing at ticker position {int(ticker_position.iloc[row])} "
            f"(frame row {row}): {previous_dates.iloc[row]} >= {frame_dates[row]}"
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

    warmup_mask = ticker_position.to_numpy() < HAR_WARMUP_DAYS
    logger.info(
        "HAR regressors: %d rows, windows %s, %d warm-up rows (first %d dates per "
        "ticker), mean daily/weekly/monthly variance %.6g/%.6g/%.6g",
        len(frame),
        HAR_WINDOWS,
        int(warmup_mask.sum()),
        HAR_WARMUP_DAYS,
        float(components[:, 0].mean()),
        float(components[:, 1].mean()),
        float(components[:, 2].mean()),
    )
    return components, warmup_mask


def fit_har_benchmark(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    target_log_train: np.ndarray,
    target_log_test: np.ndarray,
    regressors: np.ndarray,
    forecast_floor: float,
    warmup_mask: np.ndarray,
) -> dict:
    """Fit the canonical HAR-RV regression on one fold's training rows.

    OLS of the squared target (the realized variance) on the three trailing variance
    components, using training rows only. The fit set is the fold's training rows
    minus every row the warm-up mask flags — the first ``HAR_WARMUP_DAYS`` dates of
    each ticker's series, whose trailing windows are incomplete — and
    ``n_warmup_excluded`` counts the removed rows. The mask never filters scoring
    rows: a late index entrant's series can start inside a test block, and those
    warm-up rows keep their unchanged trailing-only components, so they are scored
    like every other test row; ``n_warmup_test_rows`` records how many were scored
    so the unchanged scoring set stays auditable. The fit optimizes variance-level
    squared error — Corsi's objective — so the log-scale R² below is a comparability
    metric, not the fit target; QLIKE is the headline model-vs-benchmark comparison.
    Test-row forecasts are clipped at ``forecast_floor`` so the implied volatility
    stays positive for the loss, and R² is reported on the log target against the
    fold's own test mean — the same convention the fold models use.

    Returns:
        dict with coefficients, har_r2, forecast_log (test rows), n_train (the fitted
        rows), n_test, n_warmup_excluded, n_warmup_test_rows, warmup_days.
    """
    train_indices = np.asarray(train_indices)
    test_indices = np.asarray(test_indices)
    warmup_mask = np.asarray(warmup_mask, dtype=bool)
    if warmup_mask.shape != (len(regressors),):
        raise ValueError(
            f"HAR warm-up mask has {warmup_mask.size} entries but the regressors "
            f"hold {len(regressors)} rows"
        )
    warmup_test_rows = int(np.count_nonzero(warmup_mask[test_indices]))

    retained_train = ~warmup_mask[train_indices]
    fit_indices = train_indices[retained_train]
    n_warmup_excluded = int(train_indices.size - fit_indices.size)
    if fit_indices.size == 0:
        raise ValueError(
            f"HAR fit set is empty: all {train_indices.size} training rows fall "
            f"inside the {HAR_WARMUP_DAYS}-date per-ticker warm-up"
        )

    train_design = np.column_stack([np.ones(len(fit_indices)), regressors[fit_indices]])
    test_design = np.column_stack([np.ones(len(test_indices)), regressors[test_indices]])

    target_variance = np.exp(2.0 * np.asarray(target_log_train, dtype=np.float64))
    coefficients, *_ = np.linalg.lstsq(
        train_design, target_variance[retained_train], rcond=None
    )

    forecast_variance = np.maximum(test_design @ coefficients, forecast_floor**2)
    forecast_log = 0.5 * np.log(forecast_variance)
    har_r2 = float(r2_score(target_log_test, forecast_log))

    return {
        "coefficients": [float(value) for value in coefficients],
        "har_r2": har_r2,
        "forecast_log": forecast_log,
        "n_train": int(len(fit_indices)),
        "n_test": int(len(test_indices)),
        "n_warmup_excluded": n_warmup_excluded,
        "n_warmup_test_rows": warmup_test_rows,
        "warmup_days": HAR_WARMUP_DAYS,
    }
