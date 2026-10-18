"""Canonical HAR-RV benchmark (Corsi 2008) for the 5-day-ahead volatility target.

The benchmark uses canonical 1/5/22 trailing variance components built with the
same Yang-Zhang estimator family as the target: every component is a trailing
window variance of the adjusted OHLC bars plus the prior adjusted close, so the
benchmark cannot mix a close-to-close proxy with a Yang-Zhang target. The
components are fitted by OLS in variance levels per fold on training rows only,
then evaluated on the fold's test rows. Test rows never enter the fit.

Per bar ``t`` (adjusted OHLC, per ticker, in frame order):

- ``overnight_t = ln(adj_open_t / adj_close_{t-1})``
- ``open_to_close_t = ln(adj_close_t / adj_open_t)``
- Rogers-Satchell:
  ``rs_t = ln(H_t/C_t)*ln(H_t/O_t) + ln(L_t/C_t)*ln(L_t/O_t)``

For a nominal window ``n`` the component at row ``t`` is the trailing-window
Yang-Zhang variance ending at ``t`` (inclusive; no look-ahead):

    component_n(t) = var(overnight over window)
                     + k_n * var(open_to_close over window)
                     + (1 - k_n) * mean(rs over window)
    k_n = 0.34 / (1.34 + (n + 1) / (n - 1)) for n >= 2

Degenerate rules:

- ``n = 1`` has no sample variance. The ``n -> 1`` limit of the same formula is
  used, where the open-to-close weight tends to 0 and the single observation's
  variance about zero captures the jump:
  ``component_1(t) = overnight_t**2 + rs_t``.
- A trailing window shorter than ``n`` (a ticker's series start) follows the
  ``min_periods=1`` convention: the sample variances use the available
  observations (``ddof=1``); when exactly one observation is available its
  squared value about zero is used, matching the ``n = 1`` rule; the ``rs``
  term is the available mean; the nominal ``k_n`` is applied. Warm-up rows keep
  these values and are excluded from the fit by the returned mask only.
- Each ticker's first shipped row has no predecessor close inside the frame
  (the shipped series starts after the dataset's history cut). The shipped
  ``log_return`` is the adjusted close-to-close difference, so that row's
  overnight jump is recovered as
  ``ln(adj_open_t / adj_close_t) + log_return_t``; every row stays finite.

The components are read from the dataset parquet and verified row-aligned
against the model input (ticker, date) pairs, so the benchmark cannot silently
drift from the training rows. The first ``HAR_WARMUP_DAYS`` dates of each
ticker's series carry incomplete trailing windows; the returned warm-up mask
flags those rows so the HAR fit excludes them.
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


def _yang_zhang_weight(window: int) -> float:
    """Yang-Zhang open-to-close weight for a nominal window of ``window`` days."""
    return 0.34 / (1.34 + (window + 1) / (window - 1))


def _yang_zhang_bar_quantities(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The target's per-bar quantities on the adjusted OHLC series.

    Returns ``(overnight, open_to_close, rogers_satchell)`` arrays in frame
    order, grouped per ticker. A ticker's first row has no predecessor close in
    the frame, so the shipped ``log_return`` (the adjusted close-to-close
    difference) supplies it; every other row uses the predecessor close
    directly.
    """
    adj_open = frame["_adj_open"].to_numpy(dtype=np.float64)
    adj_high = frame["_adj_high"].to_numpy(dtype=np.float64)
    adj_low = frame["_adj_low"].to_numpy(dtype=np.float64)
    adj_close = frame["adj_close"].to_numpy(dtype=np.float64)
    previous_close = frame.groupby("ticker", sort=False)["adj_close"].shift(1)
    previous_close = previous_close.to_numpy(dtype=np.float64)

    overnight = np.log(adj_open / previous_close)
    missing_predecessor = np.isnan(previous_close)
    overnight[missing_predecessor] = (
        np.log(adj_open[missing_predecessor] / adj_close[missing_predecessor])
        + frame["log_return"].to_numpy(dtype=np.float64)[missing_predecessor]
    )
    open_to_close = np.log(adj_close / adj_open)
    rogers_satchell = (
        np.log(adj_high / adj_close) * np.log(adj_high / adj_open)
        + np.log(adj_low / adj_close) * np.log(adj_low / adj_open)
    )
    return overnight, open_to_close, rogers_satchell


def _trailing_window_average(
    values: np.ndarray, tickers: pd.Series, window: int
) -> np.ndarray:
    """Trailing mean over ``window`` observations, ``min_periods=1``."""
    grouped = pd.Series(values).groupby(tickers, sort=False)
    return grouped.transform(
        lambda series: series.rolling(window, min_periods=1).mean()
    ).to_numpy(dtype=np.float64)


def _trailing_window_variance(
    values: np.ndarray, tickers: pd.Series, window: int
) -> np.ndarray:
    """Trailing sample variance (``ddof=1``) over the available observations.

    The ``min_periods=1`` convention applies: a shorter window at a ticker's
    series start uses the observations it has, and a window holding exactly one
    observation uses its squared value about zero (the ``n = 1`` rule).
    """
    grouped = pd.Series(values).groupby(tickers, sort=False)
    variance = grouped.transform(
        lambda series: series.rolling(window, min_periods=2).var()
    ).to_numpy(dtype=np.float64, copy=True)
    single_observation = (
        pd.Series(np.arange(len(values)))
        .groupby(tickers, sort=False)
        .cumcount()
        .to_numpy()
        == 0
    )
    variance[single_observation] = values[single_observation] ** 2
    return variance


def load_har_regressors(
    dates: np.ndarray, tickers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Same-estimator Yang-Zhang variance over the canonical 1/5/22 windows.

    Every component is built from the bar-level quantities the volatility target
    uses (adjusted OHLC plus the prior adjusted close) and ends on the row's own
    date, the close at which the prediction is decided. One-day rows use the
    ``n = 1`` limit (``overnight**2 + rs``); wider windows use the Yang-Zhang
    combination of trailing sample variances and the Rogers-Satchell mean, with
    shorter windows at a ticker's series start following the
    ``min_periods=1`` convention. The degenerate rules are documented in the
    module docstring.

    The first ``HAR_WARMUP_DAYS`` dates of each ticker's series (in frame order)
    are flagged by the returned mask and excluded from the fit; the component
    values keep the trailing-only math on every row.

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
    frame = pd.read_parquet(
        DATASET_PATH,
        columns=[
            "ticker",
            "date",
            "_adj_open",
            "_adj_high",
            "_adj_low",
            "adj_close",
            "log_return",
        ],
    )
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

    overnight, open_to_close, rogers_satchell = _yang_zhang_bar_quantities(frame)
    tickers_series = frame["ticker"]
    components = np.empty((len(frame), len(HAR_WINDOWS)), dtype=np.float64)
    for column, window in enumerate(HAR_WINDOWS):
        if window == 1:
            # n -> 1 limit: the open-to-close weight tends to 0, so the single
            # observation's variance about zero and the bar's rs remain.
            components[:, column] = overnight**2 + rogers_satchell
            continue
        weight = _yang_zhang_weight(window)
        components[:, column] = (
            _trailing_window_variance(overnight, tickers_series, window)
            + weight
            * _trailing_window_variance(open_to_close, tickers_series, window)
            + (1.0 - weight)
            * _trailing_window_average(rogers_satchell, tickers_series, window)
        )

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
