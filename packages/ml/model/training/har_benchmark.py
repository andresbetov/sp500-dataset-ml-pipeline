"""Canonical HAR-RV benchmark (Corsi (2009)) for the 5-day-ahead volatility target.

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

Forecast constraint
-------------------
The benchmark applies one declared forecast constraint: every scored test-row
forecast variance is clipped at ``forecast_floor**2``. The floor value comes
from the shared config (``rv_floor``, consumed by the model as
``training.RV_FLOOR``) and must equal the floor used for the model target.
Because the reported forecast is ``forecast_log = 0.5 * ln(variance)``, the
clip is equivalently a lower bound in log space,

    forecast_log >= ln(forecast_floor)

which for the configured floor ``1e-12`` is the numeric bound ``-27.631021``.
The conversions stay exact and unchanged: ``floor**2 == exp(2 * ln(floor))``
up to floating-point round-trip, and the unconstrained OLS fit remains
canonical (no constrained solver is substituted). Only the reported forecast
is clipped, and each clip is counted where it is applied — ``clip_count`` and
``clip_rate`` per fold in the returned result — so the counter cannot drift
from the applied clip.

The components are read from the dataset parquet and verified row-aligned
against the model input (ticker, date) pairs, so the benchmark cannot silently
drift from the training rows. The first ``HAR_WARMUP_DAYS`` dates of each
ticker's series carry incomplete trailing windows; the returned warm-up mask
flags those rows so the HAR fit excludes them.

Fit and report scales
---------------------
The canonical fit stays variance-level OLS and the written report declares it:
``fit_scale`` is ``"variance"`` and ``report_scale`` is
``"log-variance for R²; sigma for QLIKE"``, so the log-scale R² is never
mistaken for the fit objective. The comparator note states that QLIKE is the
headline model-vs-benchmark comparison. Coefficient standard errors are not
part of the canonical fit: Newey-West HAC standard errors (lag 4, the 5-day
label overlap) are reported only when the persisted
``coefficient_interpretation`` flag asks for them.
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

# Declared scales for the written report. The fit optimizes variance-level
# squared error as a variance-level variant of Corsi (2009) Eq. (8), which is
# specified in realized-volatility levels (variance and log are the variants he
# mentions); the reported R² is on the log target and QLIKE is in per-day sigma
# units, which is the headline comparison.
FIT_SCALE = "variance"
REPORT_SCALE = "log-variance for R²; sigma for QLIKE"
COMPARISON_NOTE = (
    "QLIKE is the headline model-vs-benchmark comparison; the log-target R² "
    "is a comparability metric only."
)

# Newey-West HAC lag: the 5-day label overlap lets serial correlation reach four
# daily steps, so the Bartlett-weighted sandwich absorbs lags 1 through 4.
HAC_LAG = 4
HAC_NOTE = (
    "Newey-West HAC standard errors (lag 4, Bartlett weights) are reported; "
    "coefficient interpretation is enabled by coefficient_interpretation=True."
)
NO_HAC_NOTE = (
    "OLS standard errors are not reported and coefficient interpretation "
    "requires coefficient_interpretation=True (Newey-West HAC, lag 4)."
)


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


def _coefficient_standard_errors(
    design: np.ndarray,
    target_variance: np.ndarray,
    coefficients: np.ndarray,
    lag: int = HAC_LAG,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS and Newey-West HAC standard errors for one fitted OLS vector.

    The OLS covariance is ``s² (X'X)^{-1}`` with ``s²`` the residual variance.
    The HAC covariance is the Bartlett-weighted sandwich
    ``(X'X)^{-1} X' Ω X (X'X)^{-1}`` with ``X' Ω X`` built from the residual
    second moments (lag 0) plus the Newey-West weighted residual
    autocovariances up to ``lag``. The pseudo-inverse, the symmetrization, and
    the nonnegative diagonal clip keep the sandwich stable in the
    rank-deficient corners that ``lstsq`` still solves.
    """
    n_rows, n_columns = design.shape
    if n_rows <= n_columns:
        raise ValueError(
            f"HAR coefficient standard errors require more fitted rows than "
            f"coefficients: {n_rows} fitted rows for {n_columns} coefficients"
        )
    residual = target_variance - design @ coefficients
    bread = np.linalg.pinv(design.T @ design)

    ols_covariance = (float(residual @ residual) / (n_rows - n_columns)) * bread
    ols_se = np.sqrt(np.clip(np.diag(ols_covariance), 0.0, None))

    meat = design.T @ (design * (residual**2)[:, None])
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1)
        cross = (
            design[offset:] * (residual[offset:] * residual[:-offset])[:, None]
        ).T @ design[:-offset]
        meat += weight * (cross + cross.T)
    hac_covariance = bread @ meat @ bread
    hac_covariance = 0.5 * (hac_covariance + hac_covariance.T)
    hac_se = np.sqrt(np.clip(np.diag(hac_covariance), 0.0, None))
    return ols_se, hac_se


def har_forecast_log(
    coefficients,
    regressors: np.ndarray,
    indices: np.ndarray,
    forecast_floor: float,
) -> tuple[np.ndarray, int]:
    """The benchmark's per-row log forecasts at ``indices`` (the one forecast path).

    ``coefficients`` is the fitted OLS vector (intercept plus the canonical
    1/5/22 components), ``regressors`` the per-row variance components, and
    ``indices`` the rows to forecast. The design is built and constrained
    exactly as the fold fit applies it: the variance-space forecast is clipped
    at ``forecast_floor**2`` (equivalently ``forecast_log >=
    ln(forecast_floor)``) and every applied clip is counted. Both the fold
    benchmark and the evaluation decomposition's floor-row forecasts call this
    function, so the forecast formula exists once and cannot drift between
    them; no coefficient is refitted here.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    regressors = np.asarray(regressors, dtype=np.float64)
    indices = np.asarray(indices)
    if regressors.ndim != 2:
        raise ValueError(
            f"HAR forecast expects a 2-D regressor matrix, got {regressors.shape}"
        )
    if coefficients.shape != (regressors.shape[1] + 1,):
        raise ValueError(
            f"HAR forecast received {coefficients.size} coefficients for "
            f"{regressors.shape[1]} regressors"
        )
    design = np.column_stack([np.ones(len(indices)), regressors[indices]])
    floor_squared = forecast_floor**2
    forecast_variance = design @ coefficients
    below_floor = forecast_variance < floor_squared
    clip_count = int(np.count_nonzero(below_floor))
    forecast_variance[below_floor] = floor_squared
    return 0.5 * np.log(forecast_variance), clip_count


def fit_har_benchmark(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    target_log_train: np.ndarray,
    target_log_test: np.ndarray,
    regressors: np.ndarray,
    forecast_floor: float,
    warmup_mask: np.ndarray,
    fold_key: str | None = None,
    coefficient_interpretation: bool = False,
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
    squared error — a variance-level variant of Corsi (2009) Eq. (8), which is
    specified in realized-volatility levels (variance and log are the variants he
    mentions) — so the log-scale R² below is a comparability metric, not the fit
    target; QLIKE is the headline model-vs-benchmark comparison.
    The result declares that split through ``fit_scale`` (``"variance"``),
    ``report_scale`` (``"log-variance for R²; sigma for QLIKE"``), and the comparator
    note. Coefficient standard errors are not part of the canonical fit: with
    ``coefficient_interpretation=False`` (the pipeline default) ``hac_se`` is None
    and the note states that OLS standard errors are not reported; with the flag
    True the result carries ``ols_se`` and the Newey-West ``hac_se`` (lag 4,
    Bartlett weights, the sandwich ``(X'X)^{-1} X' Ω X (X'X)^{-1}``), plus
    ``hac_lag = 4``. ``hac_lag`` records the configured lag whether or not the
    errors are computed; the flag and the note say whether ``hac_se`` is
    interpretable. The fitted coefficients and forecasts never depend on the flag.

    Test-row forecasts are clipped at ``forecast_floor**2`` in variance space so the
    implied volatility stays positive for the loss; equivalently the returned log
    forecast obeys ``forecast_log >= ln(forecast_floor)`` — the one declared forecast
    constraint, documented in the module docstring, where the floor comes from the
    shared config and must equal the model's floor. ``har_r2`` is reported on the log
    target against the fold's own test mean — the same fold-own-mean convention the
    fold models' ``test_r2`` uses. The model-vs-HAR comparison is the training
    phase's dual reporting layer: it keeps the returned per-row ``forecast_log`` and
    emits ``oos_r2_vs_har`` = ``1 - SSE_model/SSE_HAR`` on the same scored rows
    (alongside ``oos_r2_prevailing_mean``, the fold-own/pooled prevailing-mean
    convention under its honest name).

    The clip is counted where it is applied: ``clip_count`` of the ``n_test`` scored
    rows fall below ``forecast_floor**2`` and ``clip_rate = clip_count / n_test``. A
    non-zero count logs a warning naming ``fold_key`` and the rate; ``fold_key`` is
    optional and defaults to None for direct callers.

    Returns:
        dict with coefficients, har_r2, forecast_log (test rows), clip_count,
        clip_rate, n_train (the fitted rows), n_test, n_warmup_excluded,
        n_warmup_test_rows, warmup_days, the declared fit/report scales and
        comparator note, the persisted ``coefficient_interpretation`` flag,
        ``hac_lag``, and the opt-in ``ols_se``/``hac_se`` (None when the flag
        is off) with their note.
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

    target_variance = np.exp(2.0 * np.asarray(target_log_train, dtype=np.float64))
    coefficients, *_ = np.linalg.lstsq(
        train_design, target_variance[retained_train], rcond=None
    )

    # Coefficient standard errors are opt-in: the canonical fit (and every
    # pipeline run) reports none, and the persisted flag is what makes the
    # Newey-West errors interpretable.
    if coefficient_interpretation:
        ols_se, hac_se = _coefficient_standard_errors(
            train_design, target_variance[retained_train], coefficients
        )
    else:
        ols_se = hac_se = None

    # The one declared forecast constraint: clip the variance-space forecast at
    # floor**2, i.e. forecast_log >= ln(forecast_floor). The same mask that
    # selects the clipped rows counts them, so clip_count/clip_rate cannot
    # drift from the applied clip.
    forecast_log, clip_count = har_forecast_log(
        coefficients, regressors, test_indices, forecast_floor
    )
    har_r2 = float(r2_score(target_log_test, forecast_log))
    clip_rate = clip_count / len(test_indices) if len(test_indices) else 0.0
    if clip_count:
        logger.warning(
            "HAR %s: forecast variance fell below the floor on %d of %d scored "
            "rows (clip_rate=%.6f); applied the variance-floor clip "
            "forecast_floor**2 = %g, i.e. forecast_log >= %.6f",
            fold_key if fold_key is not None else "unlabelled fold",
            clip_count,
            len(test_indices),
            clip_rate,
            forecast_floor**2,
            np.log(forecast_floor),
        )

    return {
        "coefficients": [float(value) for value in coefficients],
        "har_r2": har_r2,
        "forecast_log": forecast_log,
        "clip_count": clip_count,
        "clip_rate": float(clip_rate),
        "n_train": int(len(fit_indices)),
        "n_test": int(len(test_indices)),
        "n_warmup_excluded": n_warmup_excluded,
        "n_warmup_test_rows": warmup_test_rows,
        "warmup_days": HAR_WARMUP_DAYS,
        "fit_scale": FIT_SCALE,
        "report_scale": REPORT_SCALE,
        "comparison_note": COMPARISON_NOTE,
        "coefficient_interpretation": bool(coefficient_interpretation),
        "hac_lag": HAC_LAG,
        "hac_se": None if hac_se is None else [float(value) for value in hac_se],
        "ols_se": None if ols_se is None else [float(value) for value in ols_se],
        "coefficient_se_note": (
            HAC_NOTE if coefficient_interpretation else NO_HAC_NOTE
        ),
    }
