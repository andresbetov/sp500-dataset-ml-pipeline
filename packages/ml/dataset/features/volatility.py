from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._common import (
    ADJUSTED_OHLC_HIGH_COLUMN,
    ADJUSTED_OHLC_LOW_COLUMN,
    ADJUSTED_OHLC_OPEN_COLUMN,
)

logger = logging.getLogger(__name__)


def _add_range_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Range-based volatility estimates over trailing windows t-w..t-1.

    Each bar-level quantity uses the adjusted OHLC columns of its own bar and the
    window mean is shifted one day, so a feature never sees its own bar. The
    Garman-Klass variance 0.5*ln(H/L)^2 - (2 ln 2 - 1)*ln(C/O)^2 can be negative on a
    strong body with a narrow range; such bars are clipped to zero (counted in the
    log) so the estimator stays defined instead of producing NaNs that would drop
    rows. Rogers-Satchell ships in variance form, not its square root.
    """
    featured = df
    required_columns = {
        ADJUSTED_OHLC_OPEN_COLUMN,
        ADJUSTED_OHLC_HIGH_COLUMN,
        ADJUSTED_OHLC_LOW_COLUMN,
        "adj_close",
    }
    missing_columns = required_columns - set(featured.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Columns required to compute range volatility are missing: {missing_list}")

    high_low_log = np.log(featured[ADJUSTED_OHLC_HIGH_COLUMN] / featured[ADJUSTED_OHLC_LOW_COLUMN])
    open_close_log = np.log(featured["adj_close"] / featured[ADJUSTED_OHLC_OPEN_COLUMN])

    atr_bar = (featured[ADJUSTED_OHLC_HIGH_COLUMN] - featured[ADJUSTED_OHLC_LOW_COLUMN]) / featured["adj_close"]
    park_bar = np.sqrt(high_low_log**2 / (4.0 * np.log(2.0)))
    gk_variance = 0.5 * high_low_log**2 - (2.0 * np.log(2.0) - 1.0) * open_close_log**2
    negative_gk_bars = int((gk_variance < 0).sum())
    if negative_gk_bars:
        logger.info(
            "_add_range_volatility_features: %d Garman-Klass bars carry a negative "
            "variance and were clipped to zero",
            negative_gk_bars,
        )
    gk_bar = np.sqrt(gk_variance.clip(lower=0.0))
    rs_bar = (
        np.log(featured[ADJUSTED_OHLC_HIGH_COLUMN] / featured["adj_close"])
        * np.log(featured[ADJUSTED_OHLC_HIGH_COLUMN] / featured[ADJUSTED_OHLC_OPEN_COLUMN])
        + np.log(featured[ADJUSTED_OHLC_LOW_COLUMN] / featured["adj_close"])
        * np.log(featured[ADJUSTED_OHLC_LOW_COLUMN] / featured[ADJUSTED_OHLC_OPEN_COLUMN])
    )
    previous_close = featured.groupby("ticker", sort=False, group_keys=False)["adj_close"].shift(1)
    gap_bar = np.log(
        featured[ADJUSTED_OHLC_OPEN_COLUMN] / previous_close
    ).abs()

    bar_series = {
        "atr": atr_bar,
        "park": park_bar,
        "gk": gk_bar,
        "rs": rs_bar,
        "gap": gap_bar,
    }
    window_by_feature = {
        "atr5": ("atr", 5),
        "atr20": ("atr", 20),
        "park5": ("park", 5),
        "park20": ("park", 20),
        "gk5": ("gk", 5),
        "rs5": ("rs", 5),
        "gap5": ("gap", 5),
    }
    for name, (source, window) in window_by_feature.items():
        featured[name] = (
            bar_series[source]
            .groupby(featured["ticker"], sort=False, group_keys=False)
            .transform(lambda s: s.rolling(window=window, min_periods=window).mean().shift(1))
            .astype("float64")
        )

    for name in window_by_feature:
        defined_values = featured[name].notna()
        if not np.isfinite(featured.loc[defined_values, name]).all():
            raise ValueError(f"Column {name} produced non-finite values")

    logger.info(
        "_add_range_volatility_features: columns=%d, negative Garman-Klass bars clipped=%d",
        len(window_by_feature),
        negative_gk_bars,
    )
    return featured


def _add_trailing_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing realized-volatility stack, HAR sums, market factor and vol-of-vol.

    Every window is shifted one day (t-w..t-1), so no feature sees its own bar:

    - rv5_past: mean |log_return| over 5 days (the established trailing-RV level).
    - rv22 / rv63: rolling close-to-close standard deviation of log_return (ddof=1);
      the dual-definition rv63 is pinned to this rolling-std convention.
    - har5 / har22 / har63: sums of squared log_return over the same windows.
    - volvol22: standard deviation of |log_return| over 22 days.
    - market_abs_ret_5d: per-ticker trailing 5-day mean of the per-date
      cross-sectional mean |log_return|; the cross-sectional mean skips tickers
      without a return on a date, and the panel's first date carries no return at
      all, so the earliest window there is blank (inside the head cut).
    """
    featured = df
    required_columns = {"ticker", "date", "log_return"}
    missing_columns = required_columns - set(featured.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Columns required to compute trailing volatility are missing: {missing_list}")

    ticker = featured["ticker"]
    grouped_log_return = featured.groupby("ticker", sort=False, group_keys=False)["log_return"]
    absolute_log_return = featured["log_return"].abs()

    def trailing_mean(values: pd.Series, window: int) -> pd.Series:
        return values.groupby(ticker, sort=False, group_keys=False).transform(
            lambda s: s.rolling(window=window, min_periods=window).mean().shift(1)
        )

    def trailing_std(values: pd.Series, window: int) -> pd.Series:
        return values.groupby(ticker, sort=False, group_keys=False).transform(
            lambda s: s.rolling(window=window, min_periods=window).std().shift(1)
        )

    def trailing_sum(values: pd.Series, window: int) -> pd.Series:
        return values.groupby(ticker, sort=False, group_keys=False).transform(
            lambda s: s.rolling(window=window, min_periods=window).sum().shift(1)
        )

    market_absolute_return = absolute_log_return.groupby(featured["date"]).transform("mean")

    featured["rv5_past"] = trailing_mean(absolute_log_return, 5).astype("float64")
    featured["rv22"] = trailing_std(featured["log_return"], 22).astype("float64")
    featured["rv63"] = trailing_std(featured["log_return"], 63).astype("float64")
    featured["har5"] = trailing_sum(featured["log_return"] ** 2, 5).astype("float64")
    featured["har22"] = trailing_sum(featured["log_return"] ** 2, 22).astype("float64")
    featured["har63"] = trailing_sum(featured["log_return"] ** 2, 63).astype("float64")
    featured["market_abs_ret_5d"] = trailing_mean(market_absolute_return, 5).astype("float64")
    featured["volvol22"] = trailing_std(absolute_log_return, 22).astype("float64")

    new_columns = (
        "rv5_past",
        "rv22",
        "rv63",
        "har5",
        "har22",
        "har63",
        "market_abs_ret_5d",
        "volvol22",
    )
    for name in new_columns:
        defined_values = featured[name].notna()
        if not np.isfinite(featured.loc[defined_values, name]).all():
            raise ValueError(f"Column {name} produced non-finite values")

    logger.info(
        "_add_trailing_volatility_features: columns=%d, longest window=%d rows",
        len(new_columns),
        63,
    )
    return featured
