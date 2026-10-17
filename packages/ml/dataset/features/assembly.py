from __future__ import annotations

import logging

import pandas as pd

from .prices import (
    _add_adjusted_ohlc_columns,
    _add_relative_normalization_features,
    _add_return_lags,
    _add_rolling_statistics_features,
    _compute_base_derived_features,
)
from .regimes import _add_market_regime_feature
from .target import _add_target_realized_volatility_5d
from .trend import _add_momentum_features, _add_trend_features
from .volatility import _add_range_volatility_features, _add_trailing_volatility_features
from .volume import _add_volume_features, _add_volume_lags, _transform_volume_lag_features

logger = logging.getLogger(__name__)


# The first row position (0-based, within a ticker) at which a feature is defined.
# Every rolling(window)+shift(1) entry needs window prior rows, so its first defined
# position equals the window length; entries built on log_return start one row later
# because each ticker's first return is undefined; lag entries need the lag count.
MIN_HISTORY_BY_FEATURE = {
    "simple_return": 1,
    "log_return": 1,
    "return_lag_1": 1,
    "return_lag_2": 2,
    "return_lag_3": 3,
    "return_lag_5": 5,
    "return_lag_10": 10,
    "volume_lag_1": 1,
    "volume_lag_3": 23,
    "volume_lag_5": 5,
    "ema_12": 1,
    "ema_26": 1,
    "rsi_14": 15,
    "macd": 1,
    "macd_signal": 1,
    "macd_hist": 1,
    "zscore_volume_20": 20,
    "price_vs_sma_10": 10,
    "price_vs_sma_20": 20,
    "price_vs_sma_50": 50,
    "zscore_price_vs_sma_20": 40,
    "rolling_mean_5": 5,
    "rolling_max_10": 10,
    "rolling_min_10": 10,
    "atr5": 5,
    "atr20": 20,
    "park5": 5,
    "park20": 20,
    "gk5": 5,
    "rs5": 5,
    "gap5": 5,
    "rv5_past": 6,
    "rv22": 23,
    "rv63": 64,
    "har5": 6,
    "har22": 23,
    "har63": 64,
    "market_abs_ret_5d": 6,
    "volvol22": 23,
}


def _validate_unique_ticker_date_pairs(df: pd.DataFrame) -> None:
    duplicated_mask = df.duplicated(subset=["ticker", "date"], keep=False)
    if not duplicated_mask.any():
        return

    duplicates = (
        df.loc[duplicated_mask, ["ticker", "date"]]
        .drop_duplicates()
        .sort_values(["ticker", "date"])
    )
    sample = duplicates.head(5)
    sample_pairs = ", ".join(
        f"({row.ticker}, {row.date})" for row in sample.itertuples(index=False)
    )
    raise ValueError(
        "Found duplicate (ticker, date) pairs before feature generation: "
        f"count={len(duplicates)}; sample=[{sample_pairs}]"
    )


def _normalize_and_validate_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Column 'date' is required before feature generation")

    date_values = df["date"]
    if pd.api.types.is_datetime64_any_dtype(date_values):
        parsed_date = date_values
    else:
        try:
            parsed_date = pd.to_datetime(date_values, errors="raise", format="mixed")
        except Exception as error:
            raise ValueError("Column 'date' must be parseable as datetime") from error

    if not pd.api.types.is_datetime64_any_dtype(parsed_date):
        raise ValueError("Column 'date' must be datetime-like after parsing")
    if isinstance(parsed_date.dtype, pd.DatetimeTZDtype):
        raise ValueError("Column 'date' must be timezone-naive")
    if parsed_date.isna().any():
        raise ValueError("Column 'date' contains invalid or missing datetime values")

    df["date"] = parsed_date.dt.normalize()
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    with_return_lags = _add_return_lags(df)
    return _add_volume_lags(with_return_lags)


def _enforce_history_based_nan_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Blank each feature before its warm-up ends, transactionally.

    Every constrained column is validated up front, and each column is replaced by a
    freshly computed Series, so a failure cannot leave a half-blanked frame behind.
    """
    unproduced_columns = [column for column in MIN_HISTORY_BY_FEATURE if column not in df.columns]
    if unproduced_columns:
        unproduced_list = ", ".join(sorted(unproduced_columns))
        raise ValueError(
            f"History-constrained columns are not produced by the feature build: {unproduced_list}"
        )

    position_in_ticker = df.groupby("ticker", sort=False, group_keys=False).cumcount()

    for column, min_history in MIN_HISTORY_BY_FEATURE.items():
        df[column] = df[column].where(position_in_ticker >= min_history)

    logger.info(
        "_enforce_history_based_nan_consistency: constrained columns=%d, longest warm-up=%d rows",
        len(MIN_HISTORY_BY_FEATURE),
        max(MIN_HISTORY_BY_FEATURE.values()),
    )

    return df


def _cast_numeric_columns_to_float64(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns] = df[numeric_columns].astype("float64")
    return df


def _sort_stably_by_ticker_date(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(["ticker", "date"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _reorder_columns_deterministically(df: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    present_base = [column for column in base_columns if column in df.columns]
    feature_columns = sorted(column for column in df.columns if column not in present_base)
    return df[present_base + feature_columns]


def build_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured = _normalize_and_validate_date_column(featured)
    _validate_unique_ticker_date_pairs(featured)
    featured = _sort_stably_by_ticker_date(featured)
    featured = _add_adjusted_ohlc_columns(featured)
    featured = _compute_base_derived_features(featured)
    featured = _add_lag_features(featured)
    featured = _transform_volume_lag_features(featured)
    featured = _add_trend_features(featured)
    featured = _add_momentum_features(featured)
    featured = _add_volume_features(featured)
    featured = _add_relative_normalization_features(featured)
    featured = _add_rolling_statistics_features(featured)
    featured = _add_range_volatility_features(featured)
    featured = _add_trailing_volatility_features(featured)
    featured = _enforce_history_based_nan_consistency(featured)
    featured = _cast_numeric_columns_to_float64(featured)
    # Target and regime are added before the final reorder so the deterministic
    # column order covers every shipped column; the regime keeps its int8 dtype
    # because it is added after the float64 cast.
    featured = _add_target_realized_volatility_5d(featured)
    featured = _add_market_regime_feature(featured)
    featured = _sort_stably_by_ticker_date(featured)
    featured = _reorder_columns_deterministically(featured)

    logger.info(
        "build_features_dataframe: rows=%d, tickers=%d",
        len(featured),
        featured["ticker"].nunique(),
    )
    return featured
