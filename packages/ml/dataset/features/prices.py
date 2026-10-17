from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._common import (
	ADJUSTED_OHLC_HIGH_COLUMN,
	ADJUSTED_OHLC_LOW_COLUMN,
	ADJUSTED_OHLC_OPEN_COLUMN,
	_compute_stable_rolling_zscore,
)

logger = logging.getLogger(__name__)


def _compute_base_derived_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	invalid_adj_close = featured["adj_close"] <= 0
	if invalid_adj_close.any():
		raise ValueError("Column 'adj_close' must be strictly positive to compute log_price")

	featured["log_price"] = np.log(featured["adj_close"]).astype("float64")
	featured["simple_return"] = (
		featured.groupby("ticker", sort=False, group_keys=False)["adj_close"].pct_change().astype("float64")
	)
	featured["log_return"] = (
		featured.groupby("ticker", sort=False, group_keys=False)["log_price"].diff().astype("float64")
	)

	return featured


def _add_adjusted_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	required_columns = {"open", "high", "low", "close", "adj_close"}
	missing_columns = required_columns - set(featured.columns)
	if missing_columns:
		missing_list = ", ".join(sorted(missing_columns))
		raise ValueError(f"Columns required to compute adjusted OHLC are missing: {missing_list}")

	invalid_close = featured["close"] <= 0
	invalid_adj_close = featured["adj_close"] <= 0
	if invalid_close.any() or invalid_adj_close.any():
		raise ValueError("Columns 'close' and 'adj_close' must be strictly positive")

	adjustment_factor = (featured["adj_close"] / featured["close"]).astype("float64")
	if not np.isfinite(adjustment_factor).all():
		raise ValueError("Adjusted OHLC factor produced non-finite values")

	featured[ADJUSTED_OHLC_OPEN_COLUMN] = (featured["open"] * adjustment_factor).astype("float64")
	featured[ADJUSTED_OHLC_HIGH_COLUMN] = (featured["high"] * adjustment_factor).astype("float64")
	featured[ADJUSTED_OHLC_LOW_COLUMN] = (featured["low"] * adjustment_factor).astype("float64")

	# Scaling every price by the same per-row positive factor preserves the OHLC ordering
	# that preparation already enforces, so re-validating it here cannot fail. Record the
	# factor once instead, so the adjustment basis stays visible in the run log.
	factor_min = float(adjustment_factor.min())
	factor_max = float(adjustment_factor.max())
	if factor_min == factor_max == 1.0:
		logger.info(
			"_add_adjusted_ohlc_columns: adjustment factor is identically 1.0 "
			"(adj_close == close); adjusted OHLC equals raw OHLC"
		)
	elif factor_min == factor_max:
		logger.info(
			"_add_adjusted_ohlc_columns: adjustment factor is a single value: %.10f",
			factor_min,
		)
	else:
		logger.info(
			"_add_adjusted_ohlc_columns: adjustment factor varies: min=%.10f max=%.10f",
			factor_min,
			factor_max,
		)

	return featured


def _add_return_lags(df: pd.DataFrame) -> pd.DataFrame:
	lagged = df
	for lag in (1, 2, 3, 5, 10):
		lagged[f"return_lag_{lag}"] = (
			lagged.groupby("ticker", sort=False, group_keys=False)["simple_return"].shift(lag).astype("float64")
		)
	return lagged


def _add_price_vs_sma_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	by_ticker_adj_close = featured.groupby("ticker", sort=False, group_keys=False)["adj_close"]

	for window in (10, 20, 50):
		lagged_sma = by_ticker_adj_close.transform(
			lambda s: s.rolling(window=window, min_periods=window).mean().shift(1)
		)
		featured[f"price_vs_sma_{window}"] = (
			featured["adj_close"] / lagged_sma
		).astype("float64")

	return featured


def _add_price_zscore_feature(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	if "price_vs_sma_20" not in featured.columns:
		raise ValueError("Column 'price_vs_sma_20' is required before computing zscore_price_vs_sma_20")

	featured["zscore_price_vs_sma_20"] = _compute_stable_rolling_zscore(
		values=featured["price_vs_sma_20"],
		group_labels=featured["ticker"],
		window=20,
		name="zscore_price_vs_sma_20",
	)

	return featured


def _add_relative_normalization_features(df: pd.DataFrame) -> pd.DataFrame:
	with_price_vs_sma = _add_price_vs_sma_features(df)
	with_price_zscore = _add_price_zscore_feature(with_price_vs_sma)
	return with_price_zscore


def _add_rolling_mean_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	by_ticker_adj_close = featured.groupby("ticker", sort=False, group_keys=False)["adj_close"]
	lagged_adj_close = by_ticker_adj_close.shift(1)
	rolling_mean_5 = by_ticker_adj_close.transform(
		lambda s: s.rolling(window=5, min_periods=5).mean().shift(1)
	)
	featured["rolling_mean_5"] = (rolling_mean_5 / lagged_adj_close).astype("float64")

	return featured


def _add_rolling_extreme_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	by_ticker_adj_close = featured.groupby("ticker", sort=False, group_keys=False)["adj_close"]
	lagged_adj_close = by_ticker_adj_close.shift(1)

	featured["rolling_max_10"] = (
		by_ticker_adj_close.transform(
			lambda s: s.rolling(window=10, min_periods=10).max().shift(1)
		).div(lagged_adj_close).astype("float64")
	)
	featured["rolling_min_10"] = (
		by_ticker_adj_close.transform(
			lambda s: s.rolling(window=10, min_periods=10).min().shift(1)
		).div(lagged_adj_close).astype("float64")
	)

	return featured


def _add_rolling_statistics_features(df: pd.DataFrame) -> pd.DataFrame:
	with_rolling_mean = _add_rolling_mean_features(df)
	return _add_rolling_extreme_features(with_rolling_mean)
