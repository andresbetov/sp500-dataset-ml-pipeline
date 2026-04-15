from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
STD_EPSILON = 1e-12


def _compute_base_derived_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	invalid_adj_close = featured["adj_close"] <= 0
	if invalid_adj_close.any():
		raise ValueError("Column 'adj_close' must be strictly positive to compute log_price")

	featured["log_price"] = np.log(featured["adj_close"]).astype("float64")
	featured["simple_return"] = (
		featured.groupby("ticker", sort=False)["adj_close"].pct_change().astype("float64")
	)
	featured["log_return"] = (
		featured.groupby("ticker", sort=False)["log_price"].diff().astype("float64")
	)

	return featured


def _add_return_lags(df: pd.DataFrame) -> pd.DataFrame:
	lagged = df.copy()
	for lag in (1, 2, 3, 5, 10):
		lagged[f"return_lag_{lag}"] = (
			lagged.groupby("ticker", sort=False)["simple_return"].shift(lag).astype("float64")
		)
	return lagged


def _add_volume_lags(df: pd.DataFrame) -> pd.DataFrame:
	lagged = df.copy()
	for lag in (1, 3, 5):
		lagged[f"volume_lag_{lag}"] = (
			lagged.groupby("ticker", sort=False)["volume"].shift(lag).astype("float64")
		)
	return lagged


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
	with_return_lags = _add_return_lags(df)
	return _add_volume_lags(with_return_lags)


def _add_sma_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_adj_close = featured.groupby("ticker", sort=False)["adj_close"]

	for window in (10, 20, 50):
		featured[f"sma_{window}"] = (
			by_ticker_adj_close.transform(
				lambda s: s.rolling(window=window, min_periods=window).mean().shift(1)
			).astype("float64")
		)

	return featured


def _add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_adj_close = featured.groupby("ticker", sort=False)["adj_close"]

	for span in (12, 26):
		featured[f"ema_{span}"] = (
			by_ticker_adj_close.transform(
				lambda s: s.ewm(span=span, adjust=False).mean().shift(1)
			).astype("float64")
		)

	return featured


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
	with_sma = _add_sma_features(df)
	return _add_ema_features(with_sma)


def _compute_rsi_14(adj_close: pd.Series) -> pd.Series:
	delta = adj_close.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)

	avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
	avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

	rs = avg_gain / avg_loss
	return 100 - (100 / (1 + rs))


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_adj_close = featured.groupby("ticker", sort=False)["adj_close"]

	featured["rsi_14"] = by_ticker_adj_close.transform(_compute_rsi_14).astype("float64")

	ema_fast_raw = by_ticker_adj_close.transform(lambda s: s.ewm(span=12, adjust=False).mean())
	ema_slow_raw = by_ticker_adj_close.transform(lambda s: s.ewm(span=26, adjust=False).mean())
	macd_raw = ema_fast_raw - ema_slow_raw
	featured["macd"] = macd_raw.groupby(featured["ticker"], sort=False).shift(1).astype("float64")

	macd_signal_raw = (
		macd_raw.groupby(featured["ticker"], sort=False)
		.transform(lambda s: s.ewm(span=9, adjust=False).mean())
	)
	featured["macd_signal"] = (
		macd_signal_raw.groupby(featured["ticker"], sort=False).shift(1)
		.astype("float64")
	)

	featured["macd_hist"] = (featured["macd"] - featured["macd_signal"]).astype("float64")

	return featured


def _add_rolling_std_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_return = featured.groupby("ticker", sort=False)["simple_return"]

	for window in (10, 20):
		featured[f"rolling_std_{window}"] = (
			by_ticker_return.transform(
				lambda s: s.rolling(window=window, min_periods=window).std().shift(1)
			).astype("float64")
		)

	return featured



def _add_atr_feature(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	prev_adj_close = featured.groupby("ticker", sort=False)["adj_close"].shift(1)
	true_range = pd.concat(
		[
			(featured["high"] - featured["low"]),
			(featured["high"] - prev_adj_close).abs(),
			(featured["low"] - prev_adj_close).abs(),
		],
		axis=1,
	).max(axis=1)

	featured["atr_14"] = (
		true_range.groupby(featured["ticker"], sort=False)
		.transform(lambda s: s.rolling(window=14, min_periods=14).mean().shift(1))
		.astype("float64")
	)
	return featured


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
	with_rolling_std = _add_rolling_std_features(df)
	return _add_atr_feature(with_rolling_std)


def _add_volume_sma_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_volume = featured.groupby("ticker", sort=False)["volume"]

	for window in (10, 20):
		featured[f"volume_sma_{window}"] = (
			by_ticker_volume.transform(
				lambda s: s.rolling(window=window, min_periods=window).mean().shift(1)
			).astype("float64")
		)

	return featured


def _add_volume_zscore_feature(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_volume = featured.groupby("ticker", sort=False)["volume"]

	rolling_mean_20 = by_ticker_volume.transform(
		lambda s: s.rolling(window=20, min_periods=20).mean()
	).shift(1)
	rolling_std_20 = by_ticker_volume.transform(
		lambda s: s.rolling(window=20, min_periods=20).std()
	).shift(1)
	rolling_std_20 = rolling_std_20.mask(rolling_std_20.abs() <= STD_EPSILON)

	featured["volume_zscore"] = (
		(featured["volume"] - rolling_mean_20) / rolling_std_20
	).astype("float64")

	return featured


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
	with_volume_sma = _add_volume_sma_features(df)
	return _add_volume_zscore_feature(with_volume_sma)


def _add_price_vs_sma_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()

	for window in (10, 20, 50):
		sma_column = f"sma_{window}"
		featured[f"price_vs_sma_{window}"] = (
			featured["adj_close"] / featured[sma_column]
		).astype("float64")

	return featured


def _add_price_zscore_feature(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_adj_close = featured.groupby("ticker", sort=False)["adj_close"]

	rolling_mean_20 = by_ticker_adj_close.transform(
		lambda s: s.rolling(window=20, min_periods=20).mean()
	).shift(1)
	rolling_std_20 = by_ticker_adj_close.transform(
		lambda s: s.rolling(window=20, min_periods=20).std()
	).shift(1)
	rolling_std_20 = rolling_std_20.mask(rolling_std_20.abs() <= STD_EPSILON)

	featured["zscore_price_20"] = (
		(featured["adj_close"] - rolling_mean_20) / rolling_std_20
	).astype("float64")

	return featured


def _add_relative_normalization_features(df: pd.DataFrame) -> pd.DataFrame:
	with_price_vs_sma = _add_price_vs_sma_features(df)
	with_price_zscore = _add_price_zscore_feature(with_price_vs_sma)
	with_price_zscore["zscore_volume_20"] = with_price_zscore["volume_zscore"].astype("float64")
	return with_price_zscore


def _add_rolling_mean_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_adj_close = featured.groupby("ticker", sort=False)["adj_close"]

	for window in (5, 10):
		featured[f"rolling_mean_{window}"] = (
			by_ticker_adj_close.transform(
				lambda s: s.rolling(window=window, min_periods=window).mean().shift(1)
			).astype("float64")
		)

	return featured


def _add_rolling_extreme_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_adj_close = featured.groupby("ticker", sort=False)["adj_close"]

	featured["rolling_max_10"] = (
		by_ticker_adj_close.transform(
			lambda s: s.rolling(window=10, min_periods=10).max().shift(1)
		).astype("float64")
	)
	featured["rolling_min_10"] = (
		by_ticker_adj_close.transform(
			lambda s: s.rolling(window=10, min_periods=10).min().shift(1)
		).astype("float64")
	)

	return featured


def _add_rolling_statistics_features(df: pd.DataFrame) -> pd.DataFrame:
	with_rolling_mean = _add_rolling_mean_features(df)
	return _add_rolling_extreme_features(with_rolling_mean)


def _add_basic_range_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	featured["high_low_range"] = (featured["high"] - featured["low"]).astype("float64")
	featured["close_open_range"] = (featured["adj_close"] - featured["open"]).astype("float64")
	return featured


def _add_true_range_feature(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	prev_adj_close = featured.groupby("ticker", sort=False)["adj_close"].shift(1)
	featured["true_range"] = pd.concat(
		[
			(featured["high"] - featured["low"]),
			(featured["high"] - prev_adj_close).abs(),
			(featured["low"] - prev_adj_close).abs(),
		],
		axis=1,
	).max(axis=1).astype("float64")
	return featured


def _add_price_action_range_features(df: pd.DataFrame) -> pd.DataFrame:
	with_basic_ranges = _add_basic_range_features(df)
	return _add_true_range_feature(with_basic_ranges)


def build_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	featured = _compute_base_derived_features(df)
	featured = _add_lag_features(featured)
	featured = _add_trend_features(featured)
	featured = _add_momentum_features(featured)
	featured = _add_volatility_features(featured)
	featured = _add_volume_features(featured)
	featured = _add_relative_normalization_features(featured)
	featured = _add_rolling_statistics_features(featured)
	featured = _add_price_action_range_features(featured)

	logger.info(
		"build_features_dataframe: rows=%d, tickers=%d",
		len(featured),
		featured["ticker"].nunique(),
	)
	return featured
