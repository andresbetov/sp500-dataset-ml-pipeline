from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
STD_EPSILON = 1e-12

MIN_HISTORY_BY_FEATURE = {
	"simple_return": 1,
	"log_return": 1,
	"return_lag_1": 2,
	"return_lag_2": 3,
	"return_lag_3": 4,
	"return_lag_5": 6,
	"return_lag_10": 11,
	"volume_lag_1": 1,
	"volume_lag_3": 3,
	"volume_lag_5": 5,
	"ema_12": 1,
	"ema_26": 1,
	"rsi_14": 28,
	"macd": 1,
	"macd_signal": 1,
	"macd_hist": 1,
	"log_return_std_10": 12,
	"log_return_std_20": 22,
	"atr_14": 16,
	"volume_sma_10": 11,
	"volume_sma_20": 21,
	"zscore_volume_20": 21,
	"price_vs_sma_10": 11,
	"price_vs_sma_20": 21,
	"price_vs_sma_50": 51,
	"zscore_price_vs_sma_20": 41,
	"rolling_mean_5": 6,
	"rolling_max_10": 11,
	"rolling_min_10": 11,
	"high_low_range": 1,
}

LEGACY_FEATURE_ALIASES = {
	"rolling_std_10": "log_return_std_10",
	"rolling_std_20": "log_return_std_20",
	"volume_zscore": "zscore_volume_20",
	"zscore_price_20": "zscore_price_vs_sma_20",
}


def _compute_stable_rolling_zscore(
	values: pd.Series,
	group_labels: pd.Series,
	window: int,
) -> pd.Series:
	grouped_values = values.groupby(group_labels, sort=False, group_keys=False)
	rolling_mean = grouped_values.transform(
		lambda s: s.rolling(window=window, min_periods=window).mean()
	).shift(1)
	rolling_std = grouped_values.transform(
		lambda s: s.rolling(window=window, min_periods=window).std()
	).shift(1)
	rolling_std = rolling_std.astype("float64")
	rolling_std = rolling_std.where(rolling_std.abs() > STD_EPSILON, np.nan)

	zscore = ((values - rolling_mean) / rolling_std).astype("float64")
	zscore = zscore.replace([np.inf, -np.inf], np.nan)
	return zscore.astype("float64")


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


def _add_return_lags(df: pd.DataFrame) -> pd.DataFrame:
	lagged = df
	for lag in (1, 2, 3, 5, 10):
		lagged[f"return_lag_{lag}"] = (
			lagged.groupby("ticker", sort=False, group_keys=False)["simple_return"].shift(lag).astype("float64")
		)
	return lagged


def _add_volume_lags(df: pd.DataFrame) -> pd.DataFrame:
	lagged = df
	for lag in (1, 3, 5):
		lagged[f"volume_lag_{lag}"] = (
			lagged.groupby("ticker", sort=False, group_keys=False)["volume"].shift(lag).astype("float64")
		)
	return lagged


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
	with_return_lags = _add_return_lags(df)
	return _add_volume_lags(with_return_lags)


def _add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	by_ticker_adj_close = featured.groupby("ticker", sort=False, group_keys=False)["adj_close"]
	lagged_adj_close = by_ticker_adj_close.shift(1)

	for span in (12, 26):
		ema_shifted = by_ticker_adj_close.transform(
			lambda s: s.ewm(span=span, adjust=False).mean().shift(1)
		)
		featured[f"ema_{span}"] = (ema_shifted / lagged_adj_close).astype("float64")

	return featured


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
	return _add_ema_features(df)


def _compute_rsi_14(adj_close: pd.Series) -> pd.Series:
	delta = adj_close.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)

	avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
	avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

	rs = avg_gain / avg_loss
	return 100 - (100 / (1 + rs))


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	by_ticker_adj_close = featured.groupby("ticker", sort=False, group_keys=False)["adj_close"]
	lagged_adj_close = by_ticker_adj_close.shift(1)

	rsi_raw = by_ticker_adj_close.transform(_compute_rsi_14)
	featured["rsi_14"] = (
		rsi_raw.groupby(featured["ticker"], sort=False, group_keys=False).shift(1).astype("float64")
	)

	ema_fast_aligned = by_ticker_adj_close.transform(
		lambda s: s.ewm(span=12, adjust=False).mean().shift(1)
	)
	ema_slow_aligned = by_ticker_adj_close.transform(
		lambda s: s.ewm(span=26, adjust=False).mean().shift(1)
	)
	macd_raw = ema_fast_aligned - ema_slow_aligned
	featured["macd"] = (macd_raw / lagged_adj_close).astype("float64")

	featured["macd_signal"] = (
		featured["macd"]
		.groupby(featured["ticker"], sort=False, group_keys=False)
		.transform(lambda s: s.ewm(span=9, adjust=False).mean())
		.astype("float64")
	)

	featured["macd_hist"] = (featured["macd"] - featured["macd_signal"]).astype("float64")

	return featured


def _add_rolling_std_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	by_ticker_return = featured.groupby("ticker", sort=False, group_keys=False)["log_return"]

	for window in (10, 20):
		featured[f"log_return_std_{window}"] = (
			by_ticker_return.transform(
				lambda s: s.rolling(window=window, min_periods=window).std().shift(1)
			).astype("float64")
		)

	return featured



def _add_atr_feature(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	by_ticker_adj_close = featured.groupby("ticker", sort=False, group_keys=False)["adj_close"]
	prev_adj_close = by_ticker_adj_close.shift(1)
	true_range = pd.concat(
		[
			(featured["high"] - featured["low"]),
			(featured["high"] - prev_adj_close).abs(),
			(featured["low"] - prev_adj_close).abs(),
		],
		axis=1,
	).max(axis=1)

	featured["atr_14"] = (
		true_range.groupby(featured["ticker"], sort=False, group_keys=False)
		.transform(lambda s: s.rolling(window=14, min_periods=14).mean().shift(1))
		.div(prev_adj_close)
		.astype("float64")
	)
	return featured


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
	with_rolling_std = _add_rolling_std_features(df)
	return _add_atr_feature(with_rolling_std)


def _add_volume_sma_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	by_ticker_volume = featured.groupby("ticker", sort=False, group_keys=False)["volume"]

	for window in (10, 20):
		featured[f"volume_sma_{window}"] = (
			by_ticker_volume.transform(
				lambda s: s.rolling(window=window, min_periods=window).mean().shift(1)
			).astype("float64")
		)

	return featured


def _add_volume_zscore_feature(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	featured["zscore_volume_20"] = _compute_stable_rolling_zscore(
		values=featured["volume"],
		group_labels=featured["ticker"],
		window=20,
	)

	return featured


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
	with_volume_sma = _add_volume_sma_features(df)
	return _add_volume_zscore_feature(with_volume_sma)


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
	)

	return featured


def _add_relative_normalization_features(df: pd.DataFrame) -> pd.DataFrame:
	with_price_vs_sma = _add_price_vs_sma_features(df)
	with_price_zscore = _add_price_zscore_feature(with_price_vs_sma)
	return with_price_zscore


def _add_legacy_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
	for legacy_name, canonical_name in LEGACY_FEATURE_ALIASES.items():
		if canonical_name in df.columns:
			df[legacy_name] = df[canonical_name]
	return df


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


def _add_basic_range_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df
	relative_range_raw = (featured["high"] - featured["low"]) / featured["adj_close"]
	featured["high_low_range"] = (
		relative_range_raw.groupby(featured["ticker"], sort=False, group_keys=False).shift(1).astype("float64")
	)
	return featured


def _add_price_action_range_features(df: pd.DataFrame) -> pd.DataFrame:
	return _add_basic_range_features(df)


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


def _enforce_history_based_nan_consistency(df: pd.DataFrame) -> pd.DataFrame:
	position_in_ticker = df.groupby("ticker", sort=False, group_keys=False).cumcount()

	for column, min_history in MIN_HISTORY_BY_FEATURE.items():
		if column in df.columns:
			df.loc[position_in_ticker < min_history, column] = np.nan

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
		"adj_close",
		"volume",
	]
	present_base = [column for column in base_columns if column in df.columns]
	feature_columns = sorted(column for column in df.columns if column not in present_base)
	return df[present_base + feature_columns]


def build_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	_validate_unique_ticker_date_pairs(featured)
	featured = _sort_stably_by_ticker_date(featured)
	featured = _compute_base_derived_features(featured)
	featured = _add_lag_features(featured)
	featured = _add_trend_features(featured)
	featured = _add_momentum_features(featured)
	featured = _add_volatility_features(featured)
	featured = _add_volume_features(featured)
	featured = _add_relative_normalization_features(featured)
	featured = _add_rolling_statistics_features(featured)
	featured = _add_price_action_range_features(featured)
	featured = _enforce_history_based_nan_consistency(featured)
	featured = _cast_numeric_columns_to_float64(featured)
	featured = _add_legacy_feature_aliases(featured)
	featured = _sort_stably_by_ticker_date(featured)
	featured = _reorder_columns_deterministically(featured)

	logger.info(
		"build_features_dataframe: rows=%d, tickers=%d",
		len(featured),
		featured["ticker"].nunique(),
	)
	return featured
