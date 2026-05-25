from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
STD_EPSILON = 1e-12
PRICE_TOLERANCE = 1e-10
DIRECTION_TOLERANCE = 0.005

ADJUSTED_OHLC_OPEN_COLUMN = "_adj_open"
ADJUSTED_OHLC_HIGH_COLUMN = "_adj_high"
ADJUSTED_OHLC_LOW_COLUMN = "_adj_low"
INTERNAL_FEATURE_COLUMNS = (
	ADJUSTED_OHLC_OPEN_COLUMN,
	ADJUSTED_OHLC_HIGH_COLUMN,
	ADJUSTED_OHLC_LOW_COLUMN,
)

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
	"realized_volatility_5d": 11,
}

LEGACY_FEATURE_ALIASES = {
	"volume_zscore": "zscore_volume_20",
	"zscore_price_20": "zscore_price_vs_sma_20",
}


def _compute_stable_rolling_zscore(
	values: pd.Series,
	group_labels: pd.Series,
	window: int,
) -> pd.Series:
	if window < 1:
		raise ValueError(f"window must be >= 1, got {window}")
	if len(values) != len(group_labels):
		raise ValueError(
			"values and group_labels must have the same length: "
			f"len(values)={len(values)}, len(group_labels)={len(group_labels)}"
		)
	if not values.index.equals(group_labels.index):
		raise ValueError("values and group_labels must share the same index")

	grouped_values = values.groupby(group_labels, sort=False, group_keys=False)
	rolling_mean = grouped_values.transform(
		lambda s: s.rolling(window=window, min_periods=window).mean().shift(1)
	)
	rolling_std = grouped_values.transform(
		lambda s: s.rolling(window=window, min_periods=window).std().shift(1)
	)
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

	valid_range = featured[ADJUSTED_OHLC_HIGH_COLUMN] >= (featured[ADJUSTED_OHLC_LOW_COLUMN] - PRICE_TOLERANCE)
	valid_open = (
		(featured[ADJUSTED_OHLC_OPEN_COLUMN] <= (featured[ADJUSTED_OHLC_HIGH_COLUMN] + PRICE_TOLERANCE))
		& (featured[ADJUSTED_OHLC_OPEN_COLUMN] >= (featured[ADJUSTED_OHLC_LOW_COLUMN] - PRICE_TOLERANCE))
	)
	valid_close = (
		(featured["adj_close"] <= (featured[ADJUSTED_OHLC_HIGH_COLUMN] + PRICE_TOLERANCE))
		& (featured["adj_close"] >= (featured[ADJUSTED_OHLC_LOW_COLUMN] - PRICE_TOLERANCE))
	)
	valid_rows = valid_range & valid_open & valid_close
	if not valid_rows.all():
		invalid_count = int((~valid_rows).sum())
		raise ValueError(
			"Adjusted OHLC consistency validation failed after aligning to adj_close: "
			f"invalid_rows={invalid_count}"
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


def _add_target_realized_volatility_5d(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Compute realized volatility: rolling std of log_return for next 5 trading days.

	For each row, this is the actual volatility that occurred in the forward window.
	Uses shift(-5) to prevent look-ahead bias: volatility is computed THEN shifted back.

	This represents the true realized volatility that would occur in the next 5 days,
	making it a perfect target for teaching the model about volatility forecasting.
	"""
	featured = df

	# Group by ticker to prevent cross-ticker leakage
	# Compute rolling 5-day std of log_return, then shift -5 to get realized volatility
	featured["realized_volatility_5d"] = featured.groupby(
		"ticker", sort=False, group_keys=False
	)["log_return"].transform(
		lambda x: x.rolling(window=5, min_periods=5).std().shift(-5)
	).astype("float64")

	return featured


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


def _drop_internal_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
	columns_to_drop = [column for column in INTERNAL_FEATURE_COLUMNS if column in df.columns]
	if columns_to_drop:
		df.drop(columns=columns_to_drop, inplace=True)
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
	featured = _add_trend_features(featured)
	featured = _add_momentum_features(featured)
	featured = _add_volume_features(featured)
	featured = _add_relative_normalization_features(featured)
	featured = _add_rolling_statistics_features(featured)
	featured = _drop_internal_feature_columns(featured)
	featured = _enforce_history_based_nan_consistency(featured)
	featured = _cast_numeric_columns_to_float64(featured)
	featured = _add_legacy_feature_aliases(featured)
	featured = _sort_stably_by_ticker_date(featured)
	featured = _reorder_columns_deterministically(featured)
	featured = _add_target_realized_volatility_5d(featured)
	featured = _add_market_regime_feature(featured)

	logger.info(
		"build_features_dataframe: rows=%d, tickers=%d",
		len(featured),
		featured["ticker"].nunique(),
	)
	return featured


def _add_market_regime_feature(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Assign market regime based on historical periods.

	Regimes:
	- 0: 2000-2007 (pre-crisis)
	- 1: 2008-2010 (financial crisis)
	- 2: 2011-2019 (post-crisis recovery)
	- 3: 2020-2021 (COVID)
	- 4: 2022-2026 (post-COVID inflation)

	Returns:
		DataFrame with new 'market_regime' column (int8: 0-4)
	"""
	def get_regime(date: pd.Timestamp) -> int:
		year = date.year
		if year <= 2007:
			return 0
		elif year <= 2010:
			return 1
		elif year <= 2019:
			return 2
		elif year <= 2021:
			return 3
		else:
			return 4

	df['market_regime'] = df['date'].apply(get_regime).astype('int8')
	return df


def _characterize_regimes(df: pd.DataFrame) -> dict:
	"""
	Calculate statistics per market regime (regime "fingerprint").

	Returns:
		Dict with regime statistics:
		{
			"regime_name": {
				"volatility_mean": float,
				"volatility_std": float,
				"volatility_min": float,
				"volatility_max": float,
				"volatility_median": float,
				"return_mean": float,
				"return_std": float,
				"return_skewness": float,
				"return_kurtosis": float,
				"volume_mean": float,
				"volume_std": float,
				"ticker_correlation_mean": float,
				"ticker_correlation_std": float,
				"n_samples": int,
				"unique_tickers": int,
				"date_range": str,
			}
		}
	"""
	regime_names = {
		0: "pre-crisis (2000-2007)",
		1: "financial-crisis (2008-2010)",
		2: "post-crisis (2011-2019)",
		3: "covid (2020-2021)",
		4: "post-covid (2022-2026)"
	}

	stats = {}

	for regime_id in range(5):
		regime_data = df[df['market_regime'] == regime_id]

		if len(regime_data) == 0:
			logger.warning(f"Regime {regime_id}: no data found")
			continue

		# Volatility statistics
		volatility_values = regime_data['realized_volatility_5d']

		# Return statistics
		returns_values = regime_data['simple_return'].dropna()

		# Volume statistics
		volume_values = regime_data['volume']

		# Ticker correlation (per regime)
		ticker_correlations = []
		for _, date_group in regime_data.groupby('date'):
			if len(date_group) > 2:
				date_returns = date_group[['ticker', 'simple_return']].dropna()
				if len(date_returns) > 2:
					try:
						corr_matrix = date_returns.set_index('ticker')['simple_return'].corr()
						# Extract upper triangle correlations
						if len(corr_matrix) > 1:
							correlations = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
							ticker_correlations.extend(correlations)
					except Exception:
						pass

		stats[regime_names[regime_id]] = {
			# Volatility characteristics
			'volatility_mean': float(volatility_values.mean()),
			'volatility_std': float(volatility_values.std()),
			'volatility_min': float(volatility_values.min()),
			'volatility_max': float(volatility_values.max()),
			'volatility_median': float(volatility_values.median()),

			# Return characteristics
			'return_mean': float(returns_values.mean()),
			'return_std': float(returns_values.std()),
			'return_skewness': float(returns_values.skew()),
			'return_kurtosis': float(returns_values.kurtosis()),

			# Volume characteristics
			'volume_mean': float(volume_values.mean()),
			'volume_std': float(volume_values.std()),

			# Correlation characteristics
			'ticker_correlation_mean': float(np.mean(ticker_correlations)) if ticker_correlations else 0.0,
			'ticker_correlation_std': float(np.std(ticker_correlations)) if ticker_correlations else 0.0,

			# Sample count
			'n_samples': len(regime_data),
			'unique_tickers': len(regime_data['ticker'].unique()),
			'date_range': f"{regime_data['date'].min()} to {regime_data['date'].max()}",
		}

	logger.info("Regime Characterization Complete:")
	for regime_name, regime_stats in stats.items():
		logger.info(f"  {regime_name}: vol_mean={regime_stats['volatility_mean']:.6f}, vol_std={regime_stats['volatility_std']:.6f}, n_samples={regime_stats['n_samples']}")

	return stats

