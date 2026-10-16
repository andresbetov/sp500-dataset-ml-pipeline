from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
STD_EPSILON = 1e-12
# A window whose value range is at or below this threshold carries no scale for a
# meaningful z-score; such windows are treated as constant instead of dividing by
# a zero standard deviation.
ZSCORE_RANGE_EPSILON = 1e-12
DIRECTION_TOLERANCE = 0.005
REGIME_CORRELATION_WINDOW = 60

TARGET_WINDOW_DAYS = 5
# Yang-Zhang's open-to-close weight for a window of TARGET_WINDOW_DAYS days.
YANG_ZHANG_OPEN_TO_CLOSE_WEIGHT = 0.34 / (
	1.34 + (TARGET_WINDOW_DAYS + 1) / (TARGET_WINDOW_DAYS - 1)
)

ADJUSTED_OHLC_OPEN_COLUMN = "_adj_open"
ADJUSTED_OHLC_HIGH_COLUMN = "_adj_high"
ADJUSTED_OHLC_LOW_COLUMN = "_adj_low"

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
	"volume_sma_10": 10,
	"volume_sma_20": 20,
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

LEGACY_FEATURE_ALIASES = {
	"volume_zscore": "zscore_volume_20",
	"zscore_price_20": "zscore_price_vs_sma_20",
}


def _compute_stable_rolling_zscore(
	values: pd.Series,
	group_labels: pd.Series,
	window: int,
	name: str,
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
	rolling_min = grouped_values.transform(
		lambda s: s.rolling(window=window, min_periods=window).min().shift(1)
	)
	rolling_max = grouped_values.transform(
		lambda s: s.rolling(window=window, min_periods=window).max().shift(1)
	)

	constant_window = (rolling_max - rolling_min) <= ZSCORE_RANGE_EPSILON
	constant_count = int(constant_window.sum())
	if constant_count:
		logger.info(
			"_compute_stable_rolling_zscore: %s constant windows=%d (range <= %.0e, z set to 0)",
			name,
			constant_count,
			ZSCORE_RANGE_EPSILON,
		)

	zscore = ((values - rolling_mean) / rolling_std).astype("float64")
	# A constant window has no dispersion to normalize by; pin its z-score to zero
	# instead of letting 0/0 and tiny-denominator ratios poison the column.
	zscore = zscore.where(~constant_window, 0.0)
	zscore = zscore.replace([np.inf, -np.inf], np.nan)

	defined_zscores = zscore.notna()
	if not np.isfinite(zscore[defined_zscores]).all():
		raise ValueError(f"Column {name} produced non-finite z-scores")
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


def _transform_volume_lag_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Put the volume-lag columns on bounded scales, in place.

	Raw share counts are heavy-tailed and scale with splits and price level, so each
	lag column keeps its name and lag but changes units: volume_lag_1 -> log1p;
	volume_lag_3 -> per-ticker trailing z-score of the log1p values over 20 days
	(constant windows are pinned to zero by the shared z-score guard); volume_lag_5
	-> per-date cross-sectional rank mapped into [-1, 1] (the rank-transform
	convention). Standardizing the raw counts instead reproduces the zero-dominated
	window blow-up already on record for the raw volume z (rows in the tens of
	thousands of sigma), so the log-first z is the zero-safe realization of the
	per-ticker z form. log1p keeps zeros at zero and ranks are finite by
	construction, so surviving zero-volume rows cannot produce -inf in any form.
	"""
	featured = df
	required_columns = {"volume_lag_1", "volume_lag_3", "volume_lag_5"}
	missing_columns = required_columns - set(featured.columns)
	if missing_columns:
		missing_list = ", ".join(sorted(missing_columns))
		raise ValueError(f"Columns required to transform volume lags are missing: {missing_list}")

	featured["volume_lag_1"] = np.log1p(featured["volume_lag_1"]).astype("float64")
	featured["volume_lag_3"] = _compute_stable_rolling_zscore(
		values=np.log1p(featured["volume_lag_3"]),
		group_labels=featured["ticker"],
		window=20,
		name="volume_lag_3",
	)
	rank_percentile = featured.groupby("date")["volume_lag_5"].rank(pct=True)
	featured["volume_lag_5"] = (2.0 * rank_percentile - 1.0).astype("float64")

	for name in ("volume_lag_1", "volume_lag_3", "volume_lag_5"):
		defined_values = featured[name].notna()
		if not np.isfinite(featured.loc[defined_values, name]).all():
			raise ValueError(f"Column {name} produced non-finite values after transformation")

	logger.info(
		"_transform_volume_lag_features: columns=3, surviving zero-volume rows=%d",
		int(featured["volume"].eq(0).sum()),
	)
	return featured


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
		name="zscore_volume_20",
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
		name="zscore_price_vs_sma_20",
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


def _forward_window_aggregate(
	series: pd.Series,
	ticker: pd.Series,
	statistic: str,
) -> pd.Series:
	"""Aggregate each ticker's next TARGET_WINDOW_DAYS values onto the current row."""
	if statistic not in {"var", "mean"}:
		raise ValueError(f"Unsupported window statistic: {statistic}")

	def aggregate(values: pd.Series) -> pd.Series:
		windowed = values.rolling(
			window=TARGET_WINDOW_DAYS, min_periods=TARGET_WINDOW_DAYS
		)
		aggregated = windowed.var() if statistic == "var" else windowed.mean()
		return aggregated.shift(-TARGET_WINDOW_DAYS)

	return series.groupby(ticker, sort=False, group_keys=False).transform(aggregate)


def _add_target_realized_volatility_5d(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute the Yang-Zhang realized-volatility target over the next 5 trading days.

	For every row j the estimator aggregates the forward window i = j+1..j+5:

	- overnight: ln(_adj_open_i / _adj_close_{i-1}); the first window day uses the
	  label row's own close, an input at the horizon start rather than leakage past it
	- open-to-close: ln(_adj_close_i / _adj_open_i)
	- Rogers-Satchell: ln(H_i/C_i)*ln(H_i/O_i) + ln(L_i/C_i)*ln(L_i/O_i)

	variance = var(overnight) + k*var(open-to-close) + (1-k)*mean(Rogers-Satchell),
	with sample variances (ddof=1) over the window and k the standard Yang-Zhang
	weight for n=5. The stored value is sqrt(max(variance, 0)), a per-day volatility
	estimate (daily sigma, not annualized - multiply by sqrt(252) to annualize).
	Rows without a complete forward window - the last 5 rows of every ticker - are NaN.
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
		raise ValueError(f"Columns required to compute the target are missing: {missing_list}")

	previous_close = featured.groupby("ticker", sort=False, group_keys=False)["adj_close"].shift(1)
	overnight_return = np.log(featured[ADJUSTED_OHLC_OPEN_COLUMN] / previous_close)
	open_to_close_return = np.log(featured["adj_close"] / featured[ADJUSTED_OHLC_OPEN_COLUMN])
	rogers_satchell = (
		np.log(featured[ADJUSTED_OHLC_HIGH_COLUMN] / featured["adj_close"])
		* np.log(featured[ADJUSTED_OHLC_HIGH_COLUMN] / featured[ADJUSTED_OHLC_OPEN_COLUMN])
		+ np.log(featured[ADJUSTED_OHLC_LOW_COLUMN] / featured["adj_close"])
		* np.log(featured[ADJUSTED_OHLC_LOW_COLUMN] / featured[ADJUSTED_OHLC_OPEN_COLUMN])
	)

	ticker = featured["ticker"]
	variance_overnight = _forward_window_aggregate(overnight_return, ticker, "var")
	variance_open_to_close = _forward_window_aggregate(open_to_close_return, ticker, "var")
	rogers_satchell_mean = _forward_window_aggregate(rogers_satchell, ticker, "mean")

	variance = (
		variance_overnight
		+ YANG_ZHANG_OPEN_TO_CLOSE_WEIGHT * variance_open_to_close
		+ (1.0 - YANG_ZHANG_OPEN_TO_CLOSE_WEIGHT) * rogers_satchell_mean
	)
	defined_variance = variance.notna()
	if not np.isfinite(variance[defined_variance]).all():
		raise ValueError("Yang-Zhang target produced non-finite variance")

	featured["realized_volatility_5d"] = np.sqrt(variance.clip(lower=0.0)).astype("float64")
	logger.info(
		"_add_target_realized_volatility_5d: window=%d, undefined rows=%d",
		TARGET_WINDOW_DAYS,
		int((~defined_variance).sum()),
	)
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
	featured = _add_legacy_feature_aliases(featured)
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


def _geyer_integrated_autocorrelation_time(values: np.ndarray) -> float:
	"""Integrated autocorrelation time via Geyer's initial positive sequence.

	Trailing windows share most of their observations, so the window-mean series is
	strongly autocorrelated and the naive standard error (sample standard deviation
	over sqrt(n)) understates the true one by roughly an order of magnitude; the
	effective sample size is n / tau.
	"""
	n = len(values)
	if n < 2:
		return 1.0
	centered = values - values.mean()
	denominator = float(np.dot(centered, centered))
	if denominator == 0.0:
		return 1.0
	autocorrelation = np.correlate(centered, centered, mode="full")[n - 1 :] / denominator
	tau = 1.0
	lag = 1
	while lag + 1 < n:
		pair_sum = autocorrelation[lag] + autocorrelation[lag + 1]
		if pair_sum <= 0.0:
			break
		tau += 2.0 * pair_sum
		lag += 2
	return tau


def _window_mean_pairwise_correlation(block: np.ndarray) -> float | None:
	"""Mean pairwise correlation of a window x ticker block, None when not computable.

	A ticker whose returns are missing anywhere in the window sits the window out.
	"""
	block = block[:, ~np.isnan(block).any(axis=0)]
	ticker_count = block.shape[1]
	if ticker_count < 2:
		return None

	standard_deviation = block.std(axis=0, ddof=1)
	if (standard_deviation > 0).all():
		# With every column standardized over the window, the correlation matrix is
		# Z'Z / (W - 1); summing its ordered pairs reduces to sum_t (row sum of Z)^2,
		# and removing the unit diagonal leaves the pairwise mean in closed form.
		standardised = (block - block.mean(axis=0)) / standard_deviation
		row_sums = standardised.sum(axis=1)
		cross_product_sum = float((row_sums**2).sum())
		return (cross_product_sum / (block.shape[0] - 1) - ticker_count) / (
			ticker_count * (ticker_count - 1)
		)

	# A ticker with zero variance over the window has an undefined correlation, so drop
	# the matrix down to the tickers that moved and average the remaining pairs.
	correlation_block = block[:, standard_deviation > 0]
	if correlation_block.shape[1] < 2:
		return None
	correlation = np.corrcoef(correlation_block, rowvar=False)
	upper_triangle = correlation[np.triu_indices_from(correlation, k=1)]
	return float(upper_triangle.mean())


def _compute_regime_correlation_stats(
	regime_data: pd.DataFrame,
	window: int = REGIME_CORRELATION_WINDOW,
) -> tuple[float, float, float, int]:
	"""Pooled cross-ticker correlation for one regime.

	The regime is pivoted into a genuine date x ticker matrix; each trailing `window`-day
	window contributes its mean pairwise cross-ticker correlation; window means are then
	pooled in Fisher-z space with equal weight per date (mean-of-date-means) and
	back-transformed. The effective standard error divides by the square root of the
	Geyer effective sample size instead of the raw window count.
	"""
	pivot = regime_data.pivot_table(
		index="date", columns="ticker", values="simple_return"
	).sort_index()
	values = pivot.to_numpy(dtype="float64")

	window_means = []
	for end in range(window, len(values) + 1):
		window_mean = _window_mean_pairwise_correlation(values[end - window : end])
		if window_mean is not None:
			window_means.append(window_mean)

	if not window_means:
		return 0.0, 0.0, 0.0, 0

	pair_means = np.asarray(window_means, dtype="float64")
	fisher_values = np.arctanh(np.clip(pair_means, -1.0 + STD_EPSILON, 1.0 - STD_EPSILON))
	pooled_mean = float(np.tanh(fisher_values.mean()))
	if len(fisher_values) < 2:
		return pooled_mean, 0.0, 0.0, len(pair_means)

	tau = _geyer_integrated_autocorrelation_time(fisher_values)
	effective_se = float(fisher_values.std(ddof=1) / np.sqrt(len(fisher_values) / tau))
	return pooled_mean, float(pair_means.std(ddof=1)), effective_se, len(pair_means)


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
				"return_excess_kurtosis": float,
				"volume_mean": float,
				"volume_std": float,
				"ticker_correlation_mean": float,
				"ticker_correlation_std": float,
				"ticker_correlation_effective_se": float,
				"ticker_correlation_windows": int,
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
		volatility_values = regime_data['realized_volatility_5d'].dropna()

		# Return statistics
		returns_values = regime_data['simple_return'].dropna()

		# Volume statistics
		volume_values = regime_data['volume'].dropna()

		# Ticker correlation (per regime)
		(
			correlation_mean,
			correlation_std,
			correlation_effective_se,
			correlation_windows,
		) = _compute_regime_correlation_stats(regime_data)
		if correlation_windows == 0:
			logger.warning(
				"Regime %s: no %d-day windows with comparable tickers; "
				"correlations reported as 0.0",
				regime_names[regime_id],
				REGIME_CORRELATION_WINDOW,
			)

		# The date span describes rows with a finite target — the window these statistics
		# can actually be used on; the target's trailing NaNs are excluded.
		target_rows = regime_data['realized_volatility_5d'].notna()

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
			'return_excess_kurtosis': float(returns_values.kurtosis()),

			# Volume characteristics
			'volume_mean': float(volume_values.mean()),
			'volume_std': float(volume_values.std()),

			# Correlation characteristics
			'ticker_correlation_mean': correlation_mean,
			'ticker_correlation_std': correlation_std,
			'ticker_correlation_effective_se': correlation_effective_se,
			'ticker_correlation_windows': correlation_windows,

			# Sample count
			'n_samples': len(regime_data),
			'unique_tickers': len(regime_data['ticker'].unique()),
			'date_range': f"{regime_data.loc[target_rows, 'date'].min()} to {regime_data.loc[target_rows, 'date'].max()}",
		}

	logger.info("Regime Characterization Complete:")
	for regime_name, regime_stats in stats.items():
		logger.info(f"  {regime_name}: vol_mean={regime_stats['volatility_mean']:.6f}, vol_std={regime_stats['volatility_std']:.6f}, n_samples={regime_stats['n_samples']}")

	return stats

