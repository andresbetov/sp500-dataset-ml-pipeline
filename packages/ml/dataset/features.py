from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)


def _compute_base_derived_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()

	featured["log_price"] = featured["close"].map(math.log).astype("float64")
	featured["simple_return"] = (
		featured.groupby("ticker", sort=False)["close"].pct_change().astype("float64")
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
	by_ticker_close = featured.groupby("ticker", sort=False)["close"]

	for window in (10, 20, 50):
		featured[f"sma_{window}"] = (
			by_ticker_close.transform(
				lambda s: s.rolling(window=window, min_periods=window).mean()
			).astype("float64")
		)

	return featured


def _add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_close = featured.groupby("ticker", sort=False)["close"]

	for span in (12, 26):
		featured[f"ema_{span}"] = (
			by_ticker_close.transform(lambda s: s.ewm(span=span, adjust=False).mean()).astype("float64")
		)

	return featured


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
	with_sma = _add_sma_features(df)
	return _add_ema_features(with_sma)


def _compute_rsi_14(close: pd.Series) -> pd.Series:
	delta = close.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)

	avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
	avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

	rs = avg_gain / avg_loss
	return 100 - (100 / (1 + rs))


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	by_ticker_close = featured.groupby("ticker", sort=False)["close"]

	featured["rsi_14"] = by_ticker_close.transform(_compute_rsi_14).astype("float64")

	ema_fast = by_ticker_close.transform(lambda s: s.ewm(span=12, adjust=False).mean())
	ema_slow = by_ticker_close.transform(lambda s: s.ewm(span=26, adjust=False).mean())
	featured["macd"] = (ema_fast - ema_slow).astype("float64")

	featured["macd_signal"] = (
		featured.groupby("ticker", sort=False)["macd"]
		.transform(lambda s: s.ewm(span=9, adjust=False).mean())
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
				lambda s: s.rolling(window=window, min_periods=window).std()
			).astype("float64")
		)

	return featured


def _compute_atr_14_by_ticker(df_ticker: pd.DataFrame) -> pd.Series:
	high = df_ticker["high"]
	low = df_ticker["low"]
	prev_close = df_ticker["close"].shift(1)

	true_range = pd.concat(
		[(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
		axis=1,
	).max(axis=1)

	return true_range.rolling(window=14, min_periods=14).mean().astype("float64")


def _add_atr_feature(df: pd.DataFrame) -> pd.DataFrame:
	featured = df.copy()
	featured["atr_14"] = (
		featured.groupby("ticker", sort=False, group_keys=False)
		.apply(_compute_atr_14_by_ticker)
		.astype("float64")
	)
	return featured


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
	with_rolling_std = _add_rolling_std_features(df)
	return _add_atr_feature(with_rolling_std)


def build_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	featured = _compute_base_derived_features(df)
	featured = _add_lag_features(featured)
	featured = _add_trend_features(featured)
	featured = _add_momentum_features(featured)
	featured = _add_volatility_features(featured)

	logger.info(
		"build_features_dataframe: rows=%d, tickers=%d",
		len(featured),
		featured["ticker"].nunique(),
	)
	return featured
