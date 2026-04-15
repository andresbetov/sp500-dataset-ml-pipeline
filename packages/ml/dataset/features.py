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


def build_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	featured = _compute_base_derived_features(df)
	featured = _add_lag_features(featured)

	logger.info(
		"build_features_dataframe: rows=%d, tickers=%d",
		len(featured),
		featured["ticker"].nunique(),
	)
	return featured
