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


def build_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	featured = _compute_base_derived_features(df)

	logger.info(
		"build_features_dataframe: rows=%d, tickers=%d",
		len(featured),
		featured["ticker"].nunique(),
	)
	return featured
