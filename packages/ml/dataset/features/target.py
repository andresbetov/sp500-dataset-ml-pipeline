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


TARGET_WINDOW_DAYS = 5
# Yang-Zhang's open-to-close weight for a window of TARGET_WINDOW_DAYS days.
YANG_ZHANG_OPEN_TO_CLOSE_WEIGHT = 0.34 / (
	1.34 + (TARGET_WINDOW_DAYS + 1) / (TARGET_WINDOW_DAYS - 1)
)


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
