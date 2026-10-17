from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
STD_EPSILON = 1e-12
REGIME_CORRELATION_WINDOW = 60

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


def characterize_regimes(df: pd.DataFrame) -> dict:
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
		4: "post-covid (2022)",
		5: "post-covid (2023-2025)",
		6: "post-covid (2026)",
	}

	stats = {}

	for regime_id in range(7):
		regime_data = df[df['market_regime'] == regime_id]

		if len(regime_data) == 0:
			logger.warning("Regime %d: no data found", regime_id)
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
		logger.info(
			"  %s: vol_mean=%.6f, vol_std=%.6f, n_samples=%d",
			regime_name,
			regime_stats['volatility_mean'],
			regime_stats['volatility_std'],
			regime_stats['n_samples'],
		)

	return stats
