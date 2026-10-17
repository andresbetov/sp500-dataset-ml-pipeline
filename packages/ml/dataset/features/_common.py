from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# A window whose value range is at or below this threshold carries no scale for a
# meaningful z-score; such windows are treated as constant instead of dividing by
# a zero standard deviation.
ZSCORE_RANGE_EPSILON = 1e-12


ADJUSTED_OHLC_OPEN_COLUMN = "_adj_open"
ADJUSTED_OHLC_HIGH_COLUMN = "_adj_high"
ADJUSTED_OHLC_LOW_COLUMN = "_adj_low"


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
