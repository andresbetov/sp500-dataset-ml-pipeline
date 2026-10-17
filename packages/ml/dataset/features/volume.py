from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._common import _compute_stable_rolling_zscore

logger = logging.getLogger(__name__)


def _add_volume_lags(df: pd.DataFrame) -> pd.DataFrame:
    lagged = df
    for lag in (1, 3, 5):
        lagged[f"volume_lag_{lag}"] = (
            lagged.groupby("ticker", sort=False, group_keys=False)["volume"].shift(lag).astype("float64")
        )
    return lagged


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
    return _add_volume_zscore_feature(df)
