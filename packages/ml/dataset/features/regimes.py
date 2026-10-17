from __future__ import annotations

import numpy as np
import pandas as pd


def _add_market_regime_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign market regime based on historical periods.

    Regimes:
    - 0: 2000-2007 (pre-crisis)
    - 1: 2008-2010 (financial crisis)
    - 2: 2011-2019 (post-crisis recovery)
    - 3: 2020-2021 (COVID)
    - 4: 2022 (post-COVID inflation)
    - 5: 2023-2025 (post-COVID recovery)
    - 6: 2026 (terminal year)

    A year past the last bucket raises instead of silently joining it, so data
    from a later source period cannot be mislabelled.

    Returns:
        DataFrame with new 'market_regime' column (int8: 0-6)
    """
    year = df['date'].dt.year
    regime = np.select(
        [
            year <= 2007,
            year <= 2010,
            year <= 2019,
            year <= 2021,
            year == 2022,
            year <= 2025,
            year == 2026,
        ],
        [0, 1, 2, 3, 4, 5, 6],
        default=-1,
    )
    uncovered = regime == -1
    if uncovered.any():
        uncovered_years = sorted(int(value) for value in year[uncovered].unique())
        raise ValueError(
            f"No market regime bucket covers year(s) {uncovered_years}; extend the "
            "calendar buckets before ingesting later data"
        )

    df['market_regime'] = regime.astype('int8')
    return df
