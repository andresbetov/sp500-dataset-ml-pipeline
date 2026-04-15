from __future__ import annotations

import pandas as pd

from loader import load_dataframe


def get_dataframe() -> pd.DataFrame:
    return load_dataframe()

print(get_dataframe().head())
