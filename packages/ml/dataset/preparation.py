from __future__ import annotations

import pandas as pd


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_column_names(df)

