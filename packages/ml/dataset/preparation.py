from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj close",
    "volume",
}

def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized

def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_column_names(df)
    _validate_required_columns(normalized)
    return normalized

