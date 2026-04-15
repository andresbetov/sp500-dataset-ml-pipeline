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

PRICE_COLUMNS = {"open", "high", "low", "close", "adj close"}

def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized

def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

def _cast_columns(df: pd.DataFrame) -> None:
    df["ticker"] = df["ticker"].astype("string").str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()

    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise").astype("float64")

    volume_numeric = pd.to_numeric(df["volume"], errors="raise")
    if volume_numeric.notna().all() and (volume_numeric % 1 == 0).all():
        df["volume"] = volume_numeric.astype("int64")
    else:
        df["volume"] = volume_numeric.astype("float64")

def _drop_invalid_ohlc_rows(df: pd.DataFrame) -> pd.DataFrame:
    valid_range = (df["high"] >= df["low"])
    valid_close = (df["close"] <= df["high"]) & (df["close"] >= df["low"])
    valid_open = (df["open"] <= df["high"]) & (df["open"] >= df["low"])
    valid_rows = valid_range & valid_close & valid_open
    return df.loc[valid_rows].reset_index(drop=True)

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_column_names(df)
    _validate_required_columns(normalized)
    _cast_columns(normalized)
    cleaned = _drop_invalid_ohlc_rows(normalized)
    return cleaned

