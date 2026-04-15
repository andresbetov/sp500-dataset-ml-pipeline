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
MIN_TRADING_DAYS = 500
MAX_ZERO_VOLUME_RATIO = 0.05


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


def _sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(by=["ticker", "date"])
    unique = ordered.drop_duplicates(subset=["ticker", "date"], keep="first")
    return unique.reset_index(drop=True)


def _drop_invalid_ohlc_rows(df: pd.DataFrame) -> pd.DataFrame:
    valid_range = (df["high"] >= df["low"])
    valid_close = (df["close"] <= df["high"]) & (df["close"] >= df["low"])
    valid_open = (df["open"] <= df["high"]) & (df["open"] >= df["low"])
    valid_rows = valid_range & valid_close & valid_open
    return df.loc[valid_rows].reset_index(drop=True)


def _filter_by_min_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    trading_days = df.groupby("ticker")["date"].transform("nunique")
    return df.loc[trading_days >= MIN_TRADING_DAYS].reset_index(drop=True)


def _filter_by_zero_volume_ratio(
    df: pd.DataFrame,
    max_zero_volume_ratio: float = MAX_ZERO_VOLUME_RATIO,
) -> pd.DataFrame:
    zero_volume_ratio = df["volume"].eq(0).groupby(df["ticker"]).transform("mean")
    return df.loc[zero_volume_ratio <= max_zero_volume_ratio].reset_index(drop=True)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_column_names(df)
    _validate_required_columns(normalized)
    _cast_columns(normalized)
    cleaned = _drop_invalid_ohlc_rows(normalized)
    deduplicated = _sort_and_deduplicate(cleaned)
    filtered_by_volume = _filter_by_zero_volume_ratio(deduplicated)
    return _filter_by_min_trading_days(filtered_by_volume)

