from __future__ import annotations

import logging

import pandas as pd


REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "adj_close",
    "volume",
}

PRICE_COLUMNS = {"open", "high", "low", "adj_close"}
MIN_TRADING_DAYS = 500
MAX_ZERO_VOLUME_RATIO = 0.05

logger = logging.getLogger(__name__)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    logger.debug(
        "_normalize_column_names: rows=%d, columns=%d",
        len(normalized),
        len(normalized.columns),
    )
    return normalized


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        logger.error("_validate_required_columns: missing columns=%s", missing_list)
        raise ValueError(f"Missing required columns: {missing_list}")
    logger.debug("_validate_required_columns: all required columns present")


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

    logger.debug(
        "_cast_columns: rows=%d, volume_dtype=%s",
        len(df),
        df["volume"].dtype,
    )


def _sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    rows_in = len(df)
    ordered = df.sort_values(by=["ticker", "date"])
    unique = ordered.drop_duplicates(subset=["ticker", "date"], keep="first")
    deduplicated = unique.reset_index(drop=True)
    logger.debug(
        "_sort_and_deduplicate: rows_in=%d, rows_out=%d, removed=%d",
        rows_in,
        len(deduplicated),
        rows_in - len(deduplicated),
    )
    return deduplicated


def _drop_invalid_ohlc_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows_in = len(df)
    valid_range = (df["high"] >= df["low"])
    valid_close = (df["adj_close"] <= df["high"]) & (df["adj_close"] >= df["low"])
    valid_open = (df["open"] <= df["high"]) & (df["open"] >= df["low"])
    valid_rows = valid_range & valid_close & valid_open
    filtered = df.loc[valid_rows].reset_index(drop=True)
    logger.debug(
        "_drop_invalid_ohlc_rows: rows_in=%d, rows_out=%d, removed=%d",
        rows_in,
        len(filtered),
        rows_in - len(filtered),
    )
    return filtered


def _filter_by_min_trading_days(
    df: pd.DataFrame,
    min_trading_days: int = MIN_TRADING_DAYS,
) -> pd.DataFrame:
    rows_in = len(df)
    tickers_in = df["ticker"].nunique()
    trading_days = df.groupby("ticker")["date"].transform("nunique")
    filtered = df.loc[trading_days >= min_trading_days].reset_index(drop=True)
    logger.debug(
        "_filter_by_min_trading_days: min_days=%d, rows_in=%d, rows_out=%d, tickers_in=%d, tickers_out=%d",
        min_trading_days,
        rows_in,
        len(filtered),
        tickers_in,
        filtered["ticker"].nunique(),
    )
    return filtered


def _filter_by_zero_volume_ratio(
    df: pd.DataFrame,
    max_zero_volume_ratio: float = MAX_ZERO_VOLUME_RATIO,
) -> pd.DataFrame:
    rows_in = len(df)
    tickers_in = df["ticker"].nunique()
    zero_volume_ratio = df["volume"].eq(0).groupby(df["ticker"]).transform("mean")
    filtered = df.loc[zero_volume_ratio <= max_zero_volume_ratio].reset_index(drop=True)
    logger.debug(
        "_filter_by_zero_volume_ratio: threshold=%.4f, rows_in=%d, rows_out=%d, tickers_in=%d, tickers_out=%d",
        max_zero_volume_ratio,
        rows_in,
        len(filtered),
        tickers_in,
        filtered["ticker"].nunique(),
    )
    return filtered


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "prepare_dataframe: start rows=%d, tickers=%d",
        len(df),
        df["ticker"].nunique() if "ticker" in df.columns else 0,
    )
    normalized = _normalize_column_names(df)
    _validate_required_columns(normalized)
    _cast_columns(normalized)
    cleaned = _drop_invalid_ohlc_rows(normalized)
    deduplicated = _sort_and_deduplicate(cleaned)
    filtered_by_volume = _filter_by_zero_volume_ratio(deduplicated)
    prepared = _filter_by_min_trading_days(filtered_by_volume)
    logger.info(
        "prepare_dataframe: end rows=%d, tickers=%d",
        len(prepared),
        prepared["ticker"].nunique(),
    )
    return prepared

