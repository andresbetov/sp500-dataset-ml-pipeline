"""
Data preparation pipeline for model inference with yfinance.

Downloads historical OHLCV data for a ticker, normalizes it to match
the expected format of the feature engineering pipeline, and computes
the 29 technical indicators required by the XGBoost model.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "ml" / "dataset"
FEATURES_DIR = SCRIPT_DIR.parent / "ml" / "model" / "features"

# Minimum days needed for the most demanding feature (price_vs_sma_50 = 51 rows)
# Add buffer for safety (rolling windows + lags)
MIN_HISTORY_DAYS = 65


def _normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance column names to match the dataset pipeline.

    Handles both formats:
      - Simple columns: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
      - MultiIndex: ('Open', 'AAPL'), ('Adj Close', 'AAPL'), etc.

    Returns:
        DataFrame with columns: open, high, low, close, adj_close, volume
    """
    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        first_ticker = result.columns.get_level_values(1).unique()
        if len(first_ticker) > 0:
            ticker_val = str(first_ticker[0])
            result.columns = [
                str(col[0]).strip().lower().replace(" ", "_")
                for col in result.columns
            ]
    else:
        result.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in result.columns
        ]

    result = result.rename(columns={
        "adj_close": "adj_close",
        "close": "close",
        "high": "high",
        "low": "low",
        "open": "open",
        "volume": "volume",
    })

    return result


def _validate_columns(df: pd.DataFrame, ticker: str) -> None:
    required = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Download for {ticker} is missing columns: "
            f"{', '.join(sorted(missing))}. "
            f"Available: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"Download for {ticker} returned empty DataFrame")


def _validate_enough_history(df: pd.DataFrame, ticker: str) -> None:
    if len(df) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Not enough history for {ticker}: "
            f"got {len(df)} trading days, need at least {MIN_HISTORY_DAYS} "
            f"(required for price_vs_sma_50 and other rolling features)"
        )


def download_ticker_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker via yfinance.

    Args:
        ticker: Stock symbol (e.g. 'AAPL', 'MSFT')
        start: Start date string (e.g. '2024-01-01'). Overrides period if set.
        end: End date string (e.g. '2026-01-01'). Defaults to today.
        period: Data period if start is not set. Default '6mo'.

    Returns:
        DataFrame with columns: ticker, date, open, high, low, close, adj_close, volume
    """
    logger.info(
        "Downloading %s data (start=%s, end=%s, period=%s)",
        ticker, start, end, period
    )

    download_kwargs = {}
    if start:
        download_kwargs["start"] = start
        if end:
            download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period

    raw = yf.download(ticker, progress=False, auto_adjust=False, **download_kwargs)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker: {ticker}")

    # Normalize columns
    df = _normalize_yfinance_columns(raw)

    # Add metadata columns
    df["ticker"] = ticker
    df["date"] = df.index.normalize()

    # Reset index for pipeline compatibility
    df = df.reset_index(drop=True)

    _validate_columns(df, ticker)
    _validate_enough_history(df, ticker)

    logger.info(
        "Downloaded %d rows for %s (%s to %s)",
        len(df), ticker,
        df["date"].min().date(), df["date"].max().date()
    )

    return df


def prepare_ticker_data(
    df: pd.DataFrame,
    auto_sort: bool = True,
) -> pd.DataFrame:
    """
    Prepare raw ticker DataFrame for feature computation.

    Mimics preparation.py steps relevant for single-ticker inference:
      1. Normalize column names
      2. Cast types (date → datetime, prices → float64)
      3. Drop invalid OHLC rows
      4. Sort by date

    Skips multi-ticker filters (min_trading_days, zero_volume_ratio).

    Args:
        df: DataFrame with ticker, date, open, high, low, close, adj_close, volume
        auto_sort: Whether to sort by date (default True)

    Returns:
        Cleaned DataFrame ready for build_features_dataframe()
    """
    prepared = df.copy()

    # Ensure correct dtypes
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise").dt.normalize()
    prepared["ticker"] = prepared["ticker"].astype("string").str.strip()

    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="raise").astype("float64")

    prepared["volume"] = pd.to_numeric(prepared["volume"], errors="raise").astype("float64")

    # Drop invalid OHLC rows (same validation as preparation.py)
    valid_range = prepared["high"] >= prepared["low"]
    valid_close = (prepared["close"] <= prepared["high"]) & (prepared["close"] >= prepared["low"])
    valid_open = (prepared["open"] <= prepared["high"]) & (prepared["open"] >= prepared["low"])
    valid_rows = valid_range & valid_close & valid_open
    invalid_count = int((~valid_rows).sum())
    if invalid_count > 0:
        logger.warning("Dropping %d rows with invalid OHLC data", invalid_count)
    prepared = prepared.loc[valid_rows]

    # Sort by date
    if auto_sort:
        prepared = prepared.sort_values("date").reset_index(drop=True)

    # Deduplicate
    n_before = len(prepared)
    prepared = prepared.drop_duplicates(subset=["ticker", "date"], keep="first").reset_index(drop=True)
    n_removed = n_before - len(prepared)
    if n_removed > 0:
        logger.warning("Removed %d duplicate (ticker, date) rows", n_removed)

    return prepared


def compute_features(
    df: pd.DataFrame,
    include_target: bool = False,
) -> pd.DataFrame:
    """
    Compute the 29 technical indicators required by the model.

    Uses the full feature engineering pipeline from the dataset module.

    Args:
        df: Cleaned DataFrame with ticker, date, OHLCV columns
        include_target: If True, includes realized_volatility_5d (default False)

    Returns:
        DataFrame with all computed features. Rows with NaN features
        (due to insufficient history) are dropped.
    """
    import sys
    sys.path.insert(0, str(DATASET_DIR))

    from features import build_features_dataframe

    featured = build_features_dataframe(df)

    if not include_target:
        featured = featured.drop(columns=["realized_volatility_5d"], errors="ignore")

    # Drop rows with NaN features (insufficient history for some indicators)
    n_before = len(featured)
    featured = featured.dropna()
    n_dropped = n_before - len(featured)
    if n_dropped > 0:
        logger.info("Dropped %d rows with NaN features (insufficient history)", n_dropped)

    return featured


def select_features(featured: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the 29 indicator columns expected by the model, in the correct order.

    Args:
        featured: DataFrame from compute_features()

    Returns:
        DataFrame with only the 29 INDICATOR columns
    """
    expected_cols = [
        "ema_12", "ema_26", "macd", "macd_hist", "macd_signal",
        "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_50",
        "return_lag_1", "return_lag_10", "return_lag_2", "return_lag_3",
        "return_lag_5", "rolling_max_10", "rolling_mean_5", "rolling_min_10",
        "rsi_14", "simple_return",
        "volume_lag_1", "volume_lag_3", "volume_lag_5",
        "volume_sma_10", "volume_sma_20", "volume_zscore",
        "zscore_price_20", "zscore_price_vs_sma_20", "zscore_volume_20",
        "log_return", "market_regime",
    ]

    missing = [c for c in expected_cols if c not in featured.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} indicator columns after feature computation: "
            f"{missing}"
        )

    return featured[expected_cols]


def prepare_for_inference(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Full pipeline: download → prepare → compute features → select indicators.

    Returns the last valid row (most recent date with complete features)
    as a single-row DataFrame ready for model.predict().

    Args:
        ticker: Stock symbol
        start: Start date (overrides period)
        end: End date
        period: Data period if start not set

    Returns:
        Single-row DataFrame with 29 indicator columns, ready for prediction
    """
    logger.info("=== Data Preparation for %s ===", ticker)

    # Step 1: Download from yfinance
    raw = download_ticker_data(ticker, start=start, end=end, period=period)

    # Step 2: Prepare (clean, cast, validate)
    prepared = prepare_ticker_data(raw)

    # Step 3: Compute all features
    featured = compute_features(prepared, include_target=False)

    # Step 4: Select the 29 indicators
    X = select_features(featured)

    # Step 5: Take the most recent row (latest prediction point)
    latest = X.iloc[[-1]].reset_index(drop=True)

    logger.info(
        "Latest features for %s: date=%s, shape=%s",
        ticker,
        featured["date"].iloc[-1].date() if "date" in featured.columns else "?",
        latest.shape,
    )

    return latest
