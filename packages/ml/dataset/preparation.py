from __future__ import annotations

import logging

import pandas as pd


REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
}

PRICE_COLUMNS = {"open", "high", "low", "close", "adj_close"}
MIN_TRADING_DAYS = 500
ZERO_VOLUME_CENSUS_RATIO = 0.05
MIN_DISTINCT_CLOSE_PRICES = 50
LOW_PRICE_CENSUS_THRESHOLD = 0.10

logger = logging.getLogger(__name__)


def _check_price_grid(df: pd.DataFrame) -> None:
    """Fail fast on a coarse close-price grid; census sub-dime closes at INFO.

    A cross-section whose closes collapse onto fewer than MIN_DISTINCT_CLOSE_PRICES
    distinct 2-decimal values cannot carry the resolution the price features assume;
    that signature means the source came back rounded or corrupt, so the frame is
    rejected. Sub-dime closes are a different matter - a back-adjusted total-return
    series legitimately prices old high-split names below a dime - so they are
    counted and logged as a census instead of rejected.
    """
    distinct_close_prices = int(df["close"].round(2).nunique())
    if distinct_close_prices < MIN_DISTINCT_CLOSE_PRICES:
        raise ValueError(
            f"Close prices collapse onto {distinct_close_prices} distinct 2-decimal "
            f"values (minimum {MIN_DISTINCT_CLOSE_PRICES}); the source looks rounded "
            "or corrupt"
        )

    below_threshold = df["close"] < LOW_PRICE_CENSUS_THRESHOLD
    below_threshold_rows = int(below_threshold.sum())
    logger.info(
        "_check_price_grid: distinct_close_prices=%d, close<%.2f rows=%d (%.4f%%), "
        "tickers=%d, min_close=%.4f",
        distinct_close_prices,
        LOW_PRICE_CENSUS_THRESHOLD,
        below_threshold_rows,
        100.0 * below_threshold_rows / len(df),
        int(df.loc[below_threshold, "ticker"].nunique()),
        float(df["close"].min()),
    )


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    colliding_columns = sorted(set(normalized.columns[normalized.columns.duplicated()]))
    if colliding_columns:
        colliding_list = ", ".join(colliding_columns)
        logger.error("_normalize_column_names: colliding columns=%s", colliding_list)
        raise ValueError(f"Column names collide after normalization: {colliding_list}")
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
    unexpected = set(df.columns) - REQUIRED_COLUMNS
    if unexpected:
        unexpected_list = ", ".join(sorted(unexpected))
        logger.error("_validate_required_columns: unexpected columns=%s", unexpected_list)
        raise ValueError(f"Unexpected columns: {unexpected_list}")
    logger.debug("_validate_required_columns: all required columns present")


def _cast_columns(df: pd.DataFrame) -> None:
    """Cast the raw columns in place; returns None by contract, so callers never rebind.

    Volume stays integer-valued while every value is a non-missing integer, and is
    kept as float64 - never coerced to zero - as soon as any value is missing or
    fractional, so volumes treated as missing survive the cast instead of silently
    counting as no-trade days.
    """
    df["ticker"] = df["ticker"].astype("string").str.strip()
    empty_tickers = int(df["ticker"].eq("").sum())
    if empty_tickers:
        raise ValueError(f"Column 'ticker' contains {empty_tickers} empty values")

    parsed_dates = pd.to_datetime(df["date"], errors="raise")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        raise ValueError(f"Column 'date' contains {invalid_dates} invalid or empty values")
    df["date"] = parsed_dates.dt.normalize()

    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise").astype("float64")

    volume_numeric = pd.to_numeric(df["volume"], errors="raise")
    missing_volume = int(volume_numeric.isna().sum())
    if missing_volume == 0 and (volume_numeric % 1 == 0).all():
        df["volume"] = volume_numeric.astype("int64")
    else:
        df["volume"] = volume_numeric.astype("float64")

    logger.debug(
        "_cast_columns: rows=%d, volume_dtype=%s, missing_volume=%d",
        len(df),
        df["volume"].dtype,
        missing_volume,
    )


def _sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    rows_in = len(df)
    # A stable sort keeps equal (ticker, date) rows in file order, so keep="first"
    # resolves a duplicate conflict deterministically instead of by quicksort's
    # internal ordering, which is not defined for ties.
    ordered = df.sort_values(by=["ticker", "date"], kind="mergesort")
    duplicate_pairs = ordered.loc[
        ordered.duplicated(subset=["ticker", "date"], keep=False), ["ticker", "date"]
    ].drop_duplicates()
    unique = ordered.drop_duplicates(subset=["ticker", "date"], keep="first")
    deduplicated = unique.reset_index(drop=True)
    removed = rows_in - len(deduplicated)
    if removed:
        sample = ", ".join(
            f"({row.ticker}, {row.date.date()})"
            for row in duplicate_pairs.head(5).itertuples(index=False)
        )
        logger.info(
            "_sort_and_deduplicate: rows_in=%d, rows_out=%d, removed=%d, "
            "conflicting pairs=%d; sample=[%s]",
            rows_in,
            len(deduplicated),
            removed,
            len(duplicate_pairs),
            sample,
        )
    else:
        logger.debug(
            "_sort_and_deduplicate: rows_in=%d, rows_out=%d, removed=0",
            rows_in,
            len(deduplicated),
        )
    return deduplicated


def _repair_invalid_ohlc_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Repair single-cell open slips in place, then drop rows that stay invalid.

    An open price outside [low, high] is a vendor rounding slip that can be repaired
    by clipping it back into its own bar; the close and the range are trusted. A row
    whose high/low range or close is invalid cannot be repaired without inventing
    prices, so it is dropped with an INFO reason line - and the frame contiguity
    assertion before the write catches any interior hole such a drop would create.
    """
    rows_in = len(df)
    valid_range = df["high"] >= df["low"]
    valid_close = (df["close"] <= df["high"]) & (df["close"] >= df["low"])
    open_outside = (df["open"] < df["low"]) | (df["open"] > df["high"])
    repairable = open_outside & valid_range & valid_close
    repaired_count = int(repairable.sum())
    if repaired_count:
        repaired = df.loc[repairable, ["ticker", "date", "open", "low", "high"]].copy()
        repaired["repaired_open"] = df.loc[repairable, "open"].clip(
            lower=df.loc[repairable, "low"], upper=df.loc[repairable, "high"]
        )
        df.loc[repairable, "open"] = repaired["repaired_open"]
        for row in repaired.itertuples(index=False):
            logger.info(
                "_repair_invalid_ohlc_rows: repaired %s %s open %.4f -> %.4f",
                row.ticker,
                row.date.date(),
                row.open,
                row.repaired_open,
            )

    valid_open = (df["open"] <= df["high"]) & (df["open"] >= df["low"])
    valid_rows = valid_range & valid_close & valid_open
    for row in df.loc[~valid_rows, ["ticker", "date", "open", "high", "low", "close"]].itertuples(
        index=False
    ):
        logger.info(
            "_repair_invalid_ohlc_rows: dropping invalid bar %s %s "
            "(open=%.4f high=%.4f low=%.4f close=%.4f)",
            row.ticker,
            row.date.date(),
            row.open,
            row.high,
            row.low,
            row.close,
        )
    filtered = df.loc[valid_rows].reset_index(drop=True)
    logger.info(
        "_repair_invalid_ohlc_rows: rows_in=%d, repaired=%d, rows_out=%d, dropped=%d",
        rows_in,
        repaired_count,
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
    logger.info(
        "_filter_by_min_trading_days: min_days=%d, rows_in=%d, rows_out=%d, tickers_in=%d, tickers_out=%d",
        min_trading_days,
        rows_in,
        len(filtered),
        tickers_in,
        filtered["ticker"].nunique(),
    )
    return filtered


def _treat_zero_volume_price_moves(
    df: pd.DataFrame,
    census_ratio: float = ZERO_VOLUME_CENSUS_RATIO,
) -> pd.DataFrame:
    """Treat zero-volume observations that moved as missing, per row, no row drops.

    A zero-volume row whose bar is flat and whose close equals the previous close is
    consistent with a genuine no-trade day. One whose bar moved or whose close moved
    against the previous close is not: the volume feed dropped out while prices kept
    moving. Those rows keep their prices, but their volume is treated as missing and
    carried forward from the ticker's last positive observed volume (backward-filled
    from the next positive volume when the ticker starts inside a missing run), so no
    row is dropped, no interior date gap can appear, and a suspect zero never gets
    propagated. Logs the treated and surviving-zero counts at INFO, plus a per-ticker
    profile for every ticker above the census ratio.
    """
    rows_in = len(df)
    previous_close = df.groupby("ticker", sort=False, group_keys=False)["close"].shift(1)
    zero_volume = df["volume"].eq(0)
    flat_bar = df["open"].eq(df["high"]) & df["high"].eq(df["low"]) & df["low"].eq(df["close"])
    price_moved = previous_close.notna() & df["close"].ne(previous_close)
    treated = zero_volume & (price_moved | ~flat_bar)

    volumes = df["volume"].where(~treated)
    observed_positive_volume = df["volume"].where(~treated & df["volume"].gt(0))
    filled_volume = observed_positive_volume.groupby(
        df["ticker"], sort=False, group_keys=False
    ).ffill()
    filled_volume = filled_volume.groupby(
        df["ticker"], sort=False, group_keys=False
    ).bfill()
    volumes = volumes.where(~treated, filled_volume)
    if volumes.isna().any():
        raise ValueError("Zero-volume treatment left missing volume values")
    df["volume"] = volumes

    logger.info(
        "_treat_zero_volume_price_moves: rows_in=%d, zero_volume_rows=%d, treated_rows=%d, "
        "surviving_zero_rows=%d, tickers_touched=%d",
        rows_in,
        int(zero_volume.sum()),
        int(treated.sum()),
        int((zero_volume & ~treated).sum()),
        int(df.loc[treated, "ticker"].nunique()),
    )

    profile = pd.DataFrame(
        {
            "zeros": zero_volume.groupby(df["ticker"]).sum(),
            "treated": treated.groupby(df["ticker"]).sum(),
            "rows": df.groupby("ticker").size(),
        }
    )
    profile["ratio"] = profile["zeros"] / profile["rows"]
    flagged = profile.loc[profile["ratio"] > census_ratio].sort_values("ratio", ascending=False)
    for ticker, row in flagged.iterrows():
        logger.info(
            "  zero-volume profile %s: zeros=%d/%d (ratio=%.4f), treated=%d, surviving=%d",
            ticker,
            int(row["zeros"]),
            int(row["rows"]),
            row["ratio"],
            int(row["treated"]),
            int(row["zeros"] - row["treated"]),
        )
    return df


def _check_survivor_cross_section(df: pd.DataFrame, survivors_only: bool) -> None:
    """Fail when a survivors-only frame's ticker cross-section shrinks over time.

    Every name in a survivors-only universe is still listed at the end of the panel, so
    the number of distinct tickers per calendar year can only grow. A decline means the
    frame contains exits, so treating it as survivors-only would silently misuse it.
    """
    if not survivors_only:
        logger.warning(
            "_check_survivor_cross_section: skipped for a membership-aware frame "
            "(survivors_only=False)"
        )
        return

    yearly_counts = df.groupby(df["date"].dt.year)["ticker"].nunique()
    if yearly_counts.empty:
        raise ValueError("Cannot check the survivors-only cross-section on a frame with no dated rows")
    if yearly_counts.is_monotonic_increasing:
        logger.info(
            "_check_survivor_cross_section: non-decreasing (%d in %d -> %d in %d)",
            yearly_counts.iloc[0],
            yearly_counts.index[0],
            yearly_counts.iloc[-1],
            yearly_counts.index[-1],
        )
        return

    summary = ", ".join(f"{year}: {count}" for year, count in yearly_counts.items())
    raise ValueError(
        "Frame labelled survivors-only has a declining per-year ticker cross-section "
        f"(unique tickers per year must be non-decreasing): {summary}"
    )


def prepare_dataframe(df: pd.DataFrame, survivors_only: bool = True) -> pd.DataFrame:
    logger.info(
        "prepare_dataframe: start rows=%d, tickers=%d",
        len(df),
        df["ticker"].nunique() if "ticker" in df.columns else 0,
    )
    normalized = _normalize_column_names(df)
    _validate_required_columns(normalized)
    _cast_columns(normalized)
    _check_price_grid(normalized)
    _check_survivor_cross_section(normalized, survivors_only)
    cleaned = _repair_invalid_ohlc_rows(normalized)
    deduplicated = _sort_and_deduplicate(cleaned)
    treated = _treat_zero_volume_price_moves(deduplicated)
    prepared = _filter_by_min_trading_days(treated)
    logger.info(
        "prepare_dataframe: end rows=%d, tickers=%d",
        len(prepared),
        prepared["ticker"].nunique(),
    )
    return prepared
