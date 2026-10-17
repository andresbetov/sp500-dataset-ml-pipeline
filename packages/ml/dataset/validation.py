"""Dataset validation checks: ticker-frame contiguity and panel staleness."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STALENESS_WARNING_DAYS = 30


def check_panel_staleness(
    df: pd.DataFrame,
    max_staleness_days: int = STALENESS_WARNING_DAYS,
) -> None:
    """Warn when the shipped panel's last date lags the run date beyond the threshold."""
    run_date = pd.Timestamp.today().normalize()
    last_date = df["date"].max()
    staleness_days = int((run_date - last_date).days)
    if staleness_days > max_staleness_days:
        logger.warning(
            "Panel is stale: last date %s is %d days before the run date %s "
            "(threshold %d days)",
            last_date.date(),
            staleness_days,
            run_date.date(),
            max_staleness_days,
        )


def check_contiguous_ticker_frames(df: pd.DataFrame) -> None:
    """Fail loudly unless every ticker's rows cover the panel dates inside its own span.

    Each ticker may begin after the panel's first date or end before its last date -
    those start/end truncations are documented - but between its own first and last
    date it must hold every date that appears anywhere in the frame. An interior hole
    means rows were silently dropped, and downstream row/date arithmetic (including
    the 5-day forward target) would mis-measure the horizon across it.
    """
    frame_dates = pd.Index(np.sort(pd.unique(df["date"])))
    if len(frame_dates) == 0:
        raise ValueError("Cannot check ticker contiguity on a frame with no dates")

    rows = df[["ticker", "date"]].copy()
    rows["_frame_position"] = frame_dates.get_indexer(rows["date"])
    for ticker, positions in rows.groupby("ticker", sort=False)["_frame_position"]:
        ordered = np.sort(positions.to_numpy())
        expected = np.arange(ordered[0], ordered[-1] + 1)
        if np.array_equal(ordered, expected):
            continue
        missing = np.setdiff1d(expected, ordered)
        missing_dates = ", ".join(str(frame_dates[index].date()) for index in missing[:5])
        raise ValueError(
            f"Ticker {ticker} has {len(missing)} interior date gap(s) between "
            f"{frame_dates[ordered[0]].date()} and {frame_dates[ordered[-1]].date()}: "
            f"{missing_dates}"
        )
    logger.info(
        "check_contiguous_ticker_frames: tickers=%d, frame dates=%d, interior gaps=0",
        df["ticker"].nunique(),
        len(frame_dates),
    )
