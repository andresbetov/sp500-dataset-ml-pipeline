from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from loader import load_dataframe
from preparation import prepare_dataframe
from features import build_features_dataframe, _characterize_regimes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_FILE_DIRECTORY = PROJECT_ROOT / "data" / "processed"

SURVIVORS_ONLY_FRAME_LABEL = "current-and-recent S&P 500 constituents, survivors only"
SURVIVORSHIP_BIAS_NOTE = (
    "Delisted constituents are absent from the source, so long-horizon return statistics "
    "are biased upward by roughly 1.4-3.9 percentage points per year (CAGR)."
)
STALENESS_WARNING_DAYS = 30


def _save_parquet(df: pd.DataFrame, file_name: str) -> None:
    file_path = Path(DEFAULT_FILE_DIRECTORY / (file_name + ".parquet"))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # The pandas string dtype writes Arrow large_string by default; pin the 32-bit
    # string type so readers without large-offset support can open the artifact.
    pinned = df.assign(ticker=df["ticker"].astype(pd.ArrowDtype(pa.string())))
    pinned.to_parquet(file_path, index=False)
    ticker_type = pq.read_schema(file_path).field("ticker").type
    if ticker_type != pa.string():
        raise ValueError(
            f"Expected the ticker column to be written as string, got {ticker_type}"
        )


def _check_panel_staleness(
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


def _assert_contiguous_ticker_frames(df: pd.DataFrame) -> None:
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
        "_assert_contiguous_ticker_frames: tickers=%d, frame dates=%d, interior gaps=0",
        df["ticker"].nunique(),
        len(frame_dates),
    )


def generate_featured_dataset() -> None:
    logger.info("Loading raw dataset...")
    raw_df = load_dataframe()
    logger.info("Loaded %d rows", len(raw_df))

    logger.info("Preparing data...")
    prepared_df = prepare_dataframe(raw_df)
    logger.info("Prepared: %d rows", len(prepared_df))

    logger.info("Building features...")
    featured_df = build_features_dataframe(prepared_df)
    logger.info("Featured: %d rows", len(featured_df))

    # Characterize regimes before dropping NaN
    logger.info("Characterizing regimes...")
    regime_stats = _characterize_regimes(featured_df)

    logger.info("Dropping NaN rows...")
    pre_dropna_rows = len(featured_df)
    featured_df = featured_df.dropna()
    logger.info(
        "Dropped %d NaN rows (%d remaining)",
        pre_dropna_rows - len(featured_df),
        len(featured_df),
    )

    _assert_contiguous_ticker_frames(featured_df)
    _save_parquet(featured_df, "dataset")
    logger.info("Dataset saved to data/processed/dataset.parquet")

    _check_panel_staleness(featured_df)

    # The JSON is written after the parquet so a failure here cannot leave a stale
    # characterization describing a dataset that was never written. The reserved
    # _frame key labels the source universe and records both row counts.
    regime_stats["_frame"] = {
        "label": SURVIVORS_ONLY_FRAME_LABEL,
        "survivors_only": True,
        "bias_note": SURVIVORSHIP_BIAS_NOTE,
        "pre_dropna_rows": pre_dropna_rows,
        "shipped_rows": len(featured_df),
    }

    artifacts_dir = PROJECT_ROOT / "packages" / "ml" / "model" / "artifacts" / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    regime_stats_path = artifacts_dir / "regime_characterization.json"
    with open(regime_stats_path, "w") as f:
        json.dump(regime_stats, f, indent=2)
    logger.info("Regime stats saved to %s", regime_stats_path)


if __name__ == "__main__":
    generate_featured_dataset()
