"""Dataset artifact writers: parquet snapshot and regime characterization JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .utils import REGIME_CHARACTERIZATION_PATH

SURVIVORS_ONLY_FRAME_LABEL = "current-and-recent S&P 500 constituents, survivors only"
SURVIVORSHIP_BIAS_NOTE = (
    "Delisted constituents are absent from the source, so long-horizon return statistics "
    "are biased upward by roughly 1.4-3.9 percentage points per year (CAGR)."
)


def write_dataset_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The pandas string dtype writes Arrow large_string by default; pin the 32-bit
    # string type so readers without large-offset support can open the artifact.
    pinned = df.assign(ticker=df["ticker"].astype(pd.ArrowDtype(pa.string())))
    pinned.to_parquet(path, index=False)
    ticker_type = pq.read_schema(path).field("ticker").type
    if ticker_type != pa.string():
        raise ValueError(
            f"Expected the ticker column to be written as string, got {ticker_type}"
        )


def write_regime_characterization(
    regime_stats: dict,
    pre_dropna_rows: int,
    shipped_rows: int,
) -> None:
    # The JSON is written after the parquet so a failure here cannot leave a stale
    # characterization describing a dataset that was never written. The reserved
    # _frame key labels the source universe and records both row counts.
    regime_stats["_frame"] = {
        "label": SURVIVORS_ONLY_FRAME_LABEL,
        "survivors_only": True,
        "bias_note": SURVIVORSHIP_BIAS_NOTE,
        "pre_dropna_rows": pre_dropna_rows,
        "shipped_rows": shipped_rows,
    }
    REGIME_CHARACTERIZATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGIME_CHARACTERIZATION_PATH, "w") as f:
        json.dump(regime_stats, f, indent=2)
