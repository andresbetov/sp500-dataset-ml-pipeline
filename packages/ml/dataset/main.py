from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

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
    df.to_parquet(file_path, index=False)


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


def generate_featured_dataset() -> None:
    logger.info("Loading raw dataset...")
    raw_df = load_dataframe()
    logger.info(f"Loaded {len(raw_df):,} rows")

    logger.info("Preparing data...")
    prepared_df = prepare_dataframe(raw_df)
    logger.info(f"Prepared: {len(prepared_df):,} rows")

    logger.info("Building features...")
    featured_df = build_features_dataframe(prepared_df)
    logger.info(f"Featured: {len(featured_df):,} rows")

    # Characterize regimes before dropping NaN
    logger.info("Characterizing regimes...")
    regime_stats = _characterize_regimes(featured_df)

    logger.info("Dropping NaN rows...")
    pre_dropna_rows = len(featured_df)
    featured_df = featured_df.dropna()
    logger.info(f"Dropped {pre_dropna_rows - len(featured_df):,} NaN rows ({len(featured_df):,} remaining)")

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
    logger.info(f"Regime stats saved to {regime_stats_path}")


if __name__ == "__main__":
    generate_featured_dataset()
