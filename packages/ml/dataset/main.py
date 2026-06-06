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


def _save_parquet(df: pd.DataFrame, file_name: str) -> None:
    file_path = Path(DEFAULT_FILE_DIRECTORY / (file_name + ".parquet"))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)


def generate_featured_dataset(
    file_name: str | None = None,
) -> None:
    logger.info("Loading raw dataset...")
    raw_df = load_dataframe(file_name=file_name)
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

    # Save regime characterization to JSON
    artifacts_dir = PROJECT_ROOT / "packages" / "ml" / "model" / "artifacts" / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    regime_stats_path = artifacts_dir / "regime_characterization.json"
    with open(regime_stats_path, "w") as f:
        json.dump(regime_stats, f, indent=2)
    logger.info(f"Regime stats saved to {regime_stats_path}")

    logger.info("Dropping NaN rows...")
    before = len(featured_df)
    featured_df = featured_df.dropna()
    logger.info(f"Dropped {before - len(featured_df):,} NaN rows ({len(featured_df):,} remaining)")

    _save_parquet(featured_df, "dataset")
    logger.info("Dataset saved to data/processed/dataset.parquet")

generate_featured_dataset()