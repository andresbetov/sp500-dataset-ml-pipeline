from __future__ import annotations

import logging
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Direct execution: python packages/ml/dataset/main.py — normalize to the
    # canonical package identity so the package-relative imports below resolve
    # without an installation (PEP 366).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "packages.ml.dataset"

from .artifacts import write_dataset_parquet, write_regime_characterization
from .cleaning import prepare_dataframe
from .features import build_features_dataframe, _characterize_regimes
from .ingestion import load_dataframe
from .utils import DATASET_PATH, REGIME_CHARACTERIZATION_PATH, setup_logging
from .validation import check_contiguous_ticker_frames, check_panel_staleness

logger = logging.getLogger(__name__)


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

    check_contiguous_ticker_frames(featured_df)
    write_dataset_parquet(featured_df, DATASET_PATH)
    logger.info("Dataset saved to data/processed/dataset.parquet")

    check_panel_staleness(featured_df)

    write_regime_characterization(regime_stats, pre_dropna_rows, len(featured_df))
    logger.info("Regime stats saved to %s", REGIME_CHARACTERIZATION_PATH)


if __name__ == "__main__":
    setup_logging()
    generate_featured_dataset()
