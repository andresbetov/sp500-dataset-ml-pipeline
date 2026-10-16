from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent.parent.parent.parent / "data" / "processed" / "dataset.parquet"

INDICATORS = [
	"atr20",
	"atr5",
	"ema_12",
	"ema_26",
	"gap5",
	"gk5",
	"har22",
	"har5",
	"har63",
	"macd",
	"macd_hist",
	"market_abs_ret_5d",
	"park20",
	"park5",
	"price_vs_sma_10",
	"price_vs_sma_20",
	"price_vs_sma_50",
	"return_lag_1",
	"return_lag_10",
	"return_lag_2",
	"return_lag_3",
	"return_lag_5",
	"rolling_max_10",
	"rolling_mean_5",
	"rolling_min_10",
	"rsi_14",
	"rv22",
	"rv5_past",
	"rv63",
	"rs5",
	"simple_return",
	"volvol22",
	"volume_lag_1",
	"volume_lag_3",
	"volume_lag_5",
	"zscore_price_vs_sma_20",
	"zscore_volume_20",
]

# The pinned model-input contract: 15 volatility inputs in, 7 redundant ones out
# (the two alias columns, both volume SMAs, and the macd_signal/log_return/
# market_regime list-only removals). Keep this list and the dataset columns in
# lockstep; feature-importance arrays are mapped onto INDICATORS by position.
EXPECTED_INDICATOR_COUNT = 37
if len(INDICATORS) != EXPECTED_INDICATOR_COUNT:
	raise ValueError(
		f"INDICATORS holds {len(INDICATORS)} entries; the pinned contract is "
		f"{EXPECTED_INDICATOR_COUNT}"
	)


def phase_2_feature_selection() -> tuple[np.ndarray, np.ndarray, dict, None]:
    """
    Load dataset and select technical indicators only (no ticker encoding).

    Returns:
        X: Feature matrix (n_samples, n_features) = the 37 pinned indicators only
        Y: Target vector (n_samples,) with continuous volatility values
        metadata: Dict with 'date' and 'ticker' columns, n_indicators, n_ticker_features
        encoder: None (no ticker encoding applied)
    """
    logger.info("Phase 2: Feature Selection (No Ticker Encoding)")

    # Load dataset (already validated and cleaned by dataset pipeline)
    logger.info(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Extract metadata (original ticker and date columns for reference)
    date_column = df["date"].values
    ticker_column = df["ticker"].values

    missing_indicators = [name for name in INDICATORS if name not in df.columns]
    if missing_indicators:
        missing_list = ", ".join(missing_indicators)
        raise ValueError(f"Dataset is missing INDICATORS columns: {missing_list}")

    # Select the pinned 37 indicators (no ticker encoding)
    logger.info(f"Selecting {len(INDICATORS)} technical indicators (no ticker encoding)")
    X = df[INDICATORS].values
    logger.info(f"Feature matrix X: {X.shape}")

    # Extract target: realized_volatility_5d (continuous volatility values)
    Y = df["realized_volatility_5d"].values
    logger.info(f"Target Y (realized volatility): shape {Y.shape}, range [{Y.min():.6f}, {Y.max():.6f}], mean {Y.mean():.6f}")

    # Create metadata dict for cross-validation
    metadata = {
        "date": date_column,
        "ticker": ticker_column,
        "n_indicators": len(INDICATORS),
        "n_ticker_features": 0,
    }

    # Create subdirectories
    (ARTIFACTS_DIR / "inputs").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "legacy").mkdir(parents=True, exist_ok=True)

    # Save X, Y, metadata to inputs
    X_path = ARTIFACTS_DIR / "inputs" / "X.npy"
    y_path = ARTIFACTS_DIR / "inputs" / "Y.npy"
    metadata_path = ARTIFACTS_DIR / "inputs" / "metadata.pkl"

    np.save(X_path, X)
    np.save(y_path, Y)
    joblib.dump(metadata, metadata_path)
    logger.info(f"Saved X to {X_path}")
    logger.info(f"Saved Y to {y_path}")
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info(f"✓ Phase 2 complete: X {X.shape}, Y {Y.shape}")

    return X, Y, metadata, None
