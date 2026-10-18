from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from packages.ml.model.utils import ARTIFACTS_DIR, get_project_paths

logger = logging.getLogger(__name__)

DATASET_PATH = get_project_paths()["dataset_path"]

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

# The pinned model-input contract: exactly the 37 INDICATORS entries below, in
# this order. Dataset columns outside the list (raw and adjusted OHLCV,
# log_price, log_return, macd_signal, market_regime, date, ticker, target) are
# recorded as extras at extraction and never fed to the model. Keep this list
# and the dataset columns in lockstep; feature-importance arrays are mapped
# onto INDICATORS by position.
EXPECTED_INDICATOR_COUNT = 37
if len(INDICATORS) != EXPECTED_INDICATOR_COUNT:
	raise ValueError(
		f"INDICATORS holds {len(INDICATORS)} entries; the pinned contract is "
		f"{EXPECTED_INDICATOR_COUNT}"
	)

# Convention: these two dataset columns are computed upstream on per-date
# cross-sections before the dataset's dropna step, so they cannot be re-derived
# from the shipped parquet. They stay in the pinned input list but are exempt
# from any model-side reconstruction check; the dataset-side fix is routed to
# the dataset pipeline, not this package. Model code must never re-derive them.
NON_RECONSTRUCTIBLE_COLUMNS = [
    "market_abs_ret_5d",
    "volume_lag_5",
]


def _stream_sha256(path: Path) -> str:
    """Streaming sha256 of a file, to pin the dataset identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indicators_order_hash() -> str:
    """Deterministic fingerprint of the pinned INDICATORS order."""
    return hashlib.sha256("\n".join(INDICATORS).encode("utf-8")).hexdigest()


def _dataset_pin(df: pd.DataFrame) -> dict:
    """Dataset identity recorded in the Phase-2 metadata."""
    extra_columns = [name for name in df.columns if name not in INDICATORS]
    return {
        "path": str(DATASET_PATH.resolve()),
        "sha256": _stream_sha256(DATASET_PATH),
        "mtime": DATASET_PATH.stat().st_mtime,
        "rows": int(df.shape[0]),
        "unique_dates": int(df["date"].nunique()),
        "extra_columns_count": len(extra_columns),
        "extra_columns": extra_columns,
        "indicators_hash": _indicators_order_hash(),
        "non_reconstructible_columns": list(NON_RECONSTRUCTIBLE_COLUMNS),
    }


def _extract_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract the pinned model inputs from a loaded dataset frame.

    Returns X (the 37 pinned indicators in contract order), Y (the realized
    volatility target), and the metadata later phases consume: the date and
    ticker columns, the feature counts, and the pinned dataset record.
    """
    # Extract metadata (original ticker and date columns for reference)
    date_column = df["date"].values
    ticker_column = df["ticker"].values

    missing_indicators = [name for name in INDICATORS if name not in df.columns]
    if missing_indicators:
        missing_list = ", ".join(missing_indicators)
        raise ValueError(f"Dataset is missing INDICATORS columns: {missing_list}")

    # Select the pinned 37 indicators (no ticker encoding)
    X = df[INDICATORS].values

    # Extract target: realized_volatility_5d (continuous volatility values)
    Y = df["realized_volatility_5d"].values

    # Create metadata dict for cross-validation
    metadata = {
        "date": date_column,
        "ticker": ticker_column,
        "n_indicators": len(INDICATORS),
        "n_ticker_features": 0,
        "dataset": _dataset_pin(df),
    }
    return X, Y, metadata


def phase_2_feature_selection() -> tuple[np.ndarray, np.ndarray, dict, None]:
    """
    Load dataset and select technical indicators only (no ticker encoding).

    Returns:
        X: Feature matrix (n_samples, n_features) = the 37 pinned indicators only
        Y: Target vector (n_samples,) with continuous volatility values
        metadata: Dict with 'date' and 'ticker' columns, n_indicators, n_ticker_features
            and the pinned dataset record
        encoder: None (no ticker encoding applied)
    """
    logger.info("Phase 2: Feature Selection (No Ticker Encoding)")

    # Load dataset (already validated and cleaned by dataset pipeline)
    logger.info(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Extract the pinned model inputs; the metadata carries the pinned dataset
    # record in addition to the date and ticker columns.
    logger.info(f"Selecting {len(INDICATORS)} technical indicators (no ticker encoding)")
    X, Y, metadata = _extract_inputs(df)
    logger.info(f"Feature matrix X: {X.shape}")
    logger.info(f"Target Y (realized volatility): shape {Y.shape}, range [{Y.min():.6f}, {Y.max():.6f}], mean {Y.mean():.6f}")

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
