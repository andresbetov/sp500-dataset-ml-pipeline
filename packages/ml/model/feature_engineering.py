from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent.parent.parent / "data" / "processed" / "dataset.parquet"

# 34 technical indicators (all engineered by dataset pipeline)
INDICATORS = [
    "atr_14",
    "ema_12",
    "ema_26",
    "high_low_range",
    "log_return",
    "log_return_std_10",
    "log_return_std_20",
    "macd",
    "macd_hist",
    "macd_signal",
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
    "rolling_std_10",
    "rolling_std_20",
    "rsi_14",
    "simple_return",
    "volume_lag_1",
    "volume_lag_3",
    "volume_lag_5",
    "volume_sma_10",
    "volume_sma_20",
    "volume_zscore",
    "zscore_price_20",
    "zscore_price_vs_sma_20",
    "zscore_volume_20",
]


def phase_2_feature_selection() -> tuple[np.ndarray, np.ndarray, dict, OneHotEncoder]:
    """
    Load dataset. Select 34 technical indicators. One-hot encode ticker.
    
    Returns:
        X: Feature matrix (n_samples, 534) = 34 indicators + 500 ticker dummies
        Y: Target vector (n_samples,) with values in {-1, 0, 1}
        metadata: Dict with 'date' and 'ticker' columns (original, pre-encoding)
        encoder: Fitted OneHotEncoder for ticker column
    """
    logger.info("Phase 2: Feature Selection & Encoding")

    # Load dataset (already validated and cleaned by dataset pipeline)
    logger.info(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Extract metadata (original ticker and date columns before encoding)
    date_column = df["date"].values
    ticker_column = df["ticker"].values

    # Select 34 technical indicators
    logger.info(f"Selecting {len(INDICATORS)} technical indicators")
    x_indicators = df[
        INDICATORS].values  # DataFrame (pandas) -> numpy.ndarray. Same structure without labels, just info.
    logger.info(f"Indicators shape: {x_indicators.shape}")

    # One-hot encode ticker column (500+ unique stocks)
    logger.info("One-hot encoding ticker column")
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        dtype=np.float32,
    )
    x_ticker = encoder.fit_transform(df[["ticker"]])
    # x_ticker: ndarray(n_rows, n_unique_tickers).Value = 1 if row's ticker matches column, else 0.
    logger.info(f"Ticker encoded shape: {x_ticker.shape}")
    # logger.info(type(x_ticker)) # <class 'numpy.ndarray'>

    # Combine indicators + ticker features
    X = np.hstack([x_indicators, x_ticker])
    # X: Combine 34 technical indicators + 500 one-hot ticker columns. Shape=(n_rows, 534)
    logger.info(f"Combined feature matrix X: {X.shape}")

    # Extract target: price_direction_5d (values: -1, 0, 1)
    Y = df["price_direction_5d"].values
    logger.info(f"Target Y: shape {Y.shape}, unique values {np.unique(Y)}")  # np.unique(Y) # [-1  0  1]

    # Create metadata dict for cross-validation
    metadata = {
        "date": date_column,
        "ticker": ticker_column,
        "n_indicators": len(INDICATORS),
        "n_ticker_features": x_ticker.shape[1],
    }

    # Create subdirectories
    (ARTIFACTS_DIR / "inputs").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "legacy").mkdir(parents=True, exist_ok=True)

    # Save encoder to legacy
    encoder_path = ARTIFACTS_DIR / "legacy" / "ticker_encoder.joblib"
    joblib.dump(encoder, encoder_path)
    logger.info(f"Saved encoder to {encoder_path}")

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

    return X, Y, metadata, encoder
