from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"


def phase_3_cross_validation() -> dict:
    """
    Setup 5-fold TimeSeriesSplit with expanding windows. Verify no temporal leakage.
    
    Returns:
        folds: Dict with 5 folds containing train/test indices and date bounds
    """
    logger.info("Phase 3: Temporal Cross-Validation Setup")

    # Load Phase 2 outputs
    logger.info("Loading Phase 2 outputs...")
    X = joblib.load(MODELS_DIR / "X.pkl")
    y = joblib.load(MODELS_DIR / "Y.pkl")
    metadata = joblib.load(MODELS_DIR / "metadata.pkl")

    logger.info(f"Loaded X: {X.shape}, y: {y.shape}")

    # Extract dates from metadata
    dates = metadata["date"]
    logger.info(f"Date range: {dates.min()} to {dates.max()}")

    # Setup TimeSeriesSplit with 5 folds (expanding windows)
    tscv = TimeSeriesSplit(n_splits=5)
    # TimeSeriesSplit creates 5 folds where:
    #   - Train always starts from first row (index 0)
    #   - Train grows each fold (expanding window)
    #   - Test is always after train (no temporal leakage)
    # Example with 2.67M rows:
    #   Fold 0: Train [0:2.14M] | Test [2.14M:2.67M]
    #   Fold 1: Train [0:2.27M] | Test [2.27M:2.67M]
    #   Fold 2: Train [0:2.40M] | Test [2.40M:2.67M]
    #   Fold 3: Train [0:2.54M] | Test [2.54M:2.67M]
    #   Fold 4: Train [0:2.67M] | Test [] (empty, last split)
    # tscv.split(X) yields (train_indices, test_indices) tuples

    logger.info(f"TimeSeriesSplit initialized with n_splits=5")

    folds = {}

    # enumerate(tscv.split(X)): iterate 5 tuples. fold_idx=0,1,2,3,4
    # Each tuple: (train_indices_array, test_indices_array) -> unpacked two arrays with row indices
    for fold_idx, (train_indices, test_indices) in enumerate(tscv.split(X)):
        # Get train/test dates
        train_dates = dates[train_indices]
        test_dates = dates[test_indices]

        # Get date bounds
        train_date_min = pd.Timestamp(train_dates.min()).strftime("%Y-%m-%d")
        train_date_max = pd.Timestamp(train_dates.max()).strftime("%Y-%m-%d")
        test_date_min = pd.Timestamp(test_dates.min()).strftime("%Y-%m-%d") if len(test_dates) > 0 else "N/A"
        test_date_max = pd.Timestamp(test_dates.max()).strftime("%Y-%m-%d") if len(test_dates) > 0 else "N/A"

        # Verify no temporal leakage
        if len(test_dates) > 0:
            if test_dates.min() <= train_dates.max():
                msg = f"Fold {fold_idx}: Temporal leakage detected! Test min ({test_dates.min()}) <= Train max ({train_dates.max()})"
                logger.error(msg)
                raise ValueError(msg)

        fold_key = f"fold_{fold_idx}"
        folds[fold_key] = {
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "train_date_min": train_date_min,
            "train_date_max": train_date_max,
            "test_date_min": test_date_min,
            "test_date_max": test_date_max,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
        }

        logger.info(
            f"Fold {fold_idx}: Train {len(train_indices):,} | Test {len(test_indices):,} | "
            f"Dates: {train_date_min} to {test_date_max}"
        )

    # Save fold metadata to JSON
    folds_metadata_path = MODELS_DIR / "folds_metadata.json"

    # Convert to JSON-serializable format
    folds_json = {
        fold_key: {
            "train_indices": fold_data["train_indices"],
            "test_indices": fold_data["test_indices"],
            "train_date_min": fold_data["train_date_min"],
            "train_date_max": fold_data["train_date_max"],
            "test_date_min": fold_data["test_date_min"],
            "test_date_max": fold_data["test_date_max"],
            "n_train": fold_data["n_train"],
            "n_test": fold_data["n_test"],
        }
        for fold_key, fold_data in folds.items()
    }

    with open(folds_metadata_path, "w") as f:
        json.dump(folds_json, f, indent=2)

    logger.info(f"Saved fold metadata to {folds_metadata_path}")
    logger.info("✓ Phase 3 complete: Temporal cross-validation setup with no leakage")

    return folds
