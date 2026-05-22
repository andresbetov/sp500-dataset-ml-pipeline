from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)
CV_GAP_DAYS = 5


def phase_3_cross_validation() -> dict:
    """
    Setup 5-fold TimeSeriesSplit with expanding windows. Verify no temporal leakage.
    
    Data is organized by ticker (not globally by date), so we must sort by date first
    before creating temporal folds.
    
    Returns:
        folds: Dict with 5 folds containing train/test indices and date bounds
    """
    logger.info("Phase 3: Temporal Cross-Validation Setup")
    
    # Load Phase 2 outputs
    logger.info("Loading Phase 2 outputs...")
    X = np.load(ARTIFACTS_DIR / "inputs" / "X.npy", mmap_mode="r")
    y = np.load(ARTIFACTS_DIR / "inputs" / "Y.npy", mmap_mode="r")
    metadata = joblib.load(ARTIFACTS_DIR / "inputs" / "metadata.pkl")
    
    logger.info(f"Loaded X: {X.shape}, y: {y.shape}")
    
    # Extract dates from metadata
    dates = metadata["date"]
    logger.info(f"Date range (original order): {dates.min()} to {dates.max()}")
    
    # Data is organized by ticker, not globally by date
    # Sort indices by date for proper temporal split
    date_sort_indices = np.argsort(dates)
    logger.info("Data organized by ticker. Sorting chronologically for temporal split...")
    
    # Re-order dates by date (X and y are not needed for splitting indices)
    dates_sorted = dates[date_sort_indices]
    
    logger.info(f"Sorted date range: {dates_sorted.min()} to {dates_sorted.max()}")
    
    # Verify sorted
    if not np.all(dates_sorted[:-1] <= dates_sorted[1:]):
        raise ValueError("Failed to sort dates chronologically")
    
    logger.info("Data chronologically sorted")
    
    # Setup TimeSeriesSplit with 5 folds (expanding windows + embargo gap)
    tscv = TimeSeriesSplit(n_splits=5, gap=CV_GAP_DAYS)
    logger.info(f"TimeSeriesSplit initialized with n_splits=5, gap={CV_GAP_DAYS}")
    
    folds = {}
    
    for fold_idx, (train_indices_sorted, test_indices_sorted) in enumerate(tscv.split(dates_sorted)):
        # Get train/test dates
        train_dates = dates_sorted[train_indices_sorted]
        test_dates = dates_sorted[test_indices_sorted]
        
        # Map back to original indices (before sorting)
        train_indices_original = date_sort_indices[train_indices_sorted]
        test_indices_original = date_sort_indices[test_indices_sorted]
        
        # Get date bounds
        train_date_min = pd.Timestamp(train_dates.min()).strftime("%Y-%m-%d")
        train_date_max = pd.Timestamp(train_dates.max()).strftime("%Y-%m-%d")
        test_date_min = pd.Timestamp(test_dates.min()).strftime("%Y-%m-%d") if len(test_dates) > 0 else "N/A"
        test_date_max = pd.Timestamp(test_dates.max()).strftime("%Y-%m-%d") if len(test_dates) > 0 else "N/A"
        
        # Verify no temporal leakage
        # Note: Same date can appear in both train and test (different stocks same date is OK)
        if len(test_dates) > 0:
            if test_dates.min() < train_dates.max():
                msg = f"Fold {fold_idx}: Temporal leakage detected! Test min ({test_dates.min()}) < Train max ({train_dates.max()})"
                logger.error(msg)
                raise ValueError(msg)
        
        fold_key = f"fold_{fold_idx}"
        folds[fold_key] = {
            "train_indices": train_indices_original.tolist(),
            "test_indices": test_indices_original.tolist(),
            "train_date_min": train_date_min,
            "train_date_max": train_date_max,
            "test_date_min": test_date_min,
            "test_date_max": test_date_max,
            "gap_days": CV_GAP_DAYS,
            "n_train": len(train_indices_original),
            "n_test": len(test_indices_original),
        }
        
        logger.info(
            f"Fold {fold_idx}: Train {len(train_indices_original):,} | Test {len(test_indices_original):,} | "
            f"{train_date_min} to {test_date_max} | gap={CV_GAP_DAYS}"
        )
    
    # Save fold metadata to joblib pickle
    (ARTIFACTS_DIR / "folds").mkdir(parents=True, exist_ok=True)
    folds_metadata_path = ARTIFACTS_DIR / "folds" / "folds_metadata.pkl"
    
    # Convert to joblib-serializable format
    folds_json = {
        fold_key: {
            "train_indices": fold_data["train_indices"],
            "test_indices": fold_data["test_indices"],
            "train_date_min": fold_data["train_date_min"],
            "train_date_max": fold_data["train_date_max"],
            "test_date_min": fold_data["test_date_min"],
            "test_date_max": fold_data["test_date_max"],
            "gap_days": fold_data["gap_days"],
            "n_train": fold_data["n_train"],
            "n_test": fold_data["n_test"],
        }
        for fold_key, fold_data in folds.items()
    }
    
    joblib.dump(folds_json, folds_metadata_path)
    
    logger.info(f"Saved fold metadata to {folds_metadata_path}")
    logger.info("Phase 3 complete: Temporal cross-validation setup with no leakage")
    
    return folds
