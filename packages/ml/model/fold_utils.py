from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"


def load_fold_data(
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple, tuple]:
    """
    Load train/test data for a specific fold.
    
    Args:
        fold_id: Fold index (0-4)
        
    Returns:
        X_train, X_test, y_train, y_test, train_indices, test_indices
    """
    # Load Phase 2 outputs
    X = joblib.load(MODELS_DIR / "X.pkl")
    y = joblib.load(MODELS_DIR / "Y.pkl")
    
    # Load fold metadata
    with open(MODELS_DIR / "folds_metadata.json") as f:
        folds_metadata = json.load(f)
    
    fold_key = f"fold_{fold_id}"
    fold_data = folds_metadata[fold_key]
    
    train_indices = np.array(fold_data["train_indices"])
    test_indices = np.array(fold_data["test_indices"])
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    logger.debug(
        f"Loaded fold {fold_id}: X_train {X_train.shape}, X_test {X_test.shape}"
    )
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def load_folds_metadata() -> dict:
    """Load cross-validation fold metadata."""
    with open(MODELS_DIR / "folds_metadata.json") as f:
        return json.load(f)
