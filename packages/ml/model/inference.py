from __future__ import annotations

import logging

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


def _unmap_predictions(predictions: np.ndarray, reverse_mapping: dict) -> np.ndarray:
    """Map predictions from {0, 1, 2} back to {-1, 0, 1}."""
    return (predictions - 1).astype(np.float32)


def _validate_predictions(fold_predictions: dict) -> None:
    """
    Validate prediction integrity. Fail-fast if data corrupted.

    Detects problems in THIS MODULE (generate_predictions output).
    Does NOT re-validate XGBoost outputs (redundant with API guarantees).

    Raises ValueError if validation fails.
    """
    # Check 1: Hard predictions are valid class labels {-1, 0, 1}
    # Detects bugs in _unmap_predictions()
    train_pred = fold_predictions["train_pred"]
    test_pred = fold_predictions["test_pred"]

    valid_labels = {-1.0, 0.0, 1.0}
    train_unique = set(np.unique(train_pred))
    test_unique = set(np.unique(test_pred))

    if not train_unique.issubset(valid_labels):
        raise ValueError(f"Invalid train_pred values. Expected {{-1, 0, 1}}, got {train_unique}")
    if not test_unique.issubset(valid_labels):
        raise ValueError(f"Invalid test_pred values. Expected {{-1, 0, 1}}, got {test_unique}")

    # Check 2: No NaN in hard predictions
    # Detects corruption in unmapping process
    if np.any(np.isnan(train_pred)):
        raise ValueError("NaN found in train_pred (unmapped predictions)")
    if np.any(np.isnan(test_pred)):
        raise ValueError("NaN found in test_pred (unmapped predictions)")

    # Check 3: Proba shapes match samples
    # Detects dimensional mismatches
    train_proba = fold_predictions["train_proba"]
    test_proba = fold_predictions["test_proba"]

    if train_proba.shape != (len(train_pred), 3):
        raise ValueError(f"Shape mismatch: train_proba {train_proba.shape} != (n={len(train_pred)}, 3)")
    if test_proba.shape != (len(test_pred), 3):
        raise ValueError(f"Shape mismatch: test_proba {test_proba.shape} != (n={len(test_pred)}, 3)")

    # Check 4: No NaN in probabilities
    # Detects XGBoost issues upstream
    if np.any(np.isnan(train_proba)):
        raise ValueError("NaN found in train_proba (XGBoost output)")
    if np.any(np.isnan(test_proba)):
        raise ValueError("NaN found in test_proba (XGBoost output)")


def generate_predictions(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_mapping: dict,
) -> dict:
    """
    Generate train and test predictions + probabilities for a fold.
    
    Converts hard predictions from {0, 1, 2} back to {-1, 0, 1}.
    Probabilities remain as-is (raw XGBoost output).
    Validates all outputs before returning.

    Args:
        model: Trained XGBClassifier
        X_train: Training features (n_train, n_features)
        X_test: Test features (n_test, n_features)
        y_train: Training labels (original: {-1, 0, 1})
        y_test: Test labels (original: {-1, 0, 1})
        train_indices: Original indices for training samples
        test_indices: Original indices for test samples
        label_mapping: Dict mapping {-1, 0, 1} to {0, 1, 2}
        
    Returns:
        Dict with hard predictions, probabilities, labels, and indices

    Raises:
        ValueError: If predictions fail validation
    """
    train_pred_raw = model.predict(X_train)
    train_proba = model.predict_proba(X_train)  # Shape: (n_train, 3)
    
    test_pred_raw = model.predict(X_test)
    test_proba = model.predict_proba(X_test)    # Shape: (n_test, 3)
    
    # Unmap hard predictions back to original label space {-1, 0, 1}
    train_pred = _unmap_predictions(train_pred_raw, label_mapping)
    test_pred = _unmap_predictions(test_pred_raw, label_mapping)
    
    # Build result dict
    result = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_pred": train_pred,
        "train_proba": train_proba,
        "train_true": y_train,
        "test_pred": test_pred,
        "test_proba": test_proba,
        "test_true": y_test,
    }

    # Validate predictions before returning (fail-fast on corruption)
    _validate_predictions(result)

    return result
