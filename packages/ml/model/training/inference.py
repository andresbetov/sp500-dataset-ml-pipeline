from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

logger = logging.getLogger(__name__)


def _validate_predictions(fold_predictions: dict) -> None:
    """
    Validate prediction integrity for regression. Fail-fast if data corrupted.

    Detects problems in THIS MODULE (generate_predictions output).
    Does NOT re-validate XGBoost outputs (redundant with API guarantees).

    Raises ValueError if validation fails.
    """
    # Check 1: Hard predictions are finite
    train_pred = fold_predictions["train_pred"]
    test_pred = fold_predictions["test_pred"]

    if not np.isfinite(train_pred).all():
        nan_count = np.isnan(train_pred).sum()
        inf_count = np.isinf(train_pred).sum()
        raise ValueError(f"Invalid train_pred values. Found {nan_count} NaNs and {inf_count} infs")
    if not np.isfinite(test_pred).all():
        nan_count = np.isnan(test_pred).sum()
        inf_count = np.isinf(test_pred).sum()
        raise ValueError(f"Invalid test_pred values. Found {nan_count} NaNs and {inf_count} infs")

    # Check 2: No NaN in true labels
    if np.any(np.isnan(fold_predictions["train_true"])):
        raise ValueError("NaN found in train_true")
    if np.any(np.isnan(fold_predictions["test_true"])):
        raise ValueError("NaN found in test_true")

    # Check 3: Shapes match
    if train_pred.shape[0] != len(fold_predictions["train_true"]):
        raise ValueError(f"Shape mismatch: train_pred {train_pred.shape[0]} != train_true {len(fold_predictions['train_true'])}")
    if test_pred.shape[0] != len(fold_predictions["test_true"]):
        raise ValueError(f"Shape mismatch: test_pred {test_pred.shape[0]} != test_true {len(fold_predictions['test_true'])}")


def generate_predictions(
    model: xgb.XGBRegressor,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_mapping: dict,
) -> dict:
    """
    Generate train and test predictions + regression metrics for a fold.

    For regression, returns continuous predictions and computes R², MAE, RMSE.

    Args:
        model: Trained XGBRegressor
        X_train: Training features (n_train, n_features)
        X_test: Test features (n_test, n_features)
        y_train: Training volatility targets (original continuous values)
        y_test: Test volatility targets (original continuous values)
        train_indices: Original indices for training samples
        test_indices: Original indices for test samples
        label_mapping: For API compatibility, unused in regression

    Returns:
        Dict with predictions, metrics, labels, and indices

    Raises:
        ValueError: If predictions fail validation
    """
    # Generate continuous predictions
    train_pred = model.predict(X_train)  # Shape: (n_train,)
    test_pred = model.predict(X_test)    # Shape: (n_test,)

    # Compute regression metrics
    train_r2 = r2_score(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # Build result dict
    result = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_pred": train_pred,
        "train_true": y_train,
        "train_r2": float(train_r2),
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "test_pred": test_pred,
        "test_true": y_test,
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
    }

    # Log metrics
    logger.info(f"Fold predictions - Train R²: {train_r2:.4f}, MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
    logger.info(f"Fold predictions - Test R²: {test_r2:.4f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")

    # Validate predictions before returning (fail-fast on corruption)
    _validate_predictions(result)

    return result
