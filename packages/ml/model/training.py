from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np

from cv_utils import load_folds_metadata
from inference import generate_predictions
from utils import ARTIFACTS_DIR
from model_trainer import train_xgboost_model

logger = logging.getLogger(__name__)


def _validate_training_prerequisites(y: np.ndarray, folds_metadata: dict) -> None:
    """
    Validate prerequisites that Phase 2-3 couldn't guarantee.

    Only validates what matters for training:
    - Target values must be {-1, 0, 1} (detect data corruption)
    - Exactly 5 folds (Phase 3 must have completed correctly)
    """
    # Validate target values (detect data corruption)
    unique_y = np.unique(y)
    if not np.all(np.isin(unique_y, [-1, 0, 1])):
        raise ValueError(f"Invalid target values. Expected {{-1, 0, 1}}, got {unique_y}")
    
    # Validate folds count (Phase 3 executed correctly)
    if len(folds_metadata) != 5:
        raise ValueError(f"Expected 5 folds, got {len(folds_metadata)}")
    
    logger.info("Pre-training validation passed")


def _train_fold_model(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    folds_metadata: dict,
) -> dict:
    """
    Train ONE independent XGBoost model on fold i.

    Returns dict with predictions, summary, and model path for aggregation.
    """
    fold_key = f"fold_{fold_idx}"
    fold_start_time = time.time()

    # Extract fold data
    fold_data = folds_metadata[fold_key]
    train_indices = np.array(fold_data["train_indices"])
    test_indices = np.array(fold_data["test_indices"])

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    train_date_min = fold_data["train_date_min"]
    train_date_max = fold_data["train_date_max"]
    test_date_min = fold_data["test_date_min"]
    test_date_max = fold_data["test_date_max"]

    # Verify no temporal overlap
    if test_date_min < train_date_max:
        raise ValueError(f"Fold {fold_idx} temporal leakage: test_date_min ({test_date_min}) < train_date_max ({train_date_max})")

    # Class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique_train, counts_train))

    # Train model and generate predictions
    model, label_mapping = train_xgboost_model(X_train, y_train)
    fold_predictions = generate_predictions(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        train_indices,
        test_indices,
        label_mapping,
    )

    # Validate prediction shapes
    if fold_predictions["train_proba"].shape != (len(X_train), 3):
        raise ValueError(f"Fold {fold_idx} train_proba shape mismatch")
    if fold_predictions["test_proba"].shape != (len(X_test), 3):
        raise ValueError(f"Fold {fold_idx} test_proba shape mismatch")

    # Save model
    model_path = ARTIFACTS_DIR / "checkpoints" / f"fold_{fold_idx}_model.pkl"
    label_mapping_path = ARTIFACTS_DIR / "checkpoints" / f"fold_{fold_idx}_label_mapping.pkl"

    joblib.dump(model, model_path)
    joblib.dump(label_mapping, label_mapping_path)

    if not model_path.exists():
        raise RuntimeError(f"Failed to save model to {model_path}")

    fold_elapsed = time.time() - fold_start_time

    # Return all data needed for aggregation
    return {
        "fold_idx": fold_idx,
        "predictions": fold_predictions,
        "summary": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_date_min": train_date_min,
            "train_date_max": train_date_max,
            "test_date_min": test_date_min,
            "test_date_max": test_date_max,
            "training_time_seconds": float(fold_elapsed),
            "model_path": str(model_path),
            "class_distribution_train": {int(k): int(v) for k, v in class_dist.items()},
        },
        "model_path": str(model_path),
    }


def _aggregate_fold_results(fold_results: list) -> tuple[dict, dict, list]:
    """
    Aggregate results from all folds. Verify consistency and temporal validity.

    Returns (validation_predictions, fold_training_summary, fold_model_paths).
    """
    validation_predictions = {}
    fold_training_summary = {}
    fold_model_paths = []

    for fold_result in fold_results:
        fold_idx = fold_result["fold_idx"]
        fold_key = f"fold_{fold_idx}"

        # Store predictions
        validation_predictions[fold_key] = {
            **fold_result["predictions"],
            "fold_model_path": fold_result["model_path"],
            "train_date_range": (fold_result["summary"]["train_date_min"], fold_result["summary"]["train_date_max"]),
            "test_date_range": (fold_result["summary"]["test_date_min"], fold_result["summary"]["test_date_max"]),
        }

        # Store summary
        fold_training_summary[fold_key] = fold_result["summary"]
        fold_model_paths.append(fold_result["model_path"])

        # Verify temporal validity (no overlaps)
        summary = fold_result["summary"]
        if summary["test_date_min"] < summary["train_date_max"]:
            raise ValueError(f"Fold {fold_idx} temporal leakage detected")

    logger.info(f"Aggregated {len(fold_results)} fold(s)")
    return validation_predictions, fold_training_summary, fold_model_paths


def _save_training_results(validation_predictions: dict, fold_training_summary: dict) -> None:
    """Save training results to disk."""
    val_pred_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"
    summary_path = ARTIFACTS_DIR / "results" / "fold_training_summary.json"
    
    joblib.dump(validation_predictions, val_pred_path)
    with open(summary_path, "w") as f:
        json.dump(fold_training_summary, f, indent=2)
    logger.info("Training outputs saved")


def phase_4_training() -> tuple[dict, dict, list]:
    """
    Orchestrate Phase 4: Validate → Train → Aggregate → Save.

    Returns:
        validation_predictions: Dict with predictions + probabilities per fold
        fold_training_summary: Dict with training metadata per fold
        fold_model_paths: List of saved model paths
    """
    logger.info("Phase 4: Model Training")
    overall_start_time = time.time()

    # Load data from Phase 2 and 3
    logger.info("Loading Phase 2-3 outputs...")
    X = joblib.load(ARTIFACTS_DIR / "inputs" / "X.pkl")
    y = joblib.load(ARTIFACTS_DIR / "inputs" / "Y.pkl")
    metadata = joblib.load(ARTIFACTS_DIR / "inputs" / "metadata.pkl")
    folds_metadata = load_folds_metadata()

    logger.info(f"Loaded X: {X.shape}, y: {y.shape}")
    logger.info(f"Features: {metadata['n_indicators']} indicators + {metadata['n_ticker_features']} ticker = {X.shape[1]} total")

    # Validate prerequisites
    _validate_training_prerequisites(y, folds_metadata)

    # Train all folds
    fold_results = []
    for fold_idx in range(1):  # Change to range(5) to train all 5 folds
        result = _train_fold_model(fold_idx, X, y, folds_metadata)
        fold_results.append(result)

    # Aggregate results
    validation_predictions, fold_training_summary, fold_model_paths = _aggregate_fold_results(fold_results)

    # Save outputs
    _save_training_results(validation_predictions, fold_training_summary)

    # Final summary
    total_time = time.time() - overall_start_time
    logger.info(f"Phase 4 complete: {len(fold_results)} model(s), {sum(s['test_samples'] for s in fold_training_summary.values()):,} test samples, {total_time:.1f}s")
    logger.info(f"Note: To train all 5 folds, change range(1) to range(5) on line 214")
    
    return validation_predictions, fold_training_summary, fold_model_paths
