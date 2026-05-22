from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd

from cross_validation.cv_utils import load_folds_metadata
from training.inference import generate_predictions
from training.model_trainer import train_xgboost_model
from utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def _validate_training_prerequisites(y: np.ndarray, folds_metadata: dict) -> None:
	"""
	Validate prerequisites that Phase 2-3 couldn't guarantee.

	For regression, validates:
	- Target is continuous (non-NaN, finite values)
	- Exactly 5 folds (Phase 3 must have completed correctly)
	"""
	# Validate target values (detect data corruption)
	if not np.isfinite(y).all():
		nan_count = np.isnan(y).sum()
		inf_count = np.isinf(y).sum()
		raise ValueError(f"Invalid target values. Found {nan_count} NaNs and {inf_count} infs")

	# Validate folds count (Phase 3 executed correctly)
	if len(folds_metadata) != 5:
		raise ValueError(f"Expected 5 folds, got {len(folds_metadata)}")

	logger.info(f"Pre-training validation passed")
	logger.info(f"Target statistics: range [{y.min():.6f}, {y.max():.6f}], mean {y.mean():.6f}, std {y.std():.6f}")


def _train_fold_model(
	fold_idx: int,
	X: np.ndarray,
	y: np.ndarray,
	metadata: dict,
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

	X_test = X[test_indices]
	y_test = y[test_indices]

	# Build a temporal validation slice from the most recent part of the train window
	if len(train_indices) < 2:
		raise ValueError(f"Fold {fold_idx}: not enough train samples for temporal validation split")

	train_dates = pd.to_datetime(metadata["date"][train_indices], errors="raise")
	temporal_order = np.argsort(train_dates.values)
	train_indices_temporal = train_indices[temporal_order]

	val_size = max(1, int(len(train_indices_temporal) * 0.10))
	val_size = min(val_size, len(train_indices_temporal) - 1)

	model_train_indices = train_indices_temporal[:-val_size]
	model_val_indices = train_indices_temporal[-val_size:]

	X_model_train = X[model_train_indices]
	y_model_train = y[model_train_indices]
	X_model_val = X[model_val_indices]
	y_model_val = y[model_val_indices]

	train_date_min = fold_data["train_date_min"]
	train_date_max = fold_data["train_date_max"]
	test_date_min = fold_data["test_date_min"]
	test_date_max = fold_data["test_date_max"]

	# Verify no temporal overlap with typed date comparisons
	if pd.Timestamp(test_date_min) < pd.Timestamp(train_date_max):
		raise ValueError(f"Fold {fold_idx} temporal leakage: test_date_min ({test_date_min}) < train_date_max ({train_date_max})")

	# Train model and generate predictions
	model, label_mapping = train_xgboost_model(
		X_model_train,
		y_model_train,
		X_val=X_model_val,
		y_val=y_model_val,
	)
	fold_predictions = generate_predictions(
		model,
		X_model_train,
		X_test,
		y_model_train,
		y_test,
		model_train_indices,
		test_indices,
		label_mapping,
	)

	# Validate prediction shapes
	if fold_predictions["train_pred"].shape[0] != len(X_model_train):
		raise ValueError(f"Fold {fold_idx} train_pred shape mismatch")
	if fold_predictions["test_pred"].shape[0] != len(X_test):
		raise ValueError(f"Fold {fold_idx} test_pred shape mismatch")

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
			"train_samples": int(len(X_model_train)),
			"validation_samples": int(len(X_model_val)),
			"test_samples": int(len(X_test)),
			"train_date_min": train_date_min,
			"train_date_max": train_date_max,
			"test_date_min": test_date_min,
			"test_date_max": test_date_max,
			"training_time_seconds": float(fold_elapsed),
			"model_path": str(model_path),
			# Regression metrics
			"train_r2": fold_predictions.get("train_r2", float('nan')),
			"train_mae": fold_predictions.get("train_mae", float('nan')),
			"train_rmse": fold_predictions.get("train_rmse", float('nan')),
			"test_r2": fold_predictions.get("test_r2", float('nan')),
			"test_mae": fold_predictions.get("test_mae", float('nan')),
			"test_rmse": fold_predictions.get("test_rmse", float('nan')),
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
		fold_predictions = fold_result["predictions"]
		validation_predictions[fold_key] = {
			"test_indices": fold_predictions["test_indices"],
			"test_pred": fold_predictions["test_pred"],
			"test_true": fold_predictions["test_true"],
			"test_r2": fold_predictions.get("test_r2", float('nan')),
			"test_mae": fold_predictions.get("test_mae", float('nan')),
			"test_rmse": fold_predictions.get("test_rmse", float('nan')),
			"fold_model_path": fold_result["model_path"],
			"train_date_range": (fold_result["summary"]["train_date_min"], fold_result["summary"]["train_date_max"]),
			"test_date_range": (fold_result["summary"]["test_date_min"], fold_result["summary"]["test_date_max"]),
		}

		# Store summary
		fold_training_summary[fold_key] = fold_result["summary"]
		fold_model_paths.append(fold_result["model_path"])

		# Verify temporal validity (no overlaps)
		summary = fold_result["summary"]
		if pd.Timestamp(summary["test_date_min"]) < pd.Timestamp(summary["train_date_max"]):
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
		validation_predictions: Dict with predictions and metrics per fold
		fold_training_summary: Dict with training metadata per fold
		fold_model_paths: List of saved model paths
	"""
    logger.info("Phase 4: Model Training")
    overall_start_time = time.time()

    # Load data from Phase 2 and 3
    logger.info("Loading Phase 2-3 outputs...")
    X = np.load(ARTIFACTS_DIR / "inputs" / "X.npy", mmap_mode="r")
    y = np.load(ARTIFACTS_DIR / "inputs" / "Y.npy", mmap_mode="r")
    metadata = joblib.load(ARTIFACTS_DIR / "inputs" / "metadata.pkl")
    folds_metadata = load_folds_metadata()

    logger.info(f"Loaded X: {X.shape}, y: {y.shape}")
    logger.info(f"Features: {metadata['n_indicators']} indicators + {metadata['n_ticker_features']} ticker = {X.shape[1]} total")

    # Validate prerequisites
    _validate_training_prerequisites(y, folds_metadata)

    # Train all folds
    fold_results = []
    for fold_idx in range(5):
        result = _train_fold_model(fold_idx, X, y, metadata, folds_metadata)
        fold_results.append(result)

    # Aggregate results
    validation_predictions, fold_training_summary, fold_model_paths = _aggregate_fold_results(fold_results)

    # Save outputs
    _save_training_results(validation_predictions, fold_training_summary)

    # Final summary
    total_time = time.time() - overall_start_time
    logger.info(f"Phase 4 complete: {len(fold_results)} model(s), {sum(s['test_samples'] for s in fold_training_summary.values()):,} test samples, {total_time:.1f}s")
    
    return validation_predictions, fold_training_summary, fold_model_paths
