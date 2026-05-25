from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
from cross_validation.cv_utils import load_folds_metadata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	fold_data: dict,
) -> tuple:
	"""
	Train one XGBoost model per fold. No temporal leakage: model trained only on fold_i_train,
	evaluated on fold_i_test (which is strictly future to train dates).

	Args:
		fold_idx: Fold index (0-4)
		X_train: Training features from fold_i_train
		y_train: Training target from fold_i_train
		X_test: Test features from fold_i_test
		y_test: Test target from fold_i_test
		fold_data: Fold metadata (dates, counts)

	Returns:
		(model, predictions_dict)
	"""
	fold_start = time.time()
	logger.info(f"Fold {fold_idx}: Training model on {len(X_train):,} samples, testing on {len(X_test):,} samples")
	logger.info(f"  Date range: {fold_data['train_date_min']} to {fold_data['test_date_max']}")

	# Train model on fold_i_train data only (no leakage)
	try:
		train_start = time.time()
		model, label_mapping = train_xgboost_model(X_train, y_train, X_val=None, y_val=None)
		train_time = time.time() - train_start
	except Exception as e:
		logger.error(f"Fold {fold_idx} model training failed: {e}")
		raise

	# Generate predictions on fold_i_test data
	inference_start = time.time()
	y_pred = model.predict(X_test)
	inference_time = time.time() - inference_start

	# Compute metrics
	try:
		test_r2 = r2_score(y_test, y_pred)
		test_mae = mean_absolute_error(y_test, y_pred)
		test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	except Exception as e:
		logger.error(f"Fold {fold_idx} metric computation failed: {e}")
		test_r2 = float('nan')
		test_mae = float('nan')
		test_rmse = float('nan')

	fold_elapsed = time.time() - fold_start
	logger.info(f"  Fold {fold_idx}: R²={test_r2:.4f}, MAE={test_mae:.6f}, RMSE={test_rmse:.6f}")
	logger.info(f"    └─ Timing: {train_time:.1f}s train + {inference_time:.1f}s inference = {fold_elapsed:.1f}s total")

	# Save model
	checkpoints_dir = ARTIFACTS_DIR / "checkpoints"
	model_path = checkpoints_dir / f"fold_{fold_idx}_model.pkl"
	joblib.dump(model, model_path)
	if not model_path.exists():
		raise RuntimeError(f"Failed to save fold {fold_idx} model to {model_path}")
	logger.info(f"  Saved fold {fold_idx} model to {model_path}")

	predictions_dict = {
		"test_pred": y_pred,
		"test_true": y_test,
		"test_r2": test_r2,
		"test_mae": test_mae,
		"test_rmse": test_rmse,
		"n_train_samples": len(X_train),
		"n_test_samples": len(X_test),
		"n_features": X_train.shape[1],
		"train_time_seconds": train_time,
		"inference_time_seconds": inference_time,
		"fold_time_seconds": fold_elapsed,
		"fold_model_path": str(model_path),
		"train_date_range": (fold_data["train_date_min"], fold_data["train_date_max"]),
		"test_date_range": (fold_data["test_date_min"], fold_data["test_date_max"]),
	}

	return model, predictions_dict


def _aggregate_fold_results(fold_predictions_list: list) -> tuple[dict, dict]:
	"""
	Aggregate predictions and metrics across all 5 folds.

	Returns:
		(validation_predictions, fold_training_summary)
	"""
	validation_predictions = {}
	fold_training_summary = {}

	for fold_idx, pred_dict in enumerate(fold_predictions_list):
		fold_key = f"fold_{fold_idx}"
		validation_predictions[fold_key] = pred_dict

		fold_training_summary[fold_key] = {
			"train_date_min": pred_dict["train_date_range"][0],
			"train_date_max": pred_dict["train_date_range"][1],
			"test_date_min": pred_dict["test_date_range"][0],
			"test_date_max": pred_dict["test_date_range"][1],
			"n_train_samples": int(pred_dict["n_train_samples"]),
			"n_test_samples": int(pred_dict["n_test_samples"]),
			"n_features": int(pred_dict["n_features"]),
			"model_path": pred_dict["fold_model_path"],
			"test_r2": pred_dict["test_r2"],
			"test_mae": pred_dict["test_mae"],
			"test_rmse": pred_dict["test_rmse"],
			"train_time_seconds": pred_dict["train_time_seconds"],
			"inference_time_seconds": pred_dict["inference_time_seconds"],
			"fold_time_seconds": pred_dict["fold_time_seconds"],
		}

	return validation_predictions, fold_training_summary


def _save_training_results(validation_predictions: dict, fold_training_summary: dict) -> None:
	"""Save training results to disk."""
	val_pred_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"
	summary_path = ARTIFACTS_DIR / "results" / "fold_training_summary.json"

	joblib.dump(validation_predictions, val_pred_path)

	# Convert fold_training_summary to JSON-serializable format
	summary_json = {}
	for fold_key, summary in fold_training_summary.items():
		summary_json[fold_key] = {
			k: v if not isinstance(v, (np.floating, np.integer)) else float(v)
			for k, v in summary.items()
		}

	with open(summary_path, "w") as f:
		json.dump(summary_json, f, indent=2, default=str)

	logger.info("Training outputs saved")


def phase_4_training() -> tuple[dict, dict, list]:
	"""
	Orchestrate Phase 4: Train 5 fold-specific XGBoost models (1 per fold) with no temporal leakage.
	Each fold_i_model trained exclusively on fold_i_train, evaluated on fold_i_test.

	Returns:
		validation_predictions, fold_training_summary, fold_model_paths
	"""
	logger.info("Phase 4: Fold-Specific Model Training (1 Model per Fold)")
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

	# Train 5 models (1 per fold) with strict temporal separation
	fold_predictions_list = []
	fold_model_paths = []

	for fold_idx in range(5):
		fold_key = f"fold_{fold_idx}"
		fold_data = folds_metadata[fold_key]

		# Extract fold-specific data
		train_indices = np.array(fold_data["train_indices"])
		test_indices = np.array(fold_data["test_indices"])

		X_train = X[train_indices]
		y_train = y[train_indices]
		X_test = X[test_indices]
		y_test = y[test_indices]

		# Train fold model (1 model per fold, only on fold_i_train data, no leakage)
		model, predictions_dict = _train_fold_model(
			fold_idx=fold_idx,
			X_train=X_train,
			y_train=y_train,
			X_test=X_test,
			y_test=y_test,
			fold_data=fold_data,
		)

		fold_predictions_list.append(predictions_dict)
		fold_model_paths.append(predictions_dict["fold_model_path"])

	# Aggregate predictions and metrics across all folds
	validation_predictions, fold_training_summary = _aggregate_fold_results(fold_predictions_list)

	# Add feature breakdown (indicators vs ticker one-hot) from metadata
	for fold_key in fold_training_summary:
		fold_training_summary[fold_key]["n_features_indicators"] = int(metadata["n_indicators"])
		fold_training_summary[fold_key]["n_features_ticker"] = int(metadata["n_ticker_features"])

	# Save outputs
	_save_training_results(validation_predictions, fold_training_summary)

	# Final summary with metrics
	total_time = time.time() - overall_start_time

	# Aggregate metrics across folds
	fold_r2_scores = [s.get("test_r2", float('nan')) for s in fold_training_summary.values()]
	fold_mae_scores = [s.get("test_mae", float('nan')) for s in fold_training_summary.values()]
	fold_rmse_scores = [s.get("test_rmse", float('nan')) for s in fold_training_summary.values()]

	valid_r2 = [r for r in fold_r2_scores if not np.isnan(r)]
	valid_mae = [m for m in fold_mae_scores if not np.isnan(m)]
	valid_rmse = [r for r in fold_rmse_scores if not np.isnan(r)]

	if valid_r2:
		mean_r2 = np.mean(valid_r2)
		std_r2 = np.std(valid_r2)
		mean_mae = np.mean(valid_mae)
		std_mae = np.std(valid_mae)
		mean_rmse = np.mean(valid_rmse)
		std_rmse = np.std(valid_rmse)

		# Aggregate timing
		total_train_time = sum(s.get("train_time_seconds", 0) for s in fold_training_summary.values())
		total_inference_time = sum(s.get("inference_time_seconds", 0) for s in fold_training_summary.values())
		total_test_samples = sum(len(p["test_pred"]) for p in fold_predictions_list)

		logger.info(f"Phase 4 complete: 5 fold model(s), {total_test_samples:,} test samples total, {total_time:.1f}s ({total_train_time:.1f}s train + {total_inference_time:.1f}s inference)")
		logger.info(f"Cross-validation results (GLOBAL): R²={mean_r2:.4f}±{std_r2:.4f}, MAE={mean_mae:.6f}±{std_mae:.6f}, RMSE={mean_rmse:.6f}±{std_rmse:.6f}")

		# Log per-fold metrics
		for fold_key in sorted(fold_training_summary.keys()):
			fold_r2 = fold_training_summary[fold_key].get("test_r2", float('nan'))
			fold_mae = fold_training_summary[fold_key].get("test_mae", float('nan'))
			fold_rmse = fold_training_summary[fold_key].get("test_rmse", float('nan'))
			train_time = fold_training_summary[fold_key].get("train_time_seconds", 0)
			inference_time = fold_training_summary[fold_key].get("inference_time_seconds", 0)
			train_date_min = fold_training_summary[fold_key].get("train_date_min", "?")
			train_date_max = fold_training_summary[fold_key].get("train_date_max", "?")
			test_date_min = fold_training_summary[fold_key].get("test_date_min", "?")
			test_date_max = fold_training_summary[fold_key].get("test_date_max", "?")

			logger.info(f"  {fold_key}: R²={fold_r2:.4f}, MAE={fold_mae:.6f}, RMSE={fold_rmse:.6f}")
			logger.info(f"    └─ Train: {train_date_min} to {train_date_max} | Test: {test_date_min} to {test_date_max}")
			logger.info(f"    └─ Timing: {train_time:.1f}s train + {inference_time:.1f}s inference")
	else:
		logger.info(f"Phase 4 complete: 5 fold model(s), {total_time:.1f}s (no valid metrics)")

	return validation_predictions, fold_training_summary, fold_model_paths
