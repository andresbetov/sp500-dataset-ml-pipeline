from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from packages.ml.model.config import get_config
from packages.ml.model.cross_validation.cv_utils import load_folds_metadata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from packages.ml.model.training.har_benchmark import fit_har_benchmark, load_har_regressors
from packages.ml.model.training.model_trainer import train_xgboost_model
from packages.ml.model.training.qlike_objective import qlike_mean
from packages.ml.model.utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# Daily-volatility floor for the log target. The Yang-Zhang target is exactly zero on
# flat windows, where log(0) is undefined; 1e-12 only binds those rows, since the
# smallest positive target in the frame is ~2.6e-10.
RV_FLOOR = float(get_config()["rv_floor"])


def qlike_loss(actual_log: np.ndarray, forecast_log: np.ndarray) -> float:
	"""QLIKE loss for log-sigma forecasts (Patton 2011; Hansen-Lunde 2005).

	QLIKE(h, f) = h/f - ln(h/f) - 1 with h = exp(actual_log) and f = exp(forecast_log),
	so in log space the loss is exp(delta) - delta - 1 for delta = actual - forecast.
	This is the robust form for volatility-forecast comparison — the same ranking as
	the Hansen-Lunde loss log(f) + h/f, up to the target-only term ln(h) + 1. The mean
	comes from ``qlike_mean``, the functional the trainer also optimizes and evaluates.

	Signed identity: QLIKE = HL - ln(h) - 1, equivalently HL = QLIKE + ln(h) + 1 for
	the Hansen-Lunde loss HL = ln(f) + h/f. The ranking is identical; the absolute
	level differs only by the target-only constant ln(h) + 1.

	Scale: the loss is in sigma (per-day volatility) units, h = exp(actual_log), not in
	variance units. The same fit scored in variance units would report roughly 4-6x
	different absolute magnitudes, so the pipeline's absolute values are not directly
	comparable to literature numbers computed on a variance scale.

	Unlike the R²/MAE/RMSE block this metric never swallows errors: non-finite inputs
	or outputs raise.
	"""
	delta = np.asarray(actual_log, dtype=np.float64) - np.asarray(forecast_log, dtype=np.float64)
	if not np.isfinite(delta).all():
		raise ValueError("QLIKE received non-finite log targets or forecasts")
	return qlike_mean(delta)


def validate_fold_minima(fold_idx: int, train_indices: np.ndarray, test_indices: np.ndarray) -> None:
	"""Require both sides of a fold to hold at least one index.

	The single per-fold minima implementation: the pre-training prerequisite pass
	runs it for every fold, and the cross-validation boundary imports it for the
	folds it constructs. Empty indices would otherwise surface much later as an
	opaque model or metric failure.
	"""
	if len(train_indices) == 0:
		raise ValueError(f"Fold {fold_idx}: empty train indices")
	if len(test_indices) == 0:
		raise ValueError(f"Fold {fold_idx}: empty test indices")


def _validate_training_prerequisites(
	X: np.ndarray,
	y: np.ndarray,
	metadata: dict,
	folds_metadata: dict,
) -> None:
	"""
	Validate prerequisites that Phase 2-3 couldn't guarantee.

	For regression, validates:
	- Target is continuous (non-NaN, finite values)
	- Exactly 5 folds (Phase 3 must have completed correctly)
	- Features are finite (a non-finite input would poison every fold fit)
	- No duplicated (ticker, date) rows (one row per ticker-day)
	- Feature width matches the metadata counts
	- The regime label exists, is 1-D, row-aligned, and inside the seven buckets
	- Every fold has non-empty train and test indices (validate_fold_minima)
	"""
	# Validate target values (detect data corruption)
	if not np.isfinite(y).all():
		nan_count = np.isnan(y).sum()
		inf_count = np.isinf(y).sum()
		raise ValueError(f"Invalid target values. Found {nan_count} NaNs and {inf_count} infs")

	# Validate folds count (Phase 3 executed correctly)
	if len(folds_metadata) != 5:
		raise ValueError(f"Expected 5 folds, got {len(folds_metadata)}")

	# Validate features; a single vectorized scan is fine, X may be a memory map.
	finite = np.isfinite(X)
	if not finite.all():
		nan_count = int(np.isnan(X).sum())
		inf_count = int(np.isinf(X).sum())
		raise ValueError(f"Invalid feature values. Found {nan_count} NaNs and {inf_count} infs")

	# Validate one row per (ticker, date): a duplicated pair would silently
	# double-weight a ticker-day in every fold fit.
	duplicate_count = int(
		pd.MultiIndex.from_arrays([metadata["ticker"], metadata["date"]]).duplicated().sum()
	)
	if duplicate_count:
		raise ValueError(f"Duplicate (ticker, date) rows: {duplicate_count} duplicated pairs")

	# Validate the width against the metadata counts Phase 2 wrote.
	expected_width = int(metadata["n_indicators"]) + int(metadata["n_ticker_features"])
	if X.shape[1] != expected_width:
		raise ValueError(
			f"Feature width mismatch: X has {X.shape[1]} columns but metadata implies "
			f"{expected_width} (n_indicators={metadata['n_indicators']} + "
			f"n_ticker_features={metadata['n_ticker_features']})"
		)

	# Validate the regime contract: the seven integer calendar buckets 0..6, one
	# label per row. Fail loud while the key is absent rather than train unlabeled.
	if "regime" not in metadata:
		raise ValueError("metadata is missing the 'regime' key (seven calendar buckets 0..6)")
	regime = np.asarray(metadata["regime"])
	if regime.ndim != 1:
		raise ValueError(f"metadata['regime'] must be a 1-D array, got shape {regime.shape}")
	if len(regime) != len(y):
		raise ValueError(f"metadata['regime'] has {len(regime)} entries but there are {len(y)} rows")
	if not (
		np.issubdtype(regime.dtype, np.integer)
		or np.issubdtype(regime.dtype, np.floating)
	):
		raise ValueError(f"metadata['regime'] must be integer-valued, got dtype {regime.dtype}")
	numeric_regime = regime.astype(np.float64)
	in_buckets = (
		(numeric_regime >= 0)
		& (numeric_regime <= 6)
		& (numeric_regime == np.floor(numeric_regime))
	)
	if not in_buckets.all():
		invalid_values = np.unique(regime[~in_buckets]).tolist()
		raise ValueError(
			f"metadata['regime'] must contain only integers in 0..6; "
			f"found invalid values: {invalid_values}"
		)

	# Validate every fold holds both sides (single minima implementation:
	# validate_fold_minima).
	for fold_idx in range(5):
		fold_data = folds_metadata[f"fold_{fold_idx}"]
		validate_fold_minima(fold_idx, fold_data["train_indices"], fold_data["test_indices"])

	logger.info(f"Pre-training validation passed")
	logger.info(f"Target statistics: range [{y.min():.6f}, {y.max():.6f}], mean {y.mean():.6f}, std {y.std():.6f}")


def _model_training_indices(fold_data: dict, y: np.ndarray) -> np.ndarray:
	"""Training row indices for one fold, with degenerate at-or-below-floor rows removed.

	The exact-zero Yang-Zhang windows are flat-price artifacts: their log target is
	undefined (floored for finiteness). A QLIKE-trained model that fits their
	extreme labels under-forecasts similar test rows by ~20 log-units, where the
	exponential term explodes. The filter uses the same floor as the target
	transform, so a positive target at floor level cannot enter training as a
	floored degenerate label. Test sets keep every row.
	"""
	train_indices = np.array(fold_data["train_indices"])
	return train_indices[y[train_indices] > RV_FLOOR]


def _train_fold_model(
	fold_idx: int,
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	fold_data: dict,
) -> tuple:
	"""
	Train one XGBoost model per fold on the log target. No temporal leakage: model
	trained only on fold_i_train, evaluated on fold_i_test (which is strictly future to
	train dates).

	Args:
		fold_idx: Fold index (0-4)
		X_train: Training features from fold_i_train
		y_train: Training log-target from fold_i_train
		X_test: Test features from fold_i_test
		y_test: Test log-target from fold_i_test
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

	# Compute metrics on the log target. R² follows sklearn's convention, whose
	# denominator is the fold's own test variance (fold-own-mean); it is not comparable
	# to a non-demeaned R²_oos. The block below swallows failures into NaN metrics.
	try:
		test_r2 = r2_score(y_test, y_pred)
		test_mae = mean_absolute_error(y_test, y_pred)
		test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	except Exception as e:
		logger.error(f"Fold {fold_idx} metric computation failed: {e}")
		test_r2 = float('nan')
		test_mae = float('nan')
		test_rmse = float('nan')

	# QLIKE is computed outside the block above on purpose: it must fail loudly rather
	# than write a NaN metric.
	test_qlike = qlike_loss(y_test, y_pred)

	fold_elapsed = time.time() - fold_start
	logger.info(f"  Fold {fold_idx}: R²={test_r2:.4f}, MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, QLIKE={test_qlike:.6f}")
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
		"test_qlike": test_qlike,
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
			"test_qlike": pred_dict["test_qlike"],
			"har_r2": pred_dict["har_r2"],
			"har_qlike": pred_dict["har_qlike"],
			"train_time_seconds": pred_dict["train_time_seconds"],
			"inference_time_seconds": pred_dict["inference_time_seconds"],
			"fold_time_seconds": pred_dict["fold_time_seconds"],
		}

	return validation_predictions, fold_training_summary


def _save_training_results(
	validation_predictions: dict,
	fold_training_summary: dict,
	har_benchmark_results: dict,
) -> None:
	"""Save training results and the HAR benchmark report to disk."""
	val_pred_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"
	summary_path = ARTIFACTS_DIR / "results" / "fold_training_summary.json"
	har_report_path = ARTIFACTS_DIR / "results" / "har_benchmark.json"

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

	# HAR benchmark report: per-fold coefficients and metrics next to the fold model's,
	# so the mandatory benchmark is auditable without reloading the predictions.
	har_report = {
		"benchmark": "HAR-RV (Corsi 2008): canonical 1/5/22 trailing variance components, OLS in variance levels",
		"target": "log(realized_volatility_5d)",
		"r2_convention": "log-target scale, fold-own-mean (sklearn r2_score)",
		"qlike": "h/f - ln(h/f) - 1 with h=exp(actual), f=exp(forecast); target in native per-day sigma units",
		"folds": {
			fold_key: {
				**result,
				"model_r2": fold_training_summary[fold_key]["test_r2"],
				"model_qlike": fold_training_summary[fold_key]["test_qlike"],
			}
			for fold_key, result in har_benchmark_results.items()
		},
	}
	with open(har_report_path, "w") as f:
		json.dump(har_report, f, indent=2, default=str)

	logger.info(f"Training outputs saved (HAR benchmark report: {har_report_path})")


def phase_4_training() -> tuple[dict, dict, list]:
	"""
	Orchestrate Phase 4: Train 5 fold-specific XGBoost models (1 per fold) with no temporal leakage.
	Each fold_i_model trains exclusively on fold_i_train and is evaluated on fold_i_test.
	The modeled target is log(RV) — the Yang-Zhang variance is closer to log-normal and
	the fold models minimize QLIKE on that log target — and every fold also fits the
	canonical HAR-RV benchmark on its training rows only. Degenerate exact-zero target
	rows are excluded from the training sets and kept in the test sets.

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
	_validate_training_prerequisites(X, y, metadata, folds_metadata)

	# Model the log of the target; the floor only touches the exact-zero Yang-Zhang
	# windows (flat 5-day prices), which would otherwise make log(0) undefined. Rows at
	# or below the floor are excluded from the fold training sets (see
	# _model_training_indices) but stay in the test sets.
	y_log = np.log(np.maximum(y, RV_FLOOR))
	logger.info(
		f"Log target: log(RV) with floor {RV_FLOOR:g}; rows at or below the floor = {int(np.count_nonzero(y <= RV_FLOOR)):,} "
		f"(excluded from training, kept in test sets)"
	)

	har_regressors = load_har_regressors(metadata["date"])

	# Train 5 models (1 per fold) with strict temporal separation
	fold_predictions_list = []
	fold_model_paths = []
	har_benchmark_results = {}

	for fold_idx in range(5):
		fold_key = f"fold_{fold_idx}"
		fold_data = folds_metadata[fold_key]

		# Extract fold-specific data; degenerate at-or-below-floor rows are dropped
		# from the training side only (the HAR benchmark shares the same rows).
		train_indices = _model_training_indices(fold_data, y)

		# An all-floored fold keeps no training rows: every label would be the
		# floor itself, so there is nothing to fit. Name the fold and the floor
		# before the per-fold log line and any fit attempt.
		if train_indices.size == 0:
			raise ValueError(
				f"Fold {fold_idx} ({fold_key}): all {int(fold_data['n_train'])} fold-train rows "
				f"are at or below the floor {RV_FLOOR:g} (RV_FLOOR); no training rows remain"
			)

		test_indices = np.array(fold_data["test_indices"])

		logger.info(
			f"  Fold {fold_idx}: excluded {int(fold_data['n_train']) - len(train_indices):,} "
			f"degenerate at-or-below-floor rows from training"
		)

		X_train = X[train_indices]
		y_train = y_log[train_indices]
		X_test = X[test_indices]
		y_test = y_log[test_indices]

		# Train fold model (1 model per fold, only on fold_i_train data, no leakage)
		model, predictions_dict = _train_fold_model(
			fold_idx=fold_idx,
			X_train=X_train,
			y_train=y_train,
			X_test=X_test,
			y_test=y_test,
			fold_data=fold_data,
		)

		# Canonical HAR-RV benchmark on the same fold rows, fitted on train only.
		har_result = fit_har_benchmark(
			train_indices=train_indices,
			test_indices=test_indices,
			target_log_train=y_train,
			target_log_test=y_test,
			regressors=har_regressors,
			forecast_floor=RV_FLOOR,
		)
		har_result["har_qlike"] = qlike_loss(y_test, har_result.pop("forecast_log"))
		predictions_dict["har_r2"] = har_result["har_r2"]
		predictions_dict["har_qlike"] = har_result["har_qlike"]
		har_benchmark_results[fold_key] = har_result

		logger.info(
			f"    └─ HAR benchmark: R²={har_result['har_r2']:.4f}, QLIKE={har_result['har_qlike']:.6f}"
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
	_save_training_results(validation_predictions, fold_training_summary, har_benchmark_results)

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

		mean_qlike = np.mean([s["test_qlike"] for s in fold_training_summary.values()])
		std_qlike = np.std([s["test_qlike"] for s in fold_training_summary.values()])
		mean_har_qlike = np.mean([s["har_qlike"] for s in fold_training_summary.values()])
		std_har_qlike = np.std([s["har_qlike"] for s in fold_training_summary.values()])
		mean_har_r2 = np.mean([s["har_r2"] for s in fold_training_summary.values()])

		logger.info(f"Phase 4 complete: 5 fold model(s), {total_test_samples:,} test samples total, {total_time:.1f}s ({total_train_time:.1f}s train + {total_inference_time:.1f}s inference)")
		logger.info(f"Cross-validation results (GLOBAL): R²={mean_r2:.4f}±{std_r2:.4f}, MAE={mean_mae:.6f}±{std_mae:.6f}, RMSE={mean_rmse:.6f}±{std_rmse:.6f}")
		logger.info(f"QLIKE (GLOBAL, log-target): model={mean_qlike:.6f}±{std_qlike:.6f} | HAR benchmark={mean_har_qlike:.6f}±{std_har_qlike:.6f} (R²={mean_har_r2:.4f})")

		# Log per-fold metrics
		for fold_key in sorted(fold_training_summary.keys()):
			fold_r2 = fold_training_summary[fold_key].get("test_r2", float('nan'))
			fold_mae = fold_training_summary[fold_key].get("test_mae", float('nan'))
			fold_rmse = fold_training_summary[fold_key].get("test_rmse", float('nan'))
			fold_qlike = fold_training_summary[fold_key].get("test_qlike", float('nan'))
			fold_har_qlike = fold_training_summary[fold_key].get("har_qlike", float('nan'))
			fold_har_r2 = fold_training_summary[fold_key].get("har_r2", float('nan'))
			train_time = fold_training_summary[fold_key].get("train_time_seconds", 0)
			inference_time = fold_training_summary[fold_key].get("inference_time_seconds", 0)
			train_date_min = fold_training_summary[fold_key].get("train_date_min", "?")
			train_date_max = fold_training_summary[fold_key].get("train_date_max", "?")
			test_date_min = fold_training_summary[fold_key].get("test_date_min", "?")
			test_date_max = fold_training_summary[fold_key].get("test_date_max", "?")

			logger.info(f"  {fold_key}: R²={fold_r2:.4f}, MAE={fold_mae:.6f}, RMSE={fold_rmse:.6f}, QLIKE={fold_qlike:.6f}")
			logger.info(f"    └─ HAR benchmark: R²={fold_har_r2:.4f}, QLIKE={fold_har_qlike:.6f}")
			logger.info(f"    └─ Train: {train_date_min} to {train_date_max} | Test: {test_date_min} to {test_date_max}")
			logger.info(f"    └─ Timing: {train_time:.1f}s train + {inference_time:.1f}s inference")
	else:
		logger.info(f"Phase 4 complete: 5 fold model(s), {total_time:.1f}s (no valid metrics)")

	return validation_predictions, fold_training_summary, fold_model_paths
