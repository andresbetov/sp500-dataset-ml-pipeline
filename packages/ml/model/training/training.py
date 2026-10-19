"""Phase 4 fold training: one QLIKE-trained XGBoost model per fold.

Each fold writes its model to the run-scoped checkpoint
``checkpoints/<run_id>/fold_<i>.pkl`` (the training artifact, loaded with
``joblib``) and its full ``get_params()`` set to the sibling
``fold_<i>_params.json`` sidecar (callables by qualified name, tool versions,
best-effort git sha; strict JSON, written atomically). XGBoost's native
``json``/``ubj`` model formats are prediction-only: a native round trip keeps
the trees but loses every training parameter (configured knobs come back
``None``/defaults and the QLIKE objective is not even declared), so the
training recipe is recovered from the pickle or from this sidecar, never from
a native reload.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from packages.ml.model.artifact_integrity import (
	REGISTRY_FILENAME,
	load_verified,
	record_artifact,
)
from packages.ml.model.config import get_config
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.cross_validation.cv_utils import load_folds_metadata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from packages.ml.model.training.har_benchmark import (
	COMPARISON_NOTE,
	FIT_SCALE,
	HAC_LAG,
	HAC_NOTE,
	HAR_WARMUP_DAYS,
	NO_HAC_NOTE,
	REPORT_SCALE,
	fit_har_benchmark,
	load_har_regressors,
)
from packages.ml.model.training.model_trainer import train_xgboost_model
from packages.ml.model.training.qlike_objective import qlike_eval_metric, qlike_mean
from packages.ml.model.run_log import (
	environment_versions,
	finish_run,
	git_sha,
	new_run,
	record_artifacts,
	record_fold,
	record_phase,
	to_repo_relative,
)
from packages.ml.model.utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# Daily-volatility floor for the log target. The Yang-Zhang target is exactly zero on
# flat windows, where log(0) is undefined; 1e-12 only binds those rows, since the
# smallest positive target in the frame is ~2.6e-10.
RV_FLOOR = float(get_config()["rv_floor"])

# The target of row t aggregates the forward window [t+1, t+5], so a row's label
# window reaches LABEL_HORIZON_DAYS trading dates past its own date. The inner split
# purges fit rows by the same horizon the outer train/test split uses.
LABEL_HORIZON_DAYS = get_config()["folds"]["label_horizon_days"]

# Fraction of each fold's floor-retained training rows held out as the purged
# inner-validation tail that early stopping consumes (config-pinned). The tail
# is withheld from the fit only, never from scoring.
VALIDATION_TAIL_FRACTION = float(get_config()["training"]["validation_tail_fraction"])

# The one declared headline row set. Fitting, model scoring, and the HAR benchmark
# all consume this same floor-retained set; the downstream with/without decomposition
# reads the per-fold masks and counts emitted below.
HEADLINE_ROW_SET = "floor-retained rows (raw RV strictly above RV_FLOOR)"


def floor_retained_mask(values: np.ndarray) -> np.ndarray:
	"""The single floor policy: True where a raw target is strictly above RV_FLOOR.

	At-or-below-floor targets are flat-window artifacts: their log label is undefined
	and their QLIKE contribution is an extreme artifact of the floor. The same
	predicate filters fold training, model scoring, and the HAR benchmark, so no row is
	fitted or scored under a different rule. The comparison is strict, so a target
	exactly at the floor is excluded and the smallest positive target is kept.
	"""
	return np.asarray(values) > RV_FLOOR


def qlike_loss(actual_log: np.ndarray, forecast_log: np.ndarray) -> float:
	"""QLIKE loss for log-sigma forecasts (Patton 2011; Hansen & Lunde (2005),
	Journal of Applied Econometrics 20(7), 873-889, doi:10.1002/jae.800; the 2005
	JAE paper is the QLIKE-origin comparison, distinct from the Hansen-Lunde (2006)
	papers).

	QLIKE(h, f) = h/f - ln(h/f) - 1 with h = exp(actual_log) and f = exp(forecast_log),
	so in log space the loss is exp(delta) - delta - 1 for delta = actual - forecast.
	This is the robust form for volatility-forecast comparison — the same ranking as
	the Hansen & Lunde (2005) loss log(f) + h/f, up to the target-only term ln(h) + 1.
	The mean comes from ``qlike_mean``, the functional the trainer also optimizes and
	evaluates.

	Signed identity: QLIKE = HL - ln(h) - 1, equivalently HL = QLIKE + ln(h) + 1 for
	the Hansen & Lunde (2005) loss HL = ln(f) + h/f. The ranking is identical; the
	absolute level differs only by the target-only constant ln(h) + 1.

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


def _fold_keys(folds_metadata: dict) -> list[str]:
	"""The payload's fold keys in fold-index order.

	The persisted payload is a schema-versioned superset, so only ``fold_*``
	keys are folds; the metadata keys (schema_version, feature_names, holdout,
	n_dates) are not. Malformed fold keys fail by name rather than surfacing as
	an opaque index error.
	"""
	fold_keys = [key for key in folds_metadata if key.startswith("fold_")]
	try:
		return sorted(fold_keys, key=lambda key: int(key.split("_", 1)[1]))
	except ValueError as exc:
		raise ValueError(
			f"fold_* metadata keys must be fold_<index>; got {sorted(fold_keys)}"
		) from exc


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
	- The configured number of folds (Phase 3 must have completed correctly)
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

	# Validate folds count against the configured single source (Phase 3
	# executed correctly). The persisted payload is a schema-versioned
	# superset, so count fold_* keys only: metadata keys (schema_version,
	# feature_names, holdout, n_dates) are not folds.
	fold_keys = _fold_keys(folds_metadata)
	n_folds = cross_validator.N_FOLDS
	if len(fold_keys) != n_folds:
		raise ValueError(f"Expected {n_folds} folds, got {len(fold_keys)}")

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
	for fold_key in fold_keys:
		fold_idx = int(fold_key.split("_", 1)[1])
		fold_data = folds_metadata[fold_key]
		validate_fold_minima(fold_idx, fold_data["train_indices"], fold_data["test_indices"])

	logger.info(f"Pre-training validation passed")
	logger.info(f"Target statistics: range [{y.min():.6f}, {y.max():.6f}], mean {y.mean():.6f}, std {y.std():.6f}")


def _model_training_indices(fold_data: dict, y: np.ndarray) -> np.ndarray:
	"""Training row indices for one fold, with floor-artifact rows removed.

	The exact-zero Yang-Zhang windows are flat-price artifacts: their log target is
	undefined (floored for finiteness). A QLIKE-trained model that fits their
	extreme labels under-forecasts similar test rows by ~20 log-units, where the
	exponential term explodes. This is the training side of the single floor policy;
	the same ``floor_retained_mask`` excludes those rows from scoring and the HAR
	benchmark too.
	"""
	train_indices = np.array(fold_data["train_indices"])
	return train_indices[floor_retained_mask(y[train_indices])]


def inner_validation_split(
	fold_idx: int,
	train_indices: np.ndarray,
	date_pos: np.ndarray,
	y: np.ndarray,
	tail_fraction: float = VALIDATION_TAIL_FRACTION,
) -> tuple[np.ndarray, np.ndarray, int]:
	"""Split one fold's floor-retained training rows into fit and a purged tail.

	The inner-validation tail is the temporally last ``tail_fraction`` of the
	fold's floor-retained training rows: rows are ordered by their trading-date
	position (ties keep row order, so the split never shuffles), and the tail
	size is the nearest integer to the fraction, at least one row. Input rows
	are stored ticker-major, so the date ordering is what makes the tail
	temporal; the stored row order is not.

	The fit set is the remainder after removing the tail and purging every fit
	row whose label window reaches the tail: kept fit rows satisfy
	``date_pos + LABEL_HORIZON_DAYS < first_tail_pos``, the same one-sided rule
	as the outer train/test purge. The last kept fit date therefore sits at
	least LABEL_HORIZON_DAYS trading dates before the tail start, and no fit
	row's 5-day label window overlaps the tail. Both sides are floor-retained:
	the split refuses at-or-below-floor input rows, and callers pass
	``_model_training_indices`` output. The tail is strictly inside
	fold-train, so it can never intersect the fold's test block or the
	reserved holdout window beyond it.

	Args:
		fold_idx: Fold index (0-based), used in error messages.
		train_indices: Floor-retained fold-train row indices.
		date_pos: Per-row position in the sorted unique trading dates.
		y: Raw target array (floor-retained rows are required on both sides).
		tail_fraction: Tail fraction of the floor-retained fold-train rows.

	Returns:
		(fit_indices, tail_indices, purge_count), both index arrays ascending.
	"""
	retained = np.asarray(train_indices)
	if retained.size == 0:
		raise ValueError(
			f"Fold {fold_idx}: no floor-retained training rows for an "
			f"inner-validation tail"
		)
	if not 0.0 < tail_fraction < 1.0:
		raise ValueError(
			f"Fold {fold_idx}: validation tail fraction must be in (0, 1), "
			f"got {tail_fraction}"
		)
	below_floor = int(np.count_nonzero(~floor_retained_mask(y[retained])))
	if below_floor:
		raise ValueError(
			f"Fold {fold_idx}: inner-validation split requires floor-retained "
			f"training rows; {below_floor} are at or below the floor {RV_FLOOR:g}"
		)
	n_tail = max(1, int(np.floor(retained.size * tail_fraction + 0.5)))
	if n_tail >= retained.size:
		raise ValueError(
			f"Fold {fold_idx}: {retained.size} floor-retained training rows cannot "
			f"hold a {tail_fraction:.0%} inner-validation tail of {n_tail} rows "
			f"plus a fit set"
		)

	temporal = retained[np.lexsort((retained, date_pos[retained]))]
	tail_indices = temporal[-n_tail:]
	fit_candidates = temporal[:-n_tail]
	tail_start_pos = int(date_pos[tail_indices].min())
	purge_mask = date_pos[fit_candidates] + LABEL_HORIZON_DAYS >= tail_start_pos
	fit_indices = fit_candidates[~purge_mask]
	if fit_indices.size == 0:
		raise ValueError(
			f"Fold {fold_idx}: the {LABEL_HORIZON_DAYS}-date label-horizon purge "
			f"leaves no fit rows before the inner-validation tail "
			f"({int(purge_mask.sum())} of {fit_candidates.size} candidates purged); "
			f"the fold is too short for a purged tail"
		)
	return np.sort(fit_indices), np.sort(tail_indices), int(purge_mask.sum())


def fold_checkpoint_path(fold_idx: int, run_id: str) -> Path:
	"""Run-scoped checkpoint location: ``checkpoints/<run_id>/fold_<i>.pkl``.

	The run id scopes every run's models into their own directory, so a new run
	never overwrites (or is mistaken for) an older run's checkpoints. The path
	is recorded repo-relative in the fold summary, the predictions, and the run
	manifest; readers resolve it from those records instead of guessing a name.
	"""
	return ARTIFACTS_DIR / "checkpoints" / run_id / f"fold_{fold_idx}.pkl"


def fold_params_path(fold_idx: int, run_id: str) -> Path:
	"""Run-scoped parameter sidecar: ``checkpoints/<run_id>/fold_<i>_params.json``.

	The sidecar sits beside the fold's checkpoint and holds its full
	``get_params()`` set, the tool versions, and the best-effort git sha, so the
	training recipe stays recoverable without loading the pickle.
	"""
	return ARTIFACTS_DIR / "checkpoints" / run_id / f"fold_{fold_idx}_params.json"


# Version keys recorded in every fold's params sidecar: the interpreter plus
# the model stack whose behavior the recorded hyperparameters describe.
SIDECAR_VERSION_KEYS = ("python", "xgboost", "numpy", "pandas", "scikit-learn")


def _sidecar_value(value: object) -> object:
	"""Project one ``get_params()`` value onto strict-JSON data.

	Callables (the QLIKE objective and eval metric) become their qualified
	name; a non-finite float (xgboost's default ``missing=nan``) becomes
	``null``, the only strict-JSON spelling available. Every other value the
	estimator reports is already JSON-native.
	"""
	if callable(value):
		return f"{value.__module__}.{value.__qualname__}"
	if isinstance(value, float) and not math.isfinite(value):
		return None
	return value


def _fold_params_payload(model: object) -> dict:
	"""The sidecar document: full params, tool versions, best-effort git sha."""
	versions = environment_versions()
	return {
		"params": {
			key: _sidecar_value(value)
			for key, value in model.get_params().items()
		},
		"environment": {key: versions.get(key) for key in SIDECAR_VERSION_KEYS},
		"git_sha": git_sha(),
	}


def write_fold_params_atomically(model: object, params_path: Path) -> None:
	"""Write the fold model's parameter sidecar via a sibling ``*.tmp`` + rename.

	The document is the model's full ``get_params()`` (callables by qualified
	name, non-finite floats as ``null``, so the file is strict JSON with
	``allow_nan=False``), plus the tool versions and the git sha. Like the
	checkpoint, the file appears only through the atomic rename: a failure
	anywhere — including in the rename itself — removes the temporary file, so
	no ``*.tmp`` residue is left behind and a previous sidecar stays intact.
	The pkl remains the training artifact; this sidecar is the parameter
	record, not a loadable model.
	"""
	params_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = params_path.with_name(params_path.name + ".tmp")
	try:
		with open(tmp_path, "w", encoding="utf-8") as handle:
			json.dump(
				_fold_params_payload(model),
				handle,
				indent=2,
				sort_keys=True,
				allow_nan=False,
			)
			handle.write("\n")
		os.replace(tmp_path, params_path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()


def write_checkpoint_atomically(model: object, model_path: Path) -> None:
	"""Write the fold model via a sibling ``*.tmp`` file + ``os.replace``.

	A reader can never observe a partially written checkpoint: the final path
	appears only through the atomic rename. Any failure — including a failure
	in the rename itself — removes the temporary file, so no ``*.tmp`` residue
	is left behind and a previous checkpoint at the target path stays intact.
	The payload is unchanged: ``joblib.dump`` of the trained ``XGBRegressor``.
	"""
	model_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = model_path.with_name(model_path.name + ".tmp")
	try:
		joblib.dump(model, tmp_path)
		os.replace(tmp_path, model_path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()


def _train_fold_model(
	fold_idx: int,
	run_id: str,
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray | None,
	y_val: np.ndarray | None,
	X_test: np.ndarray,
	y_test: np.ndarray,
	fold_data: dict,
	row_policy: dict | None = None,
	inner_purge_count: int = 0,
) -> tuple:
	"""
	Train one XGBoost model per fold on the log target. No temporal leakage: model
	trained only on fold_i_train, evaluated on the floor-retained fold_i_test rows
	(which are strictly future to train dates).

	``X_val``/``y_val`` are the fold's purged inner-validation tail when the
	phase provides one: the trainer early-stops on validation QLIKE, and the
	model's ``best_iteration`` plus per-round QLIKE history are persisted. With
	``None`` (direct callers) the fit runs to the configured round count and
	both persistence keys stay ``None``.

	Args:
		fold_idx: Fold index (0-based)
		run_id: Run id of the owning run; scopes the checkpoint directory
		X_train: Inner-fit features from fold_i_train
		y_train: Inner-fit log-target from fold_i_train
		X_val: Purged inner-validation tail features (or None)
		y_val: Purged inner-validation tail log-target (or None)
		X_test: Floor-retained test features from fold_i_test
		y_test: Floor-retained test log-target from fold_i_test
		fold_data: Fold metadata (dates, counts)
		row_policy: Per-fold floor mask, indices, counts, and headline declaration
		inner_purge_count: Fit rows removed by the inner label-horizon purge

	Returns:
		(model, predictions_dict)
	"""
	fold_start = time.time()
	tail_note = (
		f", early stopping on {len(X_val):,} inner-validation rows"
		if X_val is not None
		else ""
	)
	logger.info(
		f"Fold {fold_idx}: Training model on {len(X_train):,} samples{tail_note}, "
		f"testing on {len(X_test):,} samples"
	)
	logger.info(f"  Date range: {fold_data['train_date_min']} to {fold_data['test_date_max']}")

	# Train model on the fold's inner-fit rows only (no leakage); the purged
	# tail is the early-stopping set, never a fitting set.
	try:
		train_start = time.time()
		model, label_mapping = train_xgboost_model(
			X_train, y_train, X_val=X_val, y_val=y_val
		)
		train_time = time.time() - train_start
	except Exception as e:
		logger.error(f"Fold {fold_idx} model training failed: {e}")
		raise

	# Early stopping persists the best iteration and the per-round validation
	# QLIKE history; both stay None when no tail was passed. The QLIKE key is
	# the metric function's name, which xgboost derives from the callable.
	best_iteration = getattr(model, "best_iteration", None)
	eval_history = None
	if best_iteration is not None:
		best_iteration = int(best_iteration)
		validation_history = model.evals_result().get("validation_0", {})
		qlike_scores = validation_history.get(qlike_eval_metric.__name__)
		if qlike_scores is None:
			raise RuntimeError(
				f"Fold {fold_idx}: model reports best_iteration={best_iteration} "
				f"but no validation QLIKE history "
				f"({qlike_eval_metric.__name__})"
			)
		eval_history = {"qlike": [float(score) for score in qlike_scores]}

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
	if best_iteration is not None:
		logger.info(
			f"    └─ Early stopping: best_iteration={best_iteration} of "
			f"{model.n_estimators} configured rounds "
			f"({len(eval_history['qlike'])} rounds evaluated)"
		)
	logger.info(f"    └─ Timing: {train_time:.1f}s train + {inference_time:.1f}s inference = {fold_elapsed:.1f}s total")

	# Save the model under the run-scoped checkpoint directory, atomically:
	# the recorded path is repo-relative in the fold summary/predictions and in
	# the run manifest's artifact list.
	model_path = fold_checkpoint_path(fold_idx, run_id)
	write_checkpoint_atomically(model, model_path)
	if not model_path.exists():
		raise RuntimeError(f"Failed to save fold {fold_idx} model to {model_path}")
	# Record the checkpoint hash at write time: the run manifest is finalized
	# only at the end of the run, so the registry is what same-run readers
	# verify against.
	record_artifact(model_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)
	logger.info(f"  Saved fold {fold_idx} model to {model_path}")

	# Save the full parameter set beside the checkpoint: the pkl is the training
	# artifact, while xgboost's native json/ubj formats are prediction-only and
	# do not round-trip get_params().
	params_path = fold_params_path(fold_idx, run_id)
	write_fold_params_atomically(model, params_path)
	record_artifact(params_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)
	logger.info(f"  Saved fold {fold_idx} parameters to {params_path}")

	predictions_dict = {
		"test_pred": y_pred,
		"test_true": y_test,
		"test_r2": test_r2,
		"test_mae": test_mae,
		"test_rmse": test_rmse,
		"test_qlike": test_qlike,
		"n_train_samples": len(X_train),
		"n_inner_val_samples": len(X_val) if X_val is not None else 0,
		"n_inner_purge_rows": int(inner_purge_count),
		"best_iteration": best_iteration,
		"eval_history": eval_history,
		"n_test_samples": len(X_test),
		"n_features": X_train.shape[1],
		"train_time_seconds": train_time,
		"inference_time_seconds": inference_time,
		"fold_time_seconds": fold_elapsed,
		"fold_model_path": to_repo_relative(model_path),
		"train_date_range": (fold_data["train_date_min"], fold_data["train_date_max"]),
		"test_date_range": (fold_data["test_date_min"], fold_data["test_date_max"]),
	}
	if row_policy is not None:
		predictions_dict.update(row_policy)

	return model, predictions_dict


def _aggregate_fold_results(fold_predictions_list: list) -> tuple[dict, dict]:
	"""
	Aggregate predictions and metrics across all folds.

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
			"n_inner_val_samples": int(pred_dict["n_inner_val_samples"]),
			"n_inner_purge_rows": int(pred_dict["n_inner_purge_rows"]),
			"best_iteration": pred_dict["best_iteration"],
			"eval_history": pred_dict["eval_history"],
			"n_test_samples": int(pred_dict["n_test_samples"]),
			"n_test_total": int(pred_dict["n_test_total"]),
			"n_test_floor_excluded": int(pred_dict["n_test_floor_excluded"]),
			"n_test_scored": int(pred_dict["n_test_scored"]),
			"n_train_floor_excluded": int(pred_dict["n_train_floor_excluded"]),
			"n_warmup_excluded": int(pred_dict["n_warmup_excluded"]),
			"headline_row_set": pred_dict["headline_row_set"],
			"n_features": int(pred_dict["n_features"]),
			"model_path": pred_dict["fold_model_path"],
			"test_r2": pred_dict["test_r2"],
			"test_mae": pred_dict["test_mae"],
			"test_rmse": pred_dict["test_rmse"],
			"test_qlike": pred_dict["test_qlike"],
			"har_r2": pred_dict["har_r2"],
			"har_qlike": pred_dict["har_qlike"],
			"clip_count": int(pred_dict["clip_count"]),
			"clip_rate": float(pred_dict["clip_rate"]),
			"train_time_seconds": pred_dict["train_time_seconds"],
			"inference_time_seconds": pred_dict["inference_time_seconds"],
			"fold_time_seconds": pred_dict["fold_time_seconds"],
		}

	return validation_predictions, fold_training_summary


def _save_training_results(
	validation_predictions: dict,
	fold_training_summary: dict,
	har_benchmark_results: dict,
	coefficient_interpretation: bool = False,
) -> None:
	"""Save training results and the HAR benchmark report to disk.

	The persisted ``coefficient_interpretation`` flag must match every fold's
	HAR fit result: the report records it at the top level and next to each
	fold's HAC fields so a reader can tell whether the standard errors are
	interpretable. The fit/report scale declarations and the comparator note
	come from the HAR benchmark module and are written next to the existing
	benchmark label.
	"""
	val_pred_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"
	summary_path = ARTIFACTS_DIR / "results" / "fold_training_summary.json"
	har_report_path = ARTIFACTS_DIR / "results" / "har_benchmark.json"

	joblib.dump(validation_predictions, val_pred_path)
	record_artifact(val_pred_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)

	# Convert fold_training_summary to JSON-serializable format. Persisted paths
	# stay repo-relative: to_repo_relative passes already-relative strings through
	# and converts any absolute path built outside _train_fold_model.
	summary_json = {}
	for fold_key, summary in fold_training_summary.items():
		summary_json[fold_key] = {
			k: v if not isinstance(v, (np.floating, np.integer)) else float(v)
			for k, v in summary.items()
		}
		if "model_path" in summary_json[fold_key]:
			summary_json[fold_key]["model_path"] = to_repo_relative(
				summary_json[fold_key]["model_path"]
			)

	with open(summary_path, "w") as f:
		json.dump(summary_json, f, indent=2, default=str)
	record_artifact(summary_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)

	# HAR benchmark report: per-fold coefficients and metrics next to the fold model's,
	# so the mandatory benchmark is auditable without reloading the predictions. The
	# forecast constraint is the variance-floor clip (forecast_floor**2, equivalently
	# forecast_log >= ln(forecast_floor)); the counters are summed over the same
	# scored rows the per-fold fits reported.
	scored_rows = sum(int(result["n_test"]) for result in har_benchmark_results.values())
	clip_count = sum(int(result["clip_count"]) for result in har_benchmark_results.values())
	clip_rate = clip_count / scored_rows if scored_rows else 0.0
	clipped_folds = [
		fold_key
		for fold_key, result in har_benchmark_results.items()
		if result["clip_count"] > 0
	]
	if clipped_folds:
		logger.warning(
			f"HAR forecast constraint bound in {len(clipped_folds)} of "
			f"{len(har_benchmark_results)} folds ({', '.join(clipped_folds)}): "
			f"{clip_count:,} of {scored_rows:,} scored rows clipped "
			f"(global clip_rate={clip_rate:.6f})"
		)

	# The persisted flag must describe the actual fits: a fold fitted with a
	# different setting would make the report's coefficient errors mislabeled.
	mismatched_folds = [
		fold_key
		for fold_key, result in har_benchmark_results.items()
		if bool(result.get("coefficient_interpretation", coefficient_interpretation))
		!= bool(coefficient_interpretation)
	]
	if mismatched_folds:
		raise ValueError(
			"HAR report flag mismatch: coefficient_interpretation="
			f"{coefficient_interpretation} but folds {', '.join(mismatched_folds)} "
			"were fitted with a different setting"
		)

	har_report = {
		"benchmark": "HAR-RV (Corsi (2009)): canonical 1/5/22 trailing variance components, OLS in variance levels",
		"target": "log(realized_volatility_5d)",
		"r2_convention": "log-target scale, fold-own-mean (sklearn r2_score)",
		"qlike": "h/f - ln(h/f) - 1 with h=exp(actual), f=exp(forecast); target in native per-day sigma units",
		"fit_scale": FIT_SCALE,
		"report_scale": REPORT_SCALE,
		"comparison_note": COMPARISON_NOTE,
		"coefficient_interpretation": bool(coefficient_interpretation),
		"hac_lag": HAC_LAG,
		"coefficient_se_note": (
			HAC_NOTE if coefficient_interpretation else NO_HAC_NOTE
		),
		"forecast_clip": (
			"variance-space clip at forecast_floor**2, equivalently "
			"forecast_log >= ln(forecast_floor), counted where applied"
		),
		"clip_count": clip_count,
		"clip_rate": clip_rate,
		"folds_with_clips": clipped_folds,
		"warmup_days": HAR_WARMUP_DAYS,
		"warmup_definition": (
			f"HAR fit excludes the first {HAR_WARMUP_DAYS} dates of each ticker's "
			"series (trailing windows incomplete); scoring rows are unchanged"
		),
		"headline_row_set": HEADLINE_ROW_SET,
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
	record_artifact(har_report_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)

	logger.info(f"Training outputs saved (HAR benchmark report: {har_report_path})")


def phase_4_training(
	coefficient_interpretation: bool = False,
	run: dict | None = None,
) -> tuple[dict, dict, list]:
	"""
	Orchestrate Phase 4: Train one fold-specific XGBoost model per configured
	fold with no temporal leakage.
	Each fold_i_model trains exclusively on fold_i_train and is evaluated on fold_i_test.
	Each fold also holds out the temporally last VALIDATION_TAIL_FRACTION of its
	floor-retained training rows as a purged inner-validation tail; the model
	early-stops on the tail's QLIKE and persists best_iteration plus the
	per-round validation history. The modeled target is log(RV) — the Yang-Zhang
	variance is closer to log-normal and the fold models minimize QLIKE on that
	log target — and every fold also fits the canonical HAR-RV benchmark on its
	training rows only, minus the per-ticker warm-up (the first HAR_WARMUP_DAYS
	dates of each ticker's series, whose trailing windows are incomplete).
	Degenerate at-or-below-floor target rows are flat-window artifacts and are
	excluded from fitting and scoring under the single floor policy; the
	retained row set is declared and counted.

	``coefficient_interpretation`` defaults to False, the pipeline setting: the
	canonical variance-level OLS fit reports no coefficient standard errors, and the
	written report says so. When True, every fold also reports ordinary and
	Newey-West HAC (lag 4, the 5-day label overlap) standard errors, persisted with
	the flag so the errors stay interpretable.

	Returns:
		validation_predictions, fold_training_summary, fold_model_paths

	``run`` is the caller's run record when the pipeline orchestrator owns the
	run (one manifest and one JSONL line per pipeline run). When omitted, the
	phase starts and finalizes its own run on success.
	"""
	standalone = run is None
	if standalone:
		run = new_run("phase_4_training", ARTIFACTS_DIR)
	run_id = run["run_id"]
	phase_started = time.monotonic()
	logger.info("Phase 4: Fold-Specific Model Training (1 Model per Fold)")
	overall_start_time = time.time()

	# Load data from Phase 2 and 3
	logger.info("Loading Phase 2-3 outputs...")
	X = np.load(ARTIFACTS_DIR / "inputs" / "X.npy", mmap_mode="r")
	y = np.load(ARTIFACTS_DIR / "inputs" / "Y.npy", mmap_mode="r")
	metadata = load_verified(
		ARTIFACTS_DIR / "inputs" / "metadata.pkl",
		registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME,
	)
	folds_metadata = load_folds_metadata()

	logger.info(f"Loaded X: {X.shape}, y: {y.shape}")
	logger.info(f"Features: {metadata['n_indicators']} indicators + {metadata['n_ticker_features']} ticker = {X.shape[1]} total")

	# Validate prerequisites
	_validate_training_prerequisites(X, y, metadata, folds_metadata)

	# Model the log of the target; the floor only touches the exact-zero Yang-Zhang
	# windows (flat 5-day prices), which would otherwise make log(0) undefined. The
	# same floor policy excludes those artifact rows from fold fitting, model
	# scoring, and the HAR benchmark (see floor_retained_mask).
	y_log = np.log(np.maximum(y, RV_FLOOR))
	floor_retained = floor_retained_mask(y)
	logger.info(
		f"Log target: log(RV) with floor {RV_FLOOR:g}; rows at or below the floor = {int(y.size - np.count_nonzero(floor_retained)):,} "
		f"(excluded from fitting and scoring; headline row set: {HEADLINE_ROW_SET})"
	)

	# Pair-verified components; the mask flags each ticker's first
	# HAR_WARMUP_DAYS dates (incomplete trailing windows) for fit exclusion only.
	har_regressors, har_warmup_mask = load_har_regressors(metadata["date"], metadata["ticker"])

	# Per-row position in the sorted unique trading dates. The input rows are
	# stored ticker-major, so the inner-validation split orders fold-train rows
	# by this position (and measures the label-horizon purge in trading dates)
	# instead of by stored row order.
	_, date_pos = np.unique(metadata["date"], return_inverse=True)

	# Train one model per fold (the persisted payload's fold keys) with strict
	# temporal separation.
	fold_keys = _fold_keys(folds_metadata)
	fold_predictions_list = []
	fold_model_paths = []
	har_benchmark_results = {}

	for fold_key in fold_keys:
		fold_idx = int(fold_key.split("_", 1)[1])
		fold_data = folds_metadata[fold_key]

		# Extract fold-specific data under the single floor policy: the same
		# floor-retained rows feed model fitting, model scoring, and the HAR
		# benchmark. The HAR fit itself still sees training rows only.
		train_indices = _model_training_indices(fold_data, y)

		# An all-floored fold keeps no training rows: every label would be the
		# floor itself, so there is nothing to fit. Name the fold and the floor
		# before the per-fold log line and any fit attempt.
		if train_indices.size == 0:
			raise ValueError(
				f"Fold {fold_idx} ({fold_key}): all {int(fold_data['n_train'])} fold-train rows "
				f"are at or below the floor {RV_FLOOR:g} (RV_FLOOR); no training rows remain"
			)

		# Inner-validation tail: the temporally last fraction of the
		# floor-retained fold-train rows, purged from the fit side by the same
		# label horizon the outer train/test split uses. Both sides stay
		# floor-retained and strictly inside fold-train.
		fit_indices, inner_val_indices, n_inner_purge_rows = inner_validation_split(
			fold_idx, train_indices, date_pos, y
		)

		test_indices = np.array(fold_data["test_indices"])
		test_floor_mask = floor_retained_mask(y[test_indices])
		test_indices_scored = test_indices[test_floor_mask]
		n_test_total = int(len(test_indices))
		n_test_scored = int(len(test_indices_scored))
		n_test_floor_excluded = n_test_total - n_test_scored
		n_train_floor_excluded = int(fold_data["n_train"]) - len(train_indices)

		# An all-floored test fold has nothing left to score; name it rather than
		# letting an empty prediction array fail later in a metric.
		if test_indices_scored.size == 0:
			raise ValueError(
				f"Fold {fold_idx} ({fold_key}): all {n_test_total} fold-test rows "
				f"are at or below the floor {RV_FLOOR:g} (RV_FLOOR); no scoring rows remain"
			)

		row_policy = {
			"test_indices": test_indices,
			"test_floor_mask": test_floor_mask,
			"test_indices_scored": test_indices_scored,
			"n_test_total": n_test_total,
			"n_test_floor_excluded": n_test_floor_excluded,
			"n_test_scored": n_test_scored,
			"n_train_floor_excluded": n_train_floor_excluded,
			"headline_row_set": HEADLINE_ROW_SET,
		}

		logger.info(
			f"  Fold {fold_idx}: excluded {n_train_floor_excluded:,} degenerate "
			f"at-or-below-floor rows from training; excluded {n_test_floor_excluded:,} "
			f"of {n_test_total:,} test rows from scoring"
		)
		logger.info(
			f"  Fold {fold_idx}: inner-validation tail {len(inner_val_indices):,} rows "
			f"({metadata['date'][inner_val_indices].min()} to "
			f"{metadata['date'][inner_val_indices].max()}); purged "
			f"{n_inner_purge_rows:,} fit rows within the {LABEL_HORIZON_DAYS}-date "
			f"label horizon; fit on {len(fit_indices):,} rows"
		)

		# The model fits the inner-fit rows and early-stops on the tail; the HAR
		# benchmark keeps the full floor-retained fold-train (unchanged path).
		y_train_full = y_log[train_indices]
		X_train = X[fit_indices]
		y_train = y_log[fit_indices]
		X_val = X[inner_val_indices]
		y_val = y_log[inner_val_indices]
		X_test = X[test_indices_scored]
		y_test = y_log[test_indices_scored]

		# Train fold model (1 model per fold, only on fold_i_train data, no leakage)
		model, predictions_dict = _train_fold_model(
			fold_idx=fold_idx,
			run_id=run_id,
			X_train=X_train,
			y_train=y_train,
			X_val=X_val,
			y_val=y_val,
			X_test=X_test,
			y_test=y_test,
			fold_data=fold_data,
			row_policy=row_policy,
			inner_purge_count=n_inner_purge_rows,
		)

		# Canonical HAR-RV benchmark on the same floor-retained rows, fitted on the
		# train rows beyond the per-ticker warm-up only and scored on the retained
		# test rows (scored unchanged even when a ticker's series starts inside the
		# block).
		har_result = fit_har_benchmark(
			train_indices=train_indices,
			test_indices=test_indices_scored,
			target_log_train=y_train_full,
			target_log_test=y_test,
			regressors=har_regressors,
			forecast_floor=RV_FLOOR,
			warmup_mask=har_warmup_mask,
			fold_key=fold_key,
			coefficient_interpretation=coefficient_interpretation,
		)

		# The warm-up filter applies to the fit only: a late index entrant's first
		# dates can fall inside a test block, and those scoring rows are scored
		# unchanged (their components keep the trailing-only math). The count
		# cross-check keeps the scoring set identical to the fold model's.
		if har_result["n_test"] != n_test_scored:
			raise ValueError(
				f"Fold {fold_idx} ({fold_key}): the HAR benchmark scored "
				f"{har_result['n_test']} rows but the fold model scored "
				f"{n_test_scored}; the warm-up mask must not filter scoring rows"
			)

		har_result.update(
			{
				"n_test_total": n_test_total,
				"n_test_floor_excluded": n_test_floor_excluded,
				"n_test_scored": n_test_scored,
				"n_train_floor_excluded": n_train_floor_excluded,
				"headline_row_set": HEADLINE_ROW_SET,
			}
		)
		har_result["har_qlike"] = qlike_loss(y_test, har_result.pop("forecast_log"))
		predictions_dict["har_r2"] = har_result["har_r2"]
		predictions_dict["har_qlike"] = har_result["har_qlike"]
		predictions_dict["n_warmup_excluded"] = har_result["n_warmup_excluded"]
		predictions_dict["clip_count"] = har_result["clip_count"]
		predictions_dict["clip_rate"] = har_result["clip_rate"]
		har_benchmark_results[fold_key] = har_result

		logger.info(
			f"    └─ HAR benchmark: R²={har_result['har_r2']:.4f}, "
			f"QLIKE={har_result['har_qlike']:.6f} "
			f"(fit {har_result['n_train']:,} rows; "
			f"{har_result['n_warmup_excluded']:,} warm-up rows of "
			f"{HAR_WARMUP_DAYS} dates/ticker excluded; "
			f"{har_result['n_warmup_test_rows']:,} warm-up rows scored unchanged; "
			f"clip_count={har_result['clip_count']:,} of "
			f"{har_result['n_test']:,} scored rows, "
			f"clip_rate={har_result['clip_rate']:.6f})"
		)

		record_fold(run, fold_key, "completed", predictions_dict["fold_time_seconds"])
		fold_predictions_list.append(predictions_dict)
		fold_model_paths.append(predictions_dict["fold_model_path"])

	# Aggregate predictions and metrics across all folds
	validation_predictions, fold_training_summary = _aggregate_fold_results(fold_predictions_list)

	# Add feature breakdown (indicators vs ticker one-hot) from metadata
	for fold_key in fold_training_summary:
		fold_training_summary[fold_key]["n_features_indicators"] = int(metadata["n_indicators"])
		fold_training_summary[fold_key]["n_features_ticker"] = int(metadata["n_ticker_features"])

	# Save outputs
	_save_training_results(
		validation_predictions,
		fold_training_summary,
		har_benchmark_results,
		coefficient_interpretation=coefficient_interpretation,
	)
	# One checkpoint and one parameter sidecar per trained fold; the indices
	# come from the folds this run actually trained.
	fold_indices = sorted(
		int(fold_key.split("_", 1)[1]) for fold_key in fold_training_summary
	)
	record_artifacts(
		run,
		[fold_checkpoint_path(fold_idx, run_id) for fold_idx in fold_indices]
		+ [fold_params_path(fold_idx, run_id) for fold_idx in fold_indices]
		+ [
			ARTIFACTS_DIR / "results" / "validation_predictions.pkl",
			ARTIFACTS_DIR / "results" / "fold_training_summary.json",
			ARTIFACTS_DIR / "results" / "har_benchmark.json",
		],
	)

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

		logger.info(f"Phase 4 complete: {len(fold_training_summary)} fold model(s), {total_test_samples:,} test samples total, {total_time:.1f}s ({total_train_time:.1f}s train + {total_inference_time:.1f}s inference)")
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
		logger.info(f"Phase 4 complete: {len(fold_training_summary)} fold model(s), {total_time:.1f}s (no valid metrics)")

	if standalone:
		record_phase(
			run,
			"phase_4_training",
			"completed",
			time.monotonic() - phase_started,
		)
		finish_run(run, status="completed")

	return validation_predictions, fold_training_summary, fold_model_paths
