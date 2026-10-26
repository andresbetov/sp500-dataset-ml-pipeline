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
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from packages.ml.model.artifact_integrity import (
	REGISTRY_FILENAME,
	load_verified,
	record_artifact,
	verify_artifact,
)
from packages.ml.model.config import get_config
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.cross_validation.cv_utils import load_folds_metadata
from packages.ml.model.evaluation.report import (
	build_evaluation_summary,
	loss_vector_frame,
	write_evaluation_summary,
	write_loss_vectors,
)
from packages.ml.model.evaluation.stats import (
	AllFoldsInvalidError,
	quarantine_record,
	sanitize_json,
	spread,
)
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
	har_forecast_log,
	load_har_regressors,
)
from packages.ml.model.training.model_trainer import train_xgboost_model
from packages.ml.model.training.qlike_objective import qlike_eval_metric, qlike_mean
from packages.ml.model.run_log import (
	ResumeError,
	environment_versions,
	finish_run,
	git_sha,
	new_run,
	record_artifacts,
	record_fold,
	record_phase,
	resume_identity,
	selection_context,
	to_repo_relative,
	write_run_state,
	input_identity,
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

# The dual OOS-R² conventions reported per fold and pooled. Both live on the same
# scale over the same floor-retained scored rows; only the denominator differs:
# the prevailing-mean convention demeans by the scored rows' mean at the reporting
# level (the fold's own test mean per fold, the pooled test mean pooled), and the
# vs-HAR convention divides by the HAR benchmark's squared error on the same rows.
OOS_R2_SCALE = "log target"
OOS_R2_PREVAILING_MEAN_NOTE = (
	"oos_r2_prevailing_mean = 1 - SSE_model / SSE_mean on the log target: "
	"SSE_mean is the squared error of the prevailing mean over the scored rows "
	"at the reporting level (the fold's own test mean per fold, the pooled test "
	"mean pooled), the sklearn r2_score convention at that level"
)
OOS_R2_VS_HAR_NOTE = (
	"oos_r2_vs_har = 1 - SSE_model / SSE_HAR on the log target: SSE_HAR is the "
	"squared error of the HAR benchmark's per-row log forecasts (variance-level "
	"OLS mapped back to the log scale and clipped at the shared floor) over the "
	"same floor-retained scored rows"
)
OOS_R2_CONVENTION_NOTE = (
	"dual OOS-R² on the log target over the floor-retained scored rows: "
	+ OOS_R2_PREVAILING_MEAN_NOTE
	+ "; "
	+ OOS_R2_VS_HAR_NOTE
	+ "; test_r2 (model) and har_r2 (benchmark) keep the fold-own-mean sklearn "
	"r2_score convention"
)


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


def oos_r2_prevailing_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""OOS-R² against the prevailing mean of the scored rows (log target).

	``1 - SSE_model / SSE_mean`` with both sums over exactly the rows passed
	in: the fold's own test mean for a fold call, the pooled test mean for a
	pooled call. This is sklearn's ``r2_score`` convention restated as an
	explicit SSE ratio (the two agree to floating-point round-off); on a
	constant target (``SSE_mean == 0``) it follows ``r2_score``: 1.0 when the
	model reproduces the constant exactly, 0.0 otherwise, never a division by
	zero. The value is not comparable across aggregation levels: a fold value
	demeans by that fold's own test mean, the pooled value by the pooled mean.
	"""
	values = np.asarray(y_true, dtype=np.float64)
	predictions = np.asarray(y_pred, dtype=np.float64)
	if values.shape != predictions.shape:
		raise ValueError(
			f"oos_r2_prevailing_mean received {values.shape} targets and "
			f"{predictions.shape} predictions"
		)
	if values.size == 0:
		raise ValueError("oos_r2_prevailing_mean received no scored rows")
	sse_model = float(np.sum((values - predictions) ** 2))
	sse_mean = float(np.sum((values - values.mean()) ** 2))
	if sse_mean == 0.0:
		return 1.0 if sse_model == 0.0 else 0.0
	return 1.0 - sse_model / sse_mean


def oos_r2_vs_har(
	y_true: np.ndarray, y_pred: np.ndarray, har_forecast_log: np.ndarray
) -> float:
	"""OOS-R² against the HAR benchmark's per-row forecasts (log target).

	``1 - SSE_model / SSE_HAR`` over exactly the floor-retained scored rows
	the HAR forecasts were produced for: numerator and denominator are both
	squared errors of log-target forecasts, so the ratio compares the model
	and the benchmark on one scale. A zero HAR error (``SSE_HAR == 0``) leaves
	the ratio undefined and raises rather than fabricating a value.
	"""
	values = np.asarray(y_true, dtype=np.float64)
	predictions = np.asarray(y_pred, dtype=np.float64)
	har_predictions = np.asarray(har_forecast_log, dtype=np.float64)
	if not (values.shape == predictions.shape == har_predictions.shape):
		raise ValueError(
			"oos_r2_vs_har received misaligned rows: targets "
			f"{values.shape}, predictions {predictions.shape}, HAR forecasts "
			f"{har_predictions.shape}"
		)
	if values.size == 0:
		raise ValueError("oos_r2_vs_har received no scored rows")
	sse_model = float(np.sum((values - predictions) ** 2))
	sse_har = float(np.sum((values - har_predictions) ** 2))
	if sse_har == 0.0:
		raise ValueError(
			"oos_r2_vs_har is undefined: the HAR forecasts have zero squared "
			"error on these rows"
		)
	return 1.0 - sse_model / sse_har


def pooled_oos_r2(fold_predictions: dict) -> dict:
	"""The pooled dual OOS-R² report block over every fold's scored rows.

	Concatenates each fold record's ``test_true`` (log target), ``test_pred``
	(model forecasts), and ``har_forecast_log`` (HAR benchmark forecasts) in
	fold-key order and reports both conventions pooled:
	``oos_r2_prevailing_mean`` demeans by the pooled test mean (the pooled
	analogue of the per-fold fold-own mean), and ``oos_r2_vs_har`` divides the
	summed model error by the summed HAR error. The block states the scale,
	the row set, the pooled row count, and each convention's denominator, so a
	reader never has to guess which mean or which benchmark produced a value.
	A constant target is the documented degenerate case for the
	prevailing-mean convention: 1.0 when the model reproduces the constant
	exactly, 0.0 otherwise, never a division by zero (sklearn's ``r2_score``
	rule). The vs-HAR convention still raises when the HAR squared error is
	zero, because that ratio is undefined.
	"""
	fold_keys = sorted(fold_predictions)
	if not fold_keys:
		raise ValueError("pooled_oos_r2 received no folds")
	y_true = np.concatenate(
		[np.asarray(fold_predictions[key]["test_true"]) for key in fold_keys]
	)
	y_pred = np.concatenate(
		[np.asarray(fold_predictions[key]["test_pred"]) for key in fold_keys]
	)
	har_forecast = np.concatenate(
		[np.asarray(fold_predictions[key]["har_forecast_log"]) for key in fold_keys]
	)
	return {
		"oos_r2_prevailing_mean": oos_r2_prevailing_mean(y_true, y_pred),
		"oos_r2_vs_har": oos_r2_vs_har(y_true, y_pred, har_forecast),
		"scale": OOS_R2_SCALE,
		"row_set": HEADLINE_ROW_SET,
		"n_scored_rows": int(y_true.size),
		"prevailing_mean_convention": OOS_R2_PREVAILING_MEAN_NOTE,
		"vs_har_convention": OOS_R2_VS_HAR_NOTE,
	}


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


def fold_keys(folds_metadata: dict) -> list[str]:
	"""The payload's fold keys in fold-index order.

	The persisted payload is a schema-versioned superset, so only ``fold_*``
	keys are folds; the metadata keys (schema_version, feature_names, holdout,
	n_dates) are not. Malformed fold keys fail by name rather than surfacing as
	an opaque index error. Consumers that iterate folds (the training phase and
	the tuning harness) share this ordering.
	"""
	fold_key_list = [key for key in folds_metadata if key.startswith("fold_")]
	try:
		return sorted(fold_key_list, key=lambda key: int(key.split("_", 1)[1]))
	except ValueError as exc:
		raise ValueError(
			f"fold_* metadata keys must be fold_<index>; got {sorted(fold_key_list)}"
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
	fold_key_list = fold_keys(folds_metadata)
	n_folds = cross_validator.N_FOLDS
	if len(fold_key_list) != n_folds:
		raise ValueError(f"Expected {n_folds} folds, got {len(fold_key_list)}")

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
	for fold_key in fold_key_list:
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


def fold_summary_path(fold_idx: int, run_id: str) -> Path:
	"""Run-scoped per-fold summary record.

	The JSON record carries the fold's metric block plus the identity
	(``run_id``, config hash, fold-manifest hash, coefficient setting) a resume
	must match before trusting it. It is written last, after the predictions
	pickle, so its presence marks a complete fold record.
	"""
	return ARTIFACTS_DIR / "checkpoints" / run_id / f"fold_{fold_idx}_summary.json"


def fold_predictions_path(fold_idx: int, run_id: str) -> Path:
	"""Run-scoped per-fold predictions record used to rebuild final results."""
	return (
		ARTIFACTS_DIR
		/ "checkpoints"
		/ run_id
		/ f"fold_{fold_idx}_predictions.pkl"
	)


# Version keys recorded in every fold's params sidecar: the interpreter plus
# the model stack whose behavior the recorded hyperparameters describe.
SIDECAR_VERSION_KEYS = ("python", "xgboost", "numpy", "pandas", "scikit-learn")

# The per-run selection record: partial/subset runs name the fold rows the
# summary emits in a separate strict-JSON artifact, so the five-row summary
# contract for a full run is untouched.
RUN_SELECTION_FILENAME = "run_selection.json"


def sidecar_value(value: object) -> object:
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
			key: sidecar_value(value)
			for key, value in model.get_params().items()
		},
		"environment": {key: versions.get(key) for key in SIDECAR_VERSION_KEYS},
		"git_sha": git_sha(),
	}


def _write_atomically(path: Path, write_payload: Callable[[Path], None]) -> None:
	"""Write ``path`` via a sibling ``*.tmp`` + ``os.replace``.

	The parent directory is created before the temporary file is opened, so a
	dir-less target cannot fail with ``FileNotFoundError``. The final path
	appears only through the atomic rename: a failure anywhere — including in
	the rename itself — removes the temporary file, so no ``*.tmp`` residue is
	left behind and a previous file at the target path stays intact.
	"""
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_name(path.name + ".tmp")
	try:
		write_payload(tmp_path)
		os.replace(tmp_path, path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()


def write_json_atomically(
	path: Path,
	payload: dict,
	*,
	trailing_newline: bool = False,
	**dump_kwargs,
) -> None:
	"""Write one strict-JSON document through the shared atomic writer.

	The shared writer behind the fold params sidecar and the tuning harness's
	per-trial sidecars: strict JSON with ``allow_nan=False``, a sibling
	``*.tmp`` + ``os.replace``, and no partial final file on failure.
	"""
	def _dump(tmp_path: Path) -> None:
		with open(tmp_path, "w", encoding="utf-8") as handle:
			json.dump(payload, handle, **dump_kwargs)
			if trailing_newline:
				handle.write("\n")

	_write_atomically(path, _dump)


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
	write_json_atomically(
		params_path,
		_fold_params_payload(model),
		trailing_newline=True,
		indent=2,
		sort_keys=True,
		allow_nan=False,
	)


def parse_fold_selection(values) -> list[int] | None:
	"""Normalize a ``--folds`` selection to ascending unique integers.

	Accepts ``None`` (all folds), an ``int``, a comma-separated string
	(``"0,2"``), or an iterable of integers. An empty selection, a negative
	index, or a duplicate raises a named ``ValueError`` before any fit runs.
	"""
	if values is None:
		return None
	if isinstance(values, str):
		items = [item.strip() for item in values.split(",")]
		if not items or any(not item for item in items):
			raise ValueError(
				f"--folds must be a comma-separated list of fold indices, got {values!r}"
			)
	elif isinstance(values, (int, np.integer)):
		items = [values]
	else:
		items = list(values)
		if not items:
			raise ValueError("--folds must select at least one fold")

	indices: list[int] = []
	for item in items:
		try:
			index = int(item)
		except (TypeError, ValueError) as exc:
			raise ValueError(
				f"--folds entries must be integers, got {item!r}"
			) from exc
		if index < 0:
			raise ValueError(f"--folds entries must be non-negative, got {index}")
		if index in indices:
			raise ValueError(f"--folds lists fold {index} more than once")
		indices.append(index)
	return sorted(indices)


def _fold_record_identity(
	fold_idx: int,
	run_id: str,
	config_hash_value: str,
	fold_manifest_hash: str,
	coefficient_interpretation: bool,
) -> dict:
	"""The identity every complete per-fold sidecar set must match."""
	return {
		"fold": f"fold_{fold_idx}",
		"run_id": run_id,
		"config_hash": config_hash_value,
		"fold_manifest_hash": fold_manifest_hash,
		"coefficient_interpretation": bool(coefficient_interpretation),
	}


def write_fold_record_atomically(
	fold_idx: int,
	run_id: str,
	summary: dict,
	predictions: dict,
	har_result: dict,
	*,
	config_hash_value: str,
	fold_manifest_hash: str,
	coefficient_interpretation: bool,
) -> tuple[Path, Path]:
	"""Write one fold's summary record and predictions sidecar atomically.

	Each file is individually atomic (sibling ``*.tmp`` + ``os.replace``). The
	predictions pickle is written first; the summary JSON is written last and
	records the predictions file's byte size and SHA-256, so a stale summary
	left beside a newer predictions file is detected and the fold is
	recomputed rather than accepted. Both files carry the same identity block
	and both are recorded in the artifact registry.
	"""
	identity = _fold_record_identity(
		fold_idx,
		run_id,
		config_hash_value,
		fold_manifest_hash,
		coefficient_interpretation,
	)
	summary_path = fold_summary_path(fold_idx, run_id)
	predictions_path = fold_predictions_path(fold_idx, run_id)

	predictions_payload = {
		"identity": identity,
		"summary": sanitize_json(summary),
		"predictions": predictions,
		"har_result": sanitize_json(har_result),
	}
	_write_atomically(
		predictions_path,
		lambda tmp_path: joblib.dump(predictions_payload, tmp_path),
	)
	prediction_entry = record_artifact(
		predictions_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME
	)

	# Each file is individually atomic. The summary also records the
	# predictions file's size and SHA-256, so a summary left over from an
	# earlier pair cannot be accepted alongside a newer predictions file.
	write_json_atomically(
		summary_path,
		{
			"identity": identity,
			"predictions_size": prediction_entry["size"],
			"predictions_sha256": prediction_entry["sha256"],
			"summary": sanitize_json(summary),
		},
		trailing_newline=True,
		indent=2,
		sort_keys=True,
		allow_nan=False,
	)
	record_artifact(summary_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)
	return summary_path, predictions_path


REQUIRED_RESUMED_PREDICTION_KEYS = (
	"test_pred",
	"test_true",
	"test_pred_all",
	"test_indices",
	"test_floor_mask",
	"har_pred_all",
	"fold_model_path",
	"train_date_range",
	"test_date_range",
	"n_train_samples",
	"n_inner_val_samples",
	"n_inner_purge_rows",
	"best_iteration",
	"eval_history",
	"n_test_samples",
	"n_test_total",
	"n_test_floor_excluded",
	"n_test_scored",
	"n_train_floor_excluded",
	"n_warmup_excluded",
	"headline_row_set",
	"n_features",
	"test_r2",
	"test_mae",
	"test_rmse",
	"test_qlike",
	"oos_r2_prevailing_mean",
	"oos_r2_vs_har",
	"oos_r2_scale",
	"oos_r2_row_set",
	"har_r2",
	"har_qlike",
	"har_forecast_log",
	"clip_count",
	"clip_rate",
	"train_time_seconds",
	"inference_time_seconds",
	"har_time_seconds",
	"fold_time_seconds",
	"quarantine",
)
REQUIRED_RESUMED_HAR_KEYS = (
	"n_test",
	"clip_count",
	"clip_rate",
	"har_r2",
	"har_qlike",
)


def _load_complete_fold_record(
	fold_idx: int,
	run_id: str,
	*,
	config_hash_value: str,
	fold_manifest_hash: str,
	coefficient_interpretation: bool,
	artifacts_dir: Path | None = None,
) -> dict | None:
	"""Return a verified complete fold record, or ``None`` when incomplete.

	A complete record requires the checkpoint, parameter sidecar, summary
	JSON, and predictions pickle, each present and matching its artifact-registry
	size/SHA-256 entry; the summary must name the predictions file's size and
	hash and repeat its summary block; the identity block must match this run,
	config, fold manifest, and coefficient setting; and every key the resumed
	path later reads must be present. Any miss — including a torn pickle, a
	foreign object, or a tampered checkpoint — returns ``None`` so the caller
	recomputes the fold.
	"""
	root = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
	run_dir = root / "checkpoints" / run_id
	checkpoint_path = run_dir / f"fold_{fold_idx}.pkl"
	params_path = run_dir / f"fold_{fold_idx}_params.json"
	summary_path = run_dir / f"fold_{fold_idx}_summary.json"
	predictions_path = run_dir / f"fold_{fold_idx}_predictions.pkl"
	registry_path = root / REGISTRY_FILENAME

	for path in (checkpoint_path, params_path, summary_path, predictions_path):
		if not path.is_file():
			logger.info(
				"Fold %s: resume record incomplete (missing %s); recomputing",
				fold_idx,
				path.name,
			)
			return None

	try:
		for path in (checkpoint_path, params_path, summary_path):
			verify_artifact(path, registry_path=registry_path)
		prediction_entry = verify_artifact(
			predictions_path, registry_path=registry_path
		)
		record = load_verified(predictions_path, registry_path=registry_path)
		summary_document = json.loads(summary_path.read_text(encoding="utf-8"))
	except Exception as exc:
		# A corrupt pickle can surface as EOFError, UnpicklingError,
		# AttributeError, ImportError, ValueError, or a registry error; all are
		# "this record is unusable", so the fold is recomputed rather than
		# aborting the resume.
		logger.info(
			"Fold %s: resume record failed verification "
			"(%s: %s); recomputing",
			fold_idx,
			type(exc).__name__,
			exc,
		)
		return None

	expected_identity = _fold_record_identity(
		fold_idx,
		run_id,
		config_hash_value,
		fold_manifest_hash,
		coefficient_interpretation,
	)
	if not isinstance(record, dict) or not isinstance(summary_document, dict):
		return None
	if record.get("identity") != expected_identity:
		logger.info(
			"Fold %s: resume identity mismatch (sidecar %r); recomputing",
			fold_idx,
			record.get("identity"),
		)
		return None
	if summary_document.get("identity") != expected_identity:
		return None
	predictions = record.get("predictions")
	har_result = record.get("har_result")
	if not isinstance(predictions, dict) or not isinstance(har_result, dict):
		return None
	if (
		summary_document.get("predictions_sha256") != prediction_entry["sha256"]
		or summary_document.get("predictions_size") != prediction_entry["size"]
	):
		logger.info(
			"Fold %s: summary does not match the predictions record; recomputing",
			fold_idx,
		)
		return None
	missing_prediction_keys = [
		key for key in REQUIRED_RESUMED_PREDICTION_KEYS if key not in predictions
	]
	missing_har_keys = [
		key for key in REQUIRED_RESUMED_HAR_KEYS if key not in har_result
	]
	if missing_prediction_keys or missing_har_keys:
		logger.info(
			"Fold %s: resume record is missing key(s) %s; recomputing",
			fold_idx,
			sorted(missing_prediction_keys + missing_har_keys),
		)
		return None
	try:
		expected_summary = sanitize_json(_fold_summary_from_prediction(predictions))
		summary_matches = (
			record.get("summary") == expected_summary
			and summary_document.get("summary") == expected_summary
		)
	except Exception as exc:
		logger.info(
			"Fold %s: persisted summary block cannot be rebuilt from the "
			"predictions record (%s: %s); recomputing",
			fold_idx,
			type(exc).__name__,
			exc,
		)
		return None
	if not summary_matches:
		logger.info(
			"Fold %s: persisted summary block disagrees with predictions; "
			"recomputing",
			fold_idx,
		)
		return None
	return record


def write_checkpoint_atomically(model: object, model_path: Path) -> None:
	"""Write the fold model via a sibling ``*.tmp`` file + ``os.replace``.

	A reader can never observe a partially written checkpoint: the final path
	appears only through the atomic rename. Any failure — including a failure
	in the rename itself — removes the temporary file, so no ``*.tmp`` residue
	is left behind and a previous checkpoint at the target path stays intact.
	The payload is unchanged: ``joblib.dump`` of the trained ``XGBRegressor``.
	"""
	_write_atomically(model_path, lambda tmp_path: joblib.dump(model, tmp_path))


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

	The returned predictions carry the model-step timing (``train_time_seconds``,
	``inference_time_seconds``); the full fold timing — the HAR fit and the
	remaining fold work included — is recorded by the phase that calls this
	helper.

	Args:
		fold_idx: Fold index (0-based)
		run_id: Run id of the owning run; scopes the checkpoint directory
		X_train: Inner-fit features from fold_i_train
		y_train: Inner-fit log-target from fold_i_train
		X_val: Purged inner-validation tail features (or None)
		y_val: Purged inner-validation tail log-target (or None)
		X_test: Test features from fold_i_test. With ``row_policy`` this is the
			full test block (the one prediction call also covers the
			floor-excluded rows the loss vectors persist); without it, the
			caller passes the scored rows directly.
		y_test: Test log-target aligned with ``X_test``; the retained subset is
			selected by ``row_policy['test_floor_mask']``.
		fold_data: Fold metadata (dates, counts)
		row_policy: Per-fold floor mask, indices, counts, and headline declaration
		inner_purge_count: Fit rows removed by the inner label-horizon purge

	Returns:
		(model, predictions_dict); ``test_pred``/``test_true`` stay the
		floor-retained headline set, while ``test_pred_all`` carries the full
		test block's forecasts for the per-row loss vectors.
	"""
	model_start = time.time()
	tail_note = (
		f", early stopping on {len(X_val):,} inner-validation rows"
		if X_val is not None
		else ""
	)
	test_floor_mask = None
	if row_policy is not None and "test_floor_mask" in row_policy:
		test_floor_mask = np.asarray(row_policy["test_floor_mask"], dtype=bool)
		if test_floor_mask.shape != (len(X_test),):
			raise ValueError(
				f"Fold {fold_idx}: the floor mask holds {test_floor_mask.size} "
				f"entries but the test block holds {len(X_test)} rows"
			)
	n_scored = int(len(X_test)) if test_floor_mask is None else int(
		np.count_nonzero(test_floor_mask)
	)
	logger.info(
		f"Fold {fold_idx}: Training model on {len(X_train):,} samples{tail_note}, "
		f"testing on {n_scored:,} scored samples"
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

	# Generate predictions on the fold's test block in one call. The headline
	# metrics consume the floor-retained subset selected below; the full-block
	# forecasts are persisted in the loss vectors so the with/without-floor
	# decomposition can quantify the excluded rows without a second prediction
	# or a report-time refit.
	inference_start = time.time()
	y_pred_all = model.predict(X_test)
	inference_time = time.time() - inference_start
	if test_floor_mask is not None:
		y_pred = y_pred_all[test_floor_mask]
		y_test_scored = np.asarray(y_test)[test_floor_mask]
	else:
		y_pred = y_pred_all
		y_test_scored = np.asarray(y_test)

	# Compute the fold's model metrics inside the quarantine boundary: any
	# failure here — an R²/MAE/RMSE guard, the QLIKE guard, a degenerate test
	# block — records ``{"error": ...}`` for the fold and leaves every metric
	# value invalid (NaN in memory, ``null`` in the written summary) instead of
	# aborting the run. The loss functions themselves stay fail-loud:
	# ``qlike_loss`` still raises on non-finite input, and this boundary is the
	# only place that catches it. ``test_r2`` is sklearn's fold-own-mean R²
	# (1 - SSE_model/SSE_fold_mean); the dual OOS-R² keys are emitted after the
	# HAR fit, where the fold's per-row HAR forecasts exist:
	# ``oos_r2_prevailing_mean`` states the same fold-own-mean convention under
	# its honest name (pooled, it demeans by the pooled mean instead), and
	# ``oos_r2_vs_har`` is 1 - SSE_model/SSE_HAR on the same floor-retained
	# scored rows. The fold-own-mean value is not comparable to the pooled
	# value or to a non-demeaned R²_oos. r2_score on a constant target returns
	# 1.0/0.0, not NaN; a NaN it does return is an invalid metric value, not a
	# quarantine, and the global aggregation filters it with n_valid/n_total.
	quarantine = None
	try:
		test_r2 = float(r2_score(y_test_scored, y_pred))
		test_mae = float(mean_absolute_error(y_test_scored, y_pred))
		test_rmse = float(np.sqrt(mean_squared_error(y_test_scored, y_pred)))
		test_qlike = float(qlike_loss(y_test_scored, y_pred))
	except Exception as exc:
		quarantine = quarantine_record(exc)
		logger.warning(
			f"Fold {fold_idx} metric computation failed; quarantining the fold: "
			f"{quarantine['error']}"
		)
		test_r2 = test_mae = test_rmse = test_qlike = float("nan")

	model_elapsed = time.time() - model_start
	logger.info(f"  Fold {fold_idx}: R²={test_r2:.4f}, MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, QLIKE={test_qlike:.6f}")
	if best_iteration is not None:
		logger.info(
			f"    └─ Early stopping: best_iteration={best_iteration} of "
			f"{model.n_estimators} configured rounds "
			f"({len(eval_history['qlike'])} rounds evaluated)"
		)
	logger.info(f"    └─ Timing: {train_time:.1f}s train + {inference_time:.1f}s inference = {model_elapsed:.1f}s model step")

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
		# The headline scored set (floor-retained rows) and its metrics.
		"test_pred": y_pred,
		"test_true": y_test_scored,
		# The full test block's forecasts (retained + floor-excluded), the
		# per-row loss-vector source; never consumed by the headline metrics.
		"test_pred_all": y_pred_all,
		"test_r2": test_r2,
		"test_mae": test_mae,
		"test_rmse": test_rmse,
		"test_qlike": test_qlike,
		"n_train_samples": len(X_train),
		"n_inner_val_samples": len(X_val) if X_val is not None else 0,
		"n_inner_purge_rows": int(inner_purge_count),
		"best_iteration": best_iteration,
		"eval_history": eval_history,
		"n_test_samples": len(y_pred),
		"n_features": X_train.shape[1],
		"train_time_seconds": train_time,
		"inference_time_seconds": inference_time,
		# The full fold total is recorded by the phase, which also times the
		# HAR fit this helper does not see.
		"fold_model_path": to_repo_relative(model_path),
		"train_date_range": (fold_data["train_date_min"], fold_data["train_date_max"]),
		"test_date_range": (fold_data["test_date_min"], fold_data["test_date_max"]),
		# The fold's quarantine record (None when its metric computation
		# succeeded); the phase records it outside the five-row fold summary,
		# in the run record.
		"quarantine": quarantine,
	}
	if row_policy is not None:
		predictions_dict.update(row_policy)

	return model, predictions_dict


def _fold_summary_from_prediction(pred_dict: dict) -> dict:
	"""The per-fold summary row projected from a prediction record.

	Shared by fresh fits and resumed folds: both paths persist this exact
	projection, so a resumed final summary is built from the same fields as a
	full run.
	"""
	return {
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
		"oos_r2_prevailing_mean": pred_dict["oos_r2_prevailing_mean"],
		"oos_r2_vs_har": pred_dict["oos_r2_vs_har"],
		"oos_r2_scale": pred_dict["oos_r2_scale"],
		"oos_r2_row_set": pred_dict["oos_r2_row_set"],
		"har_r2": pred_dict["har_r2"],
		"har_qlike": pred_dict["har_qlike"],
		"clip_count": int(pred_dict["clip_count"]),
		"clip_rate": float(pred_dict["clip_rate"]),
		"train_time_seconds": pred_dict["train_time_seconds"],
		"inference_time_seconds": pred_dict["inference_time_seconds"],
		"har_time_seconds": pred_dict["har_time_seconds"],
		"fold_time_seconds": pred_dict["fold_time_seconds"],
	}


def _aggregate_fold_results(
	fold_predictions_list: list,
	fold_indices: list[int] | None = None,
) -> tuple[dict, dict]:
	"""
	Aggregate predictions and metrics across the folds actually produced.

	``fold_indices`` names each prediction record's fold index; when omitted
	the records are contiguous from fold 0 (the full-run and direct-caller
	case). A selected or resumed run passes its sparse fold indices so fold
	keys are never renumbered by list position.

	Returns:
		(validation_predictions, fold_training_summary)
	"""
	validation_predictions = {}
	fold_training_summary = {}

	indices = (
		list(range(len(fold_predictions_list)))
		if fold_indices is None
		else list(fold_indices)
	)
	if len(indices) != len(fold_predictions_list):
		raise ValueError(
			f"_aggregate_fold_results received {len(fold_predictions_list)} "
			f"records for {len(indices)} fold indices"
		)

	for fold_idx, pred_dict in zip(indices, fold_predictions_list):
		fold_key = f"fold_{fold_idx}"
		validation_predictions[fold_key] = pred_dict
		fold_training_summary[fold_key] = _fold_summary_from_prediction(pred_dict)

	return validation_predictions, fold_training_summary


# The metric keys the global block aggregates. Each is filtered to the folds
# that contributed a finite value; the headline key is the loss the fold models
# are trained on and the one the run cannot proceed without.
GLOBAL_METRIC_KEYS = (
	"test_r2",
	"test_mae",
	"test_rmse",
	"test_qlike",
	"har_qlike",
	"har_r2",
)
HEADLINE_METRIC_KEY = "test_qlike"


def _finite_metric(value: object) -> float | None:
	"""The finite float behind a metric value; ``None`` for NaN/inf/non-numbers."""
	if isinstance(value, bool) or not isinstance(
		value, (int, float, np.integer, np.floating)
	):
		return None
	number = float(value)
	return number if math.isfinite(number) else None


def _global_metrics(fold_training_summary: dict) -> dict:
	"""Aggregate the fold metrics over their finite values, with counts.

	Each metric is filtered independently: its mean and **sample standard
	deviation** (``ddof=1``, through the shared ``spread`` helper) are computed
	over the folds that contributed a finite value, and its ``n_valid`` /
	``n_total`` counts record how many did. A single finite value has no
	sample spread and is reported with ``std: null``. A fold with an invalid R²
	but a valid QLIKE therefore still contributes its QLIKE — the QLIKE
	summary is decoupled from R² validity. A warning names every metric that
	lost folds; the run raises ``AllFoldsInvalidError`` only when the headline
	QLIKE has ``n_valid == 0``, since a diagnostic metric may be entirely
	invalid while the run still has a headline result.
	"""
	n_total = len(fold_training_summary)
	metrics = {}
	for key in GLOBAL_METRIC_KEYS:
		finite = [
			value
			for summary in fold_training_summary.values()
			if (value := _finite_metric(summary.get(key))) is not None
		]
		metrics[key] = {
			"mean": float(np.mean(finite)) if finite else None,
			"std": spread(finite) if len(finite) > 1 else None,
			"n_valid": len(finite),
			"n_total": n_total,
		}

	headline = metrics[HEADLINE_METRIC_KEY]
	if headline["n_valid"] == 0:
		raise AllFoldsInvalidError(
			f"no fold contributed a finite {HEADLINE_METRIC_KEY} "
			f"(0/{n_total} valid): every fold's metric computation is "
			"quarantined or invalid"
		)
	invalid = [
		f"{key} {metrics[key]['n_valid']}/{n_total}"
		for key in GLOBAL_METRIC_KEYS
		if metrics[key]["n_valid"] < n_total
	]
	if invalid:
		logger.warning(
			f"Global metrics: {headline['n_valid']}/{n_total} folds contribute "
			f"a finite headline QLIKE; invalid metrics filtered per key: "
			+ ", ".join(invalid)
		)
	return {
		"n_valid_folds": headline["n_valid"],
		"n_total_folds": n_total,
		"metrics": metrics,
	}


def _format_metric(block: dict, fmt: str = ".4f") -> str:
	"""One global metric as ``mean±sample std`` (``ddof=1``).

	``n/a`` when no fold is valid; the mean alone when only one finite fold
	leaves the sample spread undefined.
	"""
	if block["mean"] is None:
		return "n/a"
	if block["std"] is None:
		return f"{block['mean']:{fmt}}"
	return f"{block['mean']:{fmt}}±{block['std']:{fmt}}"


def _format_scalar(value: float | None, fmt: str = ".6f", signed: bool = False) -> str:
	"""One headline scalar, ``n/a`` when no valid fold produced it."""
	if value is None:
		return "n/a"
	return format(value, f"+{fmt}" if signed else fmt)


def _save_training_results(
	validation_predictions: dict,
	fold_training_summary: dict,
	har_benchmark_results: dict,
	coefficient_interpretation: bool = False,
	pooled_oos_r2_report: dict | None = None,
) -> None:
	"""Save training results and the HAR benchmark report to disk.

	The persisted ``coefficient_interpretation`` flag must match every fold's
	HAR fit result: the report records it at the top level and next to each
	fold's HAC fields so a reader can tell whether the standard errors are
	interpretable. The fit/report scale declarations and the comparator note
	come from the HAR benchmark module and are written next to the existing
	benchmark label. When the caller passes a ``pooled_oos_r2_report`` (the
	training phase computes it from the persisted per-fold predictions), the
	HAR report carries it as the top-level ``pooled_oos_r2`` block with both
	conventions, their scale, their row set, and the pooled row count.

	Every artifact is written through the shared atomic writer: the results
	directory is created if absent (a dir-less target cannot fail with
	``FileNotFoundError``), the final path appears only through the rename, and
	a torn write leaves no partial final file and no ``*.tmp`` residue. The two
	JSON documents are projected through the shared strict sanitizer and
	written with ``allow_nan=False``: non-finite metric values (invalid folds)
	become ``null`` and an unencodable object fails the write instead of being
	masked by ``str()``.
	"""
	val_pred_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"
	summary_path = ARTIFACTS_DIR / "results" / "fold_training_summary.json"
	har_report_path = ARTIFACTS_DIR / "results" / "har_benchmark.json"

	_write_atomically(
		val_pred_path,
		lambda tmp_path: joblib.dump(validation_predictions, tmp_path),
	)
	record_artifact(val_pred_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)

	# Project the fold summary onto strict JSON through the shared sanitizer:
	# non-finite metric values (a quarantined/invalid fold) become null, numpy
	# scalars become JSON-native values, and any object the sanitizer cannot
	# encode fails the write instead of being masked by ``str()``. Persisted
	# paths stay repo-relative: to_repo_relative passes already-relative
	# strings through and converts any absolute path built outside
	# _train_fold_model.
	summary_json = {}
	for fold_key, summary in fold_training_summary.items():
		row = dict(summary)
		if "model_path" in row:
			row["model_path"] = to_repo_relative(row["model_path"])
		summary_json[fold_key] = sanitize_json(row)

	# ``allow_nan=False`` pairs with the sanitizer: even if a non-finite value
	# slipped past it, the write fails instead of emitting a NaN token.
	write_json_atomically(summary_path, summary_json, indent=2, allow_nan=False)
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
		"r2_convention": OOS_R2_CONVENTION_NOTE,
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
	# The pooled dual OOS-R² block: computed by the training phase from the
	# per-fold scored rows and written only when the caller passes it, so direct
	# callers of this writer (tests, sidecar tools) keep their minimal payloads.
	if pooled_oos_r2_report is not None:
		har_report["pooled_oos_r2"] = pooled_oos_r2_report
	# Same strict projection as the fold summary: the per-fold model metrics
	# copied into the report can be invalid (null) for a quarantined fold.
	write_json_atomically(
		har_report_path, sanitize_json(har_report), indent=2, allow_nan=False
	)
	record_artifact(har_report_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)

	logger.info(f"Training outputs saved (HAR benchmark report: {har_report_path})")


def phase_4_training(
	coefficient_interpretation: bool = False,
	run: dict | None = None,
	*,
	resume: bool = False,
	folds: list[int] | None = None,
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
	retained row set is declared and counted. Every fold persists the HAR
	benchmark's per-row log forecasts and emits the dual OOS-R² conventions —
	``oos_r2_prevailing_mean`` (the fold's own test mean per fold; the pooled
	test mean at the pooled level) and ``oos_r2_vs_har`` (against the HAR
	forecasts) — each with its scale and floor-retained row set stated, per
	fold and pooled.

	``coefficient_interpretation`` defaults to False, the pipeline setting: the
	canonical variance-level OLS fit reports no coefficient standard errors, and the
	written report says so. When True, every fold also reports ordinary and
	Newey-West HAC (lag 4, the 5-day label overlap) standard errors, persisted with
	the flag so the errors stay interpretable.

	One error policy covers the metric layer: a fold whose metric computation
	raises is quarantined — its record is ``{"error": ...}`` under
	``quarantined_folds`` in the run record (never a sixth row in the five-row
	fold summary), its metric values are invalid, and the loop continues. The
	global metrics filter invalid values with ``n_valid``/``n_total``, warn
	when a metric lost folds, and raise only when no fold contributes a finite
	headline QLIKE. The QLIKE summary is decoupled from R² validity: a fold
	with NaN R² but a valid QLIKE still contributes its QLIKE.

	Returns:
		validation_predictions, fold_training_summary, fold_model_paths

	``run`` is the caller's run record when the pipeline orchestrator owns the
	run (one manifest and one JSONL line per pipeline run). When omitted, the
	phase starts and finalizes its own run on success.

	``resume`` reuses the latest manifest's run id after validating its config
	hash and fold-manifest hash, and skips every selected fold whose verified
	sidecar set is complete (checkpoint + params + summary + predictions).
	``folds`` limits the run to the named fold indices. The manifest records
	``fold_selection`` (requested/trained/resumed) either way; omitting both
	arguments preserves the full-run default.
	"""
	standalone = run is None
	if standalone:
		if resume:
			identity = resume_identity(ARTIFACTS_DIR)
			run = new_run(
				"phase_4_training", ARTIFACTS_DIR, run_id=identity["run_id"]
			)
			run["resumed"] = True
			run["fold_manifest_hash"] = identity["fold_manifest_hash"]
			run["input_identity"] = identity["input_identity"]
		else:
			run = new_run("phase_4_training", ARTIFACTS_DIR)
	# The frozen tuning selection this run trains under is named in the run
	# record for provenance; the selection artifact remains the source of truth
	# for the parameters frozen under training.hyperparameters. A config
	# without a frozen selection records nothing.
	selection = selection_context()
	if selection is not None:
		run["selection"] = selection
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
	# Pin the manifest identity for this run: two generations from the same
	# metadata produce the same digest, so a run record names the exact fold
	# construction it trained under. A resume must match the recorded identity
	# exactly; a missing or mismatched hash refuses by name before any fit.
	fold_manifest_hash = cross_validator.fold_manifest_hash(folds_metadata)
	if resume:
		identity = resume_identity(
			ARTIFACTS_DIR, config_hash_value=run["config_hash"]
		)
		if identity["run_id"] != run["run_id"]:
			raise ResumeError(
				"cannot resume: run id mismatch — the run record names "
				f"{run['run_id']}, the manifest names {identity['run_id']}"
			)
		current_inputs = input_identity(ARTIFACTS_DIR, metadata)
		if identity["input_identity"] != current_inputs:
			raise ResumeError(
				"cannot resume: input identity mismatch — the run manifest "
				f"records {identity['input_identity']}, the loaded inputs hash "
				f"to {current_inputs}"
			)
		if identity["fold_manifest_hash"] != fold_manifest_hash:
			raise ResumeError(
				"cannot resume: fold manifest mismatch — the run manifest records "
				f"{identity['fold_manifest_hash']}, the current folds hash to "
				f"{fold_manifest_hash}; rebuild or resume the matching inputs"
			)
		run["fold_manifest_hash"] = fold_manifest_hash
		run["resumed"] = True
		logger.info(
			"Resuming run %s (config %s, folds %s)",
			run["run_id"],
			run["config_hash"],
			fold_manifest_hash[:12],
		)
	else:
		run["fold_manifest_hash"] = fold_manifest_hash
		run["input_identity"] = input_identity(ARTIFACTS_DIR, metadata)

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

	# Train the selected folds (all folds by default) with strict temporal
	# separation. A resume reuses only complete, verified sidecar sets; every
	# other selected fold is recomputed.
	all_fold_keys = fold_keys(folds_metadata)
	all_fold_indices = [
		int(fold_key.split("_", 1)[1]) for fold_key in all_fold_keys
	]
	selected_indices = (
		list(all_fold_indices)
		if folds is None
		else parse_fold_selection(folds)
	)
	unknown_folds = sorted(set(selected_indices) - set(all_fold_indices))
	if unknown_folds:
		raise ValueError(
			f"--folds selected unknown fold index(es) {unknown_folds}; "
			f"the persisted run has folds {all_fold_indices}"
		)
	if not selected_indices:
		raise ValueError("--folds must select at least one fold")

	complete_records: dict[int, dict] = {}
	pending_indices: list[int] = list(selected_indices)
	if resume:
		pending_indices = []
		for fold_idx in selected_indices:
			record = _load_complete_fold_record(
				fold_idx,
				run_id,
				config_hash_value=run["config_hash"],
				fold_manifest_hash=fold_manifest_hash,
				coefficient_interpretation=coefficient_interpretation,
				artifacts_dir=ARTIFACTS_DIR,
			)
			if record is None:
				pending_indices.append(fold_idx)
			else:
				complete_records[fold_idx] = record
	run["partial"] = len(selected_indices) != len(all_fold_indices)
	run["fold_selection"] = {
		"requested": list(selected_indices),
		"trained": list(pending_indices),
		"resumed": sorted(complete_records),
		"n_folds_total": len(all_fold_indices),
		"n_folds_selected": len(selected_indices),
	}
	logger.info(
		"Fold selection: requested %s, trained %s, resumed %s",
		selected_indices,
		pending_indices,
		sorted(complete_records),
	)

	# Every run directory carries its provenance README, written atomically.
	from packages.ml.model.retention import ensure_run_readme

	ensure_run_readme(
		ARTIFACTS_DIR / "checkpoints" / run_id, run=run, overwrite=True
	)
	# Persist the fold selection and identity before the first fit, so an
	# interrupted run is resumable from the manifest alone.
	write_run_state(run)

	fold_predictions_list = []
	fold_indices_list = []
	fold_model_paths = []
	har_benchmark_results = {}
	# Per-fold frames for the persisted per-row loss vectors: one row per
	# fold-test-block row (the whole block, floor-excluded rows included).
	loss_vector_frames = []
	# The dataset-level floor census the evaluation report decomposes: rows at
	# or below the floor anywhere in the input vs rows inside a test block.
	total_floor_rows = int(y.size - np.count_nonzero(floor_retained))
	# Fold key -> quarantine record ({"error": ...}) for every fold whose
	# metric computation failed; recorded in the run manifest/record, never in
	# the five-row fold summary.
	quarantined_folds = {
		f"fold_{fold_idx}": record["predictions"]["quarantine"]
		for fold_idx, record in complete_records.items()
		if record["predictions"].get("quarantine") is not None
	}

	for fold_idx in selected_indices:
		fold_key = f"fold_{fold_idx}"
		if fold_idx in complete_records:
			record = complete_records[fold_idx]
			predictions_dict = record["predictions"]
			har_benchmark_results[fold_key] = record["har_result"]
			fold_predictions_list.append(predictions_dict)
			fold_indices_list.append(fold_idx)
			fold_model_paths.append(predictions_dict["fold_model_path"])

			# Rebuild the fold's loss vectors from the persisted full-block
			# forecasts, so the evaluation report covers resumed and fresh folds
			# identically without refitting anything.
			test_indices = np.asarray(predictions_dict["test_indices"])
			test_floor_mask = np.asarray(
				predictions_dict["test_floor_mask"], dtype=bool
			)
			loss_vector_frames.append(
				loss_vector_frame(
					[
						{
							"fold": fold_key,
							"row_idx": test_indices,
							"date": metadata["date"][test_indices],
							"ticker": metadata["ticker"][test_indices],
							"y_true": y_log[test_indices],
							"y_pred_model": predictions_dict["test_pred_all"],
							"y_pred_har": predictions_dict["har_pred_all"],
							"is_floor": ~test_floor_mask,
						}
					]
				)
			)
			record_fold(
				run,
				fold_key,
				"quarantined" if fold_key in quarantined_folds else "resumed",
				0.0,
			)
			write_run_state(run)
			logger.info(
				"  Fold %s: resumed from complete verified sidecars", fold_key
			)
			continue

		fold_data = folds_metadata[fold_key]
		fold_started = time.time()

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
		# The model predicts the full test block in one call; its helper
		# selects the floor-retained headline rows through the row policy and
		# returns the full-block forecasts for the loss vectors.
		X_test = X[test_indices]
		y_test = y_log[test_indices]
		y_test_scored = y_test[test_floor_mask]

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
		har_started = time.time()
		har_result = fit_har_benchmark(
			train_indices=train_indices,
			test_indices=test_indices_scored,
			target_log_train=y_train_full,
			target_log_test=y_test_scored,
			regressors=har_regressors,
			forecast_floor=RV_FLOOR,
			warmup_mask=har_warmup_mask,
			fold_key=fold_key,
			coefficient_interpretation=coefficient_interpretation,
		)
		har_time_seconds = time.time() - har_started

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
		har_forecast_scored = har_result.pop("forecast_log")
		# The HAR benchmark's metric layer runs inside the same fold-level
		# quarantine boundary as the model metrics: a failure (a QLIKE guard on
		# the benchmark forecasts, a degenerate OOS-R² input) records the fold's
		# error and leaves every metric invalid instead of aborting the run.
		quarantine = predictions_dict["quarantine"]
		if quarantine is None:
			try:
				har_qlike = float(qlike_loss(y_test_scored, har_forecast_scored))
				oos_r2_prevailing = oos_r2_prevailing_mean(
					y_test_scored, predictions_dict["test_pred"]
				)
				oos_r2_vs_har_value = oos_r2_vs_har(
					y_test_scored, predictions_dict["test_pred"], har_forecast_scored
				)
			except Exception as exc:
				quarantine = quarantine_record(exc)
				logger.warning(
					f"Fold {fold_idx} HAR/OOS metric computation failed; "
					f"quarantining the fold: {quarantine['error']}"
				)
				har_qlike = oos_r2_prevailing = oos_r2_vs_har_value = float("nan")
		else:
			har_qlike = oos_r2_prevailing = oos_r2_vs_har_value = float("nan")
		if quarantine is not None:
			# A quarantined fold contributes no metric value anywhere: its
			# model metrics and the benchmark's are all invalid.
			predictions_dict["quarantine"] = quarantine
			har_result["har_r2"] = float("nan")
			for metric_key in ("test_r2", "test_mae", "test_rmse", "test_qlike"):
				predictions_dict[metric_key] = float("nan")
		har_result["har_qlike"] = har_qlike
		predictions_dict["har_r2"] = har_result["har_r2"]
		predictions_dict["har_qlike"] = har_qlike
		predictions_dict["n_warmup_excluded"] = har_result["n_warmup_excluded"]
		predictions_dict["clip_count"] = har_result["clip_count"]
		predictions_dict["clip_rate"] = har_result["clip_rate"]
		# Persist the HAR per-row log forecasts and emit the dual OOS-R²
		# conventions on the same floor-retained scored rows; the pooled block is
		# computed after the fold loop and written into the HAR report.
		predictions_dict["har_forecast_log"] = har_forecast_scored
		predictions_dict["oos_r2_prevailing_mean"] = oos_r2_prevailing
		predictions_dict["oos_r2_vs_har"] = oos_r2_vs_har_value
		predictions_dict["oos_r2_scale"] = OOS_R2_SCALE
		predictions_dict["oos_r2_row_set"] = HEADLINE_ROW_SET
		har_benchmark_results[fold_key] = har_result

		# The floor-excluded test rows' forecasts for the with/without-floor
		# decomposition: the model's full-block predictions above and the same
		# fitted HAR coefficients through the one benchmark forecast path. They
		# never enter the headline metrics; the loss vectors persist them so the
		# report can quantify the rows the floor policy excludes.
		floor_rows = ~test_floor_mask
		if floor_rows.any():
			har_forecast_floor, _floor_clip_count = har_forecast_log(
				har_result["coefficients"],
				har_regressors,
				test_indices[floor_rows],
				RV_FLOOR,
			)
		else:
			har_forecast_floor = np.empty(0, dtype=np.float64)
		y_pred_har_all = np.empty(n_test_total, dtype=np.float64)
		y_pred_har_all[test_floor_mask] = har_forecast_scored
		y_pred_har_all[floor_rows] = har_forecast_floor
		# Persist the full-block HAR forecasts alongside the model's so a resume
		# can rebuild this fold's loss vectors without refitting the benchmark.
		predictions_dict["har_pred_all"] = y_pred_har_all
		try:
			loss_vector_frames.append(
				loss_vector_frame(
					[
						{
							"fold": fold_key,
							"row_idx": test_indices,
							"date": metadata["date"][test_indices],
							"ticker": metadata["ticker"][test_indices],
							"y_true": y_test,
							"y_pred_model": predictions_dict["test_pred_all"],
							"y_pred_har": y_pred_har_all,
							"is_floor": floor_rows,
						}
					]
				)
			)
		except Exception as exc:
			if quarantine is None:
				raise
			logger.warning(
				f"  {fold_key}: per-row loss vector skipped for the quarantined "
				f"fold: {type(exc).__name__}: {exc}"
			)

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
		logger.info(
			f"    └─ OOS-R² (log target, {HEADLINE_ROW_SET}): "
			f"prevailing mean={predictions_dict['oos_r2_prevailing_mean']:.4f}, "
			f"vs HAR={predictions_dict['oos_r2_vs_har']:.4f}"
		)

		# The timing decomposition: the fold total encloses the model step
		# (train + inference), the HAR fit, and the remaining fold work, so the
		# recorded parts sum to the reported full fold cost.
		predictions_dict["har_time_seconds"] = har_time_seconds
		predictions_dict["fold_time_seconds"] = time.time() - fold_started

		# Persist the complete fold record (summary + predictions + HAR result).
		# Each file is individually atomic; the summary also records the
		# predictions hash so a crash between the two writes is detected on
		# resume and the fold is recomputed rather than half-accepted.
		fold_summary = _fold_summary_from_prediction(predictions_dict)
		write_fold_record_atomically(
			fold_idx,
			run_id,
			fold_summary,
			predictions_dict,
			har_result,
			config_hash_value=run["config_hash"],
			fold_manifest_hash=fold_manifest_hash,
			coefficient_interpretation=coefficient_interpretation,
		)

		if quarantine is not None:
			quarantined_folds[fold_key] = quarantine
			record_fold(
				run, fold_key, "quarantined", predictions_dict["fold_time_seconds"]
			)
		else:
			record_fold(
				run, fold_key, "completed", predictions_dict["fold_time_seconds"]
			)
		write_run_state(run)
		fold_predictions_list.append(predictions_dict)
		fold_indices_list.append(fold_idx)
		fold_model_paths.append(predictions_dict["fold_model_path"])

	# Aggregate predictions and metrics across the selected folds.
	validation_predictions, fold_training_summary = _aggregate_fold_results(
		fold_predictions_list, fold_indices_list
	)

	# Persist the explicit partial/selection marker beside the summary. A full
	# run keeps exactly five fold rows; a subset run is declared here instead
	# of pretending the missing folds were evaluated.
	run_selection_path = ARTIFACTS_DIR / "results" / RUN_SELECTION_FILENAME
	write_json_atomically(
		run_selection_path,
		{
			"schema_version": 1,
			"partial": bool(run["partial"]),
			**run["fold_selection"],
			"fold_keys": [f"fold_{idx}" for idx in selected_indices],
		},
		trailing_newline=True,
		indent=2,
		sort_keys=True,
		allow_nan=False,
	)
	record_artifact(
		run_selection_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME
	)

	# Add feature breakdown (indicators vs ticker one-hot) from metadata
	for fold_key in fold_training_summary:
		fold_training_summary[fold_key]["n_features_indicators"] = int(metadata["n_indicators"])
		fold_training_summary[fold_key]["n_features_ticker"] = int(metadata["n_ticker_features"])

	# The global metric block: every metric aggregated over the folds that
	# contributed a finite value, each with its n_valid/n_total count. The
	# aggregation warns when a metric lost folds and raises
	# AllFoldsInvalidError only when the headline QLIKE has no valid fold. The
	# block and the quarantine records live in the run record (manifest) —
	# never in the five-row fold summary.
	run["quarantined_folds"] = quarantined_folds
	global_metrics = _global_metrics(fold_training_summary)
	run["global_metrics"] = global_metrics
	metrics = global_metrics["metrics"]

	# The pooled dual OOS-R² block over every fold's floor-retained scored rows,
	# from the persisted per-fold predictions; written into the HAR report.
	pooled_oos_r2_report = pooled_oos_r2(validation_predictions)
	logger.info(
		f"Pooled OOS-R² (log target, {HEADLINE_ROW_SET}): "
		f"prevailing mean={pooled_oos_r2_report['oos_r2_prevailing_mean']:.4f}, "
		f"vs HAR={pooled_oos_r2_report['oos_r2_vs_har']:.4f} "
		f"({pooled_oos_r2_report['n_scored_rows']:,} scored rows)"
	)

	# Save outputs
	_save_training_results(
		validation_predictions,
		fold_training_summary,
		har_benchmark_results,
		coefficient_interpretation=coefficient_interpretation,
		pooled_oos_r2_report=pooled_oos_r2_report,
	)

	# The evaluation artifacts: the per-row loss vectors (one row per test-block
	# row) and the summary built only from them. Both are written atomically,
	# recorded in the artifact index, and named in the run manifest like every
	# other primary output. A fold already quarantined for its metric block is
	# excluded from the report deterministically rather than aborting the run
	# through the vector path.
	loss_vectors = pd.concat(loss_vector_frames, ignore_index=True)
	# The per-fold frames are copies of the same rows; release them before the
	# report aggregation so the run holds one vector copy, not two.
	del loss_vector_frames
	loss_vectors_path = ARTIFACTS_DIR / "results" / "per_row_losses.parquet"
	write_loss_vectors(loss_vectors, loss_vectors_path)
	record_artifact(loss_vectors_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME)

	evaluation_summary = build_evaluation_summary(
		loss_vectors,
		dual_oos_r2={
			"folds": {
				fold_key: {
					"oos_r2_prevailing_mean": record.get(
						"oos_r2_prevailing_mean"
					),
					"oos_r2_vs_har": record.get("oos_r2_vs_har"),
				}
				for fold_key, record in validation_predictions.items()
			},
			"pooled": pooled_oos_r2_report,
		},
		regime=metadata["regime"],
		headline_row_set=HEADLINE_ROW_SET,
		total_floor_rows=total_floor_rows,
		quarantined_folds=quarantined_folds,
	)
	evaluation_summary_path = ARTIFACTS_DIR / "results" / "evaluation_summary.json"
	write_evaluation_summary(evaluation_summary, evaluation_summary_path)
	record_artifact(
		evaluation_summary_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME
	)
	logger.info(
		f"Evaluation report saved: {len(loss_vectors):,} per-row loss vectors "
		f"({evaluation_summary['n_valid_folds']}/"
		f"{evaluation_summary['n_total_folds']} folds valid)"
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
			ARTIFACTS_DIR / "results" / "per_row_losses.parquet",
			ARTIFACTS_DIR / "results" / "evaluation_summary.json",
			run_selection_path,
		],
	)

	# Final summary with metrics
	total_time = time.time() - overall_start_time

	# Aggregate timing over the folds' rows (the per-fold log lines below).
	total_train_time = sum(s.get("train_time_seconds", 0) for s in fold_training_summary.values())
	total_inference_time = sum(s.get("inference_time_seconds", 0) for s in fold_training_summary.values())
	total_har_time = sum(s.get("har_time_seconds", 0) for s in fold_training_summary.values())
	total_test_samples = sum(len(p["test_pred"]) for p in fold_predictions_list)

	logger.info(f"Phase 4 complete: {len(fold_training_summary)} fold model(s), {total_test_samples:,} test samples total, {total_time:.1f}s ({total_train_time:.1f}s train + {total_inference_time:.1f}s inference + {total_har_time:.1f}s HAR)")

	# The promoted headline is the persisted aggregate, not a bare GLOBAL mean:
	# the unweighted mean of the fold metrics with the convention, counts, edge
	# definition, and single-seed caveat stated in the same lines. The separate
	# row-weighted pooled value is a diagnostic and is labelled as such.
	headline = evaluation_summary["headline"]
	aggregate = headline["aggregate"]
	pooled = headline["pooled"]
	logger.info(
		f"Headline aggregate QLIKE ({headline['convention']}; "
		f"{headline['n_valid']}/{headline['n_total']} folds valid; "
		f"{headline['row_set']}): "
		f"model={_format_scalar(aggregate['model_qlike'])} | "
		f"HAR={_format_scalar(aggregate['har_qlike'])} | "
		f"edge={_format_scalar(aggregate['edge'], signed=True)} "
		f"({headline['edge_definition']})"
	)
	logger.info(
		f"  Weighting: {headline['weighting']}. Caveat: {headline['caveat']}"
	)
	logger.info(
		f"  Pooled row-weighted diagnostic (not the headline; "
		f"{pooled['n_rows']:,} retained rows from persisted vectors): "
		f"model={_format_scalar(pooled['model_qlike'])} | "
		f"HAR={_format_scalar(pooled['har_qlike'])} | "
		f"edge={_format_scalar(pooled['edge'], signed=True)}"
	)
	logger.info(
		f"Fold-spread diagnostics (unweighted fold mean ± sample std; not the headline): "
		f"R²={_format_metric(metrics['test_r2'])}, "
		f"MAE={_format_metric(metrics['test_mae'], '.6f')}, "
		f"RMSE={_format_metric(metrics['test_rmse'], '.6f')}, "
		f"HAR R²={_format_metric(metrics['har_r2'])}"
	)

	# Log per-fold metrics; a quarantined fold is named with its error instead
	# of formatting its invalid metric values. Every fold is labelled as one of
	# the five equal-weight members, so fold_0 is never read as the headline.
	for fold_key in sorted(fold_training_summary.keys()):
		if fold_key in quarantined_folds:
			logger.warning(
				f"  {fold_key}: QUARANTINED — {quarantined_folds[fold_key]['error']}"
			)
			continue
		fold_r2 = fold_training_summary[fold_key].get("test_r2", float('nan'))
		fold_mae = fold_training_summary[fold_key].get("test_mae", float('nan'))
		fold_rmse = fold_training_summary[fold_key].get("test_rmse", float('nan'))
		fold_qlike = fold_training_summary[fold_key].get("test_qlike", float('nan'))
		fold_har_qlike = fold_training_summary[fold_key].get("har_qlike", float('nan'))
		fold_har_r2 = fold_training_summary[fold_key].get("har_r2", float('nan'))
		train_time = fold_training_summary[fold_key].get("train_time_seconds", 0)
		inference_time = fold_training_summary[fold_key].get("inference_time_seconds", 0)
		har_time = fold_training_summary[fold_key].get("har_time_seconds", 0)
		fold_time = fold_training_summary[fold_key].get("fold_time_seconds", 0)
		train_date_min = fold_training_summary[fold_key].get("train_date_min", "?")
		train_date_max = fold_training_summary[fold_key].get("train_date_max", "?")
		test_date_min = fold_training_summary[fold_key].get("test_date_min", "?")
		test_date_max = fold_training_summary[fold_key].get("test_date_max", "?")

		fold_label = headline["folds"][fold_key]["label"]
		fold_edge = headline["folds"][fold_key]["edge"]
		logger.info(
			f"  {fold_label}: model QLIKE={_format_scalar(fold_qlike)} | "
			f"HAR QLIKE={_format_scalar(fold_har_qlike)} | "
			f"edge={_format_scalar(fold_edge, signed=True)}"
		)
		logger.info(f"    └─ R²={fold_r2:.4f}, MAE={fold_mae:.6f}, RMSE={fold_rmse:.6f}")
		logger.info(f"    └─ HAR benchmark: R²={fold_har_r2:.4f}")
		logger.info(f"    └─ Train: {train_date_min} to {train_date_max} | Test: {test_date_min} to {test_date_max}")
		logger.info(f"    └─ Timing: {train_time:.1f}s train + {inference_time:.1f}s inference + {har_time:.1f}s HAR = {fold_time:.1f}s fold total")

	if standalone:
		record_phase(
			run,
			"phase_4_training",
			"completed",
			time.monotonic() - phase_started,
		)
		finish_run(run, status="completed")

	return validation_predictions, fold_training_summary, fold_model_paths
