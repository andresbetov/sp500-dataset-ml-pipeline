from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import xgboost as xgb
from packages.ml.model.config import get_config
from packages.ml.model.run_log import apply_thread_policy
from packages.ml.model.training.qlike_objective import (
	qlike_eval_metric,
	qlike_objective,
)

logger = logging.getLogger(__name__)

# Early stopping on the inner-validation tail: stop after this many rounds
# without a validation-QLIKE improvement. The configured n_estimators stays
# the upper bound on the number of boosting rounds.
EARLY_STOPPING_ROUNDS = 30

# The pre-freeze baseline recipe: the fallback when a config carries no
# ``training.hyperparameters`` section (a legacy or external config). The seed
# and thread count keep the pre-freeze wiring (``training.seed`` /
# ``training.n_jobs``); every other knob is the previous literal.
_FALLBACK_HYPERPARAMETERS: dict = {
	"tree_method": "hist",
	"max_depth": 6,
	"min_child_weight": 1.0,
	"gamma": 0.0,
	"learning_rate": 0.05,
	"n_estimators": 500,
	"reg_alpha": 0.1,
	"reg_lambda": 1.0,
	"subsample": 0.8,
	"colsample_bytree": 0.8,
	"colsample_bylevel": 0.8,
	"verbosity": 0,
}


def default_params() -> dict:
	"""The baseline training recipe, derived from the config at call time.

	Reads the frozen selected configuration from ``training.hyperparameters``
	and adds the QLIKE objective/eval-metric callables this module owns. The
	effective thread count is pinned by ``training.n_jobs`` (the determinism
	policy) over the selected recipe's historical value. A config without the
	hyperparameters section falls back to the pre-freeze defaults
	(:data:`_FALLBACK_HYPERPARAMETERS`, with the seed and thread count from the
	legacy ``training.seed`` / ``training.n_jobs`` keys when present), so an
	old config trains exactly as before. ``train_xgboost_model`` overlays the
	caller's ``params`` mapping on top of this recipe.
	"""
	loaded = get_config()
	training = loaded.get("training") or {}
	hyperparameters = training.get("hyperparameters")
	if not hyperparameters:
		hyperparameters = {
			**_FALLBACK_HYPERPARAMETERS,
			"random_state": training.get("seed", 42),
			"n_jobs": training.get("n_jobs", -1),
		}
	recipe = {
		**hyperparameters,
		"objective": qlike_objective,
		"eval_metric": qlike_eval_metric,
	}
	# The determinism policy pins the effective thread count over the frozen
	# selection's historical value (the selected campaign ran at -1/all cores);
	# every other recipe knob still comes from the frozen section.
	if training.get("n_jobs") is not None:
		recipe["n_jobs"] = int(training["n_jobs"])
	return recipe


def __getattr__(name: str) -> dict:
	"""Keep ``DEFAULT_PARAMS`` a module attribute that derives at call time."""
	if name == "DEFAULT_PARAMS":
		return default_params()
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _map_labels_to_classes(y: np.ndarray) -> tuple[np.ndarray, dict]:
	"""
	For regression, no label mapping needed. This function kept for API compatibility.
	Returns identity mapping (empty dict).
	"""
	return y.astype(np.float64), {}


def train_xgboost_model(
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_val: np.ndarray | None = None,
	y_val: np.ndarray | None = None,
	params: Mapping | None = None,
) -> tuple[xgb.XGBRegressor, dict]:
	"""
	Train XGBoost regression model for daily log-sigma volatility prediction.

	Target: log daily sigma (realized volatility floored before the log)
	Objective: Minimize QLIKE on the log targets, matching the reported loss

	Args:
		X_train: Training features (n_samples, n_features)
		y_train: Training target (n_samples,) in log daily sigma units
		X_val: Optional validation features for early stopping
		y_val: Optional validation target for early stopping
		params: Optional hyperparameter mapping overlaid on the config-derived
			defaults (:func:`default_params`), so tuning trials vary
			hyperparameters (learning rate, depth, min_child_weight, gamma,
			random_state, ...) without editing config or module state. An
			unknown key fails in the estimator.

	Returns:
		Trained XGBRegressor model
		Empty dict (for API compatibility with classification)
	"""
	logger.info(f"Training XGBoost Regressor on {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
	logger.info(f"Target log-sigma range: [{y_train.min():.6f}, {y_train.max():.6f}], mean: {y_train.mean():.6f}")

	# No label mapping needed for regression
	y_train_reg = y_train.astype(np.float64)
	label_mapping = {}

	# XGBoost regression configuration: the config-derived default recipe with
	# the caller's parameter mapping overlaid. The default call is that recipe.
	# The thread policy is applied first so OMP_NUM_THREADS is pinned before
	# the estimator's runtime starts a training thread pool.
	apply_thread_policy()
	model_params = default_params()
	if params:
		model_params.update(params)
	model = xgb.XGBRegressor(**model_params)

	logger.info("Training with the QLIKE objective and regression hyperparameters:")
	logger.info("  - objective: custom QLIKE gradient/Hessian, eval_metric: qlike")
	logger.info(f"  - max_depth: {model_params['max_depth']}")
	logger.info(f"  - min_child_weight: {model_params['min_child_weight']}")
	logger.info(f"  - gamma: {model_params['gamma']}")
	logger.info(f"  - learning_rate: {model_params['learning_rate']}, n_estimators: {model_params['n_estimators']}")
	logger.info(f"  - reg_alpha: {model_params['reg_alpha']}, reg_lambda: {model_params['reg_lambda']}")
	logger.info(f"  - subsample: {model_params['subsample']}, colsample: {model_params['colsample_bytree']}/{model_params['colsample_bylevel']}")

	if X_val is not None and y_val is not None:
		y_val_reg = y_val.astype(np.float64)

		logger.info(f"Using temporal early stopping set: {len(X_val):,} samples")
		model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
		model.fit(
			X_train,
			y_train_reg,
			eval_set=[(X_val, y_val_reg)],
			verbose=False,
		)
	else:
		model.fit(X_train, y_train_reg, verbose=False)

	logger.info("Regression model training complete")
	best_iteration = getattr(model, "best_iteration", None)
	if best_iteration is not None:
		logger.info(f"  - Best iteration (early stopping): {best_iteration}")
	logger.info(f"  - Configured boosting rounds: {model.n_estimators}")
	logger.info(f"  - Feature importance captured for {X_train.shape[1]} features")

	return model, label_mapping


