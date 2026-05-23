from __future__ import annotations

import logging

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


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
) -> tuple[xgb.XGBRegressor, dict]:
	"""
	Train XGBoost regression model for volatility prediction.

	Target: realized_volatility_5d (continuous volatility values)
	Objective: Minimize prediction error for future volatility

	Hyperparameter Configuration:
	─────────────────────────────
	- max_depth=8: 496 features (29 indicators + 467 ticker dummies) need deeper
	  trees to capture ticker-indicator interactions (e.g., TSLA x RSI_14).
	- min_child_weight=3: Volatility is noisy with long tails. Higher weight
	  prevents leaves from fitting outlier spikes at the cost of bias.
	- gamma=0.1: Mild split regularization to avoid fitting noise in calm periods.
	- n_estimators=1000: 2M+ training samples per fold support more iterations;
	  with learning_rate=0.05 the model sees more structure without overfitting.
	- reg_alpha=0.5: L1 regularization encourages sparsity across the 467 ticker
	  one-hot columns, forcing the model to rely more on financial indicators.
	- reg_lambda=2.0: L2 regularization controls overall complexity on noisy data.
	- colsample_bytree=0.7: With 496 features, aggressive column subsampling forces
	  diverse trees — some focus on ticker effects, others on indicator patterns.
	- subsample=0.85: Slightly more data per tree than default (0.8), justified
	  by the large dataset size reducing sampling variance.

	Args:
		X_train: Training features (n_samples, n_features)
		y_train: Training target (n_samples,) with continuous volatility values
		X_val: Optional validation features for early stopping
		y_val: Optional validation target for early stopping

	Returns:
		Trained XGBRegressor model
		Empty dict (for API compatibility with classification)
	"""
	logger.info(f"Training XGBoost Regressor on {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
	logger.info(f"Target volatility range: [{y_train.min():.6f}, {y_train.max():.6f}], mean: {y_train.mean():.6f}")

	# No label mapping needed for regression
	y_train_reg = y_train.astype(np.float64)
	label_mapping = {}

	# XGBoost regression configuration
	model = xgb.XGBRegressor(
		objective="reg:squarederror",
		random_state=42,
		verbosity=0,
		n_jobs=-1,
		tree_method="hist",
		eval_metric="rmse",

		# Tree structure
		max_depth=8,                   # Deeper trees for 496-feature interaction discovery
		min_child_weight=3.0,          # Prevents leaf overfitting on noisy vol spikes
		gamma=0.1,                     # Mild split regularization

		# Learning rate & iterations
		learning_rate=0.05,            # Conservative for stable convergence
		n_estimators=1000,             # 2M+ samples support more boosting rounds

		# Regularization
		reg_alpha=0.5,                 # L1 sparsity on 467 ticker one-hot columns
		reg_lambda=2.0,                # General regularization for noisy target

		# Subsampling
		subsample=0.85,                # Slightly more data per tree
		colsample_bytree=0.7,          # Forces diverse trees across 496 features
		colsample_bylevel=0.7,         # Consistent subsampling per level
	)

	logger.info("Training with regression hyperparameters:")
	logger.info("  - objective: reg:squarederror")
	logger.info("  - max_depth: 8 (interaction discovery)")
	logger.info("  - min_child_weight: 3.0 (outlier robustness)")
	logger.info("  - gamma: 0.1 (split regularization)")
	logger.info("  - learning_rate: 0.05, n_estimators: 1000")
	logger.info("  - reg_alpha: 0.5, reg_lambda: 2.0")
	logger.info("  - subsample: 0.85, colsample: 0.7/0.7")

	if X_val is not None and y_val is not None:
		y_val_reg = y_val.astype(np.float64)

		logger.info(f"Using temporal early stopping set: {len(X_val):,} samples")
		model.set_params(early_stopping_rounds=30)
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


