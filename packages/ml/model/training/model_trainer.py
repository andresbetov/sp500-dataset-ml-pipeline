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
	Uses Huber loss (reg:huberM) for robustness to outliers (black swan volatility events).
	Shallower trees (max_depth=6) work better for regression than classification.
	Lower regularization since no class imbalance to fight.

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
		# Objective: Squared error loss (standard for regression)
		objective="reg:squarederror",
		random_state=42,
		verbosity=0,
		n_jobs=-1,
		tree_method="hist",
		eval_metric="rmse",

		# Tree structure (shallower for regression)
		max_depth=6,                   # Shallower trees work better for continuous targets
		min_child_weight=1.0,          # Standard for regression
		gamma=0.0,                     # Min loss reduction for split

		# Learning rate & iterations
		learning_rate=0.05,            # Conservative learning rate
		n_estimators=500,              # Sufficient boosting iterations

		# Regularization (lower than classification, no class imbalance)
		reg_alpha=0.1,                 # Mild L1 regularization
		reg_lambda=1.0,                # Mild L2 regularization

		# Subsampling (reduce variance)
		subsample=0.8,                 # Use 80% of samples per iteration
		colsample_bytree=0.8,          # Use 80% of features per tree
		colsample_bylevel=0.8,         # Use 80% of features per level
	)

	logger.info("Training with regression hyperparameters:")
	logger.info("  - objective: reg:squarederror (MSE loss)")
	logger.info("  - max_depth: 6 (shallow trees for regression)")
	logger.info("  - learning_rate: 0.05 (conservative)")
	logger.info("  - n_estimators: 500")
	logger.info("  - regularization: L1=0.1, L2=1.0 (mild)")

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

	logger.info("✓ Regression model training complete")
	best_iteration = getattr(model, "best_iteration", None)
	if best_iteration is not None:
		logger.info(f"  - Best iteration (early stopping): {best_iteration}")
	logger.info(f"  - Configured boosting rounds: {model.n_estimators}")
	logger.info(f"  - Feature importance captured for {X_train.shape[1]} features")

	return model, label_mapping


