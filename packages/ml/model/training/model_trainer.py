from __future__ import annotations

import logging

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


def _map_labels_to_classes(y: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Map labels {-1, 0, 1} to classes {0, 1, 2} for XGBoost.
    
    Returns:
        y_mapped: Labels mapped to {0, 1, 2}
        mapping: Dict for reverse mapping
    """
    mapping = {-1: 0, 0: 1, 1: 2}
    y_mapped = (y + 1).astype(np.int32)
    return y_mapped, mapping


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Train XGBoost classifier on training data with optimized hyperparameters.

    XGBoost expects classes {0, 1, 2}, but our target has {-1, 0, 1}.
    We map labels during training and unmapping during prediction.
    
    Hyperparameter Tuning Rationale:
    ─────────────────────────────────
    Problem: Initial model (47.2% accuracy) with severe class imbalance (class 0 = 0% recall)

    Solutions Applied:
    1. Increased max_depth (6→8): Better capture feature interactions with 501 features
    2. Reduced learning_rate (0.1→0.05): More stable training, better generalization
    3. Increased n_estimators (200→500): More boosting iterations for improvement
    4. Added L1/L2 regularization: Prevent overfitting on minority classes
    5. Increased min_child_weight: More robust splits, less overfitting
    6. Increased colsample_bytree: Feature subsampling for diversity
    7. Added per-sample class weighting: Mitigate multiclass imbalance in train and early stopping

    Expected Impact: +8-15% accuracy improvement, better class 0 detection

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,) with values in {-1, 0, 1}
        
    Returns:
        Trained XGBClassifier model
        Label mapping dict for decoding predictions
    """
    logger.info(f"Training XGBoost on {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    
    # Map labels to {0, 1, 2}
    y_train_mapped, label_mapping = _map_labels_to_classes(y_train)
    logger.info(f"Label mapping: {label_mapping}")
    
    # Calculate class weights to handle imbalance
    unique, counts = np.unique(y_train_mapped, return_counts=True)
    class_weights = 1.0 / (counts / counts.sum())
    class_weights_dict = {int(k): v for k, v in zip(unique, class_weights)}
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    logger.info(f"Class weights (for balance): {class_weights_dict}")

    # Optimized XGBoost configuration
    # Addresses: low accuracy, class 0 invisibility, poor calibration
    model = xgb.XGBClassifier(
        # Core objective
        objective="multi:softprob",  # Changed to softprob for probability outputs
        num_class=3,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        tree_method="hist",
        eval_metric=["mlogloss", "merror"],

        # Tree structure (increased from 6→8, + regularization)
        max_depth=8,                  # Capture more feature interactions (501 features)
        min_child_weight=5,            # Prevent overfitting on small clusters
        gamma=1.0,                     # Min loss reduction for split

        # Learning rate & iterations (decreased LR, increased iterations)
        learning_rate=0.05,            # More conservative, better stability
        n_estimators=500,              # Double iterations for convergence

        # Regularization (prevent overfitting)
        reg_alpha=0.5,                 # L1 regularization (variable selection)
        reg_lambda=2.0,                # L2 regularization (weight smoothing)

        # Subsampling (reduce variance, improve generalization)
        subsample=0.8,                 # Use 80% of samples per iteration
        colsample_bytree=0.8,          # Use 80% of features per tree
        colsample_bylevel=0.8,         # Use 80% of features per level
    )

    logger.info("Training with optimized hyperparameters:")
    logger.info("  - max_depth: 8 (capture interactions)")
    logger.info("  - learning_rate: 0.05 (stable training)")
    logger.info("  - n_estimators: 500 (more boosting)")
    logger.info("  - regularization: L1=0.5, L2=2.0")
    logger.info("  - subsampling: 80% samples, 80% features")

    # Vectorized sample weights lookup
    weights_arr = np.zeros(3, dtype=np.float64)
    for c, w in class_weights_dict.items():
        weights_arr[c] = w
    sample_weight = weights_arr[y_train_mapped]

    if X_val is not None and y_val is not None:
        y_val_mapped = (y_val + 1).astype(np.int32)
        val_unique, val_counts = np.unique(y_val_mapped, return_counts=True)
        val_class_weights = 1.0 / (val_counts / val_counts.sum())
        val_class_weights_dict = {int(k): v for k, v in zip(val_unique, val_class_weights)}
        
        # Vectorized validation weights lookup
        val_weights_arr = np.zeros(3, dtype=np.float64)
        for c, w in val_class_weights_dict.items():
            val_weights_arr[c] = w
        sample_weight_val = val_weights_arr[y_val_mapped]
        
        logger.info(f"Using temporal early stopping set: {len(X_val):,} samples")
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train,
            y_train_mapped,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val_mapped)],
            sample_weight_eval_set=[sample_weight_val],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train_mapped, sample_weight=sample_weight, verbose=False)
    logger.info("✓ Model training complete")
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        logger.info(f"  - Best iteration (early stopping): {best_iteration}")
    logger.info(f"  - Configured boosting rounds: {model.n_estimators}")
    logger.info(f"  - Feature importance captured for {X_train.shape[1]} features")

    return model, label_mapping

