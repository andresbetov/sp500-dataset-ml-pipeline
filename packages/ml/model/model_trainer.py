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
    y_mapped = np.array([mapping[label] for label in y], dtype=np.int32)
    return y_mapped, mapping


def _unmap_predictions(predictions: np.ndarray, reverse_mapping: dict) -> np.ndarray:
    """Map predictions from {0, 1, 2} back to {-1, 0, 1}."""
    reverse_map = {v: k for k, v in reverse_mapping.items()}
    return np.array([reverse_map[pred] for pred in predictions], dtype=np.float32)


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Train XGBoost classifier on training data.
    
    XGBoost expects classes {0, 1, 2}, but our target has {-1, 0, 1}.
    We map labels during training and unmapping during prediction.
    
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
    
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        verbosity=0,
        n_jobs=-1,  # Use all CPU cores
    )
    
    model.fit(X_train, y_train_mapped, verbose=False)
    logger.info("Model training complete")
    
    return model, label_mapping

