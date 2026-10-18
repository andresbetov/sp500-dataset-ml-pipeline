from __future__ import annotations

import logging

import joblib

from packages.ml.model.utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def load_folds_metadata() -> dict:
    """Load cross-validation fold metadata."""
    return joblib.load(ARTIFACTS_DIR / "folds" / "folds_metadata.pkl")
