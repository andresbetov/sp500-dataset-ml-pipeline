from __future__ import annotations

import logging

from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, load_verified
from packages.ml.model.utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def load_folds_metadata() -> dict:
    """Load cross-validation fold metadata (size- and sha256-verified)."""
    return load_verified(
        ARTIFACTS_DIR / "folds" / "folds_metadata.pkl",
        registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME,
    )
