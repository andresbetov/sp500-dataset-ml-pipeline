from __future__ import annotations

import logging

from feature_engineering import phase_2_feature_selection
from cross_validator import phase_3_cross_validation
from utils import ARTIFACTS_DIR, setup_logging

logger = logging.getLogger(__name__)


def _ensure_model_directories() -> None:
    """Create all model subdirectories if they don't exist."""
    subdirs = ["inputs", "folds", "checkpoints", "results", "legacy"]
    for subdir in subdirs:
        (ARTIFACTS_DIR / subdir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Orchestrate ML pipeline: Phase 2 → Phase 3 → Phase 4."""
    setup_logging()
    _ensure_model_directories()
    logger.info("ML Pipeline - Phases 2, 3, & 4")
    
    try:
        # Phase 2: Feature selection and encoding
        logger.info("Phase 2: Feature Selection & Encoding")
        X, y, metadata, encoder = phase_2_feature_selection()
        logger.info(f"Phase 2 complete: X {X.shape}, y {y.shape}")
        
        # Phase 3: Cross-validation setup
        logger.info("Phase 3: Temporal Cross-Validation Setup")
        folds = phase_3_cross_validation()
        logger.info(f"Phase 3 complete: {len(folds)} folds created")

        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
