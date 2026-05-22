from __future__ import annotations

import logging

from features.feature_engineering import phase_2_feature_selection
from cross_validation.cross_validator import phase_3_cross_validation
from training.training import phase_4_training
# from evaluation.evaluation import phase_5_evaluation
from utils import ARTIFACTS_DIR, setup_logging

logger = logging.getLogger(__name__)


def _ensure_model_directories() -> None:
    """Create all model subdirectories if they don't exist."""
    subdirs = ["inputs", "folds", "checkpoints", "results", "legacy"]
    for subdir in subdirs:
        (ARTIFACTS_DIR / subdir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Orchestrate ML pipeline: Phase 2 → Phase 3 → Phase 4 (Phase 5 commented out)."""
    setup_logging()
    _ensure_model_directories()
    logger.info("ML Pipeline - Phases 2, 3, 4 (Volatility Regression)")

    try:
        # Phase 2: Feature selection and encoding
        logger.info("Phase 2: Feature Selection & Encoding")
        X, y, metadata, encoder = phase_2_feature_selection()
        logger.info(f"Phase 2 complete: X {X.shape}, y {y.shape}")
        
        # Phase 3: Cross-validation setup
        logger.info("Phase 3: Temporal Cross-Validation Setup")
        folds = phase_3_cross_validation()
        logger.info(f"Phase 3 complete: {len(folds)} folds created")

        # Phase 4: Model training
        logger.info("Phase 4: Model Training")
        validation_predictions, fold_training_summary, fold_model_paths = phase_4_training()
        logger.info(f"Phase 4 complete: {len(fold_model_paths)} models trained")

        # # Phase 5: Evaluation & Analytics
        # logger.info("Phase 5: Comprehensive Evaluation & Analytics")
        # evaluation_results = phase_5_evaluation()
        # logger.info(f"Phase 5 complete: {evaluation_results['execution_time']:.1f}s")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
