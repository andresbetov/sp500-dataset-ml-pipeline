from __future__ import annotations

import logging

from phase_2_feature_selection import phase_2_feature_selection
from phase_3_cross_validation import phase_3_cross_validation
from utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Orchestrate ML pipeline: Phase 2 → Phase 3."""
    setup_logging()
    logger.info("=" * 80)
    logger.info("ML PIPELINE - PHASE 2 & 3")
    logger.info("=" * 80)
    
    try:
        # Phase 2: Feature selection and encoding
        logger.info("\n[PHASE 2] Feature Selection & Encoding")
        X, y, metadata, encoder = phase_2_feature_selection()
        logger.info(f"✓ Phase 2 complete: X {X.shape}, y {y.shape}\n")
        
        # Phase 3: Cross-validation setup
        logger.info("[PHASE 3] Temporal Cross-Validation Setup")
        folds = phase_3_cross_validation()
        logger.info(f"✓ Phase 3 complete: {len(folds)} folds created\n")
        
        logger.info("=" * 80)
        logger.info("✓ PHASES 2-3 COMPLETE")
        logger.info(f"  Ready for Phase 4: Model Training")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
