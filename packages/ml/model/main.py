from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Direct execution: python packages/ml/model/main.py — normalize to the
    # canonical package identity so the package-absolute imports below resolve
    # without an installation (PEP 366).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "packages.ml.model"

import numpy as np

from packages.ml.model.features.feature_engineering import phase_2_feature_selection
from packages.ml.model.cross_validation.cross_validator import phase_3_cross_validation
from packages.ml.model.training.training import phase_4_training
from packages.ml.model.visualization.visualize import generate_all_visualizations
from packages.ml.model.run_log import finish_run, new_run, record_phase
from packages.ml.model.utils import ARTIFACTS_DIR, setup_logging

logger = logging.getLogger(__name__)


def _ensure_model_directories() -> None:
    """Create all model subdirectories if they don't exist."""
    subdirs = ["inputs", "folds", "checkpoints", "results", "legacy"]
    for subdir in subdirs:
        (ARTIFACTS_DIR / subdir).mkdir(parents=True, exist_ok=True)


def main(dry_run: bool = False) -> None:
    """Orchestrate ML pipeline: Phase 2 → Phase 3 → Phase 4 → Phase 5 (Visualization).

    With ``dry_run`` the planned phases are logged and nothing else runs. A real
    run owns one run record: each phase's status and duration, the per-fold
    outcomes, and the written artifact hashes land in
    ``artifacts/run_manifest.json`` plus one appended line in
    ``artifacts/runs.jsonl``.
    """
    setup_logging()
    if dry_run:
        logger.info(
            "Dry run: planned phases — Phase 2 (feature selection), "
            "Phase 3 (temporal cross-validation), Phase 4 (model training), "
            "Phase 5 (visualization)"
        )
        return

    _ensure_model_directories()
    logger.info("ML Pipeline - Phases 2-5 (Volatility Regression + Visualization)")

    run = new_run("pipeline", ARTIFACTS_DIR)
    logger.info(
        f"Run {run['run_id']} (config {run['config_hash']}, git {run['git_sha']})"
    )

    try:
        # Phase 2: Feature selection and encoding
        phase_started = time.perf_counter()
        logger.info("Phase 2: Feature Selection & Encoding")
        X, y, metadata, encoder = phase_2_feature_selection()
        logger.info(f"Phase 2 complete: X {X.shape}, y {y.shape}")
        # Phase 4 reloads the inputs from disk; release the Phase-2 frame copies.
        del X, y, metadata, encoder
        record_phase(
            run,
            "phase_2_feature_selection",
            "completed",
            time.perf_counter() - phase_started,
        )

        # Phase 3: Cross-validation setup
        phase_started = time.perf_counter()
        logger.info("Phase 3: Temporal Cross-Validation Setup")
        folds = phase_3_cross_validation()
        logger.info(f"Phase 3 complete: {len(folds)} folds created")
        record_phase(
            run,
            "phase_3_cross_validation",
            "completed",
            time.perf_counter() - phase_started,
        )

        # Phase 4: Model training
        phase_started = time.perf_counter()
        logger.info("Phase 4: Model Training")
        validation_predictions, fold_training_summary, fold_model_paths = phase_4_training(run=run)
        logger.info(f"Phase 4 complete: {len(fold_model_paths)} models trained")
        # Quarantined folds carry NaN metrics; the global means filter them.
        # The phase's run record holds the n_valid/n_total counts and the
        # quarantine records, and it raises before returning when no fold is
        # valid.
        mean_qlike = np.nanmean([summary["test_qlike"] for summary in fold_training_summary.values()])
        mean_har_qlike = np.nanmean([summary["har_qlike"] for summary in fold_training_summary.values()])
        logger.info(f"Evaluation: mean QLIKE {mean_qlike:.6f} (HAR benchmark {mean_har_qlike:.6f})")
        record_phase(
            run,
            "phase_4_training",
            "completed",
            time.perf_counter() - phase_started,
        )

        # Phase 5: Visualization & Diagnostics
        phase_started = time.perf_counter()
        logger.info("Phase 5: Visualization & Diagnostics")
        viz_results = generate_all_visualizations()
        total_viz = sum(len(paths) for paths in viz_results.values())
        logger.info(f"Phase 5 complete: {total_viz} visualizations generated")
        record_phase(
            run,
            "phase_5_visualization",
            "completed",
            time.perf_counter() - phase_started,
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        finish_run(run, status="failed", error=f"{type(e).__name__}: {e}")
        raise

    finish_run(run, status="completed")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args != ["--dry-run"]:
        sys.exit("usage: python -m packages.ml.model.main [--dry-run]")
    main(dry_run=args == ["--dry-run"])
