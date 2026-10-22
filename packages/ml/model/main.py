from __future__ import annotations

import argparse
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

from packages.ml.model.evaluation import holdout
from packages.ml.model.features.feature_engineering import phase_2_feature_selection
from packages.ml.model.cross_validation.cross_validator import phase_3_cross_validation
from packages.ml.model.training.training import parse_fold_selection, phase_4_training
from packages.ml.model.visualization.visualize import generate_all_visualizations
from packages.ml.model.run_log import (
    ResumeError,
    finish_run,
    input_identity,
    new_run,
    record_phase,
    resume_identity,
    write_run_state,
)
from packages.ml.model.utils import ARTIFACTS_DIR, setup_logging

logger = logging.getLogger(__name__)


def _ensure_model_directories() -> None:
    """Create all model subdirectories if they don't exist."""
    subdirs = ["inputs", "folds", "checkpoints", "results", "legacy"]
    for subdir in subdirs:
        (ARTIFACTS_DIR / subdir).mkdir(parents=True, exist_ok=True)


def main(
    dry_run: bool = False,
    resume: bool = False,
    folds: list[int] | None = None,
    evaluate_holdout: bool = False,
    force_holdout: str | None = None,
) -> None:
    """Orchestrate ML pipeline: Phase 2 → Phase 3 → Phase 4 → Phase 5 (Visualization).

    With ``dry_run`` the planned phases are logged and nothing else runs. A real
    run owns one run record: each phase's status and duration, the per-fold
    outcomes, the fold selection, and the written artifact hashes land in
    ``artifacts/run_manifest.json`` plus one appended line in
    ``artifacts/runs.jsonl``.

    ``resume`` continues the latest manifest's run id after validating its
    config hash and fold-manifest hash; ``folds`` limits training to the named
    fold indices. With neither argument the full run is unchanged.

    ``evaluate_holdout`` requests the one-shot locked holdout evaluation after
    the normal phases; without it the holdout window is never read. A resumed
    run that already holds a lock records a skipped holdout instead of
    re-scoring. ``force_holdout`` is the explicit override reason for an
    already locked holdout and implies ``evaluate_holdout``.
    """
    if force_holdout is not None:
        if not isinstance(force_holdout, str) or not force_holdout.strip():
            raise ValueError(
                "--force-holdout requires a non-empty reason"
            )
        force_holdout = force_holdout.strip()
        evaluate_holdout = True
    setup_logging()
    selection = parse_fold_selection(folds) if folds is not None else None
    if dry_run:
        planned = (
            "Dry run: planned phases — Phase 2 (feature selection), "
            "Phase 3 (temporal cross-validation), Phase 4 (model training), "
            "Phase 5 (visualization)"
        )
        if selection is not None:
            planned += f"; folds {selection}"
        if resume:
            planned += "; resume the latest manifest identity"
        if evaluate_holdout:
            planned += "; locked holdout evaluation once"
            if force_holdout is not None:
                planned += f" (override: {force_holdout})"
        logger.info(planned)
        return

    resume_source = resume_identity(ARTIFACTS_DIR) if resume else None
    _ensure_model_directories()
    logger.info("ML Pipeline - Phases 2-5 (Volatility Regression + Visualization)")

    if resume:
        run = new_run(
            "pipeline", ARTIFACTS_DIR, run_id=resume_source["run_id"]
        )
        # Preserve the identity keys even if Phase 2/3 raises: a failed
        # resumed run must remain resumable by the next invocation.
        run["resumed"] = True
        run["fold_manifest_hash"] = resume_source["fold_manifest_hash"]
        run["input_identity"] = resume_source["input_identity"]
        logger.info(
            f"Resuming run {run['run_id']} (config {run['config_hash']})"
        )
    else:
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
        try:
            current_inputs = input_identity(ARTIFACTS_DIR, metadata)
        except FileNotFoundError as exc:
            # A mocked or partial Phase 2 may not have written the arrays yet;
            # Phase 4 will fail on its own load. A resumed run must refuse.
            logger.warning("Cannot record Phase-2 input identity: %s", exc)
            current_inputs = None
        if resume:
            if current_inputs is None or current_inputs != resume_source["input_identity"]:
                raise ResumeError(
                    "cannot resume: the Phase-2 inputs/dataset changed while "
                    "rebuilding them; the manifest identity records "
                    f"{resume_source['input_identity']}, the rebuilt inputs are "
                    f"{current_inputs}"
                )
        elif current_inputs is not None:
            run["input_identity"] = current_inputs
            write_run_state(run)
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
        validation_predictions, fold_training_summary, fold_model_paths = phase_4_training(
            run=run, resume=resume, folds=selection
        )
        logger.info(f"Phase 4 complete: {len(fold_model_paths)} models trained")
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

        # Locked holdout: opt-in only, after the normal pipeline is healthy.
        # The lock and its manifest record are written before the refit and
        # every metric; a resumed run with an existing lock records a skip.
        if evaluate_holdout:
            phase_started = time.perf_counter()
            logger.info("Holdout: locked one-shot evaluation")
            holdout_result = holdout.evaluate_holdout(
                run=run,
                artifacts_dir=ARTIFACTS_DIR,
                resume=resume,
                force_reason=force_holdout,
            )
            holdout_status = holdout_result["status"]
            record_phase(
                run,
                "holdout_evaluation",
                holdout_status,
                time.perf_counter() - phase_started,
            )
            if holdout_status == "skipped":
                logger.info(
                    "Holdout: already locked; the resumed run recorded a skip "
                    "instead of re-evaluating"
                )
            else:
                logger.info(
                    f"Holdout: final report written to "
                    f"{holdout_result['report_path']}"
                )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        finish_run(run, status="failed", error=f"{type(e).__name__}: {e}")
        raise

    finish_run(run, status="completed")


def build_parser() -> argparse.ArgumentParser:
    """The documented model entry-point flags."""
    parser = argparse.ArgumentParser(
        prog="python -m packages.ml.model.main",
        description="Run model Phases 2-5 (feature selection, cross-validation, "
        "training, visualization).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="log the planned phases and exit without writing anything",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the latest run manifest, skipping folds whose complete "
        "sidecars verify against its config and fold-manifest identities",
    )
    parser.add_argument(
        "--folds",
        type=parse_fold_selection,
        default=None,
        metavar="0,2",
        help="comma-separated fold indices to train (default: every fold)",
    )
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="after the normal phases, evaluate the already reserved holdout "
        "once under the written lock and produce the final report",
    )
    parser.add_argument(
        "--force-holdout",
        default=None,
        metavar="REASON",
        help="re-evaluate an already locked holdout; the non-empty reason is "
        "appended to the override record (implies --evaluate-holdout)",
    )
    return parser


def cli(argv: list[str] | None = None) -> None:
    """Parse the CLI and run the pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.force_holdout is not None and not args.force_holdout.strip():
        parser.error("--force-holdout requires a non-empty reason")
    main(
        dry_run=args.dry_run,
        resume=args.resume,
        folds=args.folds,
        evaluate_holdout=args.evaluate_holdout,
        force_holdout=args.force_holdout,
    )


if __name__ == "__main__":
    cli()
