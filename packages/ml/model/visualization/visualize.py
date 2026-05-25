"""Orchestrator for generating all model visualization plots."""

from __future__ import annotations

import logging
from pathlib import Path

from visualization.data_loader import load_fold_training_summary
from visualization.metrics_visualizer import generate_metrics_visualizations
from visualization.temporal_analysis import generate_temporal_visualizations
from visualization.performance_comparison import generate_performance_comparisons
from visualization.training_diagnostics import generate_training_diagnostics, load_validation_predictions

logger = logging.getLogger(__name__)


def generate_all_visualizations(
    summary_path: Path | None = None,
    predictions_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, list[Path]]:
    """
    Generate all visualization plots from fold training results.

    Args:
        summary_path: Path to fold_training_summary.json. If None, uses default.
        predictions_path: Path to validation_predictions.pkl. If None, uses default.
        output_dir: Directory to save figures. If None, uses default outputs_dir.

    Returns:
        Dict mapping category names to lists of generated figure paths.
    """
    summary_df = load_fold_training_summary(summary_path)
    predictions_dict = load_validation_predictions(predictions_path)

    if output_dir is None:
        from utils import get_project_paths
        output_dir = get_project_paths()["outputs_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating all visualizations...")

    results = {
        "metrics": generate_metrics_visualizations(summary_df, output_dir),
        "temporal": generate_temporal_visualizations(summary_df, output_dir),
        "performance": generate_performance_comparisons(summary_df, output_dir),
        "diagnostics": generate_training_diagnostics(predictions_dict, summary_df, output_dir),
    }

    total = sum(len(paths) for paths in results.values())
    logger.info("Generated %d visualizations total across %d categories", total, len(results))

    for category, paths in results.items():
        logger.info("  %s: %d plots", category, len(paths))

    return results


def main() -> None:
    """Run visualization generation as standalone script."""
    from utils import setup_logging
    setup_logging()

    logger.info("Model Visualization — Generating all plots")
    results = generate_all_visualizations()

    total = sum(len(paths) for paths in results.values())
    logger.info("Done: %d visualizations saved to outputs directory", total)


if __name__ == "__main__":
    main()
