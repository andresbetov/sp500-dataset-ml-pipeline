"""Orchestrator for generating all model visualization plots."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from packages.ml.model.visualization.data_loader import (
    extract_metric_stats,
    extract_qlike_stats,
    extract_sample_stats,
    extract_training_time_stats,
    load_fold_training_summary,
)
from packages.ml.model.visualization.metrics_visualizer import generate_metrics_visualizations
from packages.ml.model.visualization.plotting import (
    collect_plot_statuses,
    write_plot_manifest,
)
from packages.ml.model.visualization.temporal_analysis import generate_temporal_visualizations
from packages.ml.model.visualization.performance_comparison import generate_performance_comparisons
from packages.ml.model.visualization.training_diagnostics import generate_training_diagnostics, load_validation_predictions

logger = logging.getLogger(__name__)


def extract_summary_stats(summary_df: pd.DataFrame) -> dict:
    """Build the manifest stats block from the loaded fold summary.

    The QLIKE block is the live consumer of the per-fold extractor; the other
    three extractors contribute the run-level metric, timing, and sample
    aggregates. All values are plain Python numbers (non-finite cells become
    ``null`` when the manifest is written).
    """
    return {
        "qlike": extract_qlike_stats(summary_df),
        "metrics": extract_metric_stats(summary_df),
        "training_time": extract_training_time_stats(summary_df),
        "samples": extract_sample_stats(summary_df),
    }


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
    stats = extract_summary_stats(summary_df)
    predictions_dict = load_validation_predictions(predictions_path)

    if output_dir is None:
        from packages.ml.model.utils import get_project_paths
        output_dir = get_project_paths()["outputs_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating all visualizations...")

    # Per-plot failures are recorded by the generators; a category that fails
    # before its plot loop still leaves the other categories running.
    results: dict[str, list[Path]] = {}
    categories = [
        (
            "metrics",
            lambda: generate_metrics_visualizations(summary_df, output_dir),
        ),
        (
            "temporal",
            lambda: generate_temporal_visualizations(summary_df, output_dir),
        ),
        (
            "performance",
            lambda: generate_performance_comparisons(summary_df, output_dir),
        ),
        (
            "diagnostics",
            lambda: generate_training_diagnostics(
                predictions_dict, summary_df, output_dir
            ),
        ),
    ]

    with collect_plot_statuses() as statuses:
        for category, generate_category in categories:
            try:
                results[category] = generate_category()
            except Exception as exc:
                logger.exception("Visualization category %s failed: %s", category, exc)
                statuses.append(
                    {
                        "category": category,
                        "plot": "*",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                results[category] = []

    manifest_path = write_plot_manifest(output_dir, statuses, stats=stats)
    logger.info("Wrote visualization manifest to %s", manifest_path)

    total = sum(len(paths) for paths in results.values())
    logger.info("Generated %d visualizations total across %d categories", total, len(results))

    for category, paths in results.items():
        logger.info("  %s: %d plots", category, len(paths))

    return results


def main() -> None:
    """Run visualization generation as standalone script."""
    from packages.ml.model.utils import setup_logging
    setup_logging()

    logger.info("Model Visualization — Generating all plots")
    results = generate_all_visualizations()

    total = sum(len(paths) for paths in results.values())
    logger.info("Done: %d visualizations saved to outputs directory", total)


if __name__ == "__main__":
    main()
