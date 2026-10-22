"""Visualizations for regression metrics (R², MAE, RMSE) across folds.

The R² panel takes data-driven limits that always contain 0 and the observed
minimum/maximum, so negative scores are shown truthfully instead of being
clipped by a ``[0, max * 1.15]`` window. The heatmap colors absolute values
against the declared per-metric reference in ``HEATMAP_REFERENCE`` rather than
re-spanning every metric over the displayed folds; a constant column has no
spread and renders at the neutral ``HEATMAP_NEUTRAL_SCORE``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from packages.ml.model.visualization.data_loader import load_fold_training_summary
from packages.ml.model.visualization.plotting import (
    render_plot,
    save_figure_atomically,
)

logger = logging.getLogger(__name__)

# R² limits add this fraction of the observed span on each side. An all-zero
# input falls back to a margin of this many R² units.
R2_AXIS_MARGIN_FRACTION = 0.05

# The stored target is log sigma, so MAE/RMSE are log-point errors.
LOG_POINT_UNIT = "log points"

# The heatmap color is the value's position on an absolute per-metric
# reference, oriented so 1 is better and clipped to [0, 1]. R² is anchored at
# 0 (no skill) in the middle of [-1, 1]; MAE/RMSE are log-point errors on
# [0, 1]. These declarations are the scale contract, and the colorbar names
# them. A column whose observed spread is at or below the tolerance is flat,
# so it carries no ordering information and renders at the neutral 0.5 instead
# of a 0/0 ratio.
HEATMAP_REFERENCE: dict[str, dict[str, float | bool]] = {
    "test_r2": {"minimum": -1.0, "maximum": 1.0, "higher_is_better": True},
    "test_mae": {"minimum": 0.0, "maximum": 1.0, "higher_is_better": False},
    "test_rmse": {"minimum": 0.0, "maximum": 1.0, "higher_is_better": False},
}
HEATMAP_NEUTRAL_SCORE = 0.5
HEATMAP_COLUMN_TOLERANCE = 1e-12
HEATMAP_SCALE_LABEL = (
    "Absolute reference score (R²: [-1, 1]; MAE/RMSE: [0, 1] log points; "
    "0.5 = neutral; green = better)"
)


def _r2_axis_limits(scores: pd.Series | np.ndarray) -> tuple[float, float]:
    """Return y-limits that contain 0 and the finite observed min/max.

    The margin is ``R2_AXIS_MARGIN_FRACTION`` of the observed span on each
    side; a flat input uses the same fraction of one R² unit. Non-finite
    scores are ignored, and an entirely non-finite input raises rather than
    yielding an inverted or meaningless axis.
    """
    values = np.asarray(scores, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("R² scores contain no finite values")
    lower = min(0.0, float(finite.min()))
    upper = max(0.0, float(finite.max()))
    span = upper - lower
    margin = span * R2_AXIS_MARGIN_FRACTION if span > 0 else R2_AXIS_MARGIN_FRACTION
    return lower - margin, upper + margin


def _heatmap_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Map absolute metrics onto the declared per-metric reference scores.

    Each cell is ``(value - reference minimum) / reference span``, inverted
    once when lower is better and clipped to ``[0, 1]``. The reference is
    declared in :data:`HEATMAP_REFERENCE`, so a color depends on the absolute
    metric value rather than on the other folds' min/max. A column whose
    observed spread is within :data:`HEATMAP_COLUMN_TOLERANCE` is constant and
    renders at :data:`HEATMAP_NEUTRAL_SCORE` (0.5), never ``0/0`` NaN.
    """
    scores = pd.DataFrame(
        index=metrics_df.index, columns=metrics_df.columns, dtype=float
    )
    for column in metrics_df.columns:
        reference = HEATMAP_REFERENCE[column]
        values = metrics_df[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0 or np.ptp(finite) <= HEATMAP_COLUMN_TOLERANCE:
            scores[column] = HEATMAP_NEUTRAL_SCORE
            continue
        position = (values - float(reference["minimum"])) / (
            float(reference["maximum"]) - float(reference["minimum"])
        )
        if not reference["higher_is_better"]:
            position = 1.0 - position
        scores[column] = np.clip(position, 0.0, 1.0)
    return scores


def plot_r2_scores(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create bar plot of R² scores across folds with mean/std bands.

    The y-limits are data-driven: they contain 0 and the observed minimum and
    maximum (plus a small margin), so negative R² values are neither clipped
    nor inverted. A dotted line marks R² = 0 (no skill).

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "r2_scores_by_fold.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    folds = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int)
    r2_scores = summary_df["test_r2"].to_numpy(dtype=float)

    bars = ax.bar(folds, r2_scores, color="steelblue", alpha=0.7, edgecolor="black")

    # Add mean line
    mean_r2 = r2_scores.mean()
    ax.axhline(y=mean_r2, color="red", linestyle="--", linewidth=2, label=f"Mean R² = {mean_r2:.4f}")
    ax.axhline(y=0.0, color="black", linestyle=":", linewidth=1, label="R² = 0 (no skill)")

    # Add value labels on bars, outside the bar for negative heights too.
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{r2:.4f}",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
    ax.set_title("Regression R² Scores by Fold (Temporal CV)", fontsize=14, fontweight="bold")
    ax.set_ylim(*_r2_axis_limits(r2_scores))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_figure_atomically(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved R² scores plot to {output_path}")
    plt.close()

    return output_path


def plot_mae_rmse_comparison(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create side-by-side comparison of MAE and RMSE across folds.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "mae_rmse_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    folds = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int)
    mae_values = summary_df["test_mae"].values
    rmse_values = summary_df["test_rmse"].values

    # MAE plot
    bars1 = ax1.bar(folds, mae_values, color="coral", alpha=0.7, edgecolor="black")
    mean_mae = mae_values.mean()
    ax1.axhline(y=mean_mae, color="red", linestyle="--", linewidth=2, label=f"Mean MAE = {mean_mae:.6f}")

    for bar, mae in zip(bars1, mae_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f"{mae:.6f}",
                ha="center", va="bottom", fontsize=9)

    ax1.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax1.set_ylabel(
        f"MAE (Mean Absolute Error, {LOG_POINT_UNIT})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_title(
        f"Test MAE by Fold ({LOG_POINT_UNIT})", fontsize=13, fontweight="bold"
    )
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    # RMSE plot
    bars2 = ax2.bar(folds, rmse_values, color="lightgreen", alpha=0.7, edgecolor="black")
    mean_rmse = rmse_values.mean()
    ax2.axhline(y=mean_rmse, color="red", linestyle="--", linewidth=2, label=f"Mean RMSE = {mean_rmse:.6f}")

    for bar, rmse in zip(bars2, rmse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f"{rmse:.6f}",
                ha="center", va="bottom", fontsize=9)

    ax2.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax2.set_ylabel(
        f"RMSE (Root Mean Squared Error, {LOG_POINT_UNIT})",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_title(
        f"Test RMSE by Fold ({LOG_POINT_UNIT})", fontsize=13, fontweight="bold"
    )
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    save_figure_atomically(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved MAE/RMSE comparison plot to {output_path}")
    plt.close()

    return output_path


def plot_metric_heatmap(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create heatmap of all regression metrics across folds.

    Color is the metric's position on the declared absolute per-metric
    reference in :data:`HEATMAP_REFERENCE`, with direction encoded once
    (green = better) and clipped to [0, 1]. Colors therefore expose absolute
    levels and do not re-span [0, 1] over the displayed folds. A constant
    column has no spread: it renders at the neutral 0.5
    (:data:`HEATMAP_NEUTRAL_SCORE`) instead of a ``0/0`` NaN, and the colorbar
    names the reference.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "metrics_heatmap.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Absolute values are annotated; colors are the declared reference scores.
    metrics_df = summary_df[["fold", "test_r2", "test_mae", "test_rmse"]].copy()
    metrics_df["fold"] = metrics_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int)
    metrics_df = metrics_df.set_index("fold")
    metrics_scores = _heatmap_scores(metrics_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        metrics_scores.T,
        annot=metrics_df.T,
        fmt=".4f",
        cmap="RdYlGn",
        cbar_kws={"label": HEATMAP_SCALE_LABEL},
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.set_title(
        "Regression Metrics Heatmap Across Folds\n"
        "(Values=Actual, Color=Absolute reference score [0.5 neutral, green=better])",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    save_figure_atomically(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved metrics heatmap to {output_path}")
    plt.close()

    return output_path


def generate_metrics_visualizations(
    summary_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Generate all regression metric visualizations.

    Args:
        summary_df: DataFrame from load_fold_training_summary(). If None, loads from default path.
        output_dir: Directory to save figures. If None, uses default outputs_dir.

    Returns:
        List of paths to generated figures
    """
    if summary_df is None:
        summary_df = load_fold_training_summary()

    if output_dir is None:
        from packages.ml.model.utils import get_project_paths
        output_dir = get_project_paths()["outputs_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("r2_scores_by_fold", plot_r2_scores),
        ("mae_rmse_comparison", plot_mae_rmse_comparison),
        ("metrics_heatmap", plot_metric_heatmap),
    ]

    paths = []
    for plot_name, plot_function in plots:
        path = render_plot(
            "metrics",
            plot_name,
            lambda output_path, render=plot_function: render(summary_df, output_path),
            output_dir,
        )
        if path is not None:
            paths.append(path)

    logger.info(f"Generated {len(paths)} metric visualizations")
    return paths

