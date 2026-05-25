"""Visualizations for regression metrics (R², MAE, RMSE) across folds."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.data_loader import load_fold_training_summary

logger = logging.getLogger(__name__)


def plot_r2_scores(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create bar plot of R² scores across folds with mean/std bands.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "r2_scores_by_fold.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    folds = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int)
    r2_scores = summary_df["test_r2"].values

    bars = ax.bar(folds, r2_scores, color="steelblue", alpha=0.7, edgecolor="black")

    # Add mean line
    mean_r2 = r2_scores.mean()
    ax.axhline(y=mean_r2, color="red", linestyle="--", linewidth=2, label=f"Mean R² = {mean_r2:.4f}")

    # Add value labels on bars
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{r2:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
    ax.set_title("Regression R² Scores by Fold (Temporal CV)", fontsize=14, fontweight="bold")
    ax.set_ylim([0, max(r2_scores) * 1.15])
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
        from utils import get_project_paths
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
    ax1.set_ylabel("MAE (Mean Absolute Error)", fontsize=12, fontweight="bold")
    ax1.set_title("Test MAE by Fold", fontsize=13, fontweight="bold")
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
    ax2.set_ylabel("RMSE (Root Mean Squared Error)", fontsize=12, fontweight="bold")
    ax2.set_title("Test RMSE by Fold", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved MAE/RMSE comparison plot to {output_path}")
    plt.close()

    return output_path


def plot_metric_heatmap(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create heatmap of all regression metrics across folds.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "metrics_heatmap.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select relevant columns and normalize for visualization
    metrics_df = summary_df[["fold", "test_r2", "test_mae", "test_rmse"]].copy()
    metrics_df["fold"] = metrics_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int)
    metrics_df = metrics_df.set_index("fold")

    # Normalize metrics to [0,1] scale for better visualization
    metrics_normalized = metrics_df.copy()
    metrics_normalized["test_r2"] = (metrics_df["test_r2"] - metrics_df["test_r2"].min()) / (metrics_df["test_r2"].max() - metrics_df["test_r2"].min())
    metrics_normalized["test_mae"] = 1 - (metrics_df["test_mae"] - metrics_df["test_mae"].min()) / (metrics_df["test_mae"].max() - metrics_df["test_mae"].min())
    metrics_normalized["test_rmse"] = 1 - (metrics_df["test_rmse"] - metrics_df["test_rmse"].min()) / (metrics_df["test_rmse"].max() - metrics_df["test_rmse"].min())

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        metrics_normalized.T,
        annot=metrics_df.T,
        fmt=".4f",
        cmap="RdYlGn",
        cbar_kws={"label": "Normalized Score (green=better)"},
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.set_title("Regression Metrics Heatmap Across Folds\n(Values=Actual, Color=Normalized [0=worst, 1=best])",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved metrics heatmap to {output_path}")
    plt.close()

    return output_path


def generate_metrics_visualizations(summary_df: pd.DataFrame | None = None, output_dir: Path | None = None) -> list[Path]:
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
        from utils import get_project_paths
        output_dir = get_project_paths()["outputs_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_r2_scores(summary_df, output_dir / "r2_scores_by_fold.png"),
        plot_mae_rmse_comparison(summary_df, output_dir / "mae_rmse_comparison.png"),
        plot_metric_heatmap(summary_df, output_dir / "metrics_heatmap.png"),
    ]

    logger.info(f"Generated {len(paths)} metric visualizations")
    return paths

