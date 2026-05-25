"""Temporal analysis visualizations for fold training results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from visualization.data_loader import load_fold_training_summary

logger = logging.getLogger(__name__)


def plot_temporal_fold_ranges(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create timeline visualization showing train/test date ranges for each fold.

    Displays temporal splits of TimeSeriesSplit with expanding windows.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "temporal_fold_ranges.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Convert date strings to datetime
    summary_df["train_date_min"] = pd.to_datetime(summary_df["train_date_min"])
    summary_df["train_date_max"] = pd.to_datetime(summary_df["train_date_max"])
    summary_df["test_date_min"] = pd.to_datetime(summary_df["test_date_min"])
    summary_df["test_date_max"] = pd.to_datetime(summary_df["test_date_max"])

    # Extract fold numbers
    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values

    y_positions = range(len(fold_nums))

    # Plot train and test ranges for each fold
    for i, (idx, row) in enumerate(summary_df.iterrows()):
        # Training period (blue)
        train_start = row["train_date_min"]
        train_end = row["train_date_max"]
        train_duration = (train_end - train_start).days

        ax.barh(i, train_duration, left=train_start, height=0.4,
                color="steelblue", alpha=0.7, label="Train" if i == 0 else "")

        # Testing period (orange)
        test_start = row["test_date_min"]
        test_end = row["test_date_max"]
        test_duration = (test_end - test_start).days

        ax.barh(i, test_duration, left=test_start, height=0.4,
                color="coral", alpha=0.7, label="Test" if i == 0 else "")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Fold {n}" for n in fold_nums])
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fold", fontsize=12, fontweight="bold")
    ax.set_title("Temporal Cross-Validation: Expanding Windows\n(Blue=Train, Orange=Test)",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right")

    # Format x-axis for better date display
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved temporal fold ranges plot to {output_path}")
    plt.close()

    return output_path


def plot_fold_sample_distribution(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create stacked bar chart showing train/test sample counts for each fold.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "fold_sample_distribution.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values
    train_samples = summary_df["n_train_samples"].values / 1e6  # Convert to millions
    test_samples = summary_df["n_test_samples"].values / 1e6

    x = range(len(fold_nums))

    # Stacked bars
    bars1 = ax.bar(x, train_samples, label="Train Samples", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x, test_samples, bottom=train_samples, label="Test Samples", color="coral", alpha=0.8)

    # Add value labels
    for i, (train, test) in enumerate(zip(train_samples, test_samples)):
        total = train + test
        ax.text(i, train/2, f"{train:.2f}M", ha="center", va="center", fontweight="bold", fontsize=10)
        ax.text(i, train + test/2, f"{test:.2f}M", ha="center", va="center", fontweight="bold", fontsize=10)
        ax.text(i, total, f"Total: {total:.2f}M", ha="center", va="bottom", fontsize=9, fontstyle="italic")

    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Samples (Millions)", fontsize=12, fontweight="bold")
    ax.set_title("Train/Test Sample Distribution Across Folds", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {n}" for n in fold_nums])
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved fold sample distribution plot to {output_path}")
    plt.close()

    return output_path


def plot_temporal_metric_trend(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create line plot showing how metrics trend across temporal folds.

    Useful for detecting performance degradation in recent periods.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "temporal_metric_trend.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values
    r2_scores = summary_df["test_r2"].values
    mae_values = summary_df["test_mae"].values
    rmse_values = summary_df["test_rmse"].values

    # R² trend
    axes[0].plot(fold_nums, r2_scores, marker="o", linewidth=2.5, markersize=8, color="steelblue")
    axes[0].fill_between(fold_nums, r2_scores, alpha=0.3, color="steelblue")
    axes[0].set_ylabel("R² Score", fontsize=11, fontweight="bold")
    axes[0].set_title("Metric Trend Across Temporal Folds", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    for x, y in zip(fold_nums, r2_scores):
        axes[0].text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=9)

    # MAE trend
    axes[1].plot(fold_nums, mae_values, marker="s", linewidth=2.5, markersize=8, color="coral")
    axes[1].fill_between(fold_nums, mae_values, alpha=0.3, color="coral")
    axes[1].set_ylabel("MAE", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    for x, y in zip(fold_nums, mae_values):
        axes[1].text(x, y, f"{y:.6f}", ha="center", va="bottom", fontsize=9)

    # RMSE trend
    axes[2].plot(fold_nums, rmse_values, marker="^", linewidth=2.5, markersize=8, color="lightgreen")
    axes[2].fill_between(fold_nums, rmse_values, alpha=0.3, color="lightgreen")
    axes[2].set_xlabel("Fold", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("RMSE", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    for x, y in zip(fold_nums, rmse_values):
        axes[2].text(x, y, f"{y:.6f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved temporal metric trend plot to {output_path}")
    plt.close()

    return output_path


def generate_temporal_visualizations(summary_df: pd.DataFrame | None = None, output_dir: Path | None = None) -> list[Path]:
    """
    Generate all temporal analysis visualizations.

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
        plot_temporal_fold_ranges(summary_df, output_dir / "temporal_fold_ranges.png"),
        plot_fold_sample_distribution(summary_df, output_dir / "fold_sample_distribution.png"),
        plot_temporal_metric_trend(summary_df, output_dir / "temporal_metric_trend.png"),
    ]

    logger.info(f"Generated {len(paths)} temporal visualizations")
    return paths

