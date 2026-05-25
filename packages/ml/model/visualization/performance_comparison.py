"""Performance comparison visualizations across folds (metrics, timing, samples, features)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.data_loader import load_fold_training_summary

logger = logging.getLogger(__name__)


def plot_metrics_comparison(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create grouped bar chart of R², MAE, and RMSE across folds with normalized overlay.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "metrics_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values
    r2_scores = summary_df["test_r2"].values
    mae_values = summary_df["test_mae"].values
    rmse_values = summary_df["test_rmse"].values

    # Normalize each metric to [0, 1] for overlay comparison
    def normalize(arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val == min_val:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)

    r2_norm = normalize(r2_scores)
    mae_norm = 1 - normalize(mae_values)  # Invert so lower=better maps to higher score
    rmse_norm = 1 - normalize(rmse_values)

    x = np.arange(len(fold_nums))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(12, 7))

    bars_r2 = ax1.bar(x - width, r2_scores, width, label="R²", color="steelblue", alpha=0.8, edgecolor="black")
    bars_mae = ax1.bar(x, mae_values, width, label="MAE", color="coral", alpha=0.8, edgecolor="black")
    bars_rmse = ax1.bar(x + width, rmse_values, width, label="RMSE", color="lightgreen", alpha=0.8, edgecolor="black")

    ax1.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual Value", fontsize=12, fontweight="bold")
    ax1.set_title("Regression Metrics Comparison Across Folds\n(Bars=Actual, Lines=Normalized [0=worst, 1=best])",
                  fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Fold {n}" for n in fold_nums])
    ax1.grid(axis="y", alpha=0.3)

    # Secondary axis for normalized scores
    ax2 = ax1.twinx()
    ax2.plot(x, r2_norm, marker="o", linewidth=2, markersize=7, color="steelblue", label="R² (norm)")
    ax2.plot(x, mae_norm, marker="s", linewidth=2, markersize=7, color="coral", label="MAE (norm)")
    ax2.plot(x, rmse_norm, marker="^", linewidth=2, markersize=7, color="lightgreen", label="RMSE (norm)")
    ax2.set_ylabel("Normalized Score (1=best)", fontsize=12, fontweight="bold")
    ax2.set_ylim([-0.1, 1.1])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved metrics comparison plot to {output_path}")
    plt.close()

    return output_path


def plot_timing_breakdown(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create stacked horizontal bar chart showing train/inference timing per fold.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "timing_breakdown.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values
    train_times = summary_df["train_time_seconds"].values
    inference_times = summary_df["inference_time_seconds"].values

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(fold_nums))
    height = 0.5

    bars_train = ax.barh(y, train_times, height, label="Train Time", color="steelblue", alpha=0.8)
    bars_inference = ax.barh(y, inference_times, height, left=train_times, label="Inference Time", color="coral", alpha=0.8)

    # Add total time labels on each bar
    for i, (train, inference) in enumerate(zip(train_times, inference_times)):
        total = train + inference
        ax.text(total + max(train_times) * 0.02, i, f"{total:.1f}s",
                ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"Fold {n}" for n in fold_nums])
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Training vs Inference Time per Fold", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved timing breakdown plot to {output_path}")
    plt.close()

    return output_path


def plot_sample_distribution(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create grouped bar chart showing train/test sample counts per fold.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "sample_distribution.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values
    train_samples = summary_df["n_train_samples"].values / 1e6
    test_samples = summary_df["n_test_samples"].values / 1e6

    x = np.arange(len(fold_nums))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_train = ax.bar(x - width / 2, train_samples, width, label="Train Samples", color="steelblue", alpha=0.8, edgecolor="black")
    bars_test = ax.bar(x + width / 2, test_samples, width, label="Test Samples", color="coral", alpha=0.8, edgecolor="black")

    # Add value labels
    for bar, val in zip(bars_train, train_samples):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f"{val:.2f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars_test, test_samples):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f"{val:.2f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Samples (Millions)", fontsize=12, fontweight="bold")
    ax.set_title("Train/Test Sample Distribution Across Folds", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {n}" for n in fold_nums])
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved sample distribution plot to {output_path}")
    plt.close()

    return output_path


def plot_feature_count_summary(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create horizontal bar chart showing indicator vs ticker feature counts per fold.

    If no ticker features exist (n_features_ticker == 0), displays a text message.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "feature_count_summary.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values
    n_indicators = summary_df["n_features_indicators"].values
    n_ticker = summary_df["n_features_ticker"].values

    fig, ax = plt.subplots(figsize=(10, 5))

    if n_ticker.sum() == 0:
        ax.text(0.5, 0.5, "Only indicator features were used\n(no ticker one-hot encoding)",
                ha="center", va="center", fontsize=13, fontstyle="italic",
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
        ax.set_title("Feature Composition: Indicators Only", fontsize=14, fontweight="bold")
        ax.axis("off")
    else:
        y = np.arange(len(fold_nums))
        height = 0.5

        bars_ind = ax.barh(y, n_indicators, height, label="Indicators", color="steelblue", alpha=0.8)
        bars_ticker = ax.barh(y, n_ticker, height, left=n_indicators, label="Ticker Features", color="coral", alpha=0.8)

        for i, (ind, ticker) in enumerate(zip(n_indicators, n_ticker)):
            total = ind + ticker
            ax.text(total + max(n_indicators) * 0.01, i, f"{int(total)}", ha="left", va="center", fontsize=10, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels([f"Fold {n}" for n in fold_nums])
        ax.set_xlabel("Feature Count", fontsize=12, fontweight="bold")
        ax.set_title("Feature Composition per Fold", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved feature count summary plot to {output_path}")
    plt.close()

    return output_path


def generate_performance_comparisons(summary_df: pd.DataFrame | None = None, output_dir: Path | None = None) -> list[Path]:
    """
    Generate all performance comparison visualizations.

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
        plot_metrics_comparison(summary_df, output_dir / "metrics_comparison.png"),
        plot_timing_breakdown(summary_df, output_dir / "timing_breakdown.png"),
        plot_sample_distribution(summary_df, output_dir / "sample_distribution.png"),
        plot_feature_count_summary(summary_df, output_dir / "feature_count_summary.png"),
    ]

    logger.info(f"Generated {len(paths)} performance comparison visualizations")
    return paths
