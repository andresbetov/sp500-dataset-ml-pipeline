"""Diagnostic visualizations for model training: predictions, residuals, feature importance, error analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from features.feature_engineering import INDICATORS
from utils import ARTIFACTS_DIR
from visualization.data_loader import load_fold_training_summary

logger = logging.getLogger(__name__)

FOLD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
FOLD_NAMES = [f"fold_{i}" for i in range(5)]


def load_validation_predictions(predictions_path: Path | None = None) -> dict:
    """Load validation predictions from pickle file.

    Args:
        predictions_path: Path to validation_predictions.pkl. If None, uses default location.

    Returns:
        Dict with fold keys containing predictions and metrics.
    """
    if predictions_path is None:
        predictions_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Validation predictions not found at {predictions_path}")

    logger.info("Loading validation predictions from %s", predictions_path)

    data: dict = joblib.load(predictions_path)
    logger.info("Loaded predictions for %d folds: %s", len(data), list(data.keys()))

    return data


def plot_actual_vs_predicted(predictions_dict: dict, output_path: Path | None = None) -> Path:
    """Create scatter plots of actual vs predicted values per fold with diagonal reference.

    Args:
        predictions_dict: Dict of fold predictions from load_validation_predictions()
        output_path: Where to save the figure. If None, uses default outputs_dir.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "actual_vs_predicted.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    all_true = []
    all_pred = []
    all_fold = []

    for i, fold_name in enumerate(FOLD_NAMES):
        ax = axes_flat[i]
        fold_data = predictions_dict[fold_name]
        y_true = fold_data["test_true"]
        y_pred = fold_data["test_pred"]

        all_true.append(y_true)
        all_pred.append(y_pred)
        all_fold.extend([i] * len(y_true))

        ax.scatter(y_true, y_pred, c=FOLD_COLORS[i], alpha=0.5, s=10, edgecolors="none")

        # Diagonal reference line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = x")

        r2 = fold_data["test_r2"]
        ax.text(0.05, 0.95, f"$R^2 = {r2:.4f}$", transform=ax.transAxes,
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Actual", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted", fontsize=11, fontweight="bold")
        ax.set_title(f"Fold {i} — Actual vs Predicted", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    # Combined plot (last subplot)
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_fold = np.array(all_fold)

    ax_combined = axes_flat[5]
    for i in range(5):
        mask = all_fold == i
        ax_combined.scatter(all_true[mask], all_pred[mask], c=FOLD_COLORS[i],
                           alpha=0.4, s=8, edgecolors="none", label=f"Fold {i}")

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    ax_combined.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = x")

    from sklearn.metrics import r2_score
    combined_r2 = r2_score(all_true, all_pred)
    ax_combined.text(0.05, 0.95, f"$R^2 = {combined_r2:.4f}$", transform=ax_combined.transAxes,
                    fontsize=12, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax_combined.set_xlabel("Actual", fontsize=11, fontweight="bold")
    ax_combined.set_ylabel("Predicted", fontsize=11, fontweight="bold")
    ax_combined.set_title("All Folds Combined", fontsize=13, fontweight="bold")
    ax_combined.grid(alpha=0.3)
    ax_combined.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved actual vs predicted plot to %s", output_path)
    plt.close()

    return output_path


def plot_residuals(predictions_dict: dict, output_path: Path | None = None) -> Path:
    """Create histogram of residuals per fold with normal distribution overlay.

    Args:
        predictions_dict: Dict of fold predictions from load_validation_predictions()
        output_path: Where to save the figure. If None, uses default outputs_dir.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "residuals_distribution.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    all_residuals = []

    for i, fold_name in enumerate(FOLD_NAMES):
        ax = axes_flat[i]
        fold_data = predictions_dict[fold_name]
        residuals = fold_data["test_true"] - fold_data["test_pred"]

        all_residuals.append(residuals)

        ax.hist(residuals, bins=80, color=FOLD_COLORS[i], alpha=0.7, density=True, edgecolor="white", linewidth=0.5)

        # Normal distribution fit
        mu, std = residuals.mean(), residuals.std()
        x_range = np.linspace(residuals.min(), residuals.max(), 200)
        y_pdf = stats.norm.pdf(x_range, mu, std)
        ax.plot(x_range, y_pdf, "k--", linewidth=2, label=f"N({mu:.4f}, {std:.4f})")

        ax.text(0.05, 0.95, f"Mean = {mu:.4f}\nStd  = {std:.4f}", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Residual (Actual - Predicted)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.set_title(f"Fold {i} — Residual Distribution", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    # Combined histogram (last subplot)
    ax_combined = axes_flat[5]
    all_residuals_concat = np.concatenate(all_residuals)

    for i, fold_residuals in enumerate(all_residuals):
        ax_combined.hist(fold_residuals, bins=80, color=FOLD_COLORS[i], alpha=0.3, density=True,
                        edgecolor="white", linewidth=0.5, label=f"Fold {i}")

    mu_all, std_all = all_residuals_concat.mean(), all_residuals_concat.std()
    x_range = np.linspace(all_residuals_concat.min(), all_residuals_concat.max(), 200)
    y_pdf = stats.norm.pdf(x_range, mu_all, std_all)
    ax_combined.plot(x_range, y_pdf, "k--", linewidth=2, label=f"Combined N({mu_all:.4f}, {std_all:.4f})")

    ax_combined.set_xlabel("Residual (Actual - Predicted)", fontsize=11, fontweight="bold")
    ax_combined.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax_combined.set_title("All Folds — Residual Distribution", fontsize=13, fontweight="bold")
    ax_combined.grid(alpha=0.3)
    ax_combined.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved residuals distribution plot to %s", output_path)
    plt.close()

    return output_path


def plot_feature_importance(predictions_dict: dict, summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Create horizontal bar charts of top 15 feature importances per fold and aggregate.

    Args:
        predictions_dict: Dict of fold predictions from load_validation_predictions()
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "feature_importance.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    models_dir = ARTIFACTS_DIR / "checkpoints"
    all_importances = []

    for i in range(5):
        model_path = models_dir / f"fold_{i}_model.pkl"
        if not model_path.exists():
            logger.warning("Model not found at %s, skipping fold %d", model_path, i)
            all_importances.append(np.zeros(len(INDICATORS)))
            continue

        model = joblib.load(model_path)
        importances = model.feature_importances_
        all_importances.append(importances)

    all_importances = np.array(all_importances)

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes_flat = axes.flatten()

    for i in range(5):
        ax = axes_flat[i]
        importances = all_importances[i]

        # Top 15
        top_indices = np.argsort(importances)[-15:][::-1]
        top_names = [INDICATORS[idx] for idx in top_indices]
        top_values = importances[top_indices]

        ax.barh(range(len(top_indices)), top_values, color=FOLD_COLORS[i], alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=11, fontweight="bold")
        ax.set_title(f"Fold {i} — Top 15 Features", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    # Aggregate importance (last subplot)
    ax_agg = axes_flat[5]
    avg_importances = all_importances.mean(axis=0)
    top_indices = np.argsort(avg_importances)[-15:][::-1]
    top_names = [INDICATORS[idx] for idx in top_indices]
    top_values = avg_importances[top_indices]
    top_stds = all_importances.std(axis=0)[top_indices]

    ax_agg.barh(range(len(top_indices)), top_values, xerr=top_stds,
                color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5,
                capsize=3)
    ax_agg.set_yticks(range(len(top_indices)))
    ax_agg.set_yticklabels(top_names, fontsize=9)
    ax_agg.invert_yaxis()
    ax_agg.set_xlabel("Average Importance", fontsize=11, fontweight="bold")
    ax_agg.set_title("Aggregate (Across Folds) — Top 15 Features", fontsize=13, fontweight="bold")
    ax_agg.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved feature importance plot to %s", output_path)
    plt.close()

    return output_path


def plot_error_by_volatility_bin(predictions_dict: dict, output_path: Path | None = None) -> Path:
    """Plot mean absolute error by actual volatility decile across folds.

    Args:
        predictions_dict: Dict of fold predictions from load_validation_predictions()
        output_path: Where to save the figure. If None, uses default outputs_dir.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        from utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "error_by_volatility_bin.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, fold_name in enumerate(FOLD_NAMES):
        fold_data = predictions_dict[fold_name]
        y_true = fold_data["test_true"]
        y_pred = fold_data["test_pred"]
        abs_errors = np.abs(y_true - y_pred)

        # Bin actual volatility into deciles
        deciles = pd.qcut(y_true, q=10, labels=False, duplicates="drop")
        n_bins = deciles.max() + 1

        bin_centers = []
        bin_means = []
        bin_stds = []

        for b in range(n_bins):
            mask = deciles == b
            if mask.sum() == 0:
                continue
            bin_errors = abs_errors[mask]
            bin_vol = y_true[mask]

            bin_centers.append(bin_vol.mean())
            bin_means.append(bin_errors.mean())
            bin_stds.append(bin_errors.std())

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                    color=FOLD_COLORS[i], marker="o", capsize=4, linewidth=2,
                    markersize=6, label=f"Fold {i}", alpha=0.8)

    ax.set_xlabel("Actual Volatility (Decile Mean)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error", fontsize=12, fontweight="bold")
    ax.set_title("Prediction Error by Volatility Decile", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved error by volatility bin plot to %s", output_path)
    plt.close()

    return output_path


def generate_training_diagnostics(predictions_dict: dict | None = None, summary_df: pd.DataFrame | None = None,
                                  output_dir: Path | None = None) -> list[Path]:
    """Generate all training diagnostic visualizations.

    Args:
        predictions_dict: Dict of fold predictions. If None, loads from default path.
        summary_df: DataFrame from load_fold_training_summary(). If None, loads from default path.
        output_dir: Directory to save figures. If None, uses default outputs_dir.

    Returns:
        List of paths to generated figures.
    """
    if predictions_dict is None:
        predictions_dict = load_validation_predictions()

    if summary_df is None:
        summary_df = load_fold_training_summary()

    if output_dir is None:
        from utils import get_project_paths
        output_dir = get_project_paths()["outputs_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_actual_vs_predicted(predictions_dict, output_dir / "actual_vs_predicted.png"),
        plot_residuals(predictions_dict, output_dir / "residuals_distribution.png"),
        plot_feature_importance(predictions_dict, summary_df, output_dir / "feature_importance.png"),
        plot_error_by_volatility_bin(predictions_dict, output_dir / "error_by_volatility_bin.png"),
    ]

    logger.info("Generated %d training diagnostic visualizations", len(paths))
    return paths
