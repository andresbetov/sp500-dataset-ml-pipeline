"""Utilities for loading and processing fold training results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_fold_training_summary(summary_path: Path | None = None) -> pd.DataFrame:
    """
    Load fold training summary from JSON and convert to DataFrame.

    Args:
        summary_path: Path to fold_training_summary.json. If None, uses default location.

    Returns:
        DataFrame with fold metrics (rows=5, cols=metrics)
    """
    if summary_path is None:
        # Use default location from model/artifacts/results/
        from utils import get_project_paths
        paths = get_project_paths()
        summary_path = paths["artifacts_dir"] / "results" / "fold_training_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Fold training summary not found at {summary_path}")

    logger.info(f"Loading fold training summary from {summary_path}")

    with open(summary_path, "r") as f:
        data = json.load(f)

    # Convert nested dict to flat DataFrame
    rows = []
    for fold_id, metrics in data.items():
        row = {"fold": fold_id}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} fold summaries")

    return df


def extract_training_time_stats(summary_df: pd.DataFrame) -> dict:
    """
    Extract training timing statistics from fold summary.

    Args:
        summary_df: DataFrame from load_fold_training_summary()

    Returns:
        Dict with timing statistics
    """
    return {
        "train_time_mean": summary_df["train_time_seconds"].mean(),
        "train_time_total": summary_df["train_time_seconds"].sum(),
        "inference_time_mean": summary_df["inference_time_seconds"].mean(),
        "inference_time_total": summary_df["inference_time_seconds"].sum(),
        "fold_time_mean": summary_df["fold_time_seconds"].mean(),
        "fold_time_total": summary_df["fold_time_seconds"].sum(),
    }


def extract_metric_stats(summary_df: pd.DataFrame) -> dict:
    """
    Extract regression metric statistics from fold summary.

    Args:
        summary_df: DataFrame from load_fold_training_summary()

    Returns:
        Dict with metric statistics (R², MAE, RMSE)
    """
    return {
        "r2_mean": summary_df["test_r2"].mean(),
        "r2_std": summary_df["test_r2"].std(),
        "r2_min": summary_df["test_r2"].min(),
        "r2_max": summary_df["test_r2"].max(),
        "mae_mean": summary_df["test_mae"].mean(),
        "mae_std": summary_df["test_mae"].std(),
        "mae_min": summary_df["test_mae"].min(),
        "mae_max": summary_df["test_mae"].max(),
        "rmse_mean": summary_df["test_rmse"].mean(),
        "rmse_std": summary_df["test_rmse"].std(),
        "rmse_min": summary_df["test_rmse"].min(),
        "rmse_max": summary_df["test_rmse"].max(),
    }


def extract_sample_stats(summary_df: pd.DataFrame) -> dict:
    """
    Extract sample count statistics from fold summary.

    Args:
        summary_df: DataFrame from load_fold_training_summary()

    Returns:
        Dict with sample statistics
    """
    return {
        "total_train_samples": summary_df["n_train_samples"].sum(),
        "total_test_samples": summary_df["n_test_samples"].sum(),
        "mean_train_samples": summary_df["n_train_samples"].mean(),
        "mean_test_samples": summary_df["n_test_samples"].mean(),
        "n_folds": len(summary_df),
    }

