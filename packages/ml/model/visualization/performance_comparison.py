"""Performance comparison visualizations across folds (metrics, timing, samples, features)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from packages.ml.model.visualization.data_loader import (
    QLIKE_GAP_SIGN_CONVENTION,
    extract_qlike_stats,
    load_fold_training_summary,
)
from packages.ml.model.visualization.plotting import render_plot

logger = logging.getLogger(__name__)

R2_UNIT = "dimensionless"
LOG_POINT_UNIT = "log points"

# The QLIKE clause is the shared extractor's definition, not a local copy, so
# the two cannot drift. R² is oriented the same way because higher R² is the
# better score.
BENCHMARK_GAP_SIGN_CONVENTION = (
    f"Gap convention: {QLIKE_GAP_SIGN_CONVENTION}; "
    "R² gap = test_r2 - har_r2 (positive = model better than HAR)."
)


def _finite_float(value: object) -> float | None:
    """Coerce a cell to a finite float, or ``None`` when it is not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _benchmark_records(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-fold absolutes and model-vs-HAR gap records for the table.

    QLIKE values and gaps come from the shared stats extractor, whose gap is
    ``har_qlike - test_qlike``. The R² gap is oriented the same way (positive =
    model better): ``test_r2 - har_r2``. Missing optional inputs stay ``None``.
    """
    qlike_stats = extract_qlike_stats(summary_df)
    records = []
    for position, fold in enumerate(summary_df["fold"].astype(str)):
        shared = qlike_stats["per_fold"].get(fold, {})
        test_r2 = _finite_float(summary_df["test_r2"].iloc[position])
        har_r2 = shared.get("har_r2")
        records.append(
            {
                "fold": fold,
                "test_r2": test_r2,
                "test_mae": _finite_float(summary_df["test_mae"].iloc[position]),
                "test_rmse": _finite_float(summary_df["test_rmse"].iloc[position]),
                "har_r2": har_r2,
                "r2_gap": (
                    test_r2 - har_r2
                    if test_r2 is not None and har_r2 is not None
                    else None
                ),
                "test_qlike": shared.get("test_qlike"),
                "har_qlike": shared.get("har_qlike"),
                "qlike_gap": shared.get("qlike_gap"),
            }
        )
    return pd.DataFrame(records)


def _format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_gap(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def plot_metrics_comparison(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Plot one absolute-value panel per metric and a per-fold benchmark-gap table.

    R² is dimensionless; MAE and RMSE are log points on the log target. Each
    metric gets its own axis so the units never share a scale. The table keeps
    the absolute values and adds the model-vs-HAR QLIKE and R² gaps under the
    figure's single sign convention.

    Only absolute values are drawn: no normalized score is rendered, so a
    constant metric cannot produce a 1.0/0.0 tie artifact.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "metrics_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values
    x = np.arange(len(fold_nums))

    panel_specs = (
        ("test_r2", "R²", f"R² ({R2_UNIT}; higher is better)", "steelblue"),
        ("test_mae", "MAE", f"MAE ({LOG_POINT_UNIT}; lower is better)", "coral"),
        ("test_rmse", "RMSE", f"RMSE ({LOG_POINT_UNIT}; lower is better)", "lightgreen"),
    )

    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.85))
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    table_axis = fig.add_subplot(grid[1, :])
    table_axis.axis("off")

    for axis, (column, metric_name, ylabel, color) in zip(axes, panel_specs):
        values = summary_df[column].to_numpy(dtype=np.float64)
        bars = axis.bar(x, values, color=color, alpha=0.8, edgecolor="black")

        axis.set_xlabel("Fold", fontsize=12, fontweight="bold")
        axis.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        axis.set_title(
            f"{metric_name} by Fold (absolute values)",
            fontsize=13,
            fontweight="bold",
        )
        axis.set_xticks(x)
        axis.set_xticklabels([f"Fold {n}" for n in fold_nums])
        axis.grid(axis="y", alpha=0.3)
        axis.margins(y=0.15)
        if column == "test_r2":
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    records = _benchmark_records(summary_df)
    headers = (
        "Fold",
        f"R² ({R2_UNIT})",
        f"MAE ({LOG_POINT_UNIT})",
        f"RMSE ({LOG_POINT_UNIT})",
        "HAR R²",
        "R² gap",
        "Model QLIKE",
        "HAR QLIKE",
        "QLIKE gap",
    )
    cells = [
        (
            str(record.fold),
            _format_value(record.test_r2),
            _format_value(record.test_mae),
            _format_value(record.test_rmse),
            _format_value(record.har_r2),
            _format_gap(record.r2_gap),
            _format_value(record.test_qlike),
            _format_value(record.har_qlike),
            _format_gap(record.qlike_gap),
        )
        for record in records.itertuples(index=False)
    ]
    table = table_axis.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=(0.07, 0.11, 0.11, 0.11, 0.10, 0.10, 0.11, 0.11, 0.11),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
    table_axis.set_title(
        "Absolute metrics and benchmark gaps per fold",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    fig.suptitle(
        "Regression Metrics and HAR Benchmark Gaps Across Folds",
        fontsize=15,
        fontweight="bold",
    )
    table_axis.text(
        0.5,
        -0.06,
        BENCHMARK_GAP_SIGN_CONVENTION,
        transform=table_axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved metrics comparison plot to {output_path}")
    plt.close()

    return output_path


def plot_timing_breakdown(summary_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """
    Create stacked horizontal bar chart showing the full fold timing per fold.

    The stacked segments are train time, inference time, the HAR benchmark fit,
    and the remaining fold work, so the bar sums to the reported
    ``fold_time_seconds``; the total label is that full fold cost.

    Args:
        summary_df: DataFrame from load_fold_training_summary()
        output_path: Where to save the figure. If None, uses default outputs_dir

    Returns:
        Path to saved figure
    """
    if output_path is None:
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "timing_breakdown.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_nums = summary_df["fold"].str.extract(r'fold_(\d+)')[0].astype(int).values
    train_times = summary_df["train_time_seconds"].values
    inference_times = summary_df["inference_time_seconds"].values
    har_times = summary_df["har_time_seconds"].values
    fold_times = summary_df["fold_time_seconds"].values

    # The remaining fold work keeps the stack equal to the reported full fold
    # cost: fold_time_seconds encloses train + inference + HAR plus the rest.
    other_times = np.maximum(
        fold_times - (train_times + inference_times + har_times), 0.0
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(fold_nums))
    height = 0.5

    bars_train = ax.barh(y, train_times, height, label="Train Time", color="steelblue", alpha=0.8)
    bars_inference = ax.barh(y, inference_times, height, left=train_times, label="Inference Time", color="coral", alpha=0.8)
    har_left = train_times + inference_times
    bars_har = ax.barh(y, har_times, height, left=har_left, label="HAR Fit", color="mediumseagreen", alpha=0.8)
    other_left = har_left + har_times
    bars_other = ax.barh(y, other_times, height, left=other_left, label="Other Fold Work", color="silver", alpha=0.8)

    # Total labels carry the full fold cost (train + inference + HAR + rest).
    for i, total in enumerate(fold_times):
        ax.text(total + max(train_times) * 0.02, i, f"{total:.1f}s",
                ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"Fold {n}" for n in fold_nums])
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Fold Time Breakdown: Train, Inference, and HAR Fit per Fold", fontsize=14, fontweight="bold")
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
        from packages.ml.model.utils import get_project_paths
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
        from packages.ml.model.utils import get_project_paths
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


def generate_performance_comparisons(
    summary_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
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
        from packages.ml.model.utils import get_project_paths
        output_dir = get_project_paths()["outputs_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("metrics_comparison", plot_metrics_comparison),
        ("timing_breakdown", plot_timing_breakdown),
        ("sample_distribution", plot_sample_distribution),
        ("feature_count_summary", plot_feature_count_summary),
    ]

    paths = []
    for plot_name, plot_function in plots:
        path = render_plot(
            "performance",
            plot_name,
            lambda output_path, render=plot_function: render(summary_df, output_path),
            output_dir,
        )
        if path is not None:
            paths.append(path)

    logger.info(f"Generated {len(paths)} performance comparison visualizations")
    return paths
