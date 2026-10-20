"""Diagnostic visualizations for model training: predictions, residuals, feature importance, error analysis."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, load_verified
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.run_log import resolve_stored_path
from packages.ml.model.utils import ARTIFACTS_DIR
from packages.ml.model.visualization.data_loader import load_fold_training_summary
from packages.ml.model.visualization.plotting import (
    render_plot,
    save_figure_atomically,
)

logger = logging.getLogger(__name__)

# The stored target is log sigma. Every axis that carries it names the unit;
# MAE/RMSE on that target are log-point errors, not raw volatility.
LOG_SIGMA_UNIT = "log σ (per-day)"
SIGMA_UNIT = "σ (per-day)"
LOG_POINT_UNIT = "log points"
EVALUATION_SUMMARY_FILENAME = "evaluation_summary.json"

# Base series palette. The per-fold colors cycle it so the color list always
# matches the configured fold count, never a hardcoded five.
_FOLD_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
FOLD_NAMES = [f"fold_{i}" for i in range(cross_validator.N_FOLDS)]
FOLD_COLORS = [
    _FOLD_PALETTE[i % len(_FOLD_PALETTE)] for i in range(len(FOLD_NAMES))
]

# The plot records how many checkpoint models it loaded so the orchestrator can
# carry that count into the visualization manifest. A context-local collector
# keeps the public plot signature source-only (the output path is its only
# argument) and avoids module-global state shared between concurrent renders.
_FEATURE_IMPORTANCE_STATS: ContextVar[dict | None] = ContextVar(
    "feature_importance_stats", default=None
)

# The fold keys the current diagnostics batch should render. A full run keeps
# every configured fold; a subset run narrows this so feature importance does
# not resolve/prune fold checkpoints that were never selected.
_SELECTED_FOLDS: ContextVar[tuple[str, ...] | None] = ContextVar(
    "selected_fold_names", default=None
)


@contextmanager
def selected_fold_names(fold_names: list[str]) -> Iterator[None]:
    """Render diagnostics for exactly these fold keys inside the context."""
    token = _SELECTED_FOLDS.set(tuple(fold_names))
    try:
        yield
    finally:
        _SELECTED_FOLDS.reset(token)


def _context_fold_names() -> list[str]:
    """The context's selected folds, or every configured fold."""
    selected = _SELECTED_FOLDS.get()
    return list(FOLD_NAMES) if selected is None else list(selected)


@contextmanager
def collect_feature_importance_stats() -> Iterator[dict]:
    """Collect the feature-importance load counts of plots in the context."""
    stats: dict = {}
    token = _FEATURE_IMPORTANCE_STATS.set(stats)
    try:
        yield stats
    finally:
        _FEATURE_IMPORTANCE_STATS.reset(token)


def _record_feature_importance_stats(
    *, n_models_loaded: int, n_folds: int | None = None
) -> None:
    """Publish the load count to the active collector, when one is open."""
    stats = _FEATURE_IMPORTANCE_STATS.get()
    if stats is None:
        return
    stats.update(
        n_models_loaded=n_models_loaded,
        n_folds=len(FOLD_NAMES) if n_folds is None else int(n_folds),
        n_features=len(INDICATORS),
    )


def _fold_panel_grid(fold_count: int) -> tuple[int, int]:
    """(rows, columns) for the per-fold panels plus the combined panel.

    Three columns, one row per three panels: the historical five folds plus
    the combined panel keep the 2x3 layout, and a different configured count
    grows the grid instead of indexing past it.
    """
    columns = 3
    return math.ceil((fold_count + 1) / columns), columns


def _available_fold_names(predictions_dict: dict) -> list[str]:
    """The configured fold names present in a (possibly partial) run.

    A subset run emits only its selected fold records; diagnostics must render
    those instead of indexing every configured fold and failing per plot.
    """
    names = [name for name in FOLD_NAMES if name in predictions_dict]
    if not names:
        raise ValueError(
            "no configured fold records in predictions_dict; "
            f"expected some of {FOLD_NAMES}"
        )
    return names


def _fold_index(fold_name: str) -> int:
    """The integer index behind a ``fold_<i>`` name."""
    return int(fold_name.split("_", 1)[1])


def load_validation_predictions(predictions_path: Path | None = None) -> dict:
    """Load validation predictions from pickle file (size- and sha256-verified).

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

    data: dict = load_verified(
        predictions_path,
        registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME,
    )
    logger.info("Loaded predictions for %d folds: %s", len(data), list(data.keys()))

    return data


def plot_actual_vs_predicted(predictions_dict: dict, output_path: Path | None = None) -> Path:
    """Create scatter plots of actual vs predicted values per fold with diagonal reference.

    The combined panel labels its pooled R² with its convention — the pooled
    prevailing-mean denominator (the pooled test mean) — instead of a bare R².
    When every fold record carries the persisted per-row HAR forecasts
    (``har_forecast_log``), the pooled model-vs-HAR value is shown beside it,
    so the panel names both dual conventions rather than implying one.

    Args:
        predictions_dict: Dict of fold predictions from load_validation_predictions()
        output_path: Where to save the figure. If None, uses default outputs_dir.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "actual_vs_predicted.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_names = _available_fold_names(predictions_dict)
    rows, columns = _fold_panel_grid(len(fold_names))
    fig, axes = plt.subplots(rows, columns, figsize=(18, 6 * rows))
    axes_flat = axes.flatten()

    all_true = []
    all_pred = []
    all_fold = []

    for i, fold_name in enumerate(fold_names):
        fold_idx = _fold_index(fold_name)
        ax = axes_flat[i]
        fold_data = predictions_dict[fold_name]
        y_true = fold_data["test_true"]
        y_pred = fold_data["test_pred"]

        all_true.append(y_true)
        all_pred.append(y_pred)
        all_fold.extend([fold_idx] * len(y_true))

        ax.scatter(y_true, y_pred, c=FOLD_COLORS[fold_idx], alpha=0.5, s=10, edgecolors="none")

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
        ax.set_title(f"Fold {fold_idx} — Actual vs Predicted", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    # Combined plot (last subplot)
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_fold = np.array(all_fold)

    ax_combined = axes_flat[len(fold_names)]
    for fold_idx in sorted(set(all_fold.tolist())):
        mask = all_fold == fold_idx
        ax_combined.scatter(all_true[mask], all_pred[mask], c=FOLD_COLORS[fold_idx],
                           alpha=0.4, s=8, edgecolors="none", label=f"Fold {fold_idx}")

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    ax_combined.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = x")

    from sklearn.metrics import r2_score
    # The combined panel names its convention: the pooled prevailing-mean
    # denominator (the pooled test mean), never a bare R². When every fold
    # record carries the persisted per-row HAR forecasts, the pooled
    # model-vs-HAR value is shown next to it, so the panel does not imply a
    # single convention.
    combined_r2 = r2_score(all_true, all_pred)
    combined_label = f"pooled $R^2$ vs pooled mean = {combined_r2:.4f}"
    if all("har_forecast_log" in predictions_dict[name] for name in fold_names):
        from packages.ml.model.training.training import oos_r2_vs_har

        all_har = np.concatenate(
            [
                np.asarray(predictions_dict[name]["har_forecast_log"])
                for name in fold_names
            ]
        )
        combined_label += (
            f"\npooled $R^2$ vs HAR = "
            f"{oos_r2_vs_har(all_true, all_pred, all_har):.4f}"
        )
    ax_combined.text(0.05, 0.95, combined_label, transform=ax_combined.transAxes,
                    fontsize=12, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax_combined.set_xlabel("Actual", fontsize=11, fontweight="bold")
    ax_combined.set_ylabel("Predicted", fontsize=11, fontweight="bold")
    ax_combined.set_title("All Folds Combined", fontsize=13, fontweight="bold")
    ax_combined.grid(alpha=0.3)
    ax_combined.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    save_figure_atomically(output_path, dpi=300, bbox_inches="tight")
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
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "residuals_distribution.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_names = _available_fold_names(predictions_dict)
    rows, columns = _fold_panel_grid(len(fold_names))
    fig, axes = plt.subplots(rows, columns, figsize=(18, 6 * rows))
    axes_flat = axes.flatten()

    all_residuals = []

    for i, fold_name in enumerate(fold_names):
        fold_idx = _fold_index(fold_name)
        ax = axes_flat[i]
        fold_data = predictions_dict[fold_name]
        residuals = fold_data["test_true"] - fold_data["test_pred"]

        all_residuals.append(residuals)

        ax.hist(residuals, bins=80, color=FOLD_COLORS[fold_idx], alpha=0.7, density=True, edgecolor="white", linewidth=0.5)

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
        ax.set_title(f"Fold {fold_idx} — Residual Distribution", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    # Combined histogram (last subplot)
    ax_combined = axes_flat[len(fold_names)]
    all_residuals_concat = np.concatenate(all_residuals)

    for fold_name, fold_residuals in zip(fold_names, all_residuals):
        fold_idx = _fold_index(fold_name)
        ax_combined.hist(fold_residuals, bins=80, color=FOLD_COLORS[fold_idx], alpha=0.3, density=True,
                        edgecolor="white", linewidth=0.5, label=f"Fold {fold_idx}")

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
    save_figure_atomically(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved residuals distribution plot to %s", output_path)
    plt.close()

    return output_path


class CheckpointResolutionError(FileNotFoundError):
    """No run record names the fold's checkpoint; refusing to guess a filename."""


def _checkpoint_from_fold_summary(summary_path: Path, fold_idx: int) -> Path | None:
    """The checkpoint path recorded in the fold summary, or None."""
    if not summary_path.is_file():
        return None
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    entry = summary.get(f"fold_{fold_idx}")
    stored = entry.get("model_path") if isinstance(entry, dict) else None
    return resolve_stored_path(stored) if stored else None


def _checkpoint_from_run_manifest(manifest_path: Path, fold_idx: int) -> Path | None:
    """The checkpoint path recorded in the run manifest's artifact list, or None.

    Run-scoped entries (``checkpoints/<run_id>/fold_<i>.pkl``) win over the
    legacy bare ``checkpoints/fold_<i>_model.pkl`` form, so a manifest that
    carries both still resolves the run-scoped checkpoint.
    """
    if not manifest_path.is_file():
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    run_scoped: list[str] = []
    legacy: list[str] = []
    for stored in manifest.get("artifacts", {}):
        path = Path(stored)
        if (
            path.name == f"fold_{fold_idx}.pkl"
            and path.parent.parent.name == "checkpoints"
        ):
            run_scoped.append(stored)
        elif (
            path.name == f"fold_{fold_idx}_model.pkl"
            and path.parent.name == "checkpoints"
        ):
            legacy.append(stored)
    for stored in run_scoped or legacy:
        return resolve_stored_path(stored)
    return None


def resolve_fold_checkpoint(
    fold_idx: int,
    summary_path: Path | None = None,
    manifest_path: Path | None = None,
) -> Path:
    """Resolve fold ``fold_idx``'s checkpoint through the run's recorded paths.

    Resolution order: the fold summary's ``model_path`` for the fold, then the
    run manifest's artifact list. The summary is the run's own record and is
    written by the training phase, while the manifest is finalized only after
    Phase 5 — so during a run the summary is the freshest record of the
    run-scoped checkpoint, and the manifest covers readers that only have the
    finalized artifact index. When neither record names the checkpoint the
    resolver raises ``CheckpointResolutionError`` instead of guessing a
    filename (the old ``fold_<i>_model.pkl`` convention is gone).

    Args:
        fold_idx: Fold index (0-based).
        summary_path: Path to fold_training_summary.json. If None, uses the
            artifacts results directory.
        manifest_path: Path to run_manifest.json. If None, uses the artifacts
            root.

    Returns:
        The recorded absolute checkpoint path. Existence is not required here:
        the feature-importance plot treats a recorded-but-absent file as the
        intentional partial-checkout case (warning + zeros for that fold).
    """
    if summary_path is None:
        summary_path = ARTIFACTS_DIR / "results" / "fold_training_summary.json"
    if manifest_path is None:
        manifest_path = ARTIFACTS_DIR / "run_manifest.json"

    recorded = _checkpoint_from_fold_summary(Path(summary_path), fold_idx)
    if recorded is not None:
        return recorded

    from_manifest = _checkpoint_from_run_manifest(Path(manifest_path), fold_idx)
    if from_manifest is not None:
        return from_manifest

    raise CheckpointResolutionError(
        f"No checkpoint recorded for fold_{fold_idx}: neither {summary_path} "
        f"nor {manifest_path} names it; refusing to guess a filename"
    )


def _load_persisted_feature_names() -> list[str] | None:
    """The fold schema's persisted feature names, or None when not recorded.

    ``folds_metadata.pkl`` is a verified artifact; when it carries
    ``feature_names`` the plot compares that order with ``INDICATORS`` before
    mapping any importance by position. A checkout without the folds artifact
    (or a schema that predates the names) has nothing to compare.
    """
    folds_path = ARTIFACTS_DIR / "folds" / "folds_metadata.pkl"
    if not folds_path.is_file():
        return None
    payload = load_verified(
        folds_path,
        registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"fold metadata at {folds_path} is not a mapping")
    names = payload.get("feature_names")
    if names is None:
        return None
    if not isinstance(names, (list, tuple)):
        raise ValueError(
            f"persisted feature_names in {folds_path} is not a sequence: {names!r}"
        )
    return [str(name) for name in names]


def _assert_feature_name_order(persisted_names: list[str] | None) -> None:
    """Refuse to map importances onto a different persisted column order."""
    if persisted_names is None:
        return
    expected = list(INDICATORS)
    if list(persisted_names) != expected:
        raise ValueError(
            "persisted feature names do not match INDICATORS order: "
            f"expected {expected!r}, actual {list(persisted_names)!r}"
        )


def _assert_importance_width(importances: np.ndarray, model_path: Path) -> None:
    """Require the model's importance vector to match the pinned feature list."""
    actual_width = len(importances)
    if importances.ndim != 1 or actual_width != len(INDICATORS):
        raise ValueError(
            "feature-importance width mismatch: "
            f"expected {len(INDICATORS)} features (INDICATORS), actual "
            f"{actual_width} for {model_path}"
        )


def plot_feature_importance(output_path: Path | None = None) -> Path:
    """Create horizontal bar charts of top 15 feature importances per fold and aggregate.

    The plot is source-only: every checkpoint comes from the run's recorded
    artifact paths (``resolve_fold_checkpoint``), so it takes no predictions or
    summary frame. The fold summary is the first record, then the run manifest.
    A fold whose checkpoint is not named by either record raises
    ``CheckpointResolutionError`` — the plot never guesses a filename. Each
    resolved checkpoint is size- and sha256-verified before loading
    (``load_verified``). A recorded path whose file is absent is the
    intentional partial-checkout case (checkpoints pruned after a run): that
    fold contributes zeros with a warning, preserving the pre-existing behavior
    so the remaining folds still render.

    A subset run narrows the rendered folds through the
    ``generate_training_diagnostics`` context; a direct call renders every
    configured fold. The docstring's guards still apply per rendered fold.

    Guards:
        Before any positional mapping, the model's importance width must equal
        ``len(INDICATORS)`` — the pinned 37 features. Both a narrower and a
        wider model raise ``ValueError`` naming the expected and actual widths;
        a wider model can no longer surface a bare ``IndexError`` while
        rendering. When the fold schema persists ``feature_names``, that list
        must equal ``INDICATORS`` in order or the plot raises ``ValueError``
        instead of mislabelling columns.

    Visibility:
        The log and the visualization manifest record ``n_models_loaded``.
        Missing checkpoints keep the existing arithmetic — each contributes a
        zero vector, so four of five loaded folds leave the aggregate at
        ``4/5 = 0.8x`` the loaded-fold mean — and the aggregate panel states
        the loaded count and the 37-feature contract instead of hiding the
        dilution.

    Limits:
        The values are XGBoost's default ``gain`` importances (the sklearn
        wrapper's default importance type), not causal effects or independent
        contributions: gain is within-fold, in-sample split improvement, and
        correlated features split credit (a lag correlated with another can
        look small even when the pair matters jointly). Treat the ranking as a
        diagnostic, not as feature-selection evidence.

    Args:
        output_path: Where to save the figure. If None, uses default outputs_dir.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "feature_importance.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fold_names = _context_fold_names()
    if not fold_names:
        raise ValueError("plot_feature_importance received no fold names")
    fold_indices = [_fold_index(name) for name in fold_names]

    all_importances = []
    n_models_loaded = 0

    try:
        _assert_feature_name_order(_load_persisted_feature_names())

        for fold_idx in fold_indices:
            model_path = resolve_fold_checkpoint(fold_idx)
            if not model_path.exists():
                logger.warning("Model not found at %s, skipping fold %d", model_path, fold_idx)
                all_importances.append(np.zeros(len(INDICATORS)))
                continue

            model = load_verified(
                model_path,
                registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME,
            )
            importances = np.asarray(model.feature_importances_, dtype=float)
            _assert_importance_width(importances, model_path)
            all_importances.append(importances)
            n_models_loaded += 1
    finally:
        _record_feature_importance_stats(
            n_models_loaded=n_models_loaded, n_folds=len(fold_names)
        )

    if n_models_loaded < len(fold_names):
        logger.warning(
            "Feature importance loaded n_models_loaded=%d of %d checkpoints; "
            "missing checkpoints contribute zeros (aggregate scaled by %.2f)",
            n_models_loaded,
            len(fold_names),
            n_models_loaded / len(fold_names),
        )
    else:
        logger.info(
            "Feature importance loaded n_models_loaded=%d of %d checkpoints",
            n_models_loaded,
            len(fold_names),
        )

    all_importances = np.array(all_importances)

    rows, columns = _fold_panel_grid(len(fold_names))
    fig, axes = plt.subplots(rows, columns, figsize=(22, 7 * rows))
    axes_flat = axes.flatten()

    for i, fold_idx in enumerate(fold_indices):
        ax = axes_flat[i]
        importances = all_importances[i]

        # Top 15
        top_indices = np.argsort(importances)[-15:][::-1]
        top_names = [INDICATORS[idx] for idx in top_indices]
        top_values = importances[top_indices]

        ax.barh(range(len(top_indices)), top_values, color=FOLD_COLORS[fold_idx], alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Fold {fold_idx} — Top 15 of {len(INDICATORS)} Features",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

    # Aggregate importance (last subplot)
    ax_agg = axes_flat[len(fold_names)]
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
    ax_agg.set_title(
        f"Aggregate (Across Folds) — Top 15 of {len(INDICATORS)} Features",
        fontsize=13,
        fontweight="bold",
    )
    # Name the dilution instead of hiding it: a missing checkpoint contributes
    # zeros, so the aggregate is scaled by n_models_loaded / n_folds (0.8x for
    # four of five). The count and the 37-feature contract stay in the figure.
    ax_agg.text(
        0.99,
        0.02,
        f"Loaded {n_models_loaded}/{len(fold_names)} checkpoints "
        f"({n_models_loaded / len(fold_names):.1f}× aggregate); "
        f"{len(INDICATORS)} features",
        transform=ax_agg.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax_agg.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_figure_atomically(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved feature importance plot to %s", output_path)
    plt.close()

    return output_path


def load_evaluation_summary(evaluation_summary_path: Path | None = None) -> dict:
    """Load the persisted evaluation tables used by the sigma-scale panel.

    The default is ``results/evaluation_summary.json`` beside the stored loss
    vectors. A missing file or a payload without ``decile_qlike`` raises a
    named error; the caller decides whether the panel can render without it.
    """
    if evaluation_summary_path is None:
        evaluation_summary_path = (
            ARTIFACTS_DIR / "results" / EVALUATION_SUMMARY_FILENAME
        )
    path = Path(evaluation_summary_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation summary not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation summary at {path} is not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("decile_qlike"), dict
    ):
        raise ValueError(
            f"Evaluation summary at {path} does not carry a decile_qlike table"
        )
    return payload


def sigma_decile_records(
    bins: list[dict],
    value_key: str = "qlike_model",
) -> list[dict]:
    """Persisted decile rows on the sigma scale, without re-binning.

    ``sigma_center`` is ``exp(mean_y_true_log)``; empty or non-finite bins are
    skipped. The persisted ``decile`` and ``n_rows`` fields are carried through
    unchanged, so the panel keeps the headline floor-retained row set and the
    global edges recorded in the evaluation summary.
    """
    records = []
    for row in bins:
        mean_log = row.get("mean_y_true_log")
        value = row.get(value_key)
        if mean_log is None or value is None:
            continue
        mean_log = float(mean_log)
        value = float(value)
        if not math.isfinite(mean_log) or not math.isfinite(value):
            continue
        records.append(
            {
                "decile": int(row["decile"]),
                "sigma_center": math.exp(mean_log),
                "value": value,
                "n_rows": int(row["n_rows"]),
            }
        )
    return records


def _plot_sigma_decile_panel(ax, evaluation_summary: dict) -> None:
    """Draw pooled QLIKE by sigma decile from the persisted decile table."""
    deciles = evaluation_summary["decile_qlike"]
    pooled = deciles.get("pooled") or {}
    bins = pooled.get("bins") or []
    model = sigma_decile_records(bins, value_key="qlike_model")
    har = sigma_decile_records(bins, value_key="qlike_har")

    if model:
        ax.plot(
            [record["sigma_center"] for record in model],
            [record["value"] for record in model],
            color="steelblue",
            marker="o",
            linewidth=2,
            markersize=5,
            label="Pooled model QLIKE",
        )
    if har:
        ax.plot(
            [record["sigma_center"] for record in har],
            [record["value"] for record in har],
            color="darkorange",
            marker="s",
            linewidth=2,
            markersize=5,
            label="Pooled HAR QLIKE",
        )

    ax.set_xlabel(
        f"Actual {SIGMA_UNIT}; decile mean, floor-retained rows, no re-binning",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("QLIKE (per-day σ units)", fontsize=12, fontweight="bold")
    ax.set_title(
        "QLIKE by σ Decile (persisted evaluation table)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)


def plot_error_by_volatility_bin(predictions_dict: dict, output_path: Path | None = None) -> Path:
    """Plot mean absolute error by log-sigma decile plus a sigma-scale panel.

    The left panel bins the stored log target with
    ``pd.qcut(..., duplicates="drop")``; its x centres are mean ``log σ`` and
    its errors are in log points, so both axes declare those units. The right
    panel renders the persisted evaluation decile table on the sigma scale:
    the x centres are ``exp()`` of the table's mean log-sigma centres, the rows
    are the headline floor-retained bins, and no re-binning happens.

    ``pd.qcut(..., duplicates="drop")`` can realize fewer than ten bins when
    the test target is constant or duplicate-heavy. The guard is warn-only:
    the fold and the realized bin count are logged and the available bins
    still render. Live folds realize ten of ten bins and never warn. A
    constant ``test_true`` also makes sklearn's ``r2_score`` return 1.0 when
    the forecasts match and 0.0 otherwise, never NaN; that metric convention
    is intentional and is not a quarantine.

    Args:
        predictions_dict: Dict of fold predictions from load_validation_predictions()
        output_path: Where to save the figure. If None, uses default outputs_dir.

    Returns:
        Path to saved figure.
    """
    if output_path is None:
        from packages.ml.model.utils import get_project_paths
        output_path = get_project_paths()["outputs_dir"] / "error_by_volatility_bin.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_sigma) = plt.subplots(1, 2, figsize=(20, 8))

    fold_names = _available_fold_names(predictions_dict)
    for fold_name in fold_names:
        fold_idx = _fold_index(fold_name)
        fold_data = predictions_dict[fold_name]
        y_true = fold_data["test_true"]
        y_pred = fold_data["test_pred"]
        abs_errors = np.abs(y_true - y_pred)

        # Bin the stored log sigma into deciles. ``duplicates="drop"`` collapses
        # duplicate quantile edges, so a constant or duplicate-heavy target can
        # realize fewer than ten bins (a constant target realizes none). The
        # guard is warn-only: name the fold and the realized count, then render
        # the bins that exist. Live folds are 10/10.
        deciles = np.asarray(
            pd.qcut(y_true, q=10, labels=False, duplicates="drop"),
            dtype=np.float64,
        )
        finite_deciles = deciles[np.isfinite(deciles)]
        n_bins = int(finite_deciles.max()) + 1 if finite_deciles.size else 0
        if n_bins < 10:
            logger.warning(
                "log σ decile collapse for %s: pd.qcut realized %d of 10 "
                "bins; rendering the bins that exist",
                fold_name,
                n_bins,
            )

        bin_centers = []
        bin_means = []
        bin_stds = []

        for b in range(n_bins):
            mask = deciles == b
            if mask.sum() == 0:
                continue
            bin_errors = abs_errors[mask]
            bin_log_sigma = y_true[mask]

            bin_centers.append(bin_log_sigma.mean())
            bin_means.append(bin_errors.mean())
            bin_stds.append(bin_errors.std())

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                    color=FOLD_COLORS[fold_idx], marker="o", capsize=4, linewidth=2,
                    markersize=6, label=f"Fold {fold_idx}", alpha=0.8)

    ax.set_xlabel(
        f"Actual {LOG_SIGMA_UNIT}; decile mean",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        f"Mean Absolute Error ({LOG_POINT_UNIT})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        "Prediction Error by log σ Decile (log scale)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    try:
        evaluation_summary = load_evaluation_summary()
    except (FileNotFoundError, ValueError) as exc:
        logger.info("Sigma decile panel unavailable: %s", exc)
        ax_sigma.text(
            0.5,
            0.5,
            "Persisted evaluation decile table unavailable",
            ha="center",
            va="center",
            transform=ax_sigma.transAxes,
            fontsize=11,
        )
        ax_sigma.set_xlabel(
            f"Actual {SIGMA_UNIT}; decile mean",
            fontsize=12,
            fontweight="bold",
        )
        ax_sigma.set_ylabel(
            "QLIKE (per-day σ units)", fontsize=12, fontweight="bold"
        )
        ax_sigma.set_title(
            "QLIKE by σ Decile (persisted evaluation table)",
            fontsize=14,
            fontweight="bold",
        )
        ax_sigma.grid(alpha=0.3)
    else:
        _plot_sigma_decile_panel(ax_sigma, evaluation_summary)

    plt.tight_layout()
    save_figure_atomically(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved error by log σ decile plot to %s", output_path)
    plt.close()

    return output_path


def generate_training_diagnostics(
    predictions_dict: dict | None = None,
    summary_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    stats: dict | None = None,
) -> list[Path]:
    """Generate all training diagnostic visualizations.

    Args:
        predictions_dict: Dict of fold predictions. If None, loads from default path.
        summary_df: DataFrame from load_fold_training_summary(). If None, loads from default path.
        output_dir: Directory to save figures. If None, uses default outputs_dir.
        stats: Optional manifest stats block. When provided, the
            feature-importance load count is recorded under
            ``feature_importance``.

    Returns:
        List of paths to generated figures.
    """
    if predictions_dict is None:
        predictions_dict = load_validation_predictions()

    if summary_df is None:
        summary_df = load_fold_training_summary()

    if output_dir is None:
        from packages.ml.model.utils import get_project_paths
        output_dir = get_project_paths()["outputs_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_names = _available_fold_names(predictions_dict)

    plots = [
        (
            "actual_vs_predicted",
            lambda output_path: plot_actual_vs_predicted(
                predictions_dict, output_path
            ),
        ),
        (
            "residuals_distribution",
            lambda output_path: plot_residuals(predictions_dict, output_path),
        ),
        (
            "feature_importance",
            lambda output_path: plot_feature_importance(output_path),
        ),
        (
            "error_by_volatility_bin",
            lambda output_path: plot_error_by_volatility_bin(
                predictions_dict, output_path
            ),
        ),
    ]

    paths = []
    with selected_fold_names(fold_names), collect_feature_importance_stats() as importance_stats:
        for plot_name, plot_function in plots:
            path = render_plot("diagnostics", plot_name, plot_function, output_dir)
            if path is not None:
                paths.append(path)
    if stats is not None and importance_stats:
        stats["feature_importance"] = dict(importance_stats)

    logger.info("Generated %d training diagnostic visualizations", len(paths))
    return paths
