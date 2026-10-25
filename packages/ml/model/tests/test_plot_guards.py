"""Guards for metric-plot axis limits and absolute heatmap scales.

The R² panel must show every observed score, including negatives, with zero
inside the visible range. The heatmap must color raw metrics against a declared
per-metric reference instead of re-spanning [0, 1] over the displayed folds,
and a constant column must render at the declared neutral value rather than
dividing by zero.

Every rendering writes under ``tmp_path``; no live output is touched.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import conftest  # noqa: F401  (puts the repository root on sys.path)
from matplotlib.collections import QuadMesh

from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.visualization import (
    metrics_visualizer,
    training_diagnostics,
    visualize,
)
from packages.ml.model.visualization.data_loader import load_fold_training_summary


METRIC_COLUMNS = ("test_r2", "test_mae", "test_rmse")

EXPECTED_PLOTS = {
    "metrics": {"r2_scores_by_fold", "mae_rmse_comparison", "metrics_heatmap"},
    "temporal": {
        "temporal_fold_ranges",
        "fold_sample_distribution",
        "temporal_metric_trend",
    },
    "performance": {
        "metrics_comparison",
        "timing_breakdown",
        "sample_distribution",
        "feature_count_summary",
    },
    "diagnostics": {
        "actual_vs_predicted",
        "residuals_distribution",
        "feature_importance",
        "error_by_volatility_bin",
    },
}
ALL_PLOTS = set().union(*EXPECTED_PLOTS.values())


def _summary_frame(
    *,
    r2: list[float] | None = None,
    mae: list[float] | None = None,
    rmse: list[float] | None = None,
) -> pd.DataFrame:
    """Five live-shaped folds with optional metric overrides."""
    r2 = list(r2) if r2 is not None else [0.10, 0.11, 0.12, 0.13, 0.14]
    mae = list(mae) if mae is not None else [0.20, 0.21, 0.22, 0.23, 0.24]
    rmse = list(rmse) if rmse is not None else [0.30, 0.31, 0.32, 0.33, 0.34]
    rows = []
    for index, (test_r2, test_mae, test_rmse) in enumerate(zip(r2, mae, rmse)):
        rows.append(
            {
                "fold": f"fold_{index}",
                "train_date_min": f"200{index}-01-03",
                "train_date_max": f"200{index}-06-29",
                "test_date_min": f"200{index}-07-01",
                "test_date_max": f"200{index}-12-31",
                "n_train_samples": 1_000 + index,
                "n_test_samples": 200 + index,
                "test_r2": test_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "train_time_seconds": 10.0 + index,
                "inference_time_seconds": 1.0 + index,
                "har_time_seconds": 2.0 + index,
                "fold_time_seconds": 13.0 + index,
                "n_features_indicators": 37,
                "n_features_ticker": 0,
            }
        )
    return pd.DataFrame(rows)


def _render_and_capture(
    plot_function: Callable[[pd.DataFrame, Path], Path],
    frame: pd.DataFrame,
    output_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Render one plot and keep the figure open for inspection."""
    captured: dict[str, object] = {}
    real_savefig = metrics_visualizer.plt.savefig

    def capture(*args, **kwargs):
        captured["figure"] = metrics_visualizer.plt.gcf()
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(metrics_visualizer.plt, "savefig", capture)
    returned = plot_function(frame, output_path)
    assert returned == output_path
    assert "figure" in captured, "plot did not reach savefig"
    return captured["figure"]


def _heatmap_values(figure) -> np.ndarray:
    """The values seaborn mapped through the main heatmap colormap."""
    meshes = [
        collection
        for axis in figure.axes
        for collection in axis.collections
        if isinstance(collection, QuadMesh)
        and np.shape(collection.get_array()) == (len(METRIC_COLUMNS), 5)
    ]
    assert len(meshes) == 1, [np.shape(m.get_array()) for m in meshes]
    values = np.ma.filled(np.asarray(meshes[0].get_array(), dtype=float), np.nan)
    return values


@pytest.mark.parametrize(
    "r2_values",
    [
        [-0.30, -0.20, -0.25, -0.10, -0.15],
        [0.10, 0.20, 0.30, 0.40, 0.35],
    ],
)
def test_r2_axis_contains_zero_and_the_observed_range(
    r2_values: list[float], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Limits contain 0 and min/max; negatives are neither clipped nor inverted."""
    figure = _render_and_capture(
        metrics_visualizer.plot_r2_scores,
        _summary_frame(r2=r2_values),
        tmp_path / "r2_scores_by_fold.png",
        monkeypatch,
    )
    bar_axes = [axis for axis in figure.axes if axis.patches]
    assert len(bar_axes) == 1
    axis = bar_axes[0]

    lower, upper = axis.get_ylim()
    observed_min = min(r2_values)
    observed_max = max(r2_values)
    assert lower < upper, (lower, upper)
    assert lower <= observed_min, (lower, observed_min)
    assert upper >= observed_max, (upper, observed_max)
    assert lower <= 0.0 <= upper, (lower, upper)

    heights = np.array([patch.get_height() for patch in axis.patches])
    np.testing.assert_allclose(heights, r2_values, rtol=0.0, atol=1e-12)
    assert any(
        line.get_label() == "R² = 0 (no skill)" for line in axis.lines
    ), [line.get_label() for line in axis.lines]


def test_constant_heatmap_column_is_finite_and_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flat metric column maps to neutral 0.5, never 0/0 NaN."""
    frame = _summary_frame(
        r2=[0.10, 0.12, 0.14, 0.16, 0.18],
        mae=[0.25, 0.25, 0.25, 0.25, 0.25],
        rmse=[0.30, 0.31, 0.32, 0.33, 0.34],
    )
    figure = _render_and_capture(
        metrics_visualizer.plot_metric_heatmap,
        frame,
        tmp_path / "metrics_heatmap.png",
        monkeypatch,
    )
    values = _heatmap_values(figure)
    assert np.isfinite(values).all(), values
    neutral = metrics_visualizer.HEATMAP_NEUTRAL_SCORE
    assert neutral == pytest.approx(0.5)
    constant_row = values[METRIC_COLUMNS.index("test_mae")]
    np.testing.assert_allclose(
        constant_row, np.full(5, neutral), rtol=0.0, atol=1e-12
    )

    mesh = next(
        collection
        for axis in figure.axes
        for collection in axis.collections
        if isinstance(collection, QuadMesh)
        and np.shape(collection.get_array()) == (len(METRIC_COLUMNS), 5)
    )
    assert mesh.norm.vmin == pytest.approx(0.0)
    assert mesh.norm.vmax == pytest.approx(1.0)
    rendered = mesh.get_facecolors().reshape(len(METRIC_COLUMNS), 5, 4)
    assert np.isfinite(rendered).all()
    np.testing.assert_allclose(
        rendered[METRIC_COLUMNS.index("test_mae")],
        np.tile(mesh.cmap(neutral), (5, 1)),
        rtol=0.0,
        atol=1e-12,
    )


def test_heatmap_colors_follow_the_declared_absolute_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Colors are hand-computable from the declared reference, not fold ranks."""
    frame = _summary_frame(
        r2=[-0.50, 0.00, 0.25, 0.50, 1.00],
        mae=[0.05, 0.10, 0.20, 0.40, 0.80],
        rmse=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    expected = np.array(
        [
            [0.25, 0.500, 0.625, 0.75, 1.00],
            [0.95, 0.900, 0.800, 0.60, 0.20],
            [0.90, 0.800, 0.700, 0.60, 0.50],
        ]
    )

    figure = _render_and_capture(
        metrics_visualizer.plot_metric_heatmap,
        frame,
        tmp_path / "metrics_heatmap.png",
        monkeypatch,
    )
    values = _heatmap_values(figure)
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-12)

    assert metrics_visualizer.HEATMAP_REFERENCE["test_r2"] == {
        "minimum": -1.0,
        "maximum": 1.0,
        "higher_is_better": True,
    }
    assert metrics_visualizer.HEATMAP_REFERENCE["test_mae"] == {
        "minimum": 0.0,
        "maximum": 1.0,
        "higher_is_better": False,
    }
    assert metrics_visualizer.HEATMAP_REFERENCE["test_rmse"] == {
        "minimum": 0.0,
        "maximum": 1.0,
        "higher_is_better": False,
    }

    colorbar_labels = [
        axis.get_ylabel() for axis in figure.axes if axis.get_ylabel()
    ]
    assert any(
        "reference" in label.lower()
        and "R²" in label
        and "MAE/RMSE" in label
        for label in colorbar_labels
    ), colorbar_labels

    docstring = metrics_visualizer.plot_metric_heatmap.__doc__ or ""
    assert "reference" in docstring.lower()
    assert "0.5" in docstring


def _predictions() -> dict:
    rng = np.random.default_rng(7)
    payload = {}
    for index in range(5):
        truth = rng.normal(size=120)
        prediction = truth + rng.normal(scale=0.1, size=120)
        payload[f"fold_{index}"] = {
            "test_true": truth,
            "test_pred": prediction,
            "test_r2": 0.10 + index * 0.01,
            "har_forecast_log": prediction + 0.05,
        }
    return payload


def _write_summary(tmp_path: Path, frame: pd.DataFrame) -> Path:
    payload = {
        row["fold"]: {key: value for key, value in row.items() if key != "fold"}
        for row in frame.to_dict(orient="records")
    }
    path = tmp_path / "fold_training_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stub_checkpoint_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "stub_model.pkl"
    checkpoint.write_bytes(b"stub")
    model = SimpleNamespace(
        feature_importances_=np.arange(len(INDICATORS), dtype=np.float64)
    )
    monkeypatch.setattr(
        training_diagnostics, "resolve_fold_checkpoint", lambda _index: checkpoint
    )
    monkeypatch.setattr(
        training_diagnostics, "load_verified", lambda *args, **kwargs: model
    )
    # Persisted-name checks are covered by test_importance_guards.py; keep this
    # stub independent of the live fold schema.
    monkeypatch.setattr(
        training_diagnostics, "_load_persisted_feature_names", lambda: None
    )


