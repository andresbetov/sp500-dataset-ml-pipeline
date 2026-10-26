"""Tests for honest log units and the sigma-scale decile panel.

The stored target is log sigma. The diagnostics decile axis and the MAE/RMSE
metric labels therefore declare ``log σ`` / ``log points`` rather than an
unqualified volatility. A second diagnostics panel renders the persisted
evaluation decile table on the sigma scale: its centres are ``exp()`` of the
table's mean log-sigma centres, its rows are the persisted floor-retained
bins, and no re-binning happens. Existing decile errorbar values and the
fourteen-plot manifest are unchanged.

Every rendering writes under ``tmp_path``; the live evaluation table is only
read when a test explicitly uses it. No live artifact is written.
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import conftest  # noqa: F401  (puts the repository root on sys.path)
from matplotlib.text import Text

from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.visualization import (
    metrics_visualizer,
    training_diagnostics,
    visualize,
)

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

MODEL_LINE_LABEL = "Pooled model QLIKE"
HAR_LINE_LABEL = "Pooled HAR QLIKE"


def _summary_frame() -> pd.DataFrame:
    rows = []
    for index in range(5):
        rows.append(
            {
                "fold": f"fold_{index}",
                "train_date_min": f"200{index}-01-03",
                "train_date_max": f"200{index}-06-29",
                "test_date_min": f"200{index}-07-01",
                "test_date_max": f"200{index}-12-31",
                "n_train_samples": 1_000 + index,
                "n_test_samples": 200 + index,
                "test_r2": 0.10 + index * 0.01,
                "test_mae": 0.20 + index * 0.01,
                "test_rmse": 0.30 + index * 0.01,
                "train_time_seconds": 10.0 + index,
                "inference_time_seconds": 1.0 + index,
                "har_time_seconds": 2.0 + index,
                "fold_time_seconds": 13.0 + index,
                "n_features_indicators": 37,
                "n_features_ticker": 0,
            }
        )
    return pd.DataFrame(rows)


def _predictions() -> dict:
    rng = np.random.default_rng(7)
    payload = {}
    for index in range(5):
        truth = rng.normal(loc=-3.0, scale=0.7, size=120)
        prediction = truth + rng.normal(scale=0.1, size=120)
        payload[f"fold_{index}"] = {
            "test_true": truth,
            "test_pred": prediction,
            "test_r2": 0.10 + index * 0.01,
            "har_forecast_log": prediction + 0.05,
        }
    return payload


def _evaluation_summary() -> dict:
    """A persisted-shaped decile table with ten bins per fold and pooled."""
    folds: dict[str, dict] = {}
    for fold_index in range(5):
        bins = []
        for decile in range(10):
            mean_log = -5.0 + 0.4 * decile + 0.01 * fold_index
            bins.append(
                {
                    "decile": decile,
                    "edge_low": mean_log - 0.2,
                    "edge_high": mean_log + 0.2,
                    "n_rows": 100 * (decile + 1) + fold_index,
                    "mean_y_true_log": mean_log,
                    "sigma_center": math.exp(mean_log),
                    "qlike_model": 0.02 + 0.001 * decile + 0.0001 * fold_index,
                    "qlike_har": 0.08 - 0.005 * decile + 0.0001 * fold_index,
                }
            )
        folds[f"fold_{fold_index}"] = {
            "n_rows": sum(row["n_rows"] for row in bins),
            "quarantined": False,
            "bins": bins,
        }

    pooled_bins = []
    for decile in range(10):
        mean_log = -4.95 + 0.39 * decile
        pooled_bins.append(
            {
                "decile": decile,
                "edge_low": mean_log - 0.2,
                "edge_high": mean_log + 0.2,
                "n_rows": 500 * (decile + 1),
                "mean_y_true_log": mean_log,
                "sigma_center": math.exp(mean_log),
                "qlike_model": 0.021 + 0.0009 * decile,
                "qlike_har": 0.079 - 0.0048 * decile,
            }
        )

    return {
        "schema_version": 1,
        "scale": "log target (log daily sigma); QLIKE in per-day sigma units",
        "row_set": "floor-retained rows (raw RV strictly above RV_FLOOR)",
        "floor_decomposition": {
            "row_set": (
                "headline = without_floor (floor-retained rows); with_floor adds "
                "the persisted at-or-below-floor test-block rows"
            ),
            "counts": {"n_rows": sum(row["n_rows"] for row in pooled_bins)},
        },
        "decile_qlike": {
            "n_deciles": 10,
            "edges": [row["edge_low"] for row in pooled_bins]
            + [pooled_bins[-1]["edge_high"]],
            "edges_source": (
                "global pooled edges over the headline (floor-retained) rows of "
                "the non-quarantined folds, applied to every fold"
            ),
            "folds": folds,
            "pooled": {
                "n_rows": sum(row["n_rows"] for row in pooled_bins),
                "n_valid_folds": 5,
                "n_total_folds": 5,
                "excluded_folds": [],
                "bins": pooled_bins,
            },
        },
    }


def _write_evaluation_summary(tmp_path: Path) -> Path:
    path = tmp_path / "artifacts" / "results" / "evaluation_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_evaluation_summary()), encoding="utf-8")
    return path


def _use_temp_evaluation_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    summary_path = _write_evaluation_summary(tmp_path)
    monkeypatch.setattr(
        training_diagnostics, "ARTIFACTS_DIR", summary_path.parents[1]
    )
    return summary_path


def _render_and_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Render the diagnostics decile plot and keep the figure for inspection."""
    captured: dict[str, object] = {}
    real_savefig = training_diagnostics.plt.savefig

    def capture(*args, **kwargs):
        captured["figure"] = training_diagnostics.plt.gcf()
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(training_diagnostics.plt, "savefig", capture)
    output_path = tmp_path / "error_by_volatility_bin.png"
    returned = training_diagnostics.plot_error_by_volatility_bin(
        _predictions(), output_path=output_path
    )
    assert returned == output_path
    assert "figure" in captured, "plot did not reach savefig"
    return output_path, captured["figure"]


def _figure_text(figure) -> str:
    return "\n".join(text.get_text() for text in figure.findobj(Text))


def _sigma_axis(figure):
    axes = [
        axis
        for axis in figure.axes
        if "σ" in axis.get_xlabel() and "log σ" not in axis.get_xlabel().lower()
    ]
    assert len(axes) == 1, [axis.get_xlabel() for axis in figure.axes]
    return axes[0]


def _write_summary(tmp_path: Path) -> Path:
    payload = {
        row["fold"]: {key: value for key, value in row.items() if key != "fold"}
        for row in _summary_frame().to_dict(orient="records")
    }
    path = tmp_path / "fold_training_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stub_checkpoint_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_log_axes_declare_log_sigma_and_log_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decile axis and its error use log-sigma/log-point units."""
    _use_temp_evaluation_table(tmp_path, monkeypatch)
    _, figure = _render_and_capture(tmp_path, monkeypatch)

    log_axis = figure.axes[0]
    xlabel = log_axis.get_xlabel().lower()
    ylabel = log_axis.get_ylabel().lower()
    title = log_axis.get_title().lower()

    assert "log σ" in xlabel, log_axis.get_xlabel()
    assert "per-day" in xlabel, log_axis.get_xlabel()
    assert "log points" in ylabel, log_axis.get_ylabel()
    assert "volatility" not in xlabel, log_axis.get_xlabel()
    assert "volatility" not in title, log_axis.get_title()
    assert "volatility" not in _figure_text(figure).lower()


def test_mae_rmse_labels_declare_log_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAE/RMSE are log-target errors, not raw volatility units."""
    captured: dict[str, object] = {}
    real_savefig = metrics_visualizer.plt.savefig

    def capture(*args, **kwargs):
        captured["figure"] = metrics_visualizer.plt.gcf()
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(metrics_visualizer.plt, "savefig", capture)
    output_path = tmp_path / "mae_rmse_comparison.png"
    returned = metrics_visualizer.plot_mae_rmse_comparison(
        _summary_frame(), output_path
    )
    assert returned == output_path
    figure = captured["figure"]

    bar_axes = [axis for axis in figure.axes if axis.patches]
    labels = [axis.get_ylabel().lower() for axis in bar_axes]
    assert len(bar_axes) == 2, labels
    mae_labels = [label for label in labels if "mae" in label]
    rmse_labels = [label for label in labels if "rmse" in label]
    assert len(mae_labels) == 1, labels
    assert len(rmse_labels) == 1, labels
    assert "log points" in mae_labels[0], mae_labels[0]
    assert "log points" in rmse_labels[0], rmse_labels[0]


def test_sigma_panel_uses_exp_of_persisted_log_centres(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sigma panel reads the persisted table and never re-bins."""
    summary = _evaluation_summary()
    _use_temp_evaluation_table(tmp_path, monkeypatch)
    _, figure = _render_and_capture(tmp_path, monkeypatch)

    axis = _sigma_axis(figure)
    assert "per-day" in axis.get_xlabel().lower(), axis.get_xlabel()
    assert "qlike" in axis.get_ylabel().lower(), axis.get_ylabel()

    lines = {line.get_label(): line for line in axis.lines}
    assert MODEL_LINE_LABEL in lines, list(lines)
    assert HAR_LINE_LABEL in lines, list(lines)

    pooled = summary["decile_qlike"]["pooled"]["bins"]
    expected_sigma = np.array(
        [math.exp(row["mean_y_true_log"]) for row in pooled], dtype=float
    )
    expected_model = np.array([row["qlike_model"] for row in pooled], dtype=float)
    expected_har = np.array([row["qlike_har"] for row in pooled], dtype=float)

    np.testing.assert_allclose(
        np.asarray(lines[MODEL_LINE_LABEL].get_xdata(), dtype=float),
        expected_sigma,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(lines[MODEL_LINE_LABEL].get_ydata(), dtype=float),
        expected_model,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(lines[HAR_LINE_LABEL].get_xdata(), dtype=float),
        expected_sigma,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(lines[HAR_LINE_LABEL].get_ydata(), dtype=float),
        expected_har,
        rtol=0.0,
        atol=1e-12,
    )

    # The panel carries the persisted bins unchanged: same deciles, same row
    # counts, and the floor-retained row set declared from the table.
    records = training_diagnostics.sigma_decile_records(
        pooled, value_key="qlike_model"
    )
    assert [record["decile"] for record in records] == [
        row["decile"] for row in pooled
    ]
    assert [record["n_rows"] for record in records] == [
        row["n_rows"] for row in pooled
    ]
    np.testing.assert_allclose(
        [record["sigma_center"] for record in records],
        expected_sigma,
        rtol=0.0,
        atol=1e-12,
    )
    assert sum(record["n_rows"] for record in records) == summary["decile_qlike"][
        "pooled"
    ]["n_rows"]

    text = _figure_text(figure).lower()
    assert "floor-retained" in text
    assert "no re-binning" in text


def test_existing_log_decile_errorbars_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log-scale decile centres/means/stds stay numerically identical."""
    predictions = _predictions()
    expected = []
    for fold_name in training_diagnostics.FOLD_NAMES:
        y_true = np.asarray(predictions[fold_name]["test_true"], dtype=float)
        y_pred = np.asarray(predictions[fold_name]["test_pred"], dtype=float)
        abs_errors = np.abs(y_true - y_pred)
        deciles = np.asarray(
            pd.qcut(y_true, q=10, labels=False, duplicates="drop")
        )
        expected.append(
            (
                np.array([y_true[deciles == b].mean() for b in range(10)]),
                np.array([abs_errors[deciles == b].mean() for b in range(10)]),
                np.array([abs_errors[deciles == b].std() for b in range(10)]),
            )
        )

    captured: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    real_errorbar = training_diagnostics.plt.Axes.errorbar

    def spy_errorbar(self, x, y, *args, **kwargs):
        captured.append(
            (
                np.asarray(x, dtype=float),
                np.asarray(y, dtype=float),
                np.asarray(kwargs["yerr"], dtype=float),
            )
        )
        return real_errorbar(self, x, y, *args, **kwargs)

    _use_temp_evaluation_table(tmp_path, monkeypatch)
    monkeypatch.setattr(training_diagnostics.plt.Axes, "errorbar", spy_errorbar)
    _, figure = _render_and_capture(tmp_path, monkeypatch)

    assert len(captured) == len(training_diagnostics.FOLD_NAMES)
    for (centers, means, stds), (expected_centers, expected_means, expected_stds) in zip(
        captured, expected
    ):
        np.testing.assert_allclose(
            centers, expected_centers, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(means, expected_means, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(stds, expected_stds, rtol=0.0, atol=1e-12)

    # The sigma panel is a second panel, not a replacement or a new PNG.
    assert len(figure.axes) == 2


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            if (
                "tmp_path" in _fn.__code__.co_varnames
                or "monkeypatch" in _fn.__code__.co_varnames
            ):
                continue
            try:
                _fn()
                print(f"PASS {_name}")
            except pytest.skip.Exception as exc:
                skipped += 1
                print(f"SKIP {_name}: {exc}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    suffix = f" ({skipped} skipped)" if skipped else ""
    print(f"All tests passed{suffix}")
