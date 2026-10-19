"""Tests for per-category/per-plot isolation and the visualization manifest.

A failing plot must be contained: its siblings in the same category still
render, the other categories still run, and ``manifest.json`` records one
``{"category", "plot", "status", "error"}`` entry per attempted plot. Plot
generation never mutates the caller's frame. The live check renders the real
14-plot set into a temporary directory and only reads existing artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR

from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.visualization import (
    metrics_visualizer,
    performance_comparison,
    temporal_analysis,
    training_diagnostics,
    visualize,
)

LIVE_SUMMARY = ARTIFACTS_DIR / "results" / "fold_training_summary.json"
LIVE_PREDICTIONS = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"

EXPECTED_PLOTS = {
    "metrics": {
        "r2_scores_by_fold",
        "mae_rmse_comparison",
        "metrics_heatmap",
    },
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
MANIFEST_KEYS = {"category", "plot", "status", "error"}


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
        truth = rng.normal(size=120)
        prediction = truth + rng.normal(scale=0.1, size=120)
        payload[f"fold_{index}"] = {
            "test_true": truth,
            "test_pred": prediction,
            "test_r2": 0.10 + index * 0.01,
            "har_forecast_log": prediction + 0.05,
        }
    return payload


def _write_summary(tmp_path: Path) -> Path:
    payload = {
        row["fold"]: {key: value for key, value in row.items() if key != "fold"}
        for row in _summary_frame().to_dict(orient="records")
    }
    path = tmp_path / "fold_training_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stub_checkpoint_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the diagnostics plots hermetic: no live checkpoint required."""
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


def _manifest(output_dir: Path) -> list[dict]:
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    return payload["plots"]


def _assert_manifest_shape(entries: list[dict]) -> None:
    for entry in entries:
        assert set(entry) == MANIFEST_KEYS, entry
        assert entry["status"] in {"ok", "error"}, entry
        if entry["status"] == "ok":
            assert entry["error"] is None, entry
        else:
            assert isinstance(entry["error"], str) and entry["error"], entry


def test_failing_metrics_plot_leaves_its_siblings_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("metrics plot boom")

    monkeypatch.setattr(metrics_visualizer, "plot_mae_rmse_comparison", fail)
    output_dir = tmp_path / "plots"

    paths = metrics_visualizer.generate_metrics_visualizations(
        _summary_frame(), output_dir
    )

    assert {path.name for path in paths} == {
        "r2_scores_by_fold.png",
        "metrics_heatmap.png",
    }
    assert (output_dir / "r2_scores_by_fold.png").is_file()
    assert (output_dir / "metrics_heatmap.png").is_file()
    assert not (output_dir / "mae_rmse_comparison.png").exists()


def test_failing_temporal_plot_leaves_its_siblings_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("temporal plot boom")

    monkeypatch.setattr(temporal_analysis, "plot_fold_sample_distribution", fail)
    output_dir = tmp_path / "plots"

    paths = temporal_analysis.generate_temporal_visualizations(
        _summary_frame(), output_dir
    )

    assert {path.name for path in paths} == {
        "temporal_fold_ranges.png",
        "temporal_metric_trend.png",
    }
    assert (output_dir / "temporal_fold_ranges.png").is_file()
    assert (output_dir / "temporal_metric_trend.png").is_file()


def test_failing_performance_plot_leaves_its_siblings_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("performance plot boom")

    monkeypatch.setattr(performance_comparison, "plot_timing_breakdown", fail)
    output_dir = tmp_path / "plots"

    paths = performance_comparison.generate_performance_comparisons(
        _summary_frame(), output_dir
    )

    assert {path.name for path in paths} == {
        "metrics_comparison.png",
        "sample_distribution.png",
        "feature_count_summary.png",
    }
    assert (output_dir / "metrics_comparison.png").is_file()
    assert (output_dir / "sample_distribution.png").is_file()
    assert (output_dir / "feature_count_summary.png").is_file()


def test_failing_diagnostics_plot_leaves_its_siblings_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_checkpoint_loads(monkeypatch, tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("diagnostics plot boom")

    monkeypatch.setattr(training_diagnostics, "plot_residuals", fail)
    output_dir = tmp_path / "plots"

    paths = training_diagnostics.generate_training_diagnostics(
        _predictions(), _summary_frame(), output_dir
    )

    assert {path.name for path in paths} == {
        "actual_vs_predicted.png",
        "feature_importance.png",
        "error_by_volatility_bin.png",
    }
    assert (output_dir / "actual_vs_predicted.png").is_file()
    assert (output_dir / "feature_importance.png").is_file()
    assert (output_dir / "error_by_volatility_bin.png").is_file()


def test_manifest_lists_every_plot_with_ok_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_checkpoint_loads(monkeypatch, tmp_path)
    monkeypatch.setattr(
        visualize, "load_validation_predictions", lambda _path=None: _predictions()
    )
    output_dir = tmp_path / "plots"

    results = visualize.generate_all_visualizations(
        summary_path=_write_summary(tmp_path),
        predictions_path=tmp_path / "unused.pkl",
        output_dir=output_dir,
    )

    assert set(results) == set(EXPECTED_PLOTS)
    assert sum(len(paths) for paths in results.values()) == 14
    assert {path.name for paths in results.values() for path in paths} == {
        f"{plot}.png" for plot in ALL_PLOTS
    }

    entries = _manifest(output_dir)
    assert len(entries) == 14
    _assert_manifest_shape(entries)
    assert all(entry["status"] == "ok" for entry in entries)
    assert {entry["category"] for entry in entries} == set(EXPECTED_PLOTS)
    assert {entry["plot"] for entry in entries} == ALL_PLOTS
    for entry in entries:
        assert (output_dir / f"{entry['plot']}.png").is_file()


def test_manifest_records_a_failing_plot_without_dropping_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_checkpoint_loads(monkeypatch, tmp_path)
    monkeypatch.setattr(
        visualize, "load_validation_predictions", lambda _path=None: _predictions()
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("manifest plot boom")

    monkeypatch.setattr(metrics_visualizer, "plot_mae_rmse_comparison", fail)
    output_dir = tmp_path / "plots"

    results = visualize.generate_all_visualizations(
        summary_path=_write_summary(tmp_path),
        predictions_path=tmp_path / "unused.pkl",
        output_dir=output_dir,
    )

    assert sum(len(paths) for paths in results.values()) == 13
    entries = _manifest(output_dir)
    assert len(entries) == 14
    _assert_manifest_shape(entries)

    errors = [entry for entry in entries if entry["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["category"] == "metrics"
    assert errors[0]["plot"] == "mae_rmse_comparison"
    assert "manifest plot boom" in errors[0]["error"]
    assert sum(entry["status"] == "ok" for entry in entries) == 13


def test_failing_category_leaves_the_other_categories_and_records_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_checkpoint_loads(monkeypatch, tmp_path)
    monkeypatch.setattr(
        visualize, "load_validation_predictions", lambda _path=None: _predictions()
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("category boom")

    monkeypatch.setattr(visualize, "generate_temporal_visualizations", fail)
    output_dir = tmp_path / "plots"

    results = visualize.generate_all_visualizations(
        summary_path=_write_summary(tmp_path),
        predictions_path=tmp_path / "unused.pkl",
        output_dir=output_dir,
    )

    assert set(results) == set(EXPECTED_PLOTS)
    assert results["temporal"] == []
    assert sum(len(paths) for paths in results.values()) == 11
    entries = _manifest(output_dir)
    _assert_manifest_shape(entries)
    assert len(entries) == 12
    errors = [entry for entry in entries if entry["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["category"] == "temporal"
    assert errors[0]["plot"] == "*"
    assert "category boom" in errors[0]["error"]
    assert sum(entry["status"] == "ok" for entry in entries) == 11


def test_temporal_generation_does_not_mutate_the_caller_frame(tmp_path: Path) -> None:
    frame = _summary_frame()
    snapshot = frame.copy(deep=True)

    temporal_analysis.generate_temporal_visualizations(frame, tmp_path / "plots")

    pd.testing.assert_frame_equal(frame, snapshot)


def test_live_artifacts_render_all_fourteen_plots(tmp_path: Path) -> None:
    if not LIVE_SUMMARY.is_file() or not LIVE_PREDICTIONS.is_file():
        pytest.skip("requires live model artifacts absent from this checkout")

    output_dir = tmp_path / "plots"
    results = visualize.generate_all_visualizations(
        summary_path=LIVE_SUMMARY,
        predictions_path=LIVE_PREDICTIONS,
        output_dir=output_dir,
    )

    assert sum(len(paths) for paths in results.values()) == 14
    assert {path.name for paths in results.values() for path in paths} == {
        f"{plot}.png" for plot in ALL_PLOTS
    }
    entries = _manifest(output_dir)
    assert len(entries) == 14
    _assert_manifest_shape(entries)
    assert all(entry["status"] == "ok" for entry in entries), entries
