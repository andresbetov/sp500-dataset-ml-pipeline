"""Tests for QLIKE/HAR statistics at the visualization-manifest consumer.

The visualization path must call the previously dead ``extract_*`` helpers and
write their results plus the per-fold headline statistics into
``manifest.json``. The manifest's QLIKE gap is defined once as
``har_qlike - test_qlike``: a positive gap means the model's QLIKE is lower
(better) than the HAR benchmark.

The live check reads the generated fold summary read-only; every rendering
uses a temporary output directory. No live artifact is written.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR

from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.visualization import (
    data_loader,
    training_diagnostics,
    visualize,
)

MODEL_ROOT = Path(__file__).resolve().parents[1]
VISUALIZE_PATH = MODEL_ROOT / "visualization" / "visualize.py"

# Current live fold-0 values, recorded at the branch point. Interim pins:
# re-record them when the generated artifacts are regenerated.
LIVE_FOLD_0 = {
    "test_qlike": 0.057832782932382025,
    "har_qlike": 0.12572092515717045,
    "har_r2": -0.3149361367920722,
    "qlike_gap": 0.06788814222478842,
}

PER_FOLD_KEYS = {"test_qlike", "har_qlike", "har_r2", "qlike_gap"}
MEAN_KEYS = {
    "test_qlike_mean",
    "har_qlike_mean",
    "har_r2_mean",
    "qlike_gap_mean",
}
EXPECTED_PLOTS = {
    ("metrics", "r2_scores_by_fold"),
    ("metrics", "mae_rmse_comparison"),
    ("metrics", "metrics_heatmap"),
    ("temporal", "temporal_fold_ranges"),
    ("temporal", "fold_sample_distribution"),
    ("temporal", "temporal_metric_trend"),
    ("performance", "metrics_comparison"),
    ("performance", "timing_breakdown"),
    ("performance", "sample_distribution"),
    ("performance", "feature_count_summary"),
    ("diagnostics", "actual_vs_predicted"),
    ("diagnostics", "residuals_distribution"),
    ("diagnostics", "feature_importance"),
    ("diagnostics", "error_by_volatility_bin"),
}


def _summary_frame() -> pd.DataFrame:
    """Five live-shaped folds; fold 0 carries the branch-point QLIKE values."""
    qlike = [
        (
            LIVE_FOLD_0["test_qlike"],
            LIVE_FOLD_0["har_qlike"],
            LIVE_FOLD_0["har_r2"],
        ),
        (0.060, 0.090, 0.510),
        (0.050, 0.080, -0.640),
        (0.065, 0.095, -0.460),
        (0.055, 0.085, 0.350),
    ]
    rows = []
    for index in range(5):
        test_qlike, har_qlike, har_r2 = qlike[index]
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
                "test_qlike": test_qlike,
                "har_qlike": har_qlike,
                "har_r2": har_r2,
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


def _write_summary(tmp_path: Path, frame: pd.DataFrame | None = None) -> Path:
    frame = _summary_frame() if frame is None else frame
    payload = {
        row["fold"]: {key: value for key, value in row.items() if key != "fold"}
        for row in frame.to_dict(orient="records")
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
    # Persisted-name checks are covered by test_importance_guards.py; keep this
    # stub independent of the live fold schema.
    monkeypatch.setattr(
        training_diagnostics, "_load_persisted_feature_names", lambda: None
    )


def _reject_json_constant(token: str) -> None:
    raise AssertionError(f"manifest contains a non-strict JSON constant: {token}")


def _load_manifest(output_dir: Path) -> dict:
    text = (output_dir / "manifest.json").read_text(encoding="utf-8")
    return json.loads(text, parse_constant=_reject_json_constant)


def _called_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return called


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def test_qlike_gap_sign_convention_matches_a_synthetic_fixture() -> None:
    """A positive gap means the model beats HAR: gap = har_qlike - test_qlike."""
    frame = _summary_frame()
    frame.loc[0, "test_qlike"] = 0.20
    frame.loc[0, "har_qlike"] = 0.10

    stats = data_loader.extract_qlike_stats(frame)

    assert stats["per_fold"]["fold_0"]["qlike_gap"] == pytest.approx(
        -0.10, rel=0.0, abs=1e-12
    )
    assert "har_qlike - test_qlike" in stats["sign_convention"]
    assert "positive" in stats["sign_convention"]
    assert "better" in stats["sign_convention"]
    # A model with a lower QLIKE than HAR produces a positive gap.
    assert stats["per_fold"]["fold_1"]["qlike_gap"] > 0
    assert stats["means"]["qlike_gap_mean"] == pytest.approx(
        _mean(
            record["qlike_gap"] for record in stats["per_fold"].values()
        ),
        rel=0.0,
        abs=1e-12,
    )


def test_stats_extraction_does_not_mutate_the_fold_frame() -> None:
    frame = _summary_frame()
    snapshot = frame.copy(deep=True)

    data_loader.extract_training_time_stats(frame)
    data_loader.extract_metric_stats(frame)
    data_loader.extract_sample_stats(frame)
    data_loader.extract_qlike_stats(frame)

    pd.testing.assert_frame_equal(frame, snapshot)


