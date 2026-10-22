"""Guards for feature-importance width, persisted names, and load visibility.

Feature importances are mapped onto ``INDICATORS`` positionally, so a model
whose importance vector is narrower is silently mislabelled and a wider one
raises a bare ``IndexError`` at the last possible moment. The plot now names
the expected and actual widths before any positional mapping, verifies the
persisted fold-schema feature names when they exist, reports how many
checkpoint models loaded in the visualization manifest and log, and keeps the
existing missing-checkpoint arithmetic: zeros for the absent fold, so one
missing checkpoint out of five leaves the aggregate at ``4/5 = 0.8x``.

Every rendering writes under ``tmp_path``; the live check only reads existing
artifacts. No live artifact or output is written.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import conftest  # noqa: F401  (puts the repository root on sys.path)
from matplotlib.text import Text

from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.visualization import training_diagnostics, visualize

EXPECTED_FEATURES = len(INDICATORS)


def _stub_model(width: int) -> SimpleNamespace:
    """A checkpoint stand-in with one dominant feature at position zero."""
    importances = np.zeros(width, dtype=np.float64)
    if width:
        importances[0] = 1.0
    return SimpleNamespace(
        feature_importances_=importances,
        n_features_in_=width,
    )


def _stub_plot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    width: int,
    missing: set[int] | None = None,
    persisted_names: list[str] | None = None,
    folds_schema: bool = False,
) -> list[Path]:
    """Point the plot at stub checkpoints and an optional fold schema."""
    missing = missing or set()
    root = tmp_path / "artifacts"
    if folds_schema:
        (root / "folds").mkdir(parents=True, exist_ok=True)
        (root / "folds" / "folds_metadata.pkl").write_bytes(b"stub")
    paths = []
    for index in range(len(training_diagnostics.FOLD_NAMES)):
        path = tmp_path / f"fold_{index}.pkl"
        if index not in missing:
            path.write_bytes(b"stub")
        paths.append(path)
    model = _stub_model(width)

    def load(path, *args, **kwargs):
        if Path(path).name == "folds_metadata.pkl":
            return {"feature_names": persisted_names}
        return model

    monkeypatch.setattr(training_diagnostics, "ARTIFACTS_DIR", root)
    monkeypatch.setattr(
        training_diagnostics, "resolve_fold_checkpoint", lambda index: paths[index]
    )
    monkeypatch.setattr(training_diagnostics, "load_verified", load)
    return paths


def _capture_figure(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Capture the figure at save time so its bars can be inspected."""
    captured: dict[str, object] = {}
    real_savefig = training_diagnostics.plt.savefig

    def capture(*args, **kwargs):
        captured["figure"] = training_diagnostics.plt.gcf()
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(training_diagnostics.plt, "savefig", capture)
    return captured


def _figure_text(figure) -> str:
    return "\n".join(text.get_text() for text in figure.findobj(Text))


def _aggregate_bars(figure) -> tuple[object, dict[str, float]]:
    axis = figure.axes[len(training_diagnostics.FOLD_NAMES)]
    labels = [tick.get_text() for tick in axis.get_yticklabels()]
    heights = {
        label: patch.get_width() for label, patch in zip(labels, axis.patches)
    }
    return axis, heights


def _summary_payload() -> dict[str, dict]:
    rows = {}
    for index in range(5):
        rows[f"fold_{index}"] = {
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
            "n_features_indicators": EXPECTED_FEATURES,
            "n_features_ticker": 0,
        }
    return rows


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


def test_narrower_model_raises_a_named_width_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narrower model is refused instead of silently mislabelled."""
    output = tmp_path / "feature_importance.png"
    actual_width = EXPECTED_FEATURES - 7
    _stub_plot(monkeypatch, tmp_path, width=actual_width)

    with pytest.raises(
        ValueError,
        match=rf"expected {EXPECTED_FEATURES} features.*actual {actual_width}",
    ):
        training_diagnostics.plot_feature_importance(output)

    assert not output.exists(), "the plot must not render a narrower model"


def test_wider_model_raises_a_named_width_error_not_an_index_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wider model raises the named ValueError, never a bare IndexError."""
    output = tmp_path / "feature_importance.png"
    actual_width = EXPECTED_FEATURES + 3
    _stub_plot(monkeypatch, tmp_path, width=actual_width)

    with pytest.raises(
        ValueError,
        match=rf"expected {EXPECTED_FEATURES} features.*actual {actual_width}",
    ):
        training_diagnostics.plot_feature_importance(output)

    assert not output.exists(), "the plot must not render a wider model"


def test_correct_width_and_matching_names_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pinned width renders, and matching fold-schema names are accepted."""
    output = tmp_path / "feature_importance.png"
    _stub_plot(
        monkeypatch,
        tmp_path,
        width=EXPECTED_FEATURES,
        persisted_names=list(INDICATORS),
        folds_schema=True,
    )
    captured = _capture_figure(monkeypatch)

    returned = training_diagnostics.plot_feature_importance(output)

    assert returned == output
    assert output.is_file() and output.stat().st_size > 0
    axis, heights = _aggregate_bars(captured["figure"])
    assert len(axis.patches) == 15
    assert heights[INDICATORS[0]] == pytest.approx(1.0, abs=1e-12)
    assert f"{EXPECTED_FEATURES} features" in _figure_text(captured["figure"])


def test_persisted_feature_name_reorder_raises_a_named_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted name order that differs from INDICATORS refuses to render."""
    output = tmp_path / "feature_importance.png"
    reordered = list(INDICATORS)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    _stub_plot(
        monkeypatch,
        tmp_path,
        width=EXPECTED_FEATURES,
        persisted_names=reordered,
        folds_schema=True,
    )

    with pytest.raises(ValueError) as excinfo:
        training_diagnostics.plot_feature_importance(output)

    message = str(excinfo.value)
    assert "feature names" in message, message
    assert "INDICATORS" in message, message
    assert "expected" in message and "actual" in message, message
    assert not output.exists(), "the plot must not render a reordered schema"


def test_one_missing_checkpoint_is_visible_and_keeps_the_0_8x_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Four of five checkpoints load; the missing fold still contributes zeros."""
    output = tmp_path / "feature_importance.png"
    _stub_plot(monkeypatch, tmp_path, width=EXPECTED_FEATURES, missing={4})
    captured = _capture_figure(monkeypatch)
    caplog.set_level(logging.WARNING, logger=training_diagnostics.__name__)

    returned = training_diagnostics.plot_feature_importance(output)

    assert returned == output and output.is_file()
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "n_models_loaded=4" in message and "5" in message for message in messages
    ), messages

    axis, heights = _aggregate_bars(captured["figure"])
    assert len(axis.patches) == 15
    assert heights[INDICATORS[0]] == pytest.approx(0.8, abs=1e-12)
    text = _figure_text(captured["figure"])
    assert "0.8" in text, text
    assert f"{EXPECTED_FEATURES} features" in text, text


def test_manifest_records_n_models_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The visualization manifest carries the load count beside the plot status."""
    _stub_plot(monkeypatch, tmp_path, width=EXPECTED_FEATURES, missing={4})
    summary_path = tmp_path / "fold_training_summary.json"
    summary_path.write_text(json.dumps(_summary_payload()), encoding="utf-8")
    monkeypatch.setattr(
        visualize, "load_validation_predictions", lambda _path=None: _predictions()
    )
    output_dir = tmp_path / "plots"

    results = visualize.generate_all_visualizations(
        summary_path=summary_path,
        predictions_path=tmp_path / "unused.pkl",
        output_dir=output_dir,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    importance = manifest["stats"]["feature_importance"]
    assert importance["n_models_loaded"] == 4
    assert importance["n_folds"] == len(training_diagnostics.FOLD_NAMES)
    assert importance["n_features"] == EXPECTED_FEATURES
    assert sum(len(paths) for paths in results.values()) == 14
    status = next(
        entry
        for entry in manifest["plots"]
        if entry["category"] == "diagnostics"
        and entry["plot"] == "feature_importance"
    )
    assert status["status"] == "ok", status


def test_live_feature_importance_figure_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-regression: a live-shaped importance figure renders read-only."""
    artifacts = conftest.ARTIFACTS_DIR
    required = (
        artifacts / "folds" / "folds_metadata.pkl",
        artifacts / "results" / "fold_training_summary.json",
        artifacts / "run_manifest.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("requires live model artifacts absent from this checkout")
    captured = _capture_figure(monkeypatch)
    output = tmp_path / "feature_importance.png"

    returned = training_diagnostics.plot_feature_importance(output)

    assert returned == output and output.is_file()
    axis, _heights = _aggregate_bars(captured["figure"])
    assert len(axis.patches) == 15
    assert f"{EXPECTED_FEATURES} features" in _figure_text(captured["figure"])
