"""Tests for the per-fold timing decomposition and its timing-plot consumer.

The reported fold cost must tell the truth: ``fold_time_seconds`` is the full
fold — the model fit (train + inference), the HAR benchmark fit, and the
remaining fold work — and ``har_time_seconds`` records the HAR fit that used to
sit outside the fold timer. The checks drive a real ``phase_4_training`` against
synthetic artifacts under a temporary root, so the live inputs and the stored
artifacts are neither read nor written; the consumer check runs the real timing
plot against a synthetic summary frame and spies on the columns it reads and
the stacked bars it draws.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_timing.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.training import training
from packages.ml.model.visualization import performance_comparison

N_ROWS = 60
BLOCK_ROWS = 10
N_INDICATORS = 3
N_TICKER_FEATURES = 1
# The enclosing fold timer and the nested component timers share one clock, so
# the decomposition holds exactly; this tolerance only absorbs measurement
# noise between the nested reads.
TIMING_TOLERANCE = 1e-6

# The per-fold summary keys that predate the HAR timing component. The new
# schema adds ``har_time_seconds`` and changes nothing else.
PREEXISTING_SUMMARY_KEYS = frozenset(
    {
        "train_date_min",
        "train_date_max",
        "test_date_min",
        "test_date_max",
        "n_train_samples",
        "n_inner_val_samples",
        "n_inner_purge_rows",
        "best_iteration",
        "eval_history",
        "n_test_samples",
        "n_test_total",
        "n_test_floor_excluded",
        "n_test_scored",
        "n_train_floor_excluded",
        "n_warmup_excluded",
        "headline_row_set",
        "n_features",
        "model_path",
        "test_r2",
        "test_mae",
        "test_rmse",
        "test_qlike",
        "har_r2",
        "har_qlike",
        "clip_count",
        "clip_rate",
        "train_time_seconds",
        "inference_time_seconds",
        "fold_time_seconds",
        "n_features_indicators",
        "n_features_ticker",
    }
)


class _StubModel:
    """Picklable stand-in fold model with the surface the training path reads."""

    def get_params(self) -> dict:
        return {"n_estimators": 5}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), -2.0, dtype=np.float64)


def _stub_trainer(X_train, y_train, X_val=None, y_val=None):
    return _StubModel(), {}


def _stub_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Constant components with no warm-up rows; the HAR fit stays real."""
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _write_synthetic_artifacts(root: Path) -> None:
    """Write a small live-shaped Phase 2-3 input set under ``root``."""
    rng = np.random.default_rng(11)
    y = rng.uniform(0.05, 0.6, size=N_ROWS)
    X = rng.normal(size=(N_ROWS, N_INDICATORS + N_TICKER_FEATURES))
    X[:, 0] = np.arange(N_ROWS, dtype=np.float64)
    dates = np.datetime64("2000-01-03") + np.arange(N_ROWS)

    metadata = {
        "ticker": np.array([f"T{row % 3}" for row in range(N_ROWS)]),
        "date": dates,
        "regime": (np.arange(N_ROWS) % 7).astype(np.int8),
        "n_indicators": N_INDICATORS,
        "n_ticker_features": N_TICKER_FEATURES,
    }

    folds_metadata = {}
    for fold_idx in range(5):
        train_stop = BLOCK_ROWS * (fold_idx + 1)
        test_stop = BLOCK_ROWS * (fold_idx + 2)
        folds_metadata[f"fold_{fold_idx}"] = {
            "train_indices": list(range(train_stop)),
            "test_indices": list(range(train_stop, test_stop)),
            "n_train": train_stop,
            "n_test": test_stop - train_stop,
            "train_date_min": str(dates[0]),
            "train_date_max": str(dates[train_stop - 1]),
            "test_date_min": str(dates[train_stop]),
            "test_date_max": str(dates[test_stop - 1]),
        }

    inputs = root / "inputs"
    folds = root / "folds"
    inputs.mkdir(parents=True)
    folds.mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "results").mkdir()
    np.save(inputs / "X.npy", X)
    np.save(inputs / "Y.npy", y)
    joblib.dump(metadata, inputs / "metadata.pkl")
    joblib.dump(folds_metadata, folds / "folds_metadata.pkl")
    # The live readers verify recorded hashes before loading; the synthetic
    # inputs must carry their registry entries like the real writers'.
    record_artifact(inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME)
    record_artifact(
        folds / "folds_metadata.pkl", registry_path=root / REGISTRY_FILENAME
    )


def _run_phase(root: Path) -> tuple:
    """Drive one real ``phase_4_training`` run against the temporary root."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
        stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
        stack.enter_context(
            mock.patch.object(training, "load_har_regressors", _stub_har_regressors)
        )
        stack.enter_context(
            mock.patch.object(training, "train_xgboost_model", _stub_trainer)
        )
        return training.phase_4_training()


def _prepared_root(tmpdir: str) -> Path:
    """A temporary artifacts root with the synthetic Phase 2-3 inputs."""
    root = Path(tmpdir) / "artifacts"
    assert not root.resolve().is_relative_to(REPO_ROOT), (
        "the phase run must target a temporary root, never live artifacts"
    )
    root.mkdir()
    _write_synthetic_artifacts(root)
    return root


def _exercised_run(root: Path) -> tuple:
    """One phase run plus its persisted summary JSON under ``root``."""
    validation, summary, model_paths = _run_phase(root)
    summary_json = json.loads(
        (root / "results" / "fold_training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    run_manifest = json.loads(
        (root / "run_manifest.json").read_text(encoding="utf-8")
    )
    return validation, summary, summary_json, model_paths, run_manifest


def _assert_decomposes(prediction: dict, fold_key: str) -> None:
    """The full fold cost encloses train + inference + HAR."""
    assert prediction["har_time_seconds"] > 0, fold_key
    components = (
        prediction["train_time_seconds"]
        + prediction["inference_time_seconds"]
        + prediction["har_time_seconds"]
    )
    assert prediction["fold_time_seconds"] >= components - TIMING_TOLERANCE, fold_key


def test_fold_timing_decomposes_into_train_inference_and_har() -> None:
    """Every fold records the HAR fit and a total that encloses the parts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        validation, summary, summary_json, _model_paths, _manifest = _exercised_run(
            root
        )

        for fold_idx in range(5):
            fold_key = f"fold_{fold_idx}"
            _assert_decomposes(validation[fold_key], fold_key)
            _assert_decomposes(summary[fold_key], fold_key)

            stored = summary_json[fold_key]
            assert "har_time_seconds" in stored, fold_key
            _assert_decomposes(stored, fold_key)

            # The model step's fields keep their meanings; only the fold total
            # grew to enclose the HAR fit and the remaining fold work.
            assert stored["train_time_seconds"] == validation[fold_key][
                "train_time_seconds"
            ]
            assert stored["inference_time_seconds"] == validation[fold_key][
                "inference_time_seconds"
            ]
            assert (
                stored["fold_time_seconds"]
                >= stored["train_time_seconds"]
                + stored["inference_time_seconds"]
            )


def test_summary_schema_gains_only_the_har_timing_key() -> None:
    """The per-fold summary keeps every pre-existing key and value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        validation, _summary, summary_json, _model_paths, manifest = _exercised_run(
            root
        )

        for fold_key, stored in summary_json.items():
            assert set(stored) == PREEXISTING_SUMMARY_KEYS | {
                "har_time_seconds"
            }, (
                fold_key,
                sorted(
                    set(stored) ^ (PREEXISTING_SUMMARY_KEYS | {"har_time_seconds"})
                ),
            )
            prediction = validation[fold_key]
            for key in (
                "train_time_seconds",
                "inference_time_seconds",
                "fold_time_seconds",
                "test_r2",
                "test_qlike",
                "clip_count",
                "n_test_samples",
            ):
                assert stored[key] == prediction[key], (fold_key, key)
            assert stored["model_path"] == prediction["fold_model_path"], fold_key

        # The checkpoint payload is unchanged: loadable, and it predicts exactly
        # what the fold recorded.
        checkpoint = root / "checkpoints" / manifest["run_id"] / "fold_0.pkl"
        loaded = joblib.load(checkpoint)
        scored = validation["fold_0"]["test_pred"]
        assert np.array_equal(
            loaded.predict(
                np.zeros((len(scored), N_INDICATORS + N_TICKER_FEATURES))
            ),
            scored,
        )


def _summary_frame() -> pd.DataFrame:
    """A complete summary frame whose fold totals exceed their components."""
    return pd.DataFrame(
        {
            "fold": [f"fold_{fold_idx}" for fold_idx in range(5)],
            "train_time_seconds": [10.0, 12.0, 14.0, 16.0, 18.0],
            "inference_time_seconds": [1.0, 1.2, 1.4, 1.6, 1.8],
            "har_time_seconds": [0.5, 0.6, 0.7, 0.8, 0.9],
            "fold_time_seconds": [12.0, 14.0, 16.5, 18.9, 21.0],
            "test_r2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "test_mae": [0.1, 0.2, 0.3, 0.4, 0.5],
            "test_rmse": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_train_samples": [100, 200, 300, 400, 500],
            "n_test_samples": [50, 60, 70, 80, 90],
            "n_features_indicators": [37] * 5,
            "n_features_ticker": [0] * 5,
        }
    )


def test_timing_plot_reads_the_har_component_and_keeps_the_figure_name() -> None:
    """The stacked bars include HAR and the figure name and caller are stable."""
    frame = _summary_frame()
    accessed: list = []
    barh_calls: list = []
    text_labels: list = []

    real_getitem = pd.DataFrame.__getitem__
    real_barh = plt.Axes.barh
    real_text = plt.Axes.text

    def recording_getitem(self, key):
        accessed.append(key)
        return real_getitem(self, key)

    def recording_barh(ax, y, width, height, **kwargs):
        barh_calls.append(
            (np.asarray(width, dtype=float), kwargs.get("left"), kwargs.get("label"))
        )
        return real_barh(ax, y, width, height, **kwargs)

    def recording_text(ax, x, y, s, *args, **kwargs):
        text_labels.append((float(x), s))
        return real_text(ax, x, y, s, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "model_outputs"
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(pd.DataFrame, "__getitem__", recording_getitem)
            )
            stack.enter_context(mock.patch.object(plt.Axes, "barh", recording_barh))
            stack.enter_context(mock.patch.object(plt.Axes, "text", recording_text))
            figure = performance_comparison.plot_timing_breakdown(
                frame, output_dir / "timing_breakdown.png"
            )

        assert figure == output_dir / "timing_breakdown.png"
        assert figure.is_file() and figure.stat().st_size > 0

        assert "har_time_seconds" in accessed, (
            "the timing plot must read the HAR timing component; "
            f"read columns: {sorted({str(key) for key in accessed})}"
        )

        har_segments = [
            call for call in barh_calls if call[2] and "HAR" in call[2]
        ]
        assert len(har_segments) == 1, barh_calls
        widths, left, _label = har_segments[0]
        assert np.allclose(widths, frame["har_time_seconds"].values)
        assert np.allclose(
            np.asarray(left, dtype=float),
            frame["train_time_seconds"].values
            + frame["inference_time_seconds"].values,
        )

        # The total label is the full fold cost, not just the model step.
        for fold_time in frame["fold_time_seconds"].values:
            assert any(
                label == f"{fold_time:.1f}s" for _x, label in text_labels
            ), (fold_time, text_labels)

        # The caller keeps the same figure name.
        paths = performance_comparison.generate_performance_comparisons(
            frame, output_dir
        )
        assert output_dir / "timing_breakdown.png" in paths
        assert len(paths) == 4
        assert not list(Path(tmpdir).rglob("*.tmp"))


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
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
