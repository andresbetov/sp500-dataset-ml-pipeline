"""Tests for the persisted per-row loss vectors and their alignment contract.

The training phase writes one row per fold-test-block row to
``results/per_row_losses.parquet``:

``fold, row_idx, date, ticker, y_true, y_pred_model, y_pred_har, loss_model,
loss_har, is_floor``

The vector covers the fold's whole test block: ``test_indices`` in fold order,
including the at-or-below-floor rows that the headline scored set excludes.
``is_floor`` is the persisted ``test_floor_mask`` inverted (True marks the
floor-excluded rows); the reporting layer never re-derives floor membership
from the target values. ``loss_model``/``loss_har`` are the elementwise
``exp(delta) - delta - 1`` terms of the same QLIKE functional the headline
uses (``qlike_loss``), so the mean over a row set reconciles exactly with the
fold metric.

These checks pin:

- the artifact's schema, per-fold length (``n_test`` of the fold metadata),
  and the total recomputed from the live artifacts behind the skip guard;
- ``row_idx``/``date``/``ticker`` alignment against ``metadata.pkl``;
- ``y_true`` against ``log(max(y, RV_FLOOR))``;
- ``is_floor`` against the persisted mask (no re-derivation);
- per-row loss parity with the QLIKE formula and with ``qlike_loss`` on the
  retained rows (the model/HAR QLIKE formula is reused unchanged);
- the retained forecasts equal the persisted ``test_pred``/``har_forecast_log``
  and carry no non-finite values;
- the evaluation summary is written beside the vectors as strict JSON.

All phase checks drive a real ``phase_4_training`` under a temporary artifact
root with a stub trainer and constant HAR regressors, so no live
``artifacts/**`` or ``data/model_outputs/**`` file is written. The live check
reads the stored artifacts only.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_loss_vectors.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR, requires_live_artifacts
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.training import training

N_ROWS = 60
BLOCK_ROWS = 10
N_INDICATORS = 3
N_TICKER_FEATURES = 1
# One at-or-below-floor row in every fold's test block (rows 10-19, 20-29,
# ..., 50-59), so every fold exercises the floor side of the vector.
TEST_FLOOR_ROWS = (12, 25, 38, 42, 57)
VECTOR_COLUMNS = [
    "fold",
    "row_idx",
    "date",
    "ticker",
    "y_true",
    "y_pred_model",
    "y_pred_har",
    "loss_model",
    "loss_har",
    "is_floor",
]


class _RecordingModel:
    """Picklable fold-model double with deterministic row-indexed forecasts."""

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        rows = np.asarray(X)[:, 0].astype(np.float64)
        return -1.4 + 0.05 * np.sin(rows)


def _stub_trainer(X_train, y_train, X_val=None, y_val=None):
    return _RecordingModel(), {}


def _stub_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Constant components with no warm-up rows; the HAR fit stays real."""
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _write_synthetic_artifacts(root: Path) -> None:
    """Write a live-shaped five-fold Phase 2-3 input set under ``root``."""
    rng = np.random.default_rng(23)
    y = rng.uniform(0.05, 0.6, size=N_ROWS)
    for row in TEST_FLOOR_ROWS:
        y[row] = training.RV_FLOOR
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
    record_artifact(inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME)
    record_artifact(
        folds / "folds_metadata.pkl", registry_path=root / REGISTRY_FILENAME
    )


def _run_phase() -> dict:
    """Drive one real ``phase_4_training`` run and read its outputs back.

    The temporary root is removed only after every payload has been read, so
    the checks never dereference a deleted directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        assert not root.resolve().is_relative_to(
            Path(__file__).resolve().parents[4]
        ), "the phase run must target a temporary root, never live artifacts"
        root.mkdir()
        _write_synthetic_artifacts(root)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
            stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(training, "load_har_regressors", _stub_har_regressors)
            )
            stack.enter_context(
                mock.patch.object(training, "train_xgboost_model", _stub_trainer)
            )
            validation, summary, _model_paths = training.phase_4_training()

        vectors_path = root / "results" / "per_row_losses.parquet"
        summary_path = root / "results" / "evaluation_summary.json"
        evidence = {
            "validation": validation,
            "summary": summary,
            "y": np.load(root / "inputs" / "Y.npy"),
            "metadata": joblib.load(root / "inputs" / "metadata.pkl"),
            "folds_metadata": joblib.load(root / "folds" / "folds_metadata.pkl"),
            "summary_json": json.loads(
                (root / "results" / "fold_training_summary.json").read_text(
                    encoding="utf-8"
                )
            ),
            "vectors_path": None,
            "evaluation_summary": None,
        }
        if vectors_path.is_file():
            evidence["vectors"] = pd.read_parquet(vectors_path)
        else:
            evidence["vectors"] = None
        if summary_path.is_file():
            evidence["evaluation_summary"] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        assert not list(root.rglob("*.tmp")), "the report writers left *.tmp residue"
        return evidence


def _fold_frame(vectors: pd.DataFrame, fold_key: str) -> pd.DataFrame:
    return vectors[vectors["fold"] == fold_key].reset_index(drop=True)


def _elementwise_qlike(actual_log: np.ndarray, forecast_log: np.ndarray) -> np.ndarray:
    delta = np.asarray(actual_log) - np.asarray(forecast_log)
    return np.exp(delta) - delta - 1.0


def test_phase_4_persists_the_per_row_loss_vector() -> None:
    """One vector row per fold-test-block row, aligned and QLIKE-consistent."""
    evidence = _run_phase()
    vectors = evidence["vectors"]
    assert vectors is not None, (
        "phase_4_training did not write results/per_row_losses.parquet"
    )
    assert list(vectors.columns) == VECTOR_COLUMNS, list(vectors.columns)

    validation = evidence["validation"]
    metadata = evidence["metadata"]
    folds_metadata = evidence["folds_metadata"]
    y = evidence["y"]
    fold_keys = sorted(
        key for key in folds_metadata if key.startswith("fold_")
    )
    expected_total = 0
    for fold_key in fold_keys:
        fold_data = folds_metadata[fold_key]
        test_indices = np.asarray(fold_data["test_indices"])
        expected_total += len(test_indices)
        frame = _fold_frame(vectors, fold_key)

        # The vector covers the fold's whole test block, in fold order, not
        # only the floor-retained rows the headline scores.
        assert len(frame) == len(test_indices) == fold_data["n_test"], fold_key
        assert frame["row_idx"].to_numpy().tolist() == test_indices.tolist(), fold_key

        prediction = validation[fold_key]
        persisted_mask = np.asarray(prediction["test_floor_mask"], dtype=bool)
        persisted_scored = np.asarray(prediction["test_indices_scored"])
        assert np.array_equal(
            frame["is_floor"].to_numpy(), ~persisted_mask
        ), fold_key

        # Alignment: date/ticker are the metadata rows at row_idx.
        assert np.array_equal(
            frame["date"].to_numpy(), np.asarray(metadata["date"])[test_indices]
        ), fold_key
        assert (
            frame["ticker"].to_numpy().tolist()
            == np.asarray(metadata["ticker"])[test_indices].tolist()
        ), fold_key

        # The target is the floored log target, including the floor rows.
        expected_true = np.log(np.maximum(y[test_indices], training.RV_FLOOR))
        assert np.allclose(
            frame["y_true"].to_numpy(), expected_true, rtol=0.0, atol=1e-12
        ), fold_key

        # The retained rows carry exactly the persisted headline forecasts.
        retained = ~frame["is_floor"].to_numpy()
        assert np.array_equal(
            frame.loc[retained, "y_pred_model"].to_numpy(),
            np.asarray(prediction["test_pred"]),
        ), fold_key
        assert np.array_equal(
            frame.loc[retained, "y_pred_har"].to_numpy(),
            np.asarray(prediction["har_forecast_log"]),
        ), fold_key
        assert np.array_equal(
            test_indices[retained], persisted_scored
        ), fold_key

        # Per-row losses are the unchanged QLIKE terms, and their retained-row
        # means reproduce the headline fold metrics.
        expected_model = _elementwise_qlike(
            frame["y_true"].to_numpy(), frame["y_pred_model"].to_numpy()
        )
        expected_har = _elementwise_qlike(
            frame["y_true"].to_numpy(), frame["y_pred_har"].to_numpy()
        )
        assert np.allclose(
            frame["loss_model"].to_numpy(), expected_model, rtol=0.0, atol=1e-12
        ), fold_key
        assert np.allclose(
            frame["loss_har"].to_numpy(), expected_har, rtol=0.0, atol=1e-12
        ), fold_key
        assert np.mean(frame.loc[retained, "loss_model"]) == pytest.approx(
            training.qlike_loss(
                frame.loc[retained, "y_true"].to_numpy(),
                frame.loc[retained, "y_pred_model"].to_numpy(),
            ),
            rel=1e-12,
            abs=1e-12,
        )
        assert np.mean(frame.loc[retained, "loss_model"]) == pytest.approx(
            evidence["summary"][fold_key]["test_qlike"], rel=1e-12, abs=1e-12
        )
        assert np.mean(frame.loc[retained, "loss_har"]) == pytest.approx(
            evidence["summary"][fold_key]["har_qlike"], rel=1e-12, abs=1e-12
        )

        # No non-finite value reaches the persisted vector.
        numeric = frame[
            ["y_true", "y_pred_model", "y_pred_har", "loss_model", "loss_har"]
        ].to_numpy(dtype=np.float64)
        assert np.isfinite(numeric).all(), fold_key

    live_total = sum(
        int(validation[fold_key]["n_test_total"]) for fold_key in fold_keys
    )
    assert len(vectors) == expected_total == live_total


def test_phase_4_writes_the_evaluation_summary_beside_the_vectors() -> None:
    """The evaluation summary exists, is strict JSON, and names every block."""
    evidence = _run_phase()
    payload = evidence["evaluation_summary"]
    assert payload is not None, (
        "phase_4_training did not write results/evaluation_summary.json"
    )
    for key in (
        "dual_oos_r2",
        "loss_distributions",
        "floor_decomposition",
        "decile_qlike",
        "regime_qlike",
        "date_clustered_gap",
        "dm_cw_gw",
        "mz",
        "expanding_origin",
        "calibration",
    ):
        assert key in payload, sorted(payload)

    # The summary is projected through the shared sanitizer: no NaN/Infinity
    # tokens (json.loads would have accepted them, so re-scan the text).
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        root.mkdir()
        _write_synthetic_artifacts(root)
        path = root / "results" / "evaluation_summary.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text

    assert payload["calibration"]["status"] == "not applied"
    assert payload["dm_cw_gw"]["hac_lag"] == 4
    assert payload["row_set"] == training.HEADLINE_ROW_SET
    for fold_key, block in payload["expanding_origin"]["folds"].items():
        assert block["n_points"] == evidence["folds_metadata"][fold_key]["n_test"]
    for fold_key in evidence["folds_metadata"]:
        if not fold_key.startswith("fold_"):
            continue
        assert len(payload["decile_qlike"]["folds"][fold_key]["bins"]) == 10


def test_vector_writer_creates_the_results_dir_and_leaves_no_tmp() -> None:
    """The parquet writer creates a missing parent and writes atomically."""
    from packages.ml.model.evaluation import report

    frame = report.loss_vector_frame(
        [
            {
                "fold": "fold_0",
                "row_idx": np.array([0, 1]),
                "date": np.array(
                    [np.datetime64("2000-01-03"), np.datetime64("2000-01-04")]
                ),
                "ticker": np.array(["AAA", "BBB"]),
                "y_true": np.array([-1.0, -2.0]),
                "y_pred_model": np.array([-1.1, -1.9]),
                "y_pred_har": np.array([-1.05, -2.05]),
                "is_floor": np.array([False, False]),
            }
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "absent" / "results" / "per_row_losses.parquet"
        report.write_loss_vectors(frame, path)

        assert path.is_file()
        stored = pd.read_parquet(path)
        assert list(stored.columns) == VECTOR_COLUMNS
        assert len(stored) == 2
        assert np.allclose(
            stored["loss_model"].to_numpy(),
            _elementwise_qlike(
                stored["y_true"].to_numpy(), stored["y_pred_model"].to_numpy()
            ),
            rtol=0.0,
            atol=1e-12,
        )
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
