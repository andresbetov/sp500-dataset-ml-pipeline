"""Tests for parent-directory creation and torn-write safety at writer sites.

Every writer must create its parent directory before writing, so a dir-less
target (checkpoint dir, params sidecar dir, folds dir, results dir) cannot fail
with ``FileNotFoundError``, and an injected failure must leave no partial final
file. The checks exercise the real writers against temporary roots only: no
live ``artifacts/**`` or ``data/model_outputs/**`` file is read or written.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_io_dirs.py
"""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.training import training

# Phase 3 fixture: 500 reserved holdout dates plus a 100-date fold remainder
# exactly divisible by the six blocks, mirroring the live reservation.
N_DATES = 600
TICKERS = ("AAA", "BBB", "CCC")
N_REGIME_BUCKETS = 7


class _StubModel:
    """Picklable stand-in model with the surface the writers read."""

    def get_params(self) -> dict:
        return {"n_estimators": 5, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), -2.0, dtype=np.float64)


def _temp_root(tmpdir: str) -> Path:
    """A temporary artifacts root, guarded to stay outside the live tree."""
    root = Path(tmpdir) / "artifacts"
    assert not root.resolve().is_relative_to(REPO_ROOT), (
        "writer tests must target temporary roots, never live artifacts"
    )
    return root


def _minimal_results_inputs() -> tuple[dict, dict, dict]:
    """The smallest payloads ``_save_training_results`` accepts."""
    validation = {
        "fold_0": {
            "test_pred": np.array([0.0, 0.1]),
            "test_true": np.array([0.05, 0.15]),
        }
    }
    summary = {
        "fold_0": {
            "test_r2": 0.5,
            "test_qlike": 0.25,
            "train_time_seconds": 1.0,
        }
    }
    har_results = {
        "fold_0": {
            "n_test": 2,
            "clip_count": 0,
            "clip_rate": 0.0,
            "har_r2": 0.1,
            "har_qlike": 0.2,
            "coefficient_interpretation": False,
        }
    }
    return validation, summary, har_results


def test_results_writer_creates_a_missing_results_dir() -> None:
    """A results write into an absent artifacts root creates the parents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)  # deliberately never created
        validation, summary, har_results = _minimal_results_inputs()

        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            training._save_training_results(validation, summary, har_results)

        results = root / "results"
        assert results.is_dir(), "the results writer must create its parent"
        val_pred_path = results / "validation_predictions.pkl"
        summary_path = results / "fold_training_summary.json"
        har_report_path = results / "har_benchmark.json"
        assert val_pred_path.is_file()
        assert summary_path.is_file()
        assert har_report_path.is_file()

        stored = joblib.load(val_pred_path)
        assert stored["fold_0"]["test_pred"].tolist() == [0.0, 0.1]
        summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary_json["fold_0"]["test_r2"] == 0.5
        report = json.loads(har_report_path.read_text(encoding="utf-8"))
        assert report["folds"]["fold_0"]["har_r2"] == 0.1

        assert not list(Path(tmpdir).rglob("*.tmp")), "results write left *.tmp"


def test_checkpoint_writer_creates_a_missing_checkpoint_dir() -> None:
    """A checkpoint write into an absent directory creates the parents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = _temp_root(tmpdir) / "checkpoints" / "run1" / "fold_0.pkl"

        training.write_checkpoint_atomically(_StubModel(), target)

        assert target.is_file()
        loaded = joblib.load(target)
        assert np.array_equal(
            loaded.predict(np.zeros((2, 1))), np.full(2, -2.0)
        )
        assert not list(Path(tmpdir).rglob("*.tmp")), "checkpoint write left *.tmp"


def test_params_writer_creates_a_missing_params_dir() -> None:
    """A params sidecar write into an absent directory creates the parents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = _temp_root(tmpdir) / "checkpoints" / "run1" / "fold_0_params.json"

        training.write_fold_params_atomically(_StubModel(), target)

        assert target.is_file()
        document = json.loads(target.read_text(encoding="utf-8"))
        assert document["params"]["n_estimators"] == 5
        assert not list(Path(tmpdir).rglob("*.tmp")), "params write left *.tmp"


def test_torn_results_dump_leaves_no_partial_final_file() -> None:
    """A failing results pickle write leaves no final file and no tmp residue."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        (root / "results").mkdir(parents=True)
        validation, summary, har_results = _minimal_results_inputs()

        def failing_dump(obj, path, *args, **kwargs):
            Path(path).write_bytes(b"partial results payload")
            raise RuntimeError("injected failure during results dump")

        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            with mock.patch.object(training.joblib, "dump", failing_dump):
                with pytest.raises(RuntimeError, match="injected failure"):
                    training._save_training_results(
                        validation, summary, har_results
                    )

        assert not (
            root / "results" / "validation_predictions.pkl"
        ).exists(), "a torn results dump left a final file"
        assert not list(root.rglob("*.tmp")), "a torn results dump left *.tmp"


def _failing_json_dump(target_name: str):
    """A ``json.dump`` double that tears only the named results document."""
    real_dump = json.dump

    def failing_dump(obj, fp, *args, **kwargs):
        name = str(getattr(fp, "name", ""))
        if name.endswith(target_name) or name.endswith(target_name + ".tmp"):
            fp.write('{"partial":')
            raise RuntimeError(f"injected failure during {target_name} dump")
        return real_dump(obj, fp, *args, **kwargs)

    return failing_dump


def test_torn_results_summary_json_leaves_no_partial_final_file() -> None:
    """A failing summary JSON write leaves no partial final summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        (root / "results").mkdir(parents=True)
        validation, summary, har_results = _minimal_results_inputs()

        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            with mock.patch.object(
                training.json,
                "dump",
                _failing_json_dump("fold_training_summary.json"),
            ):
                with pytest.raises(RuntimeError, match="injected failure"):
                    training._save_training_results(
                        validation, summary, har_results
                    )

        assert not (
            root / "results" / "fold_training_summary.json"
        ).exists(), "a torn summary dump left a final file"
        assert not (
            root / "results" / "har_benchmark.json"
        ).exists(), "a torn summary dump left a final report"
        assert not list(root.rglob("*.tmp")), "a torn summary dump left *.tmp"


def test_torn_results_report_json_leaves_no_partial_final_file() -> None:
    """A failing HAR report JSON write leaves no partial final report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        (root / "results").mkdir(parents=True)
        validation, summary, har_results = _minimal_results_inputs()

        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            with mock.patch.object(
                training.json, "dump", _failing_json_dump("har_benchmark.json")
            ):
                with pytest.raises(RuntimeError, match="injected failure"):
                    training._save_training_results(
                        validation, summary, har_results
                    )

        assert not (
            root / "results" / "har_benchmark.json"
        ).exists(), "a torn report dump left a final file"
        assert not list(root.rglob("*.tmp")), "a torn report dump left *.tmp"


def test_torn_checkpoint_into_a_missing_dir_leaves_no_final_file() -> None:
    """A failing checkpoint write creates no final file, even dir-less."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        target = root / "checkpoints" / "run1" / "fold_0.pkl"

        def failing_dump(obj, path, *args, **kwargs):
            Path(path).write_bytes(b"partial checkpoint payload")
            raise RuntimeError("injected failure during checkpoint dump")

        with mock.patch.object(training.joblib, "dump", failing_dump):
            with pytest.raises(RuntimeError, match="injected failure"):
                training.write_checkpoint_atomically(_StubModel(), target)

        assert not target.exists(), "a torn checkpoint dump left a final file"
        assert not list(root.rglob("*.tmp")), "a torn checkpoint dump left *.tmp"


def test_torn_params_into_a_missing_dir_leaves_no_final_file() -> None:
    """A failing params write creates no final sidecar, even dir-less."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        target = root / "checkpoints" / "run1" / "fold_0_params.json"

        def failing_dump(obj, fp, *args, **kwargs):
            fp.write('{"params": {"partial"')
            raise RuntimeError("injected failure during params dump")

        with mock.patch.object(training.json, "dump", failing_dump):
            with pytest.raises(RuntimeError, match="injected failure"):
                training.write_fold_params_atomically(_StubModel(), target)

        assert not target.exists(), "a torn params dump left a final sidecar"
        assert not list(root.rglob("*.tmp")), "a torn params dump left *.tmp"


def _write_cv_inputs(root: Path) -> None:
    """Write a minimal Phase 2 input set; the folds dir stays absent."""
    dates = np.repeat(
        pd.bdate_range("2000-01-03", periods=N_DATES).values, len(TICKERS)
    )
    n_rows = len(dates)
    X = np.zeros((n_rows, 4), dtype=np.float64)
    Y = np.full(n_rows, 0.25, dtype=np.float64)
    metadata = {
        "date": dates,
        "ticker": np.tile(np.array(TICKERS), N_DATES),
        "regime": np.repeat(
            np.arange(N_DATES) % N_REGIME_BUCKETS, len(TICKERS)
        ).astype(np.int8),
        "n_indicators": 4,
        "n_ticker_features": 0,
    }
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    np.save(inputs / "X.npy", X)
    np.save(inputs / "Y.npy", Y)
    joblib.dump(metadata, inputs / "metadata.pkl")
    # The live Phase 3 reader verifies recorded hashes before loading; a
    # synthetic input set must carry its registry entry like the real writer's.
    record_artifact(inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME)


def test_folds_writer_creates_a_missing_folds_dir() -> None:
    """Phase 3 writes its fold payload into an absent folds directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        root.mkdir()
        _write_cv_inputs(root)
        assert not (root / "folds").exists()

        with mock.patch.object(cross_validator, "ARTIFACTS_DIR", root):
            folds = cross_validator.phase_3_cross_validation()

        folds_dir = root / "folds"
        assert folds_dir.is_dir(), "the folds writer must create its parent"
        payload_path = folds_dir / "folds_metadata.pkl"
        assert payload_path.is_file()
        payload = joblib.load(payload_path)
        assert set(folds) == {f"fold_{fold_idx}" for fold_idx in range(5)}
        assert payload["schema_version"] == cross_validator.SCHEMA_VERSION
        assert not list(root.rglob("*.tmp")), "folds write left *.tmp"


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
