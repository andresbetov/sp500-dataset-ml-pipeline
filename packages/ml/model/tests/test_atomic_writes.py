"""Atomic-writer contract for every remaining model-side write site.

Each converted writer is exercised against a temporary root only: no live
``artifacts/**`` or ``data/model_outputs/**`` file is read or written. The
checks inject a mid-write failure and assert the previous file stays
byte-identical and no ``*.tmp`` sibling is left behind; the success paths also
assert the final file exists and no residue remains.

Writers covered: the shared primitive, Phase-2 inputs (``X.npy``/``Y.npy``/
``metadata.pkl``), Phase-3 fold metadata, the results bundle, the run manifest,
and PNG figures.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_atomic_writes.py
"""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from conftest import REPO_ROOT  # noqa: F401  (puts the repository root on sys.path)
from packages.ml.model import run_log
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.features import feature_engineering
from packages.ml.model.training import training
from packages.ml.model.visualization import plotting


def _temp_root(tmpdir: str) -> Path:
    root = Path(tmpdir) / "artifacts"
    root.mkdir()
    return root


def _prior_bytes(path: Path, payload: bytes = b"previous payload") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path.read_bytes()


def _assert_no_tmp(root: Path) -> None:
    residue = [str(path) for path in root.rglob("*.tmp")]
    assert not residue, f"atomic write left *.tmp residue: {residue}"


def _minimal_results() -> tuple[dict, dict, dict]:
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


# --- shared primitive -------------------------------------------------------


def test_shared_bytes_writer_replaces_atomically() -> None:
    from packages.ml.model import atomic

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "nested" / "payload.bin"
        atomic.write_bytes_atomically(target, b"first")
        assert target.read_bytes() == b"first"
        atomic.write_bytes_atomically(target, b"second")
        assert target.read_bytes() == b"second"
        _assert_no_tmp(Path(tmpdir))


def test_shared_bytes_writer_keeps_prior_file_on_torn_rename() -> None:
    from packages.ml.model import atomic

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target = root / "payload.bin"
        prior = _prior_bytes(target)

        with mock.patch.object(
            atomic.os, "replace", side_effect=OSError("injected rename failure")
        ):
            with pytest.raises(OSError, match="injected rename failure"):
                atomic.write_bytes_atomically(target, b"new payload")

        assert target.read_bytes() == prior
        _assert_no_tmp(root)


def test_shared_json_writer_keeps_prior_file_on_torn_dump() -> None:
    from packages.ml.model import atomic

    def failing_dump(payload, handle, *args, **kwargs):
        handle.write('{"partial":')
        raise RuntimeError("injected json failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target = root / "document.json"
        prior = _prior_bytes(target, b'{"previous": true}\n')

        with mock.patch.object(atomic.json, "dump", failing_dump):
            with pytest.raises(RuntimeError, match="injected json failure"):
                atomic.write_json_atomically(target, {"new": True})

        assert target.read_bytes() == prior
        _assert_no_tmp(root)


def test_shared_npy_writer_keeps_prior_file_on_torn_save() -> None:
    from packages.ml.model import atomic

    def failing_save(handle, array, *args, **kwargs):
        handle.write(b"partial npy payload")
        raise RuntimeError("injected npy failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target = root / "X.npy"
        prior = _prior_bytes(target, b"previous npy payload")

        with mock.patch.object(atomic.np, "save", failing_save):
            with pytest.raises(RuntimeError, match="injected npy failure"):
                atomic.write_npy_atomically(target, np.zeros(3))

        assert target.read_bytes() == prior
        _assert_no_tmp(root)


def test_shared_pickle_writer_keeps_prior_file_on_torn_dump() -> None:
    from packages.ml.model import atomic

    def failing_dump(payload, path, *args, **kwargs):
        Path(path).write_bytes(b"partial pickle payload")
        raise RuntimeError("injected pickle failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target = root / "record.pkl"
        prior = _prior_bytes(target, b"previous pickle payload")

        with mock.patch.object(atomic.joblib, "dump", failing_dump):
            with pytest.raises(RuntimeError, match="injected pickle failure"):
                atomic.write_pickle_atomically(target, {"x": 1})

        assert target.read_bytes() == prior
        _assert_no_tmp(root)


# --- Phase 2 inputs ---------------------------------------------------------


def test_phase_2_inputs_writer_is_atomic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        inputs = root / "inputs"
        prior = {
            "X.npy": _prior_bytes(inputs / "X.npy", b"previous X"),
            "Y.npy": _prior_bytes(inputs / "Y.npy", b"previous Y"),
            "metadata.pkl": _prior_bytes(inputs / "metadata.pkl", b"previous meta"),
        }

        with pytest.raises(RuntimeError, match="injected npy failure"):
            with mock.patch.object(
                feature_engineering.atomic.np,
                "save",
                side_effect=RuntimeError("injected npy failure"),
            ):
                feature_engineering.write_inputs_atomically(
                    np.zeros((2, 1)), np.zeros(2), {"ticker": ["A", "B"]}, inputs
                )

        for name, payload in prior.items():
            assert (inputs / name).read_bytes() == payload, name
        _assert_no_tmp(root)


# --- Phase 3 fold metadata --------------------------------------------------


def test_fold_metadata_writer_is_atomic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        target = root / "folds" / "folds_metadata.pkl"
        prior = _prior_bytes(target, b"previous folds")

        def failing_dump(payload, path, *args, **kwargs):
            Path(path).write_bytes(b"partial folds payload")
            raise RuntimeError("injected folds failure")

        with mock.patch.object(cross_validator.joblib, "dump", failing_dump):
            with pytest.raises(RuntimeError, match="injected folds failure"):
                cross_validator.write_folds_metadata_atomically(
                    {"fold_0": {}}, target
                )

        assert target.read_bytes() == prior
        _assert_no_tmp(root)


def test_fold_metadata_writer_creates_a_missing_directory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        target = root / "folds" / "folds_metadata.pkl"
        cross_validator.write_folds_metadata_atomically({"fold_0": {}}, target)
        assert target.is_file()
        _assert_no_tmp(root)


def test_fold_record_predictions_failure_keeps_the_prior_pair() -> None:
    def failing_dump(payload, path, *args, **kwargs):
        Path(path).write_bytes(b"partial predictions payload")
        raise RuntimeError("injected predictions failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        run_id = "20260101T000000Z-deadbeef"
        run_dir = root / "checkpoints" / run_id
        summary_path = run_dir / "fold_0_summary.json"
        predictions_path = run_dir / "fold_0_predictions.pkl"
        prior_summary = _prior_bytes(summary_path, b"previous summary")
        prior_predictions = _prior_bytes(predictions_path, b"previous predictions")

        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            with mock.patch.object(training.joblib, "dump", failing_dump):
                with pytest.raises(
                    RuntimeError, match="injected predictions failure"
                ):
                    training.write_fold_record_atomically(
                        0,
                        run_id,
                        {"test_qlike": 0.1},
                        {"test_pred": np.zeros(2)},
                        {"n_test": 0},
                        config_hash_value="deadbeef",
                        fold_manifest_hash="0" * 64,
                        coefficient_interpretation=False,
                    )

        assert summary_path.read_bytes() == prior_summary
        assert predictions_path.read_bytes() == prior_predictions
        _assert_no_tmp(root)


# --- results bundle ---------------------------------------------------------


def test_results_summary_keeps_the_prior_file_on_torn_write() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        results = root / "results"
        summary_path = results / "fold_training_summary.json"
        har_path = results / "har_benchmark.json"
        prior_summary = _prior_bytes(summary_path, b'{"previous": true}\n')
        prior_har = _prior_bytes(har_path, b'{"previous_har": true}\n')
        validation, summary, har_results = _minimal_results()

        real_dump = json.dump

        def failing_dump(payload, handle, *args, **kwargs):
            name = str(getattr(handle, "name", ""))
            if name.endswith("fold_training_summary.json.tmp"):
                handle.write('{"partial":')
                raise RuntimeError("injected summary failure")
            return real_dump(payload, handle, *args, **kwargs)

        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            with mock.patch.object(training.json, "dump", failing_dump):
                with pytest.raises(RuntimeError, match="injected summary failure"):
                    training._save_training_results(validation, summary, har_results)

        assert summary_path.read_bytes() == prior_summary
        assert har_path.read_bytes() == prior_har
        _assert_no_tmp(root)


# --- run manifest -----------------------------------------------------------


def test_run_manifest_keeps_the_prior_file_on_torn_write() -> None:
    def failing_dump(payload, handle, *args, **kwargs):
        handle.write('{"partial":')
        raise RuntimeError("injected manifest failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        target = root / "run_manifest.json"
        prior = _prior_bytes(target, b'{"previous": true}\n')

        with mock.patch.object(run_log.json, "dump", failing_dump):
            with pytest.raises(RuntimeError, match="injected manifest failure"):
                run_log.write_manifest({"run_id": "new"}, target)

        assert target.read_bytes() == prior
        _assert_no_tmp(root)


def test_run_manifest_write_is_atomic_on_success() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _temp_root(tmpdir)
        target = root / "run_manifest.json"
        run_log.write_manifest({"run_id": "new"}, target)
        assert json.loads(target.read_text(encoding="utf-8")) == {"run_id": "new"}
        _assert_no_tmp(root)


# --- PNG figures ------------------------------------------------------------


def test_figure_writer_keeps_the_prior_png_on_torn_save() -> None:
    def failing_savefig(path, *args, **kwargs):
        Path(path).write_bytes(b"partial png payload")
        raise RuntimeError("injected png failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target = root / "figure.png"
        prior = _prior_bytes(target, b"previous png payload")

        with pytest.raises(RuntimeError, match="injected png failure"):
            plotting.save_figure_atomically(target, savefig=failing_savefig)

        assert target.read_bytes() == prior
        _assert_no_tmp(root)


def test_figure_writer_creates_a_missing_directory_and_png() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target = root / "nested" / "figure.png"

        def save_png(path, *args, **kwargs):
            Path(path).write_bytes(b"\x89PNG\r\n\x1a\npayload")
            return path

        returned = plotting.save_figure_atomically(target, savefig=save_png)
        assert returned == target
        assert target.read_bytes().startswith(b"\x89PNG")
        _assert_no_tmp(root)


if __name__ == "__main__":
    failures = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All tests passed")
