"""Tests for the guarded native-format model loader.

Handing a zero-length buffer to the xgboost native decoder
(``Booster.load_model(bytearray(b""))``) aborts the interpreter (SIGABRT,
uncatchable) instead of raising, so the loader reads the source bytes first
and refuses empty or unrecognized input with a named error before the decoder
runs. The empty-input checks run in subprocesses, so the pre-guard behavior is
an observable abort (rc 134 / -6) rather than a dead test session; the
wrong-format and valid-model checks run in-process. A real tiny model is
saved in both native formats (json, ubj) into a temporary directory, so no
live ``artifacts/**`` or ``data/model_outputs/**`` file is touched and the
real pipeline is never run.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_native_loader.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xgboost as xgb

from conftest import REPO_ROOT
from packages.ml.model import artifact_integrity
from packages.ml.model.artifact_integrity import (
    ArtifactIntegrityError,
    EmptyModelBufferError,
    UnrecognizedModelFormatError,
    load_native_model,
)

CHILD_TIMEOUT_SECONDS = 60

# The child imports the loader from the repository root and feeds it the
# malformed source. An unguarded decoder aborts the child (SIGABRT) instead
# of raising; the parent observes that as rc 134 (shell) or -6 (subprocess).
_CHILD_SCRIPT = """
import sys

from packages.ml.model.artifact_integrity import load_native_model

mode = sys.argv[1]
source = bytearray(b"") if mode == "buffer" else sys.argv[2]
load_native_model(source)
"""


def _run_child(*argv: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-B", "-c", _CHILD_SCRIPT, *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        env=env,
    )


def _assert_guard_refused(
    completed: subprocess.CompletedProcess[str], argv: tuple[str, ...]
) -> None:
    """The child exited normally with the named error on stderr."""
    assert completed.returncode not in (134, -6), (
        argv,
        completed.returncode,
        completed.stderr,
    )
    assert completed.returncode != 0, (argv, completed.stdout, completed.stderr)
    assert "EmptyModelBufferError" in completed.stderr, (argv, completed.stderr)


def _tiny_model() -> tuple[xgb.XGBRegressor, np.ndarray]:
    rng = np.random.default_rng(17)
    X = rng.normal(size=(200, 5))
    y = rng.uniform(-3.0, -1.0, size=200)
    model = xgb.XGBRegressor(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        random_state=7,
        verbosity=0,
        tree_method="hist",
    )
    model.fit(X, y, verbose=False)
    return model, X


def test_empty_buffer_survives_the_guard_in_a_subprocess() -> None:
    completed = _run_child("buffer")
    _assert_guard_refused(completed, ("buffer",))


def test_empty_file_survives_the_guard_in_a_subprocess() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_file = Path(tmpdir) / "empty.model"
        empty_file.write_bytes(b"")
        completed = _run_child("file", str(empty_file))
    _assert_guard_refused(completed, ("file", str(empty_file)))


def test_wrong_format_buffer_is_refused_before_any_decode() -> None:
    spy = mock.Mock(
        side_effect=AssertionError("the native decoder must not be reached")
    )
    with mock.patch.object(artifact_integrity.xgb.Booster, "load_model", spy):
        for garbage in (bytearray(b"this is not a native model"), b"\x00\x01\x02\x03"):
            with pytest.raises(UnrecognizedModelFormatError) as excinfo:
                load_native_model(garbage)
            assert isinstance(excinfo.value, ArtifactIntegrityError)
            message = str(excinfo.value)
            assert "unrecognized" in message, message
            assert "format" in message, message
    assert not spy.called, "a wrong-format buffer must be refused before the decoder"


def test_missing_path_is_refused_before_any_decode() -> None:
    spy = mock.Mock(
        side_effect=AssertionError("the native decoder must not be reached")
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "missing.model"
        with mock.patch.object(artifact_integrity.xgb.Booster, "load_model", spy):
            with pytest.raises(FileNotFoundError) as excinfo:
                load_native_model(missing)
    assert str(missing) in str(excinfo.value)
    assert not spy.called, "a missing path must be refused before the decoder"


def test_valid_native_model_loads_and_predicts_identically() -> None:
    model, X = _tiny_model()
    expected = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for fmt in ("json", "ubj"):
            native_path = root / f"tiny_model.{fmt}"
            model.save_model(native_path)
            booster = load_native_model(native_path)
            assert isinstance(booster, xgb.Booster)
            np.testing.assert_array_equal(booster.predict(xgb.DMatrix(X)), expected)

        # The guard normalizes bytes/bytearray for the decoder, which rejects
        # plain bytes; the same model must load from an in-memory buffer.
        raw = (root / "tiny_model.ubj").read_bytes()
        for source in (raw, bytearray(raw)):
            booster = load_native_model(source)
            np.testing.assert_array_equal(booster.predict(xgb.DMatrix(X)), expected)


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
