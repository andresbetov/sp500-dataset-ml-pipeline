"""Tests for run-scoped checkpoints, params sidecars, atomic writes, and readers.

Every exercised run drives the real ``phase_4_training`` against a temporary
artifact root with a stub trainer and constant HAR regressors, so no live
pipeline runs and no live ``artifacts/**`` or ``data/model_outputs/**`` file is
touched. The checks pin the naming contract (``checkpoints/<run_id>/fold_<i>.pkl``
plus the ``fold_<i>_params.json`` sidecar, both recorded in the run manifest),
the atomic tmp + rename write for both files (an injected failure leaves no
final file and no ``*.tmp`` residue, and a previous file survives), the
sidecar's strict-JSON round trip of the full ``get_params()`` (callables by
qualified name, non-finite floats as ``null``, tool versions, git sha), the
native ``json``/``ubj`` round trip that loses every training parameter (the
reason the sidecar exists), the reader's resolution through the recorded
artifact index (fold summary, then run manifest; a named error instead of a
guessed filename when neither names the checkpoint), and the unchanged pickle
payload surface.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_sidecars.py
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model import artifact_integrity
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.training import training
from packages.ml.model.training.qlike_objective import (
    qlike_eval_metric,
    qlike_objective,
)
from packages.ml.model.visualization import training_diagnostics

N_ROWS = 60
BLOCK_ROWS = 10
N_INDICATORS = 3
N_TICKER_FEATURES = 1
TEST_RUN_ID = "20260101T000000Z-deadbeef"

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"

# The version keys every params sidecar must carry: the interpreter plus the
# model stack whose behavior the recorded hyperparameters describe.
SIDECAR_VERSION_KEYS = ("python", "xgboost", "numpy", "pandas", "scikit-learn")
NATIVE_MODEL_CALL_RE = re.compile(r"\.\s*(save_model|load_model)\s*\(")


class _StubModel:
    """Picklable stand-in fold model with the surface the training path reads."""

    def get_params(self) -> dict:
        return {
            "objective": qlike_objective,
            "eval_metric": qlike_eval_metric,
            "n_estimators": 5,
            "missing": float("nan"),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.zeros(len(INDICATORS), dtype=np.float64)


class _WarningRecorder(logging.Handler):
    """Collect formatted log messages without pytest's caplog fixture."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _stub_trainer(X_train, y_train, X_val=None, y_val=None):
    return _StubModel(), {}


def _tiny_xgb_trainer(X_train, y_train, X_val=None, y_val=None):
    """Trainer double: a real tiny XGBRegressor with the QLIKE objective/metric."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective=qlike_objective,
        eval_metric=qlike_eval_metric,
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        random_state=7,
        verbosity=0,
        tree_method="hist",
    )
    model.fit(X_train, y_train, verbose=False)
    return model, {}


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


def _exercise_run(root: Path) -> tuple:
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
    root = Path(tmpdir) / "artifacts"
    root.mkdir()
    _write_synthetic_artifacts(root)
    return root


def _read_manifest(root: Path) -> dict:
    return json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))


def _fold_inputs() -> tuple:
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(24, N_INDICATORS))
    y_train = rng.uniform(-3.0, -1.0, size=24)
    X_test = rng.normal(size=(8, N_INDICATORS))
    y_test = rng.uniform(-3.0, -1.0, size=8)
    fold_data = {
        "train_date_min": "2000-01-03",
        "train_date_max": "2000-02-01",
        "test_date_min": "2000-02-02",
        "test_date_max": "2000-02-11",
    }
    return X_train, y_train, X_test, y_test, fold_data


def _qualified_name(value: object) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _reject_constant(token: str):
    raise AssertionError(f"non-strict JSON constant emitted: {token}")


def _assert_params_roundtrip(stored: dict, live: dict) -> None:
    """The sidecar's params equal the estimator's, under the sidecar's encoding."""
    assert set(stored) == set(live), (sorted(stored), sorted(live))
    for key, value in live.items():
        if callable(value):
            assert stored[key] == _qualified_name(value), key
        elif isinstance(value, float) and not math.isfinite(value):
            assert stored[key] is None, key
        else:
            assert stored[key] == value, key


def test_fold_write_is_run_scoped_and_recorded_everywhere() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        validation, summary, model_paths = _exercise_run(root)
        run_id = _read_manifest(root)["run_id"]

        for fold_idx in range(5):
            expected = root / "checkpoints" / run_id / f"fold_{fold_idx}.pkl"
            assert expected.is_file(), f"run-scoped checkpoint missing: {expected}"
            assert not (
                root / "checkpoints" / f"fold_{fold_idx}_model.pkl"
            ).exists(), "the legacy bare checkpoint name must not be written"

            stored = summary[f"fold_{fold_idx}"]["model_path"]
            assert not Path(stored).is_absolute(), stored
            assert stored.endswith(f"checkpoints/{run_id}/fold_{fold_idx}.pkl")
            assert model_paths[fold_idx] == stored

            stored_prediction = validation[f"fold_{fold_idx}"]["fold_model_path"]
            assert stored_prediction == stored

        bare = sorted(path.name for path in (root / "checkpoints").glob("*.pkl"))
        assert bare == [], f"bare checkpoint files written: {bare}"
        assert not list(root.rglob("*.tmp")), "checkpoint write left *.tmp residue"

        manifest = _read_manifest(root)
        for fold_idx in range(5):
            assert any(
                key.endswith(f"checkpoints/{run_id}/fold_{fold_idx}.pkl")
                for key in manifest["artifacts"]
            ), manifest["artifacts"]


def test_reader_resolves_checkpoints_through_the_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root)
        run_id = _read_manifest(root)["run_id"]

        # A reader given the manifest but no fold summary still resolves the
        # run-scoped checkpoints from the manifest's artifact list.
        missing_summary = root / "results" / "missing_summary.json"
        for fold_idx in range(5):
            resolved = training_diagnostics.resolve_fold_checkpoint(
                fold_idx,
                summary_path=missing_summary,
                manifest_path=root / "run_manifest.json",
            )
            assert resolved == root / "checkpoints" / run_id / f"fold_{fold_idx}.pkl"
            assert resolved.is_file()

        # With the default sources (summary + manifest) the reader resolves the
        # same files the writer recorded.
        with mock.patch.object(training_diagnostics, "ARTIFACTS_DIR", root):
            for fold_idx in range(5):
                resolved = training_diagnostics.resolve_fold_checkpoint(fold_idx)
                assert resolved == root / "checkpoints" / run_id / f"fold_{fold_idx}.pkl"


def test_manifest_missing_entry_raises_named_error() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        (root / "results").mkdir(parents=True)
        manifest_path = root / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_id": TEST_RUN_ID,
                    "artifacts": {"results/har_benchmark.json": "0" * 64},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(training_diagnostics.CheckpointResolutionError) as excinfo:
            training_diagnostics.resolve_fold_checkpoint(
                0,
                summary_path=root / "results" / "fold_training_summary.json",
                manifest_path=manifest_path,
            )

        message = str(excinfo.value)
        assert "fold_0" in message, message
        assert "run_manifest.json" in message, message
        assert "fold_0_model.pkl" not in message, message


def test_plot_refuses_to_guess_a_missing_checkpoint_record() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        (root / "results").mkdir(parents=True)
        (root / "run_manifest.json").write_text(
            json.dumps({"run_id": TEST_RUN_ID, "artifacts": {}}), encoding="utf-8"
        )
        output = Path(tmpdir) / "feature_importance.png"

        with mock.patch.object(training_diagnostics, "ARTIFACTS_DIR", root):
            with pytest.raises(training_diagnostics.CheckpointResolutionError):
                training_diagnostics.plot_feature_importance({}, None, output)

        assert not output.exists(), "the plot must not render after a refusal"


def test_plot_loads_the_run_scoped_checkpoints_the_manifest_names() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        validation, _summary, _model_paths = _exercise_run(root)
        run_id = _read_manifest(root)["run_id"]

        loaded: list[Path] = []
        real_load = joblib.load

        def spy(path, *args, **kwargs):
            loaded.append(Path(path))
            return real_load(path, *args, **kwargs)

        output = Path(tmpdir) / "feature_importance.png"
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(training_diagnostics, "ARTIFACTS_DIR", root)
            )
            stack.enter_context(
                mock.patch.object(artifact_integrity.joblib, "load", spy)
            )
            training_diagnostics.plot_feature_importance(validation, None, output)

        expected = {
            root / "checkpoints" / run_id / f"fold_{fold_idx}.pkl"
            for fold_idx in range(5)
        }
        assert set(loaded) == expected, loaded
        assert output.is_file() and output.stat().st_size > 0


def test_plot_renders_zeros_only_when_a_recorded_checkpoint_file_is_absent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        (root / "results").mkdir(parents=True)
        # The summary records run-scoped paths whose files were pruned: the
        # intentional partial-checkout case keeps the warning + zeros behavior.
        summary = {}
        for fold_idx in range(5):
            stored = Path(
                os.path.relpath(
                    root / "checkpoints" / "r1" / f"fold_{fold_idx}.pkl", REPO_ROOT
                )
            ).as_posix()
            summary[f"fold_{fold_idx}"] = {"model_path": stored}
        (root / "results" / "fold_training_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        output = Path(tmpdir) / "feature_importance.png"
        handler = _WarningRecorder()
        logger = training_diagnostics.logger
        logger.addHandler(handler)
        try:
            with mock.patch.object(training_diagnostics, "ARTIFACTS_DIR", root):
                training_diagnostics.plot_feature_importance({}, None, output)
        finally:
            logger.removeHandler(handler)

        assert output.is_file(), "the plot must still render the remaining folds"
        warnings = [
            message for message in handler.messages if "Model not found" in message
        ]
        assert len(warnings) == 5, warnings


def test_torn_dump_leaves_no_final_file_and_no_tmp_residue() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        X_train, y_train, X_test, y_test, fold_data = _fold_inputs()

        def failing_dump(obj, path, *args, **kwargs):
            Path(path).write_bytes(b"partial checkpoint payload")
            raise RuntimeError("injected failure during checkpoint dump")

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(training, "train_xgboost_model", _stub_trainer)
            )
            stack.enter_context(
                mock.patch.object(training.joblib, "dump", failing_dump)
            )
            with pytest.raises(RuntimeError, match="injected failure"):
                training._train_fold_model(
                    fold_idx=0,
                    run_id=TEST_RUN_ID,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_train[:8],
                    y_val=y_train[:8],
                    X_test=X_test,
                    y_test=y_test,
                    fold_data=fold_data,
                )

        target = root / "checkpoints" / TEST_RUN_ID / "fold_0.pkl"
        assert not target.exists(), "a torn dump left a final checkpoint"
        assert not list(root.rglob("*.tmp")), "a torn dump left *.tmp residue"


def test_torn_rename_keeps_the_previous_checkpoint_intact() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        X_train, y_train, X_test, y_test, fold_data = _fold_inputs()
        target = root / "checkpoints" / TEST_RUN_ID / "fold_0.pkl"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"previous checkpoint payload")

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(training, "train_xgboost_model", _stub_trainer)
            )
            stack.enter_context(
                mock.patch("os.replace", side_effect=OSError("injected rename failure"))
            )
            with pytest.raises(OSError, match="injected rename failure"):
                training._train_fold_model(
                    fold_idx=0,
                    run_id=TEST_RUN_ID,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_train[:8],
                    y_val=y_train[:8],
                    X_test=X_test,
                    y_test=y_test,
                    fold_data=fold_data,
                )

        assert target.read_bytes() == b"previous checkpoint payload"
        assert not list(root.rglob("*.tmp")), "a torn rename left *.tmp residue"


def test_checkpoint_payload_surface_unchanged() -> None:
    from xgboost import XGBRegressor

    X_train, y_train, X_test, y_test, fold_data = _fold_inputs()
    X_train = np.tile(X_train, (2, 1))
    y_train = np.tile(y_train, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            model, predictions = training._train_fold_model(
                fold_idx=0,
                run_id=TEST_RUN_ID,
                X_train=X_train,
                y_train=y_train,
                X_val=X_train[:8],
                y_val=y_train[:8],
                X_test=X_test,
                y_test=y_test,
                fold_data=fold_data,
            )
        checkpoint = root / "checkpoints" / TEST_RUN_ID / "fold_0.pkl"
        loaded = joblib.load(checkpoint)

        # The params sidecar sits beside the checkpoint and is strict JSON even
        # for this real estimator's callables and NaN defaults.
        sidecar = checkpoint.with_name("fold_0_params.json")
        assert sidecar.is_file(), "the params sidecar must sit beside the checkpoint"
        json.loads(sidecar.read_text(encoding="utf-8"), parse_constant=_reject_constant)

    assert isinstance(loaded, XGBRegressor)
    assert type(loaded) is type(model)
    loaded_params = loaded.get_params()
    assert set(loaded_params) == set(model.get_params())
    for key, value in model.get_params().items():
        other = loaded_params[key]
        if isinstance(value, float) and np.isnan(value):
            assert isinstance(other, float) and np.isnan(other), key
        else:
            assert other == value, key
    np.testing.assert_array_equal(loaded.predict(X_test), model.predict(X_test))

    stored = predictions["fold_model_path"]
    assert not Path(stored).is_absolute(), stored
    assert stored.endswith(f"checkpoints/{TEST_RUN_ID}/fold_0.pkl")


def test_params_sidecar_roundtrip() -> None:
    rng = np.random.default_rng(31)
    X_train = rng.normal(size=(200, N_INDICATORS))
    y_train = rng.uniform(-3.0, -1.0, size=200)
    X_test = rng.normal(size=(40, N_INDICATORS))
    y_test = rng.uniform(-3.0, -1.0, size=40)
    fold_data = _fold_inputs()[-1]

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(training, "train_xgboost_model", _tiny_xgb_trainer)
            )
            model, _predictions = training._train_fold_model(
                fold_idx=0,
                run_id=TEST_RUN_ID,
                X_train=X_train,
                y_train=y_train,
                X_val=X_train[:8],
                y_val=y_train[:8],
                X_test=X_test,
                y_test=y_test,
                fold_data=fold_data,
            )

        sidecar = root / "checkpoints" / TEST_RUN_ID / "fold_0_params.json"
        assert sidecar.is_file(), f"params sidecar missing: {sidecar}"
        text = sidecar.read_text(encoding="utf-8")
        document = json.loads(text, parse_constant=_reject_constant)

        # The minimal sidecar schema: the full parameter set, the tool versions,
        # and the best-effort git sha.
        assert set(document) == {"params", "environment", "git_sha"}, sorted(document)
        _assert_params_roundtrip(document["params"], model.get_params())
        assert document["params"]["objective"] == (
            "packages.ml.model.training.qlike_objective.qlike_objective"
        )
        assert document["params"]["eval_metric"] == (
            "packages.ml.model.training.qlike_objective.qlike_eval_metric"
        )

        environment = document["environment"]
        for key in SIDECAR_VERSION_KEYS:
            assert isinstance(environment.get(key), str), (key, environment)
            assert environment[key], (key, environment)
        git_sha = document["git_sha"]
        assert git_sha is None or re.fullmatch(r"[0-9a-f]{40}", git_sha), git_sha

        # Strict JSON: no non-standard constants, no *.tmp residue.
        assert "NaN" not in text and "Infinity" not in text, text
        assert not list(root.rglob("*.tmp")), "the sidecar write left *.tmp residue"


def test_native_json_ubj_roundtrip_loses_training_params() -> None:
    from xgboost import XGBRegressor

    rng = np.random.default_rng(17)
    X = rng.normal(size=(200, N_INDICATORS))
    y = rng.uniform(-3.0, -1.0, size=200)
    model = XGBRegressor(
        objective=qlike_objective,
        eval_metric=qlike_eval_metric,
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        random_state=7,
        verbosity=0,
        tree_method="hist",
    )
    model.fit(X, y, verbose=False)
    live_params = model.get_params()
    predictions = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for fmt in ("json", "ubj"):
            native_path = root / f"fold_0_model.{fmt}"
            model.save_model(native_path)
            reloaded = XGBRegressor()
            reloaded.load_model(native_path)
            stored_params = reloaded.get_params()

            # The trees survive (predictions unchanged) but the training recipe
            # does not: every configured knob reloads as None or a default, and
            # the QLIKE objective is not even declared anymore. This is the
            # documented reason the params sidecar exists.
            np.testing.assert_array_equal(reloaded.predict(X), predictions)
            for key in (
                "max_depth",
                "n_estimators",
                "learning_rate",
                "random_state",
                "tree_method",
                "verbosity",
            ):
                assert stored_params[key] is None, (fmt, key, stored_params[key])
            assert stored_params["eval_metric"] is None, fmt
            assert stored_params["objective"] != live_params["objective"], fmt
            assert stored_params["objective"] == "reg:squarederror", fmt


def test_params_sidecars_are_recorded_in_the_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root)
        manifest = _read_manifest(root)
        run_id = manifest["run_id"]

        for fold_idx in range(5):
            sidecar = root / "checkpoints" / run_id / f"fold_{fold_idx}_params.json"
            assert sidecar.is_file(), f"params sidecar missing: {sidecar}"
            assert any(
                key.endswith(f"checkpoints/{run_id}/fold_{fold_idx}_params.json")
                for key in manifest["artifacts"]
            ), manifest["artifacts"]

        # The run-level write applies the same strict-JSON encoding as the
        # round-trip check: the stub's callable and NaN knobs are recorded by
        # qualified name and as null.
        document = json.loads(
            (root / "checkpoints" / run_id / "fold_0_params.json").read_text(
                encoding="utf-8"
            ),
            parse_constant=_reject_constant,
        )
        assert document["params"]["objective"] == _qualified_name(qlike_objective)
        assert document["params"]["missing"] is None
        assert document["git_sha"] is None or re.fullmatch(
            r"[0-9a-f]{40}", document["git_sha"]
        )


def test_torn_params_dump_leaves_no_sidecar_and_no_tmp_residue() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        params_path = root / "checkpoints" / TEST_RUN_ID / "fold_0_params.json"

        def failing_dump(obj, fp, *args, **kwargs):
            fp.write('{"params": {"partial"')
            raise RuntimeError("injected failure during params dump")

        with mock.patch.object(training.json, "dump", failing_dump):
            with pytest.raises(RuntimeError, match="injected failure"):
                training.write_fold_params_atomically(_StubModel(), params_path)

        assert not params_path.exists(), "a torn dump left a final sidecar"
        assert not list(root.rglob("*.tmp")), "a torn dump left *.tmp residue"


def test_torn_params_rename_keeps_the_previous_sidecar_intact() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        params_path = root / "checkpoints" / TEST_RUN_ID / "fold_0_params.json"
        params_path.parent.mkdir(parents=True)
        params_path.write_bytes(b"previous sidecar payload")

        with mock.patch("os.replace", side_effect=OSError("injected rename failure")):
            with pytest.raises(OSError, match="injected rename failure"):
                training.write_fold_params_atomically(_StubModel(), params_path)

        assert params_path.read_bytes() == b"previous sidecar payload"
        assert not list(root.rglob("*.tmp")), "a torn rename left *.tmp residue"


def test_native_model_calls_are_confined_to_the_guarded_loader() -> None:
    """Native decode calls live only in the guarded loader module.

    The guarded loader (``artifact_integrity.load_native_model``) reads the
    source bytes, refuses empty or wrong-format input with a named error, and
    only then calls the native decoder. Everywhere else native save/load
    calls stay banned: the native formats drop the training recipe, so the
    pkl remains the training artifact and the params sidecar carries
    ``get_params()``.
    """
    guarded_loader = MODEL_ROOT / "artifact_integrity.py"
    hits = []
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        if TESTS_DIR in path.parents or path == guarded_loader:
            continue
        if NATIVE_MODEL_CALL_RE.search(path.read_text(encoding="utf-8")):
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert not hits, (
        "native xgboost save/load calls outside the guarded loader: "
        + ", ".join(hits)
    )


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
