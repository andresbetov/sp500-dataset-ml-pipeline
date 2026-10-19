"""Tests for size- and sha256-verified artifact loads.

Every live ``joblib.load`` in the model package must verify the artifact's byte
size and SHA-256 against the entry recorded for it in the artifact registry
before the pickle is deserialized: a tampered artifact, a missing entry, or a
missing file is refused by name with ``ArtifactIntegrityError`` and
``joblib.load`` is never called. The writer sites record the entries at write
time (atomic registry update), so a reader later in the same run finds the
entry the moment the artifact exists on disk.

The exercised pipeline drives the real Phase 2 -> 3 -> 4 writers and the
diagnostics readers against a temporary dataset and artifact root with a stub
fold trainer and constant HAR regressors, so no live ``artifacts/**`` or
``data/model_outputs/**`` file is written and the real pipeline is never run.
``X.npy``/``Y.npy`` ``np.load`` paths stay direct and unverified.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_verified_loads.py
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
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
from packages.ml.model import artifact_integrity, run_log
from packages.ml.model.artifact_integrity import (
    REGISTRY_FILENAME,
    ArtifactIntegrityError,
    load_verified,
    record_artifact,
)
from packages.ml.model.cross_validation import cross_validator, cv_utils
from packages.ml.model.features import feature_engineering
from packages.ml.model.training import training
from packages.ml.model.visualization import training_diagnostics

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"

# The synthetic calendar mirrors the live config's reserved holdout window:
# 600 business dates, the final 500 reserved, the 100-date remainder split
# into the six fold blocks (edges 0, 16, 33, 50, 66, 83, 100).
N_DATES = 600
HOLDOUT_N_DATES = 500
TICKERS = ("AAA", "BBB", "CCC")
N_REGIME_BUCKETS = 7

# A long literal run so one flipped byte inside the pickled string keeps the
# pickle loadable while changing the deserialized payload.
MARKER = "A" * 64


def _registry_path(root: Path) -> Path:
    return root / REGISTRY_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _reject_constant(token: str):
    raise AssertionError(f"non-strict JSON constant emitted: {token}")


def _write_marker_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"marker": MARKER}, path)


def _tamper_marker_artifact(path: Path) -> None:
    """Flip one byte inside the pickled marker; the pickle still loads."""
    raw = bytearray(path.read_bytes())
    raw[raw.index(MARKER.encode("ascii")) + 5] ^= 0x01
    path.write_bytes(bytes(raw))


def test_tampered_artifact_is_refused_before_any_load() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        artifact = root / "artifact.pkl"
        _write_marker_artifact(artifact)
        record_artifact(artifact, registry_path=_registry_path(root))
        _tamper_marker_artifact(artifact)

        # Without verification the tampered bytes still deserialize — the gap
        # this module closes. With verification the load is refused.
        assert joblib.load(artifact)["marker"] != MARKER

        spy = mock.Mock(side_effect=AssertionError("joblib.load must not be called"))
        with mock.patch.object(artifact_integrity.joblib, "load", spy):
            with pytest.raises(ArtifactIntegrityError) as excinfo:
                load_verified(artifact, registry_path=_registry_path(root))

        assert not spy.called, "the load must be refused before joblib.load"
        message = str(excinfo.value)
        assert "sha256" in message, message
        assert str(artifact) in message, message


def test_missing_entry_is_refused_before_any_load() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        artifact = root / "artifact.pkl"
        _write_marker_artifact(artifact)
        # A registry exists, but it records a different artifact only.
        other = root / "other.pkl"
        _write_marker_artifact(other)
        record_artifact(other, registry_path=_registry_path(root))

        spy = mock.Mock(side_effect=AssertionError("joblib.load must not be called"))
        with mock.patch.object(artifact_integrity.joblib, "load", spy):
            with pytest.raises(ArtifactIntegrityError) as excinfo:
                load_verified(artifact, registry_path=_registry_path(root))

        assert not spy.called, "an unrecorded artifact must not reach joblib.load"
        message = str(excinfo.value)
        assert "no integrity entry" in message, message
        assert str(artifact) in message, message


def test_missing_file_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        artifact = root / "artifact.pkl"
        _write_marker_artifact(artifact)
        record_artifact(artifact, registry_path=_registry_path(root))
        artifact.unlink()

        spy = mock.Mock(side_effect=AssertionError("joblib.load must not be called"))
        with mock.patch.object(artifact_integrity.joblib, "load", spy):
            with pytest.raises(ArtifactIntegrityError) as excinfo:
                load_verified(artifact, registry_path=_registry_path(root))

        assert not spy.called, "a recorded-but-missing file must not reach joblib.load"
        message = str(excinfo.value)
        assert "missing" in message, message
        assert str(artifact) in message, message


def test_recorded_artifact_loads_like_a_direct_joblib_load() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        artifact = root / "artifact.pkl"
        payload = {"values": np.arange(5), "label": "verified"}
        joblib.dump(payload, artifact)
        record_artifact(artifact, registry_path=_registry_path(root))

        loaded = load_verified(artifact, registry_path=_registry_path(root))
        direct = joblib.load(artifact)
        assert loaded["label"] == direct["label"]
        assert np.array_equal(loaded["values"], direct["values"])


def test_registry_entry_carries_size_and_sha256() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        artifact = root / "artifact.pkl"
        joblib.dump({"value": 1}, artifact)

        entry = record_artifact(artifact, registry_path=_registry_path(root))
        assert entry == {"size": artifact.stat().st_size, "sha256": _sha256(artifact)}

        registry = json.loads(
            _registry_path(root).read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
        assert len(registry) == 1
        stored_key = next(iter(registry))
        assert registry[stored_key] == entry
        assert run_log.resolve_stored_path(stored_key) == artifact
        assert not list(root.rglob("*.tmp")), "the registry update left *.tmp residue"


def _synthetic_frame() -> pd.DataFrame:
    """A minimal extraction input frame with unique (ticker, date) pairs."""
    dates = pd.bdate_range("2000-01-03", periods=N_DATES)
    n_rows = len(dates) * len(TICKERS)
    data = {
        "date": np.repeat(dates.values, len(TICKERS)),
        "ticker": np.tile(np.array(TICKERS), N_DATES),
        "realized_volatility_5d": np.linspace(0.05, 0.6, n_rows),
        "market_regime": np.repeat(
            np.arange(N_DATES) % N_REGIME_BUCKETS, len(TICKERS)
        ).astype(np.int8),
    }
    for column, name in enumerate(feature_engineering.INDICATORS):
        data[name] = np.arange(n_rows, dtype=np.float64) + column + 1.0
    return pd.DataFrame(data)


def _without_dataset_pin(frame: pd.DataFrame) -> dict:
    """Stub for the dataset pin, which reads the live parquet from disk."""
    return {"rows": int(frame.shape[0])}


class _StubModel:
    """Picklable stand-in fold model with the surface the training path reads."""

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.zeros(len(feature_engineering.INDICATORS), dtype=np.float64)


def _stub_trainer(X_train, y_train, X_val=None, y_val=None):
    return _StubModel(), {}


def _constant_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _run_full_pipeline(root: Path, spies: dict | None = None) -> dict:
    """Drive the real Phase 2 -> 3 -> 4 writers and readers under ``root``.

    ``spies`` maps module names to call lists; when given, each module's
    ``load_verified`` reference is wrapped so the caller can assert that the
    live load sites go through verification.
    """
    artifacts = root / "artifacts"
    artifacts.mkdir()
    # The orchestrator creates the subdirectories before the phases run; the
    # result writer relies on its output directory existing.
    (artifacts / "results").mkdir()
    frame = _synthetic_frame()
    dataset = root / "dataset.parquet"
    frame.to_parquet(dataset, index=False)

    def _spy(module, calls):
        real = module.load_verified

        def wrapper(path, **kwargs):
            calls.append(Path(path))
            return real(path, **kwargs)

        return wrapper

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(feature_engineering, "DATASET_PATH", dataset)
        )
        stack.enter_context(
            mock.patch.object(feature_engineering, "_dataset_pin", _without_dataset_pin)
        )
        stack.enter_context(
            mock.patch.object(feature_engineering, "ARTIFACTS_DIR", artifacts)
        )
        stack.enter_context(
            mock.patch.object(cross_validator, "ARTIFACTS_DIR", artifacts)
        )
        stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", artifacts))
        stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", artifacts))
        stack.enter_context(
            mock.patch.object(training_diagnostics, "ARTIFACTS_DIR", artifacts)
        )
        stack.enter_context(
            mock.patch.object(training, "load_har_regressors", _constant_har_regressors)
        )
        stack.enter_context(
            mock.patch.object(training, "train_xgboost_model", _stub_trainer)
        )
        if spies is not None:
            for name, module in (
                ("cv_utils", cv_utils),
                ("cross_validator", cross_validator),
                ("training", training),
                ("training_diagnostics", training_diagnostics),
            ):
                stack.enter_context(
                    mock.patch.object(module, "load_verified", _spy(module, spies[name]))
                )

        _X, _Y, metadata, _encoder = feature_engineering.phase_2_feature_selection()
        folds = cross_validator.phase_3_cross_validation()
        validation, summary, model_paths = training.phase_4_training()

        predictions = training_diagnostics.load_validation_predictions()
        plot_path = root / "feature_importance.png"
        training_diagnostics.plot_feature_importance(validation, None, plot_path)

    return {
        "artifacts": artifacts,
        "folds": folds,
        "validation": validation,
        "summary": summary,
        "model_paths": model_paths,
        "predictions": predictions,
        "plot_path": plot_path,
        "metadata": metadata,
    }


def test_pipeline_run_verifies_all_five_converted_sites() -> None:
    spies = {
        name: []
        for name in ("cv_utils", "cross_validator", "training", "training_diagnostics")
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_full_pipeline(Path(tmpdir), spies=spies)

        assert cross_validator.HOLDOUT_N_DATES == HOLDOUT_N_DATES
        assert len(result["folds"]) == 5
        assert set(result["summary"]) == {f"fold_{idx}" for idx in range(5)}
        assert len(result["validation"]) == 5
        assert len(result["model_paths"]) == 5
        assert set(result["predictions"]) == set(result["validation"])
        for fold_key, fold_data in result["validation"].items():
            assert (
                result["predictions"][fold_key]["test_qlike"]
                == fold_data["test_qlike"]
            )
        assert result["plot_path"].is_file()

    def _names(calls: list[Path]) -> list[str]:
        return [path.name for path in calls]

    # Site 1: cv_utils.load_folds_metadata.
    assert "folds_metadata.pkl" in _names(spies["cv_utils"]), spies["cv_utils"]
    # Site 2: the cross-validation metadata load.
    assert "metadata.pkl" in _names(spies["cross_validator"]), spies["cross_validator"]
    # Site 3: the training metadata load.
    assert "metadata.pkl" in _names(spies["training"]), spies["training"]
    # Site 4: the diagnostics predictions load.
    assert "validation_predictions.pkl" in _names(spies["training_diagnostics"]), (
        spies["training_diagnostics"]
    )
    # Site 5: the diagnostics checkpoint load, once per fold.
    checkpoint_names = {f"fold_{fold_idx}.pkl" for fold_idx in range(5)}
    assert checkpoint_names <= set(_names(spies["training_diagnostics"])), (
        spies["training_diagnostics"]
    )


def test_writer_sites_record_verifiable_entries() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_full_pipeline(Path(tmpdir))
        artifacts = result["artifacts"]
        registry = json.loads(
            (artifacts / REGISTRY_FILENAME).read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
        run_id = json.loads(
            (artifacts / "run_manifest.json").read_text(encoding="utf-8")
        )["run_id"]

        expected = {
            # Phase 2 inputs metadata (loaded by Phases 3 and 4).
            artifacts / "inputs" / "metadata.pkl",
            # Phase 3 fold metadata (loaded by cv_utils and the reader).
            artifacts / "folds" / "folds_metadata.pkl",
            # Phase 4 prediction/result files.
            artifacts / "results" / "validation_predictions.pkl",
            artifacts / "results" / "fold_training_summary.json",
            artifacts / "results" / "har_benchmark.json",
            # The evaluation report's per-row loss vectors and summary.
            artifacts / "results" / "per_row_losses.parquet",
            artifacts / "results" / "evaluation_summary.json",
        }
        expected |= {
            artifacts / "checkpoints" / run_id / f"fold_{fold_idx}.pkl"
            for fold_idx in range(5)
        }
        expected |= {
            artifacts / "checkpoints" / run_id / f"fold_{fold_idx}_params.json"
            for fold_idx in range(5)
        }

        resolved = {run_log.resolve_stored_path(key) for key in registry}
        assert resolved == expected, sorted(str(path) for path in resolved ^ expected)

        for key, entry in registry.items():
            path = run_log.resolve_stored_path(key)
            assert isinstance(entry, dict), (key, entry)
            assert set(entry) == {"size", "sha256"}, (key, entry)
            assert entry["size"] == path.stat().st_size, key
            assert entry["sha256"] == _sha256(path), key

        # The recorded entries verify: reloads through the helper return the
        # payloads the run produced.
        registry_path = artifacts / REGISTRY_FILENAME
        metadata = load_verified(
            artifacts / "inputs" / "metadata.pkl", registry_path=registry_path
        )
        assert metadata["n_indicators"] == len(feature_engineering.INDICATORS)
        folds = load_verified(
            artifacts / "folds" / "folds_metadata.pkl", registry_path=registry_path
        )
        assert folds["schema_version"] == 2
        reloaded = load_verified(
            artifacts / "results" / "validation_predictions.pkl",
            registry_path=registry_path,
        )
        assert set(reloaded) == {f"fold_{fold_idx}" for fold_idx in range(5)}
        checkpoint = load_verified(
            artifacts / "checkpoints" / run_id / "fold_0.pkl",
            registry_path=registry_path,
        )
        assert isinstance(checkpoint, _StubModel)

        assert not list(artifacts.rglob("*.tmp")), "a writer left *.tmp residue"


def test_x_and_y_loads_stay_direct_np_load() -> None:
    for name in ("cross_validation/cross_validator.py", "training/training.py"):
        tree = ast.parse(
            (MODEL_ROOT / name).read_text(encoding="utf-8"), filename=name
        )
        np_load_targets: list[str] = []
        verified_calls: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.args
            ):
                np_load_targets.append(ast.unparse(node.args[0]))
            if isinstance(node.func, ast.Name) and node.func.id == "load_verified":
                verified_calls.append(ast.unparse(node))
        assert any("X.npy" in target for target in np_load_targets), (
            name,
            np_load_targets,
        )
        assert any("Y.npy" in target for target in np_load_targets), (
            name,
            np_load_targets,
        )
        assert not any(".npy" in call for call in verified_calls), verified_calls


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
    print("All tests passed")
