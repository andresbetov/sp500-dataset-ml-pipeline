"""Resumable-run contract: ``--resume``, ``--folds``, and identity checks.

Every exercised run drives the real ``phase_4_training`` against a temporary
artifact root with a recording stub trainer and constant HAR regressors, so no
live pipeline runs and no live ``artifacts/**`` or ``data/model_outputs/**``
file is touched.

The checks pin:

- a partial run (``folds=[0, 2]``) writes per-fold records, and a resume trains
  only the missing folds while producing the same metrics and predictions as a
  full run on the same fixture;
- complete per-fold records are byte-identical across the resume (they were
  not retrained or rewritten);
- a missing or mismatched config hash / fold-manifest identity refuses by name;
- an incomplete record (missing sidecar or tampered checkpoint hash) is
  recomputed;
- the CLI exposes ``--resume`` and ``--folds`` and records the fold selection;
- the no-argument dry-run plan still works.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_resume.py
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model import config as model_config
from packages.ml.model import run_log
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cross_validator, cv_utils
from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.training import training

N_ROWS = 60
BLOCK_ROWS = 10
N_INDICATORS = 3
N_TICKER_FEATURES = 1

# Timing and run-scoped paths differ between a full and a resumed run; every
# other summary field is a result and must match exactly.
NON_RESULT_SUMMARY_KEYS = {
    "train_time_seconds",
    "inference_time_seconds",
    "har_time_seconds",
    "fold_time_seconds",
    "model_path",
}


class _RecordingModel:
    """Picklable fold model: deterministic row-indexed log forecasts."""

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        rows = np.asarray(X)[:, 0].astype(np.float64)
        return -1.4 + 0.05 * np.sin(rows)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.zeros(len(INDICATORS), dtype=np.float64)


class _TrackingTrainer:
    """Stub trainer that records one entry per fit (the fit row count)."""

    def __init__(self) -> None:
        self.folds: list[int] = []

    def __call__(self, X_train, y_train, X_val=None, y_val=None):
        self.folds.append(len(X_train))
        return _RecordingModel(), {}


def _constant_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Constant components with no warm-up rows; the HAR fit stays real."""
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_synthetic_artifacts(root: Path) -> None:
    """Write a small live-shaped Phase 2-3 input set under ``root``."""
    rng = np.random.default_rng(11)
    y = rng.uniform(0.05, 0.6, size=N_ROWS)
    X = rng.normal(size=(N_ROWS, N_INDICATORS + N_TICKER_FEATURES))
    X[:, 0] = np.arange(N_ROWS, dtype=np.float64)
    dates = np.datetime64("2000-01-03") + np.arange(N_ROWS)

    dataset_path = root / "dataset.parquet"
    dataset_bytes = b"synthetic dataset v1"
    dataset_path.write_bytes(dataset_bytes)

    metadata = {
        "ticker": np.array([f"T{row % 3}" for row in range(N_ROWS)]),
        "date": dates,
        "regime": (np.arange(N_ROWS) % 7).astype(np.int8),
        "n_indicators": N_INDICATORS,
        "n_ticker_features": N_TICKER_FEATURES,
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_bytes(dataset_bytes),
        },
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


def _prepared_root(tmpdir: str) -> Path:
    root = Path(tmpdir) / "artifacts"
    root.mkdir()
    _write_synthetic_artifacts(root)
    return root


def _exercise_run(
    root: Path,
    *,
    resume: bool = False,
    folds: list[int] | None = None,
    trainer: _TrackingTrainer | None = None,
    run: dict | None = None,
) -> tuple:
    """Drive one real ``phase_4_training`` run against the temporary root."""
    active_trainer = trainer if trainer is not None else _TrackingTrainer()
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
        stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
        stack.enter_context(
            mock.patch.object(
                training, "load_har_regressors", _constant_har_regressors
            )
        )
        stack.enter_context(
            mock.patch.object(training, "train_xgboost_model", active_trainer)
        )
        kwargs = {"resume": resume}
        if folds is not None:
            kwargs["folds"] = folds
        if run is not None:
            kwargs["run"] = run
        result = training.phase_4_training(**kwargs)
    return result, active_trainer


def _manifest(root: Path) -> dict:
    return json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _sidecar_bytes(root: Path, run_id: str, fold_idx: int) -> dict[str, bytes]:
    run_dir = root / "checkpoints" / run_id
    names = [
        f"fold_{fold_idx}.pkl",
        f"fold_{fold_idx}_params.json",
        f"fold_{fold_idx}_summary.json",
        f"fold_{fold_idx}_predictions.pkl",
    ]
    return {name: (run_dir / name).read_bytes() for name in names}


def _resume_error_type() -> type[Exception]:
    """The named resume error, looked up lazily so the red run stays observable."""
    return getattr(training, "ResumeError", RuntimeError)


def test_resume_trains_only_missing_folds_and_matches_a_full_run() -> None:
    with tempfile.TemporaryDirectory() as full_tmp, tempfile.TemporaryDirectory() as part_tmp:
        full_root = _prepared_root(full_tmp)
        full_validation, full_summary, _ = _exercise_run(full_root)[0]
        full_manifest = _manifest(full_root)
        assert full_manifest["partial"] is False
        assert (full_root / "results" / "run_selection.json").is_file()

        partial_root = _prepared_root(part_tmp)
        _partial_validation, partial_summary, _ = _exercise_run(
            partial_root, folds=[0, 2]
        )[0]
        assert set(partial_summary) == {"fold_0", "fold_2"}
        partial_manifest = _manifest(partial_root)
        run_id = partial_manifest["run_id"]
        assert partial_manifest["fold_selection"]["requested"] == [0, 2]
        assert partial_manifest["fold_selection"]["trained"] == [0, 2]

        before = {
            fold_idx: _sidecar_bytes(partial_root, run_id, fold_idx)
            for fold_idx in (0, 2)
        }

        trainer = _TrackingTrainer()
        resumed_validation, resumed_summary, _ = _exercise_run(
            partial_root, resume=True, trainer=trainer
        )[0]

        assert len(trainer.folds) == 3, trainer.folds
        assert set(resumed_summary) == {f"fold_{idx}" for idx in range(5)}
        resumed_manifest = _manifest(partial_root)
        assert resumed_manifest["run_id"] == run_id
        assert resumed_manifest["resumed"] is True
        assert resumed_manifest["fold_selection"]["trained"] == [1, 3, 4]
        assert resumed_manifest["fold_selection"]["resumed"] == [0, 2]

        # Reused folds were neither retrained nor rewritten.
        for fold_idx in (0, 2):
            after = _sidecar_bytes(partial_root, run_id, fold_idx)
            assert after == before[fold_idx], f"fold_{fold_idx} sidecar changed"

        # Every result the full run produced is reproduced by the resume.
        for fold_key, base in full_summary.items():
            resumed = resumed_summary[fold_key]
            for key, value in base.items():
                if key in NON_RESULT_SUMMARY_KEYS:
                    continue
                if isinstance(value, float) and np.isnan(value):
                    assert isinstance(resumed[key], float) and np.isnan(resumed[key])
                else:
                    assert resumed[key] == value, (fold_key, key)
            np.testing.assert_array_equal(
                np.asarray(resumed_validation[fold_key]["test_pred"]),
                np.asarray(full_validation[fold_key]["test_pred"]),
            )
            np.testing.assert_array_equal(
                np.asarray(resumed_validation[fold_key]["test_true"]),
                np.asarray(full_validation[fold_key]["test_true"]),
            )


def test_resume_without_a_manifest_refuses_by_name() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        assert not (root / "run_manifest.json").exists()

        with pytest.raises(_resume_error_type()) as excinfo:
            _exercise_run(root, resume=True)

        message = str(excinfo.value)
        assert "resume" in message.lower(), message
        assert "run_manifest.json" in message, message


def test_resume_refuses_a_missing_config_hash() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0])
        manifest = _manifest(root)
        manifest.pop("config_hash")
        _write_manifest(root, manifest)

        with pytest.raises(_resume_error_type()) as excinfo:
            _exercise_run(root, resume=True)

        message = str(excinfo.value)
        assert "config_hash" in message, message
        assert "resume" in message.lower(), message


def test_resume_refuses_a_config_hash_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0])
        manifest = _manifest(root)
        manifest["config_hash"] = "deadbeef"
        _write_manifest(root, manifest)

        with pytest.raises(_resume_error_type()) as excinfo:
            _exercise_run(root, resume=True)

        message = str(excinfo.value)
        assert "config" in message.lower(), message
        assert "deadbeef" in message, message


def test_resume_refuses_a_fold_manifest_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0])
        manifest = _manifest(root)
        assert manifest["fold_manifest_hash"]
        manifest["fold_manifest_hash"] = "0" * 64
        _write_manifest(root, manifest)

        with pytest.raises(_resume_error_type()) as excinfo:
            _exercise_run(root, resume=True)

        message = str(excinfo.value)
        assert "fold" in message.lower(), message
        assert "manifest" in message.lower(), message


def test_resume_recomputes_a_fold_with_a_missing_summary_record() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0, 2])
        run_id = _manifest(root)["run_id"]
        (root / "checkpoints" / run_id / "fold_0_summary.json").unlink()

        trainer = _TrackingTrainer()
        _, summary, _ = _exercise_run(root, resume=True, trainer=trainer)[0]

        assert len(trainer.folds) == 4, trainer.folds
        assert set(summary) == {f"fold_{idx}" for idx in range(5)}


def test_resume_recomputes_a_fold_with_a_tampered_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0, 2])
        run_id = _manifest(root)["run_id"]
        checkpoint = root / "checkpoints" / run_id / "fold_0.pkl"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

        trainer = _TrackingTrainer()
        _, summary, _ = _exercise_run(root, resume=True, trainer=trainer)[0]

        assert len(trainer.folds) == 4, trainer.folds
        assert set(summary) == {f"fold_{idx}" for idx in range(5)}


def test_folds_subset_records_the_selection() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _, summary, paths = _exercise_run(root, folds=[0, 2])[0]

        assert set(summary) == {"fold_0", "fold_2"}
        assert len(paths) == 2
        manifest = _manifest(root)
        assert manifest["partial"] is True
        assert manifest["fold_selection"] == {
            "requested": [0, 2],
            "trained": [0, 2],
            "resumed": [],
            "n_folds_total": 5,
            "n_folds_selected": 2,
        }
        selection = json.loads(
            (root / "results" / "run_selection.json").read_text(encoding="utf-8")
        )
        assert selection["partial"] is True
        assert selection["requested"] == [0, 2]
        assert selection["trained"] == [0, 2]
        assert selection["resumed"] == []
        assert selection["fold_keys"] == ["fold_0", "fold_2"]
        assert selection["n_folds_total"] == 5
        assert selection["n_folds_selected"] == 2

        source = (
            REPO_ROOT / "packages" / "ml" / "model" / "main.py"
        ).read_text(encoding="utf-8")
        assert "--resume" in source
        assert "--folds" in source


def test_resume_with_folds_skips_only_selected_complete_records() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0, 2])

        trainer = _TrackingTrainer()
        _, summary, _ = _exercise_run(
            root, resume=True, folds=[1, 2, 3], trainer=trainer
        )[0]

        assert len(trainer.folds) == 2, trainer.folds
        assert set(summary) == {"fold_1", "fold_2", "fold_3"}
        manifest = _manifest(root)
        assert manifest["partial"] is True
        assert manifest["fold_selection"]["requested"] == [1, 2, 3]
        assert manifest["fold_selection"]["trained"] == [1, 3]
        assert manifest["fold_selection"]["resumed"] == [2]


def test_failed_resume_preserves_identity_and_can_retry() -> None:
    from packages.ml.model import main as main_module

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0, 2])
        before = _manifest(root)
        before_fold_hash = before["fold_manifest_hash"]
        before_inputs = before["input_identity"]

        with mock.patch.object(main_module, "ARTIFACTS_DIR", root), mock.patch.object(
            main_module,
            "phase_2_feature_selection",
            side_effect=RuntimeError("phase-2 boom"),
        ):
            with pytest.raises(RuntimeError, match="phase-2 boom"):
                main_module.main(resume=True)

        failed = _manifest(root)
        assert failed["status"] == "failed"
        assert failed["run_id"] == before["run_id"]
        assert failed["fold_manifest_hash"] == before_fold_hash
        assert failed["input_identity"] == before_inputs

        trainer = _TrackingTrainer()
        _, summary, _ = _exercise_run(root, resume=True, trainer=trainer)[0]
        assert len(trainer.folds) == 3, trainer.folds
        assert set(summary) == {f"fold_{idx}" for idx in range(5)}


def test_dataset_change_refuses_before_phase_2_rewrites_inputs() -> None:
    from packages.ml.model import main as main_module

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0, 2])
        x_before = (root / "inputs" / "X.npy").read_bytes()
        (root / "dataset.parquet").write_bytes(b"synthetic dataset v2")

        phase_two = mock.Mock(side_effect=AssertionError("Phase 2 must not run"))
        with mock.patch.object(main_module, "ARTIFACTS_DIR", root), mock.patch.object(
            main_module, "phase_2_feature_selection", phase_two
        ):
            with pytest.raises(_resume_error_type()) as excinfo:
                main_module.main(resume=True)

        phase_two.assert_not_called()
        assert (root / "inputs" / "X.npy").read_bytes() == x_before
        assert "dataset digest mismatch" in str(excinfo.value)


def test_input_change_refuses_before_any_fit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0, 2])
        x_path = root / "inputs" / "X.npy"
        x_path.write_bytes(x_path.read_bytes() + b"changed")

        trainer = _TrackingTrainer()
        with pytest.raises(_resume_error_type()) as excinfo:
            _exercise_run(root, resume=True, trainer=trainer)

        assert trainer.folds == []
        assert "input digest mismatch" in str(excinfo.value)


def test_persisted_fold_change_refuses_before_phase_2_rewrites_folds() -> None:
    from packages.ml.model import main as main_module

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0, 2])
        folds_path = root / "folds" / "folds_metadata.pkl"
        payload = joblib.load(folds_path)
        payload["fold_0"]["n_train"] = int(payload["fold_0"]["n_train"]) + 1
        cross_validator.write_folds_metadata_atomically(payload, folds_path)
        record_artifact(folds_path, registry_path=root / REGISTRY_FILENAME)

        phase_two = mock.Mock(side_effect=AssertionError("Phase 2 must not run"))
        with mock.patch.object(main_module, "ARTIFACTS_DIR", root), mock.patch.object(
            main_module, "phase_2_feature_selection", phase_two
        ):
            with pytest.raises(_resume_error_type()) as excinfo:
                main_module.main(resume=True)

        phase_two.assert_not_called()
        assert "fold manifest mismatch" in str(excinfo.value)


def test_fold_record_rejects_a_stale_summary_predictions_pair() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0])
        manifest = _manifest(root)
        run_id = manifest["run_id"]
        run_dir = root / "checkpoints" / run_id
        summary_path = run_dir / "fold_0_summary.json"
        predictions_path = run_dir / "fold_0_predictions.pkl"
        old_summary = summary_path.read_bytes()
        payload = joblib.load(predictions_path)
        modified_predictions = dict(payload["predictions"])
        modified_predictions["test_pred"] = (
            np.asarray(modified_predictions["test_pred"]) + 0.25
        )

        with mock.patch.object(training, "ARTIFACTS_DIR", root), mock.patch.object(
            training, "write_json_atomically", side_effect=RuntimeError("summary boom")
        ):
            with pytest.raises(RuntimeError, match="summary boom"):
                training.write_fold_record_atomically(
                    0,
                    run_id,
                    payload["summary"],
                    modified_predictions,
                    payload["har_result"],
                    config_hash_value=manifest["config_hash"],
                    fold_manifest_hash=manifest["fold_manifest_hash"],
                    coefficient_interpretation=False,
                )

        assert summary_path.read_bytes() == old_summary
        assert not list(root.rglob("*.tmp"))
        assert (
            training._load_complete_fold_record(
                0,
                run_id,
                config_hash_value=manifest["config_hash"],
                fold_manifest_hash=manifest["fold_manifest_hash"],
                coefficient_interpretation=False,
                artifacts_dir=root,
            )
            is None
        ), "a summary left beside newer predictions must not be trusted"


def test_fold_record_rejects_a_torn_predictions_pickle() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0])
        manifest = _manifest(root)
        run_id = manifest["run_id"]
        predictions_path = (
            root / "checkpoints" / run_id / "fold_0_predictions.pkl"
        )
        predictions_path.write_bytes(b"not a pickle")
        record_artifact(
            predictions_path, registry_path=root / REGISTRY_FILENAME
        )

        assert (
            training._load_complete_fold_record(
                0,
                run_id,
                config_hash_value=manifest["config_hash"],
                fold_manifest_hash=manifest["fold_manifest_hash"],
                coefficient_interpretation=False,
                artifacts_dir=root,
            )
            is None
        ), "an unreadable predictions pickle must be recomputed, not raised"


def test_fold_record_rejects_missing_resume_keys() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root, folds=[0])
        manifest = _manifest(root)
        run_id = manifest["run_id"]
        run_dir = root / "checkpoints" / run_id
        payload = joblib.load(run_dir / "fold_0_predictions.pkl")
        broken = dict(payload["predictions"])
        del broken["har_pred_all"]

        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            training.write_fold_record_atomically(
                0,
                run_id,
                payload["summary"],
                broken,
                payload["har_result"],
                config_hash_value=manifest["config_hash"],
                fold_manifest_hash=manifest["fold_manifest_hash"],
                coefficient_interpretation=False,
            )

        assert (
            training._load_complete_fold_record(
                0,
                run_id,
                config_hash_value=manifest["config_hash"],
                fold_manifest_hash=manifest["fold_manifest_hash"],
                coefficient_interpretation=False,
                artifacts_dir=root,
            )
            is None
        ), "a record missing keys the resumed path reads must be recomputed"


def test_resume_recomputes_a_record_missing_only_har_forecast_log() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root)
        manifest = _manifest(root)
        run_id = manifest["run_id"]
        run_dir = root / "checkpoints" / run_id
        payload = joblib.load(run_dir / "fold_0_predictions.pkl")
        broken = dict(payload["predictions"])
        del broken["har_forecast_log"]

        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            training.write_fold_record_atomically(
                0,
                run_id,
                payload["summary"],
                broken,
                payload["har_result"],
                config_hash_value=manifest["config_hash"],
                fold_manifest_hash=manifest["fold_manifest_hash"],
                coefficient_interpretation=False,
            )

        assert (
            training._load_complete_fold_record(
                0,
                run_id,
                config_hash_value=manifest["config_hash"],
                fold_manifest_hash=manifest["fold_manifest_hash"],
                coefficient_interpretation=False,
                artifacts_dir=root,
            )
            is None
        ), "a record missing har_forecast_log must be recomputed"

        trainer = _TrackingTrainer()
        _, summary, _ = _exercise_run(root, resume=True, trainer=trainer)[0]
        assert len(trainer.folds) == 1, trainer.folds
        assert set(summary) == {f"fold_{idx}" for idx in range(5)}


def test_config_hash_ignores_the_operational_run_section() -> None:
    loaded = model_config.load_config()
    changed = json.loads(json.dumps(loaded))
    changed["run"]["retention"] = 1
    assert run_log.config_hash(loaded) == run_log.config_hash(changed)

    changed["folds"]["n_folds"] = 4
    assert run_log.config_hash(loaded) != run_log.config_hash(changed)


def test_resume_validates_before_creating_artifact_directories() -> None:
    from packages.ml.model import main as main_module

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        with mock.patch.object(main_module, "ARTIFACTS_DIR", root):
            with pytest.raises(_resume_error_type()):
                main_module.main(resume=True)
        assert not root.exists(), "a failed resume probe created the artifact tree"


def test_subset_phase_5_renders_selected_folds_without_plot_errors() -> None:
    from packages.ml.model.visualization import training_diagnostics
    from packages.ml.model.visualization.data_loader import (
        load_fold_training_summary,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        validation, _summary, _ = _exercise_run(root, folds=[0, 2])[0]
        summary_df = load_fold_training_summary(
            root / "results" / "fold_training_summary.json"
        )
        output_dir = Path(tmpdir) / "plots"

        with mock.patch.object(
            training_diagnostics, "ARTIFACTS_DIR", root
        ):
            paths = training_diagnostics.generate_training_diagnostics(
                validation, summary_df, output_dir
            )

        assert {path.name for path in paths} == {
            "actual_vs_predicted.png",
            "residuals_distribution.png",
            "feature_importance.png",
            "error_by_volatility_bin.png",
        }
        assert all(path.is_file() for path in paths)


def test_cli_help_lists_resume_and_folds() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "packages.ml.model.main",
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout, result.stdout
    assert "--folds" in result.stdout, result.stdout


def test_no_argument_dry_run_plan_is_unchanged() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "packages.ml.model.main",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout.lower(), result.stdout
    assert "Phase 2" in result.stdout, result.stdout


def test_agent_and_readme_docs_name_the_new_flags() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    for document in (readme, agents):
        assert "--resume" in document, "docs must name --resume"
        assert "--folds" in document, "docs must name --folds"
        assert "partial" in document.lower(), "docs must describe partial runs"
    assert "run.retention" in readme, "README must document the retention key"
    assert "run_selection.json" in readme


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
