"""Tests for the run manifest, the append-only run log, and relative paths.

Every exercised run drives the real ``phase_4_training`` against a temporary
artifact root with a stub trainer and constant HAR regressors, so no live
pipeline runs and no live ``artifacts/**`` or ``data/model_outputs/**`` file is
touched. The checks pin one strict-JSON line per run in ``runs.jsonl`` with the
required provenance fields (run id, git sha, config hash, environment versions,
phase/fold statuses, durations, artifact hashes), append-only behaviour,
repo-relative persisted paths with a resolver, and the reader's resolution of
the stored live summary.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_manifest.py
"""

from __future__ import annotations

import contextlib
import json
import re
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.training import training

N_ROWS = 60
BLOCK_ROWS = 10
N_INDICATORS = 3
N_TICKER_FEATURES = 1

LIVE_SUMMARY = (
    REPO_ROOT / "packages" / "ml" / "model" / "artifacts" / "results"
    / "fold_training_summary.json"
)

RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
CONFIG_HASH_RE = re.compile(r"^[0-9a-f]{8}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The manifest/log schema: every line and the manifest carry these keys.
REQUIRED_RECORD_KEYS = {
    "run_id",
    "phase",
    "status",
    "git_sha",
    "config_hash",
    "environment",
    "started_at",
    "finished_at",
    "duration_seconds",
    "phases",
    "folds",
    "artifacts",
    "error",
}


class _ConstantModel:
    """Picklable stand-in fold model: predicts the log-target mean (zero)."""

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float64)


def _constant_trainer(X_train, y_train, X_val=None, y_val=None):
    return _ConstantModel(), {}


def _constant_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
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


def _exercise_run(root: Path, run: dict | None = None) -> tuple:
    """Drive one real ``phase_4_training`` run against the temporary root."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
        stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
        stack.enter_context(
            mock.patch.object(
                training, "load_har_regressors", _constant_har_regressors
            )
        )
        stack.enter_context(
            mock.patch.object(training, "train_xgboost_model", _constant_trainer)
        )
        if run is None:
            return training.phase_4_training()
        return training.phase_4_training(run=run)


def _prepared_root(tmpdir: str) -> Path:
    root = Path(tmpdir) / "artifacts"
    root.mkdir()
    _write_synthetic_artifacts(root)
    return root


def _reject_constant(token: str):
    raise AssertionError(f"non-strict JSON constant emitted: {token}")


def _walk_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def test_two_runs_write_two_jsonl_lines_with_required_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root)
        runs_path = root / "runs.jsonl"
        assert runs_path.is_file(), "no runs.jsonl after an exercised run"

        _exercise_run(root)
        lines = runs_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, f"expected one line per run, got {len(lines)}"

        from packages.ml.model import run_log

        records = [json.loads(line) for line in lines]
        for record in records:
            missing = REQUIRED_RECORD_KEYS - set(record)
            assert not missing, f"record is missing fields: {sorted(missing)}"
            assert not any(key.startswith("_") for key in record), (
                "internal run-state keys leaked into the log"
            )
            assert record["phase"] == "phase_4_training"
            assert RUN_ID_RE.fullmatch(record["run_id"]), record["run_id"]
            assert CONFIG_HASH_RE.fullmatch(record["config_hash"])
            assert record["run_id"].endswith(record["config_hash"])
            assert record["config_hash"] == run_log.config_hash()
            if record["git_sha"] is not None:
                assert GIT_SHA_RE.fullmatch(record["git_sha"]), record["git_sha"]
            assert record["environment"]["python"]
            assert record["environment"]["numpy"]
            assert record["status"] == "completed"
            assert record["error"] is None
            assert record["duration_seconds"] >= 0
            assert record["phases"], "no phase status recorded"
            assert all(
                phase["status"] == "completed" and phase["duration_seconds"] >= 0
                for phase in record["phases"]
            ), record["phases"]
            assert [fold["fold"] for fold in record["folds"]] == [
                f"fold_{fold_idx}" for fold_idx in range(5)
            ]
            assert all(
                fold["status"] == "completed" and fold["duration_seconds"] >= 0
                for fold in record["folds"]
            ), record["folds"]
            assert record["artifacts"], "no artifact hashes recorded"
            for stored, digest in record["artifacts"].items():
                assert not Path(stored).is_absolute(), stored
                assert SHA256_RE.fullmatch(digest), (stored, digest)
                assert run_log.resolve_stored_path(stored).is_file()

        manifest = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
        assert manifest == records[-1], "the manifest must match the last run line"


def test_run_log_lines_are_strict_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root)

        text = (root / "runs.jsonl").read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text, text
        lines = text.splitlines()
        assert lines, "no runs.jsonl lines to check"
        for line in lines:
            record = json.loads(line, parse_constant=_reject_constant)
            assert isinstance(record, dict), line


def test_persisted_paths_are_repo_relative_and_resolvable() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root)

        results_dir = root / "results"
        documents = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(results_dir.glob("*.json"))
        }
        documents["run_manifest.json"] = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )
        for name, document in documents.items():
            for value in _walk_strings(document):
                assert "/home/" not in value, (name, value)
                assert not value.startswith("/"), (name, value)

        summary = documents["fold_training_summary.json"]
        stored = summary["fold_0"]["model_path"]
        assert not Path(stored).is_absolute(), stored

        from packages.ml.model import run_log

        resolved = run_log.resolve_stored_path(stored)
        run_id = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )["run_id"]
        assert resolved == root / "checkpoints" / run_id / "fold_0.pkl"
        assert resolved.is_file()

        from packages.ml.model.visualization.data_loader import (
            load_fold_training_summary,
        )

        summary_df = load_fold_training_summary(
            root / "results" / "fold_training_summary.json"
        )
        loaded = Path(
            summary_df.loc[summary_df["fold"] == "fold_0", "model_path"].iloc[0]
        )
        assert loaded.is_absolute(), loaded
        assert loaded.is_file()


def test_second_run_appends_without_truncating_the_first_line() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root)
        runs_path = root / "runs.jsonl"
        assert runs_path.is_file(), "no runs.jsonl after an exercised run"

        first_bytes = runs_path.read_bytes()
        _exercise_run(root)
        after = runs_path.read_bytes()

        assert after.startswith(first_bytes), (
            "the second run truncated or rewrote the first line"
        )
        assert after != first_bytes
        assert len(after.splitlines()) == 2


def test_phase_4_records_folds_and_artifacts_into_the_caller_run() -> None:
    from packages.ml.model import run_log

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        run = run_log.new_run("pipeline", root)
        _exercise_run(root, run=run)
        run_log.finish_run(run, status="completed")

        record = json.loads(
            (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert record["phase"] == "pipeline"
        assert record["status"] == "completed"
        assert [fold["fold"] for fold in record["folds"]] == [
            f"fold_{fold_idx}" for fold_idx in range(5)
        ]
        assert record["phases"] == [], "phase timing belongs to the orchestrator"
        # 5 checkpoints + 5 parameter sidecars + the three training results +
        # the per-row loss vectors and the evaluation summary.
        assert len(record["artifacts"]) == 15, record["artifacts"]
        assert any(
            key.endswith(f"checkpoints/{run['run_id']}/fold_0.pkl")
            for key in record["artifacts"]
        )
        assert any(
            key.endswith(f"checkpoints/{run['run_id']}/fold_0_params.json")
            for key in record["artifacts"]
        )
        assert any(
            key.endswith("results/fold_training_summary.json")
            for key in record["artifacts"]
        )


def test_phase_4_run_names_the_frozen_selection_in_its_manifest() -> None:
    from packages.ml.model import config as model_config

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _exercise_run(root)
        record = json.loads(
            (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )

    loaded = model_config.get_config()
    selected = loaded["tuning"]["selected"]
    assert record["selection"] == {
        "run_id": selected["run_id"],
        "trial_index": selected["trial_index"],
        "config_hash": selected["config_hash"],
        "artifact": (
            f"{loaded['paths']['artifacts']}/tuning/{selected['run_id']}"
            "/selection.json"
        ),
    }
    # The context names the campaign; the parameters stay in the selection
    # artifact (the source of truth) and the frozen config section.
    assert "params" not in record["selection"]


def test_selection_context_is_absent_without_a_frozen_selection() -> None:
    from packages.ml.model import config as model_config
    from packages.ml.model import run_log

    without = {"training": {}, "paths": {"artifacts": "packages/ml/model/artifacts"}}
    with mock.patch.object(model_config, "get_config", return_value=without):
        assert run_log.selection_context() is None


def test_pipeline_run_writes_exactly_one_line_and_manifest() -> None:
    from packages.ml.model import main as main_module

    summary = {
        f"fold_{fold_idx}": {"test_qlike": 0.1, "har_qlike": 0.2}
        for fold_idx in range(5)
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(main_module, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(
                    main_module,
                    "phase_2_feature_selection",
                    return_value=(np.zeros((2, 2)), np.zeros(2), {}, None),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main_module, "phase_3_cross_validation", return_value=[{}] * 5
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main_module,
                    "phase_4_training",
                    return_value=({}, summary, [f"model_{idx}" for idx in range(5)]),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main_module,
                    "generate_all_visualizations",
                    return_value={"metrics": [1]},
                )
            )
            main_module.main()

        lines = (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, "a pipeline run must append exactly one line"
        record = json.loads(lines[0])
        assert record["phase"] == "pipeline"
        assert record["status"] == "completed"
        assert record["error"] is None
        assert [phase["phase"] for phase in record["phases"]] == [
            "phase_2_feature_selection",
            "phase_3_cross_validation",
            "phase_4_training",
            "phase_5_visualization",
        ]
        assert all(phase["status"] == "completed" for phase in record["phases"])
        manifest = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest == record


def test_pipeline_failure_is_recorded_as_failed() -> None:
    from packages.ml.model import main as main_module

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(main_module, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(
                    main_module,
                    "phase_2_feature_selection",
                    side_effect=RuntimeError("boom"),
                )
            )
            with pytest.raises(RuntimeError, match="boom"):
                main_module.main()

        lines = (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, "a failed pipeline run must still append one line"
        record = json.loads(lines[0])
        assert record["phase"] == "pipeline"
        assert record["status"] == "failed"
        assert record["error"] == "RuntimeError: boom"
        assert record["phases"] == []
        manifest = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest == record


def test_run_id_is_utc_stamp_plus_config_hash() -> None:
    from datetime import datetime, timezone

    from packages.ml.model import run_log

    run_id = run_log.make_run_id(
        "deadbeef", now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    )
    assert run_id == "20260102T030405Z-deadbeef"


def test_git_sha_is_null_outside_a_repository() -> None:
    from packages.ml.model import run_log

    with tempfile.TemporaryDirectory() as tmpdir:
        assert run_log.git_sha(project_root=Path(tmpdir)) is None


def test_live_summary_model_path_resolves_for_the_reader() -> None:
    if not LIVE_SUMMARY.is_file():
        pytest.skip(
            "requires the live fold training summary absent from this checkout"
        )

    stored = json.loads(LIVE_SUMMARY.read_text(encoding="utf-8"))["fold_0"][
        "model_path"
    ]

    from packages.ml.model import run_log

    resolved = run_log.resolve_stored_path(stored)
    if not resolved.is_file():
        pytest.skip(f"live fold-0 checkpoint absent from this checkout: {resolved}")

    # The branch-point artifact stores an absolute path; newly written summaries
    # store a repo-relative one. The reader must resolve both to the same file.
    from packages.ml.model.visualization.data_loader import (
        load_fold_training_summary,
    )

    summary_df = load_fold_training_summary(LIVE_SUMMARY)
    loaded = Path(
        summary_df.loc[summary_df["fold"] == "fold_0", "model_path"].iloc[0]
    )
    assert loaded.resolve() == resolved.resolve()


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
