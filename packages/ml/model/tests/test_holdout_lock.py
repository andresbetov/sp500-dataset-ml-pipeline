"""Tests for the locked holdout, the one-shot protocol, and the final report.

The fixture reserves the configured final 500 trading dates with the production
fold builder, writes Phase-2/3-shaped inputs under a temporary root, and drives
the real evaluation/refit/report path with a small deterministic trainer and
constant HAR regressors. No live ``artifacts/**`` or ``data/model_outputs/**``
file is read or written; the one live check is the CLI help.

Pinned here:

- the reserved holdout is disjoint from every fold and the final refit is
  separated from it by the configured purge horizon;
- the lock is written to the manifest before any holdout metric is computed;
- a second evaluation raises ``HoldoutLockError`` and leaves the first report
  byte-identical;
- a post-lock resume skips with a manifest record instead of re-evaluating;
- a tampered config, dataset pin, fold hash, or holdout bound mismatches the
  lock and raises;
- the final report and ``SHA256SUMS`` verify by hash, and the protocol and all
  four caveats are present;
- no tuning trial's fit or validation view touches the holdout window.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_holdout_lock.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR, REPO_ROOT, requires_live_artifacts
from packages.ml.model import config as model_config
from packages.ml.model import run_log
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.evaluation import holdout
from packages.ml.model.training import training
from packages.ml.model.tuning import harness

TICKERS = ("AAA", "BBB", "CCC")
HOLDOUT_N_DATES = 500
FOLD_N_DATES = 100
N_DATES = HOLDOUT_N_DATES + FOLD_N_DATES
N_FEATURES = 4
TINY_SEARCH_SPACE = {
    "coarse_grid": {
        "learning_rate": [0.03],
        "max_depth": [4],
        "min_child_weight": [1],
    },
    "finalist_refinement": {"gamma": [0.0]},
    "seed_finalists": [42],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_root(tmpdir: str) -> Path:
    """A temporary artifact root shaped like Phases 2-3 with a real holdout."""
    root = Path(tmpdir) / "artifacts"
    assert not root.resolve().is_relative_to(REPO_ROOT)
    rng = np.random.default_rng(54)
    calendar = pd.bdate_range("2000-01-03", periods=N_DATES).values
    row_dates = np.tile(calendar, len(TICKERS))
    tickers = np.repeat(np.asarray(TICKERS), N_DATES)
    n_rows = row_dates.size
    X = rng.normal(size=(n_rows, N_FEATURES))
    # The first feature is the row index, so the stub model can map rows back.
    X[:, 0] = np.arange(n_rows, dtype=np.float64)
    y = rng.uniform(0.05, 0.6, size=n_rows)
    regime = (np.arange(n_rows) % len(cross_validator.REGIME_NAMES)).astype(
        np.int8
    )
    folds_metadata = cross_validator.build_fold_payload(row_dates, regime)

    dataset_path = root / "dataset.parquet"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(b"synthetic dataset for the holdout fixture")
    metadata = {
        "ticker": tickers,
        "date": row_dates,
        "regime": regime,
        "n_indicators": N_FEATURES,
        "n_ticker_features": 0,
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
        },
    }

    inputs = root / "inputs"
    folds = root / "folds"
    inputs.mkdir(parents=True)
    folds.mkdir(parents=True)
    np.save(inputs / "X.npy", X)
    np.save(inputs / "Y.npy", y)
    joblib.dump(metadata, inputs / "metadata.pkl")
    joblib.dump(folds_metadata, folds / "folds_metadata.pkl")
    record_artifact(inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME)
    record_artifact(
        folds / "folds_metadata.pkl", registry_path=root / REGISTRY_FILENAME
    )
    return root


def _load(root: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    metadata = joblib.load(root / "inputs" / "metadata.pkl")
    folds_metadata = joblib.load(root / "folds" / "folds_metadata.pkl")
    return (
        np.load(root / "inputs" / "X.npy"),
        np.load(root / "inputs" / "Y.npy"),
        metadata,
        folds_metadata,
    )


def _constant_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Smooth deterministic components with no warm-up rows."""
    rows = np.arange(len(dates), dtype=np.float64)
    components = np.column_stack(
        [
            0.02 + 0.005 * np.sin(rows / 7.0),
            0.03 + 0.005 * np.cos(rows / 11.0),
            0.04 + 0.005 * np.sin(rows / 17.0),
        ]
    )
    return components, np.zeros(len(tickers), dtype=bool)


class _RowModel:
    """Picklable deterministic model keyed on the row-index feature."""

    def get_params(self) -> dict:
        return {"n_estimators": 1}

    def predict(self, X: np.ndarray) -> np.ndarray:
        rows = np.asarray(X)[:, 0].astype(np.float64)
        return -1.25 + 0.04 * np.sin(rows / 23.0)


class _RecordingTrainer:
    """Trainer double recording each fit's shape and validation arguments."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, X_train, y_train, X_val=None, y_val=None, params=None):
        self.calls.append(
            {
                "n_fit": int(len(X_train)),
                "n_features": int(X_train.shape[1]),
                "X_val": X_val,
                "y_val": y_val,
                "rows": np.asarray(X_train)[:, 0].astype(np.int64),
            }
        )
        return _RowModel(), {}


def _evaluate(
    root: Path,
    trainer: _RecordingTrainer | None = None,
    **kwargs,
) -> dict:
    active = trainer if trainer is not None else _RecordingTrainer()
    with mock.patch.object(holdout, "load_har_regressors", _constant_har_regressors):
        return holdout.evaluate_holdout(
            artifacts_dir=root, trainer=active, **kwargs
        )


def _manifest(root: Path) -> dict:
    return json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))


def _write_folds(root: Path, payload: dict) -> None:
    path = root / "folds" / "folds_metadata.pkl"
    cross_validator.write_folds_metadata_atomically(payload, path)
    record_artifact(path, registry_path=root / REGISTRY_FILENAME)


def _write_metadata(root: Path, payload: dict) -> None:
    path = root / "inputs" / "metadata.pkl"
    joblib.dump(payload, path)
    record_artifact(path, registry_path=root / REGISTRY_FILENAME)


def _parse_sums(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        entries[name.strip()] = digest.strip()
    return entries


def test_holdout_is_disjoint_from_every_fold_and_purged_on_entry() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        _X, y, metadata, folds_metadata = _load(root)

        reservation = holdout.validate_holdout_reservation(
            metadata["date"], y, folds_metadata
        )

        assert reservation["n_holdout_dates"] == HOLDOUT_N_DATES
        assert reservation["bounds"] == folds_metadata["holdout"]["bounds"]
        holdout_rows = set(reservation["holdout_indices"].tolist())
        holdout_start = np.datetime64(reservation["bounds"]["start"])
        assert (
            metadata["date"][reservation["holdout_indices"]] >= holdout_start
        ).all()

        for fold_key in training.fold_keys(folds_metadata):
            fold = folds_metadata[fold_key]
            train = set(fold["train_indices"])
            test = set(fold["test_indices"])
            assert not holdout_rows & train, fold_key
            assert not holdout_rows & test, fold_key
            assert (metadata["date"][fold["train_indices"]] < holdout_start).all()
            assert (metadata["date"][fold["test_indices"]] < holdout_start).all()

        # The final refit drops the rows whose five-day label window reaches the
        # first holdout date; exactly the configured purge horizon separates it.
        separation = reservation["separation"]
        assert separation >= cross_validator.SEPARATION_MIN
        assert separation == training.LABEL_HORIZON_DAYS + 1
        assert reservation["n_purged_fit_rows"] > 0
        fit_rows = set(reservation["fit_indices"].tolist())
        assert not fit_rows & holdout_rows
        assert reservation["n_scored_rows"] > 0
        assert set(reservation["scored_indices"].tolist()) <= holdout_rows

        # Poisoned metadata is re-checked: a holdout row in fold-train, and a
        # window that no longer reserves the configured final date count.
        poisoned = deepcopy(folds_metadata)
        poisoned["fold_0"]["train_indices"] = (
            poisoned["fold_0"]["train_indices"]
            + [int(reservation["holdout_indices"][0])]
        )
        with pytest.raises(holdout.HoldoutLockError, match="fold"):
            holdout.validate_holdout_reservation(
                metadata["date"], y, poisoned
            )

        poisoned_bounds = deepcopy(folds_metadata)
        poisoned_bounds["holdout"]["n_dates"] = HOLDOUT_N_DATES - 1
        with pytest.raises(holdout.HoldoutLockError, match="holdout"):
            holdout.validate_holdout_reservation(
                metadata["date"], y, poisoned_bounds
            )


def test_lock_is_recorded_before_metrics_and_report_and_sums_verify() -> None:
    events: list[str] = []
    real_metrics = holdout.compute_holdout_metrics
    real_write_state = holdout.write_run_state

    def spy_metrics(*args, **kwargs):
        assert (root / "results" / holdout.LOCK_FILENAME).is_file(), (
            "the lock file must exist before any holdout metric is computed"
        )
        events.append("metrics")
        return real_metrics(*args, **kwargs)

    def spy_write_state(run, *args, **kwargs):
        record = run.get("holdout")
        if isinstance(record, dict) and record.get("lock_sha256"):
            if "metrics" not in events:
                assert record.get("status") == "locked", record
            events.append("manifest-lock")
        return real_write_state(run, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        trainer = _RecordingTrainer()
        har_calls: list[tuple[np.ndarray, np.ndarray]] = []
        real_har = holdout.fit_har_benchmark

        def spy_har(*args, **kwargs):
            har_calls.append(
                (
                    np.asarray(kwargs["train_indices"]).copy(),
                    np.asarray(kwargs["test_indices"]).copy(),
                )
            )
            return real_har(*args, **kwargs)

        with mock.patch.object(
            holdout, "compute_holdout_metrics", side_effect=spy_metrics
        ), mock.patch.object(
            holdout, "write_run_state", side_effect=spy_write_state
        ), mock.patch.object(
            holdout, "fit_har_benchmark", side_effect=spy_har
        ):
            result = _evaluate(root, trainer)

        assert "manifest-lock" in events, events
        assert "metrics" in events, events
        assert events.index("manifest-lock") < events.index("metrics"), events

        results = root / "results"
        lock_path = results / holdout.LOCK_FILENAME
        report_path = results / holdout.REPORT_FILENAME
        sums_path = results / holdout.SUMS_FILENAME
        assert lock_path.is_file()
        assert report_path.is_file()
        assert sums_path.is_file()
        assert not list(root.rglob("*.tmp"))

        lock_document = json.loads(lock_path.read_text(encoding="utf-8"))
        assert lock_document["sha256"] == holdout.lock_sha256(
            lock_document["payload"]
        )
        assert lock_document["payload"]["derived_inputs"] == {
            "x_sha256": _sha256(root / "inputs" / "X.npy"),
            "y_sha256": _sha256(root / "inputs" / "Y.npy"),
        }
        manifest = _manifest(root)
        assert manifest["holdout"]["lock_sha256"] == lock_document["sha256"]
        assert manifest["holdout"]["status"] == "completed"
        assert manifest["holdout"]["bounds"] == lock_document["payload"]["holdout"][
            "bounds"
        ]

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["config_hash"] == run_log.config_hash()
        assert report["git_sha"] == run_log.git_sha()
        assert report["environment"]["python"]
        assert report["environment_reference"].endswith("run_manifest.json#environment")
        assert report["lock"]["sha256"] == lock_document["sha256"]
        assert report["protocol"]["one_shot"]["lock_before_metrics"] is True
        assert report["protocol"]["refit"]["rows"] == reservation_rows(root)
        assert report["protocol"]["refit"]["validation_tail"] is False
        assert report["headline"]["aggregate"]["edge"] == (
            report["headline"]["aggregate"]["har_qlike"]
            - report["headline"]["aggregate"]["model_qlike"]
        )
        assert set(report["headline"]["caveats"]) == {
            "single_seed",
            "floor_policy",
            "holdout",
            "adjustment_not_applied",
        }
        assert all(report["headline"]["caveats"].values())
        assert "exactly once" in report["headline"]["caveats"]["holdout"]
        assert "not applied" in report["headline"]["caveats"][
            "adjustment_not_applied"
        ].lower()
        assert report["post_hoc_adjustment"]["status"] == "not applied"
        assert report["metrics"]["n_scored_rows"] > 0
        assert report["metrics"]["n_floor_excluded"] >= 0
        assert report["holdout_metrics"] == report["metrics"]
        assert report["artifact_hashes"] == report["artifacts"]
        assert report["benchmark"]["har_qlike"] == report["metrics"]["har_qlike"]
        for path, digest in report["artifacts"].items():
            resolved = run_log.resolve_stored_path(path)
            assert _sha256(resolved) == digest, path

        sums = _parse_sums(sums_path)
        assert sums[holdout.REPORT_FILENAME] == _sha256(report_path)
        assert sums[holdout.PREDICTIONS_FILENAME] == _sha256(
            results / holdout.PREDICTIONS_FILENAME
        )
        assert sums[holdout.LOCK_FILENAME] == _sha256(lock_path)
        for name, digest in sums.items():
            assert _sha256(results / name) == digest, name

        predictions = joblib.load(results / holdout.PREDICTIONS_FILENAME)
        assert len(predictions["holdout_indices"]) == len(
            predictions["y_pred_model_log"]
        )
        assert predictions["lock_sha256"] == lock_document["sha256"]
        # The final refit is one model on exactly the purged remainder, with no
        # inner-validation tail; the HAR benchmark receives the same fit rows
        # and forecasts the same scored holdout rows.
        reservation = holdout.validate_holdout_reservation(
            *_target_arrays(root)
        )
        assert len(trainer.calls) == 1
        assert trainer.calls[0]["X_val"] is None
        assert trainer.calls[0]["n_fit"] == len(reservation["fit_indices"])
        np.testing.assert_array_equal(
            trainer.calls[0]["rows"], reservation["fit_indices"]
        )
        assert len(har_calls) == 1
        np.testing.assert_array_equal(
            har_calls[0][0], reservation["fit_indices"]
        )
        np.testing.assert_array_equal(
            har_calls[0][1], reservation["scored_indices"]
        )
        assert result["status"] == "completed"


def reservation_rows(root: Path) -> int:
    reservation = holdout.validate_holdout_reservation(*_target_arrays(root))
    return int(len(reservation["fit_indices"]))


def _target_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    _X, y, metadata, folds_metadata = _load(root)
    return metadata["date"], y, folds_metadata


def test_second_evaluation_raises_and_never_rewrites_the_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        _evaluate(root)
        report_path = root / "results" / holdout.REPORT_FILENAME
        before = report_path.read_bytes()

        trainer = _RecordingTrainer()
        with pytest.raises(holdout.HoldoutLockError, match="already"):
            _evaluate(root, trainer)

        assert trainer.calls == []
        assert report_path.read_bytes() == before


def test_post_lock_resume_skips_with_a_manifest_record() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        first = _evaluate(root)
        report_path = root / "results" / holdout.REPORT_FILENAME
        before = report_path.read_bytes()

        run = run_log.new_run("pipeline", root)
        trainer = _RecordingTrainer()
        skipped = _evaluate(root, trainer, run=run, resume=True)

        assert skipped["status"] == "skipped"
        assert trainer.calls == []
        assert report_path.read_bytes() == before
        manifest = _manifest(root)
        assert manifest["holdout"]["status"] == "skipped"
        assert manifest["holdout"]["lock_sha256"] == first["lock"]["sha256"]
        assert manifest["holdout"]["reason"]
        assert "re-evaluating" in manifest["holdout"]["reason"]


def test_missing_results_lock_still_binds_via_manifest_record() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        first = _evaluate(root)
        lock_path = root / "results" / holdout.LOCK_FILENAME
        manifest = _manifest(root)
        run = run_log.new_run(
            "pipeline", root, run_id=manifest["run_id"]
        )  # run_id reuse does not rewrite the manifest

        # A fresh/rerun with the results lock gone must not silently score.
        lock_path.unlink()
        trainer = _RecordingTrainer()
        with pytest.raises(holdout.HoldoutLockError, match="already"):
            _evaluate(root, trainer, run=run)
        assert trainer.calls == []
        assert lock_path.is_file(), "the lock must be restored from the manifest"
        restored = json.loads(lock_path.read_text(encoding="utf-8"))
        assert restored["sha256"] == first["lock"]["sha256"]

        # A resume with the results lock gone skips with a manifest record.
        lock_path.unlink()
        trainer = _RecordingTrainer()
        skipped = _evaluate(root, trainer, run=run, resume=True)
        assert skipped["status"] == "skipped"
        assert trainer.calls == []
        assert lock_path.is_file()
        manifest = _manifest(root)
        assert manifest["holdout"]["status"] == "skipped"
        assert manifest["holdout"]["lock_sha256"] == first["lock"]["sha256"]
        assert manifest["holdout"]["lock_payload"]["config_hash"] == (
            first["lock"]["payload"]["config_hash"]
        )

        # A forced re-evaluation may use the manifest's lock record when the
        # results lock is gone, and appends the override.
        lock_path.unlink()
        trainer = _RecordingTrainer()
        forced = _evaluate(
            root, trainer, run=run, force_reason="recover a lost lock artifact"
        )
        assert forced["status"] == "overridden"
        assert len(trainer.calls) == 1
        assert lock_path.is_file()
        overrides = [
            json.loads(line)
            for line in (root / "results" / holdout.OVERRIDES_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert overrides[-1]["reason"] == "recover a lost lock artifact"


def test_force_holdout_without_lock_or_manifest_record_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        _evaluate(root)
        (root / "results" / holdout.LOCK_FILENAME).unlink()
        manifest = _manifest(root)
        manifest.pop("holdout")
        (root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        # Remove the append-only fallback too: no durable lock record remains.
        (root / run_log.RUNS_FILENAME).unlink()

        trainer = _RecordingTrainer()
        with pytest.raises(ValueError, match="existing holdout lock"):
            _evaluate(root, trainer, force_reason="no recorded lock")
        assert trainer.calls == []


def test_force_holdout_requires_a_manifest_record() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        _evaluate(root)
        manifest = _manifest(root)
        manifest.pop("holdout")
        (root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        # The results lock survives, but the append-only fallback must not
        # stand in for the manifest record force is required to name.
        (root / run_log.RUNS_FILENAME).unlink()

        trainer = _RecordingTrainer()
        with pytest.raises(holdout.HoldoutLockError, match="manifest"):
            _evaluate(root, trainer, force_reason="no manifest record")
        assert trainer.calls == []
        assert not (root / "results" / holdout.OVERRIDES_FILENAME).exists()


def test_ordinary_run_carries_the_holdout_block() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        first = _evaluate(root)

        ordinary = run_log.new_run("pipeline", root)
        assert ordinary["holdout"]["lock_sha256"] == first["lock"]["sha256"]
        run_log.finish_run(ordinary, status="completed")
        manifest = _manifest(root)
        assert manifest["holdout"]["lock_sha256"] == first["lock"]["sha256"]

        lock_path = root / "results" / holdout.LOCK_FILENAME
        lock_path.unlink()
        trainer = _RecordingTrainer()
        with pytest.raises(holdout.HoldoutLockError, match="already"):
            _evaluate(root, trainer)
        assert trainer.calls == []
        assert lock_path.is_file(), "the lock must be restored from the manifest"


def test_runs_log_fallback_restores_a_lost_manifest_holdout() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        first = _evaluate(root)

        # Simulate a historical ordinary run whose manifest was written
        # without the holdout block; runs.jsonl is the only remaining record.
        ordinary = run_log.new_run("pipeline", root)
        ordinary.pop("holdout")
        run_log.finish_run(ordinary, status="completed")
        assert "holdout" not in _manifest(root)

        lock_path = root / "results" / holdout.LOCK_FILENAME
        lock_path.unlink()
        trainer = _RecordingTrainer()
        with pytest.raises(holdout.HoldoutLockError, match="already"):
            _evaluate(root, trainer)
        assert trainer.calls == []
        assert lock_path.is_file(), "the lock must be restored from runs.jsonl"
        restored = json.loads(lock_path.read_text(encoding="utf-8"))
        assert restored["sha256"] == first["lock"]["sha256"]

        # The same fallback supports a resume skip.
        lock_path.unlink()
        run = run_log.new_run("pipeline", root)
        assert "holdout" not in run
        trainer = _RecordingTrainer()
        skipped = _evaluate(root, trainer, run=run, resume=True)
        assert skipped["status"] == "skipped"
        assert trainer.calls == []
        assert lock_path.is_file()
        assert _manifest(root)["holdout"]["status"] == "skipped"


def test_force_holdout_refuses_a_changed_lock_identity() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        _evaluate(root)
        _X, _y, _metadata, folds_metadata = _load(root)
        folds_metadata["fold_0"]["n_train"] = (
            int(folds_metadata["fold_0"]["n_train"]) + 1
        )
        _write_folds(root, folds_metadata)

        trainer = _RecordingTrainer()
        with pytest.raises(holdout.HoldoutLockError, match="fold"):
            _evaluate(root, trainer, force_reason="should not bypass identity")
        assert trainer.calls == []
        assert not (root / "results" / holdout.OVERRIDES_FILENAME).exists()


def test_lock_payload_refuses_an_empty_hyperparameter_mapping() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        _X, _y, metadata, folds_metadata = _load(root)
        config = deepcopy(model_config.get_config())
        config["training"]["hyperparameters"] = {}
        with pytest.raises(holdout.HoldoutLockError, match="hyperparameters"):
            holdout.build_lock_payload(
                config=config,
                metadata=metadata,
                folds_metadata=folds_metadata,
                input_hashes={"x_sha256": "0" * 64, "y_sha256": "1" * 64},
            )


def test_compute_holdout_metrics_reports_supplied_floor_counts() -> None:
    target = np.log(np.array([0.2, 0.3, 0.4, 0.5]))
    model = target + 0.01
    har = target + 0.02
    dates = np.datetime64("2020-01-01") + np.arange(target.size)
    regime = np.arange(target.size) % len(cross_validator.REGIME_NAMES)

    metrics = holdout.compute_holdout_metrics(
        y_true=target,
        y_pred_model=model,
        y_pred_har=har,
        dates=dates,
        regime=regime,
        n_floor_excluded=3,
        n_holdout_rows=7,
    )

    assert metrics["n_scored_rows"] == 4
    assert metrics["n_floor_excluded"] == 3
    assert metrics["n_holdout_rows"] == 7


@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("config", "config"),
        ("dataset", "dataset"),
        ("inputs", "input"),
        ("folds", "fold"),
        ("bounds", "holdout"),
    ],
)
def test_tampered_lock_identity_mismatches_and_raises(
    target: str, field: str
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        _evaluate(root)
        _X, _y, metadata, folds_metadata = _load(root)

        config_patch = None
        if target == "config":
            changed = deepcopy(model_config.get_config())
            changed["training"]["hyperparameters"]["max_depth"] = 3
            config_patch = mock.patch.object(
                model_config, "get_config", return_value=changed
            )
        elif target == "dataset":
            metadata["dataset"]["sha256"] = "0" * 64
            _write_metadata(root, metadata)
        elif target == "inputs":
            X_path = root / "inputs" / "X.npy"
            tampered = np.load(X_path)
            tampered[0, 0] += 1.0
            np.save(X_path, tampered)
        elif target == "folds":
            folds_metadata["fold_0"]["n_train"] = (
                int(folds_metadata["fold_0"]["n_train"]) + 1
            )
            _write_folds(root, folds_metadata)
        else:
            folds_metadata["holdout"]["bounds"]["start"] = str(
                np.datetime64(folds_metadata["holdout"]["bounds"]["start"])
                + np.timedelta64(1, "D")
            )
            _write_folds(root, folds_metadata)

        trainer = _RecordingTrainer()
        if config_patch is not None:
            with config_patch:
                with pytest.raises(holdout.HoldoutLockError, match=field):
                    _evaluate(root, trainer)
        else:
            with pytest.raises(holdout.HoldoutLockError, match=field):
                _evaluate(root, trainer)
        assert trainer.calls == []


def test_force_holdout_appends_an_override_and_requires_a_reason() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        _evaluate(root)
        report_path = root / "results" / holdout.REPORT_FILENAME
        before = report_path.read_bytes()

        with pytest.raises(ValueError, match="reason"):
            _evaluate(root, force_reason="   ")

        overrides_path = root / "results" / holdout.OVERRIDES_FILENAME
        overrides_path.parent.mkdir(parents=True, exist_ok=True)
        overrides_path.write_text('{"torn":', encoding="utf-8")

        trainer = _RecordingTrainer()
        forced = _evaluate(root, trainer, force_reason="re-run after review")
        assert forced["status"] == "overridden"
        assert len(trainer.calls) == 1

        lines = [
            json.loads(line)
            for line in overrides_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 1
        assert lines[0]["reason"] == "re-run after review"
        assert lines[0]["previous_lock_sha256"] == forced["lock"]["sha256"]

        manifest = _manifest(root)
        assert manifest["holdout"]["override"]["reason"] == "re-run after review"
        assert manifest["holdout"]["status"] == "overridden"
        # A forced re-evaluation is still a full report rewrite, not a silent
        # second score: the report records the override and fresh hashes.
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["overrides"][0]["reason"] == "re-run after review"
        assert "override" in report["headline"]["caveats"]["holdout"]
        assert "exactly once" not in report["headline"]["caveats"]["holdout"]
        assert report_path.read_bytes() != before
        sums = _parse_sums(root / "results" / holdout.SUMS_FILENAME)
        assert sums[holdout.REPORT_FILENAME] == _sha256(report_path)
        for name, digest in sums.items():
            assert _sha256(root / "results" / name) == digest, name


def test_no_tuning_trial_view_touches_the_declared_holdout() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _synthetic_root(tmpdir)
        X, y, metadata, folds_metadata = _load(root)
        holdout_rows = set(
            harness.holdout_row_indices(metadata["date"], folds_metadata).tolist()
        )
        assert holdout_rows

        seen: list[np.ndarray] = []

        def recording_trainer(X_train, y_train, X_val=None, y_val=None, params=None):
            seen.append(np.asarray(X_train)[:, 0].astype(np.int64))
            seen.append(np.asarray(X_val)[:, 0].astype(np.int64))
            return _RowModel(), {}

        output_dir = Path(tmpdir) / "tuning"
        result = harness.run_tuning_campaign(
            X=X,
            y=y,
            dates=metadata["date"],
            folds_metadata=folds_metadata,
            output_dir=output_dir,
            search_space=TINY_SEARCH_SPACE,
            trainer=recording_trainer,
        )

        assert seen
        for rows in seen:
            assert not set(rows.tolist()) & holdout_rows

        # The persisted per-trial fingerprints match the locked accessor's
        # views, so the recorded evidence covers exactly the rows above.
        expected = {
            fold_key: harness.build_locked_fold_views(
                fold_key,
                X=X,
                y=y,
                dates=metadata["date"],
                folds_metadata=folds_metadata,
            )
            for fold_key in training.fold_keys(folds_metadata)
        }
        sidecars = sorted(result["run_dir"].glob("trial_*_sidecar.json"))
        assert len(sidecars) == 3
        for path in sidecars:
            document = json.loads(path.read_text(encoding="utf-8"))
            for fold_key, record in document["folds"].items():
                assert (
                    record["fit_fingerprint"]
                    == expected[fold_key]["fit"]["fingerprint"]
                )
                assert (
                    record["validation_fingerprint"]
                    == expected[fold_key]["validation"]["fingerprint"]
                )
                assert not set(
                    expected[fold_key]["fit"]["indices"].tolist()
                ) & holdout_rows
                assert not set(
                    expected[fold_key]["validation"]["indices"].tolist()
                ) & holdout_rows

        # Tuning never creates the holdout lock or the final report.
        assert not list(output_dir.rglob(holdout.LOCK_FILENAME))
        assert not list(output_dir.rglob(holdout.REPORT_FILENAME))


def test_cli_exposes_the_holdout_entry_and_override_flags() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "packages.ml.model.main", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--evaluate-holdout" in result.stdout, result.stdout
    assert "--force-holdout" in result.stdout, result.stdout
    assert "REASON" in result.stdout, result.stdout

    planned = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "packages.ml.model.main",
            "--dry-run",
            "--evaluate-holdout",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert planned.returncode == 0, planned.stderr
    assert "holdout" in planned.stdout.lower(), planned.stdout

    default_plan = subprocess.run(
        [sys.executable, "-B", "-m", "packages.ml.model.main", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert default_plan.returncode == 0, default_plan.stderr
    assert "holdout" not in default_plan.stdout.lower(), default_plan.stdout


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            if inspect.signature(_fn).parameters:
                print(f"SKIP {_name}: pytest parametrization only")
                skipped += 1
                continue
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
