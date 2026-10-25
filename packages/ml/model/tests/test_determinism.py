"""Tests for the determinism pack: policy, environment capture, and equality.

The checks pin the declared policy (single-threaded training, recorded
``OMP_NUM_THREADS``, the existing seed), the per-run environment manifest
(versions, ``PYTHONHASHSEED``, thread settings, git sha, lock hash), the
byte-equality of two tiny fits with the same seed, the hash-equality of two
in-memory fold constructions from the same metadata, and a seed sweep that
writes per-seed/per-fold validation results with the policy that produced
them. Every fit and every write uses a temporary directory; no live
``artifacts/**`` or ``data/model_outputs/**`` file is touched.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_determinism.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import tempfile
import traceback
from pathlib import Path

import joblib
import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model import config as model_config
from packages.ml.model.training import model_trainer

# Tiny synthetic calendar: 100 reserved holdout dates plus a 120-date fold
# remainder (six 20-date blocks). Three tickers keep every fold's inner
# validation tail and fit set non-empty.
TICKERS = ("AAA", "BBB", "CCC")
N_DATES = 220
HOLDOUT_DATES = 100
N_FEATURES = 4
SWEEP_SEEDS = (42, 43, 44)

# The tiny fit: the frozen recipe's shape with a handful of boosting rounds.
TINY_TREES = 10
SWEEP_TREES = 8


def _synthetic_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-aligned X, y, dates, and regime for the synthetic fold calendar."""
    rng = np.random.default_rng(20261019)
    dates = np.repeat(
        np.datetime64("2000-01-03") + np.arange(N_DATES), len(TICKERS)
    )
    n_rows = dates.size
    X = rng.normal(size=(n_rows, N_FEATURES))
    X[:, 0] = np.arange(n_rows, dtype=np.float64)
    y = rng.uniform(0.05, 0.6, size=n_rows)
    regime = (np.arange(n_rows) % 7).astype(np.int8)
    return X, y, dates, regime


def _tiny_fit_params(seed: int = 42) -> dict:
    """The frozen recipe with tiny round count and the pinned thread settings."""
    params = model_trainer.default_params()
    params.update(
        {
            "n_estimators": TINY_TREES,
            "max_depth": 3,
            "n_jobs": 1,
            "random_state": seed,
        }
    )
    return params


def _model_bytes(model: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def test_config_declares_the_determinism_policy() -> None:
    """The shipped config pins the thread settings and keeps the baseline seed."""
    cfg = model_config.load_config()
    declared = cfg["training"]

    assert declared["seed"] == 42
    assert declared["n_jobs"] == 1
    assert declared["omp_num_threads"] == 1
    assert declared["determinism_policy"] == "bit-reproducible"
    # The frozen selection keeps its historical all-cores value; the policy
    # overrides the effective recipe rather than rewriting the selection.
    assert declared["hyperparameters"]["n_jobs"] == -1
    assert model_trainer.default_params()["n_jobs"] == 1


def test_run_record_captures_environment_manifest(monkeypatch) -> None:
    """A run record carries versions, hash seed, threads, git sha, lock hash."""
    from packages.ml.model import run_log

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    loaded = model_config.get_config()
    declared = loaded["training"]

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        run = run_log.new_run("determinism", root)
        environment = run["environment"]

        # Versions match the process environment.
        assert environment["python"] == platform.python_version()
        assert environment["numpy"] == importlib.metadata.version("numpy")

        # Values match the process environment after the policy is applied.
        assert environment["PYTHONHASHSEED"] == os.environ.get("PYTHONHASHSEED")
        assert environment["threads"] == {
            "policy": "bit-reproducible",
            "n_jobs": declared["n_jobs"],
            "omp_num_threads": str(declared["omp_num_threads"]),
        }
        assert os.environ["OMP_NUM_THREADS"] == str(declared["omp_num_threads"])
        assert environment["seed"] == declared["seed"]
        assert (
            environment["random_state"]
            == declared["hyperparameters"]["random_state"]
        )

        # Provenance hashes: the git sha is the record's top-level sha and the
        # lock hash is the bytes of the repository's uv.lock.
        assert environment["git_sha"] == run["git_sha"]
        assert environment["git_sha"] is not None
        lock_path = REPO_ROOT / "uv.lock"
        assert environment["lock_hash"] == hashlib.sha256(
            lock_path.read_bytes()
        ).hexdigest()

        # Persisted in both forms of the run record: the manifest and the
        # append-only JSONL line.
        run_log.finish_run(run)
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        line = json.loads(
            (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert manifest == line
        assert manifest["environment"] == environment


def test_two_tiny_fits_with_the_same_seed_are_byte_identical() -> None:
    """Same seed/params/threads -> identical model bytes and predictions."""
    X, y, _dates, _regime = _synthetic_inputs()
    params = _tiny_fit_params(seed=42)

    first, _ = model_trainer.train_xgboost_model(X, y, params=params)
    second, _ = model_trainer.train_xgboost_model(X, y, params=params)

    first_bytes = _model_bytes(first)
    second_bytes = _model_bytes(second)
    assert first_bytes == second_bytes, "two identical fits produced different bytes"
    assert np.array_equal(first.predict(X), second.predict(X))
    assert first.get_params()["random_state"] == second.get_params()["random_state"] == 42


def test_two_fold_manifest_generations_are_hash_equal() -> None:
    """The in-memory fold construction is deterministic and content-sensitive."""
    from packages.ml.model.cross_validation import cross_validator

    _X, _y, dates, regime = _synthetic_inputs()
    first = cross_validator.build_fold_payload(
        dates, regime, holdout_n_dates=HOLDOUT_DATES
    )
    second = cross_validator.build_fold_payload(
        dates, regime, holdout_n_dates=HOLDOUT_DATES
    )

    first_hash = cross_validator.fold_manifest_hash(first)
    assert first_hash == cross_validator.fold_manifest_hash(second)
    assert len(first_hash) == 64
    assert set(first) == {
        "schema_version",
        "feature_names",
        "holdout",
        "n_dates",
        *{f"fold_{fold_idx}" for fold_idx in range(cross_validator.N_FOLDS)},
    }

    # The hash is not a constant: changing one fold's membership changes it.
    changed = json.loads(json.dumps(first))
    changed["fold_0"]["train_indices"].append(changed["fold_0"]["train_indices"][-1])
    assert cross_validator.fold_manifest_hash(changed) != first_hash


def test_seed_sweep_writes_at_least_three_seeds_with_per_fold_results(
    tmp_path: Path,
) -> None:
    """The sweep entry trains seeds x folds and writes seed_sweep.json."""
    from packages.ml.model import determinism
    from packages.ml.model.cross_validation import cross_validator

    X, y, dates, regime = _synthetic_inputs()
    folds_metadata = cross_validator.build_fold_payload(
        dates, regime, holdout_n_dates=HOLDOUT_DATES
    )
    output_path = tmp_path / "seed_sweep.json"

    result = determinism.run_seed_sweep(
        X=X,
        y=y,
        dates=dates,
        folds_metadata=folds_metadata,
        output_path=output_path,
        seeds=SWEEP_SEEDS,
        n_estimators=SWEEP_TREES,
    )

    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == document
    assert output_path.name == determinism.SEED_SWEEP_FILENAME

    # The policy that produced the sweep is persisted with it.
    policy = document["policy"]
    declared = model_config.get_config()["training"]
    assert policy["policy"] == "bit-reproducible"
    assert policy["n_jobs"] == declared["n_jobs"] == 1
    assert policy["omp_num_threads"] == str(declared["omp_num_threads"])
    assert policy["PYTHONHASHSEED"] == os.environ.get("PYTHONHASHSEED")
    assert policy["seeds"] == list(SWEEP_SEEDS)
    assert document["environment"]["lock_hash"] is not None
    assert document["fold_manifest_hash"] == cross_validator.fold_manifest_hash(
        folds_metadata
    )

    # Per-seed/per-fold results, one record for each seed, no seed reported
    # under another seed's identity.
    fold_keys = {f"fold_{fold_idx}" for fold_idx in range(cross_validator.N_FOLDS)}
    assert [record["seed"] for record in document["seeds"]] == list(SWEEP_SEEDS)
    for record in document["seeds"]:
        assert record["params"]["random_state"] == record["seed"]
        assert set(record["folds"]) == fold_keys
        for fold in record["folds"].values():
            assert np.isfinite(fold["validation_qlike"])
            assert len(fold["predictions_sha256"]) == 64
            assert len(fold["model_sha256"]) == 64
        assert record["mean_validation_qlike"] == pytest.approx(
            float(
                np.mean(
                    [fold["validation_qlike"] for fold in record["folds"].values()]
                )
            )
        )

    # Distinct seeds produce distinct recipes and distinct forecasts on at
    # least one fold; no pair can be mistaken for the same run.
    params_hashes = [record["params_sha256"] for record in document["seeds"]]
    assert len(set(params_hashes)) == len(SWEEP_SEEDS)
    for left in range(len(document["seeds"])):
        for right in range(left + 1, len(document["seeds"])):
            left_folds = document["seeds"][left]["folds"]
            right_folds = document["seeds"][right]["folds"]
            assert any(
                left_folds[fold_key]["predictions_sha256"]
                != right_folds[fold_key]["predictions_sha256"]
                for fold_key in fold_keys
            ), "two seeds produced identical forecasts on every fold"


if __name__ == "__main__":
    import inspect

    failures = 0
    monkeypatch = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmpdir:
        fixtures = {"tmp_path": Path(tmpdir), "monkeypatch": monkeypatch}
        try:
            for _name, _fn in list(globals().items()):
                if _name.startswith("test_") and callable(_fn):
                    kwargs = {
                        parameter: fixtures[parameter]
                        for parameter in inspect.signature(_fn).parameters
                    }
                    try:
                        _fn(**kwargs)
                        print(f"PASS {_name}")
                    except Exception:
                        failures += 1
                        print(f"FAIL {_name}")
                        traceback.print_exc()
        finally:
            monkeypatch.undo()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All tests passed")
