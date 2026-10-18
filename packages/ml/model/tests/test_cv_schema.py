"""Tests for the regime metadata contract between extraction and cross-validation.

Phase 2 must write the calendar ``market_regime`` labels into the extracted
metadata so the Phase 3 consumer can report per-fold regime distributions. The
synthetic checks build a temporary dataset and artifact tree, so they neither
read nor write the live generated data; the non-regression check compares the
extraction X/Y and the fold payload's key/type contract against the live
branch-point artifacts behind the existing skip guard (the stored fold
membership is interim until the later integration re-run, so it is not compared
index-for-index).

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_cv_schema.py
"""

from __future__ import annotations

import contextlib
import hashlib
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR, DATASET_PATH, requires_live_artifacts
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.features import feature_engineering
from packages.ml.model.training import training

N_REGIME_BUCKETS = 7
N_DATES = 70
TICKERS = ("AAA", "BBB", "CCC")


def _synthetic_frame() -> pd.DataFrame:
    """A small extraction input frame with unique (ticker, date) pairs."""
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


def _array_digest(array: np.ndarray) -> str:
    """Streaming sha256 over an array's bytes, without materializing a copy."""
    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def test_extraction_metadata_carries_row_aligned_regime() -> None:
    frame = _synthetic_frame()
    with mock.patch.object(feature_engineering, "_dataset_pin", _without_dataset_pin):
        _, Y, metadata = feature_engineering._extract_inputs(frame)

    assert "regime" in metadata, "extraction metadata carries no 'regime' key"
    regime = np.asarray(metadata["regime"])
    assert regime.ndim == 1
    assert len(regime) == len(metadata["date"]) == len(metadata["ticker"]) == len(Y)
    assert np.array_equal(metadata["date"], frame["date"].to_numpy(dtype="datetime64[ns]"))
    assert np.array_equal(metadata["ticker"], frame["ticker"].to_numpy())
    assert np.array_equal(regime, frame["market_regime"].to_numpy())
    assert np.issubdtype(regime.dtype, np.integer)
    assert set(np.unique(regime)) == set(range(N_REGIME_BUCKETS))


def test_cv_reports_per_fold_regime_distributions_from_extracted_metadata() -> None:
    frame = _synthetic_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        dataset_path = root / "dataset.parquet"
        frame.to_parquet(dataset_path, index=False)
        artifacts = root / "artifacts"

        with mock.patch.object(feature_engineering, "DATASET_PATH", dataset_path):
            with mock.patch.object(feature_engineering, "ARTIFACTS_DIR", artifacts):
                X, Y, metadata, _ = feature_engineering.phase_2_feature_selection()

        persisted = joblib.load(artifacts / "inputs" / "metadata.pkl")

        with mock.patch.object(cross_validator, "ARTIFACTS_DIR", artifacts):
            folds = cross_validator.phase_3_cross_validation()

        assert (artifacts / "folds" / "folds_metadata.pkl").is_file()
        assert set(folds) == {f"fold_{fold_idx}" for fold_idx in range(5)}
        for fold_data in folds.values():
            train_distribution = fold_data["regime_distribution_train"]
            test_distribution = fold_data["regime_distribution_test"]
            assert set(train_distribution) == set(cross_validator.REGIME_NAMES.values())
            assert set(test_distribution) == set(cross_validator.REGIME_NAMES.values())
            assert sum(train_distribution.values()) == fold_data["n_train"] > 0
            assert sum(test_distribution.values()) == fold_data["n_test"] > 0

        assert "regime" in persisted
        assert np.array_equal(persisted["regime"], metadata["regime"])

        # The prerequisite guard from the training phase accepts this freshly
        # extracted metadata; it stays fail-loud for malformed regime arrays.
        training._validate_training_prerequisites(X, Y, metadata, folds)


def test_indicator_contract_count_unchanged() -> None:
    assert len(feature_engineering.INDICATORS) == 37


def test_live_extraction_and_fold_contract_match_branch_point_artifacts() -> None:
    requires_live_artifacts(ARTIFACTS_DIR)
    if not DATASET_PATH.is_file():
        pytest.skip(
            "requires the generated dataset absent from this checkout: "
            f"{DATASET_PATH}"
        )

    frame = pd.read_parquet(DATASET_PATH)
    with mock.patch.object(feature_engineering, "_dataset_pin", _without_dataset_pin):
        X, Y, _ = feature_engineering._extract_inputs(frame)

    for name, array in (("X", X), ("Y", Y)):
        snapshot = np.load(ARTIFACTS_DIR / "inputs" / f"{name}.npy", mmap_mode="r")
        assert array.shape == snapshot.shape
        assert array.dtype == snapshot.dtype
        assert _array_digest(array) == _array_digest(snapshot), (
            f"{name} drifted from the branch-point snapshot"
        )

    live_metadata = joblib.load(ARTIFACTS_DIR / "inputs" / "metadata.pkl")
    live_folds = joblib.load(ARTIFACTS_DIR / "folds" / "folds_metadata.pkl")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        inputs = root / "inputs"
        inputs.mkdir()
        for name in ("X", "Y"):
            (inputs / f"{name}.npy").symlink_to(
                (ARTIFACTS_DIR / "inputs" / f"{name}.npy").resolve()
            )
        joblib.dump(live_metadata, inputs / "metadata.pkl")

        with mock.patch.object(cross_validator, "ARTIFACTS_DIR", root):
            folds = cross_validator.phase_3_cross_validation()

    assert set(folds) == set(live_folds)

    # The stored test-side membership is interim: the test blocks predate the
    # train-only purge and are regenerated at the later integration re-run, so
    # only the fields the purge change leaves untouched are compared by value.
    # The membership contract is asserted from in-memory recomputation in
    # tests/test_purge.py.
    canonical_keys = {
        "train_indices",
        "test_indices",
        "train_date_min",
        "train_date_max",
        "test_date_min",
        "test_date_max",
        "gap_days",
        "n_train",
        "n_test",
        "first_test_idx",
        "last_train_idx",
        "purge_train_count",
    }
    stable_keys = {
        "train_indices",
        "train_date_min",
        "train_date_max",
        "gap_days",
        "n_train",
        "last_train_idx",
        "purge_train_count",
    }
    for fold_key, expected in live_folds.items():
        fresh = folds[fold_key]
        assert canonical_keys <= set(fresh), fold_key
        assert canonical_keys <= set(expected), fold_key
        assert "purge_test_count" not in fresh, fold_key
        for field in stable_keys:
            assert fresh[field] == expected[field], f"{fold_key} {field} drifted"
        for field in canonical_keys:
            assert type(fresh[field]) is type(expected[field]), (
                f"{fold_key} {field} type drifted"
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
