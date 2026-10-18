"""Tests for the regime metadata and versioned fold-payload contract.

Phase 2 must write the calendar ``market_regime`` labels into the extracted
metadata so the Phase 3 consumer can report per-fold regime distributions.
Phase 3 persists a schema-versioned superset: the schema version, the pinned
feature names, the reserved final-500-date holdout window, the remainder date
count, and the five fold records with typed gap keys. The synthetic checks
build a temporary dataset and artifact tree, so they neither read nor write the
live generated data; the non-regression check compares the extraction X/Y and
the freshly computed fold payload against the regenerated stored artifacts
behind the existing skip guard.

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
# 500 reserved holdout dates plus a 100-date fold remainder that is exactly
# divisible by the six blocks (edges 0, 16, 33, 50, 66, 83, 100).
N_DATES = 600
HOLDOUT_N_DATES = 500
TICKERS = ("AAA", "BBB", "CCC")

FOLD_RECORD_KEYS = frozenset(
    {
        "train_indices",
        "test_indices",
        "train_date_min",
        "train_date_max",
        "test_date_min",
        "test_date_max",
        "n_train",
        "n_test",
        "first_test_idx",
        "last_train_idx",
        "purge_train_count",
        "regime_distribution_train",
        "regime_distribution_test",
        "n_dates",
        "purge_horizon_days",
        "separation_units",
    }
)


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


def _write_synthetic_inputs(root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Write a minimal Phase 2 input set to a temporary artifact root."""
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
    return X, Y, metadata


def _run_cv(root: Path) -> dict:
    """Run Phase 3 against a temporary artifact root."""
    with mock.patch.object(cross_validator, "ARTIFACTS_DIR", root):
        return cross_validator.phase_3_cross_validation()


def _date_string(value: np.datetime64) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


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


def test_persisted_payload_reserves_the_final_holdout_and_carries_fold_schema() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        X, Y, metadata = _write_synthetic_inputs(root)
        folds = _run_cv(root)

        persisted = joblib.load(root / "folds" / "folds_metadata.pkl")
        unique_dates = np.sort(np.unique(metadata["date"]))
        assert cross_validator.HOLDOUT_N_DATES == HOLDOUT_N_DATES == 500
        fold_dates = unique_dates[:-HOLDOUT_N_DATES]
        holdout_dates = unique_dates[-HOLDOUT_N_DATES:]
        fold_n_dates = len(fold_dates)

        fold_keys = {f"fold_{idx}" for idx in range(cross_validator.N_FOLDS)}
        assert set(folds) == fold_keys
        assert set(persisted) == fold_keys | {
            "schema_version",
            "feature_names",
            "holdout",
            "n_dates",
        }
        assert len(holdout_dates) == HOLDOUT_N_DATES == 500
        assert persisted["schema_version"] == 2
        assert persisted["feature_names"] == list(feature_engineering.INDICATORS)
        assert persisted["holdout"] == {
            "bounds": {
                "start": _date_string(holdout_dates[0]),
                "end": _date_string(holdout_dates[-1]),
            },
            "n_dates": HOLDOUT_N_DATES,
        }
        assert persisted["n_dates"] == fold_n_dates == len(unique_dates) - HOLDOUT_N_DATES
        assert fold_n_dates > 0

        # Recompute the exact split from the persisted remainder and compare
        # index-for-index; the fold construction is reproducible from the schema.
        tpos = np.searchsorted(fold_dates, metadata["date"])
        boundaries = cross_validator._fold_boundaries(persisted["n_dates"])

        for fold_idx in range(cross_validator.N_FOLDS):
            fold_key = f"fold_{fold_idx}"
            record = persisted[fold_key]
            assert set(record) == FOLD_RECORD_KEYS, fold_key
            assert "gap_days" not in record, fold_key
            assert "purge_test_count" not in record, fold_key
            assert record["n_dates"] == fold_n_dates, fold_key
            assert record["purge_horizon_days"] == 5, fold_key
            assert record["separation_units"] == 6, fold_key
            assert record["n_train"] == len(record["train_indices"]) > 0, fold_key
            assert record["n_test"] == len(record["test_indices"]) > 0, fold_key
            assert (
                sum(record["regime_distribution_train"].values())
                == record["n_train"]
            ), fold_key
            assert (
                sum(record["regime_distribution_test"].values()) == record["n_test"]
            ), fold_key
            assert record["first_test_idx"] - record["last_train_idx"] == 6, fold_key

            # No fold row may fall inside the reserved window.
            train_dates = metadata["date"][np.asarray(record["train_indices"])]
            test_dates = metadata["date"][np.asarray(record["test_indices"])]
            assert (train_dates < holdout_dates[0]).all(), fold_key
            assert (test_dates < holdout_dates[0]).all(), fold_key

            expected_train, expected_test, expected_purge = (
                cross_validator._split_date_blocks(
                    tpos,
                    int(boundaries[fold_idx + 1]),
                    int(boundaries[fold_idx + 2]),
                )
            )
            assert record["train_indices"] == expected_train.tolist(), fold_key
            assert record["test_indices"] == expected_test.tolist(), fold_key
            assert record["purge_train_count"] == expected_purge, fold_key

        # Fold construction reads the inputs but never rewrites X/Y.
        assert np.array_equal(np.load(root / "inputs" / "X.npy"), X)
        assert np.array_equal(np.load(root / "inputs" / "Y.npy"), Y)


def test_training_guard_counts_fold_keys_and_accepts_schema_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        X, Y, metadata = _write_synthetic_inputs(root)
        _run_cv(root)
        persisted = joblib.load(root / "folds" / "folds_metadata.pkl")

    # Metadata keys must not be mistaken for folds.
    training._validate_training_prerequisites(X, Y, metadata, persisted)

    # A wrong fold count still fails loudly.
    reduced = dict(persisted)
    del reduced["fold_4"]
    with pytest.raises(ValueError, match="Expected 5 folds, got 4"):
        training._validate_training_prerequisites(X, Y, metadata, reduced)


def test_indicator_contract_count_unchanged() -> None:
    assert len(feature_engineering.INDICATORS) == 37


def test_live_extraction_and_fold_contract_match_stored_artifacts() -> None:
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
            f"{name} drifted from the stored input snapshot"
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

        fresh_payload = joblib.load(root / "folds" / "folds_metadata.pkl")

    fold_keys = {f"fold_{idx}" for idx in range(cross_validator.N_FOLDS)}
    assert set(folds) == fold_keys
    assert set(fresh_payload) == fold_keys | {
        "schema_version",
        "feature_names",
        "holdout",
        "n_dates",
    }

    unique_dates = np.sort(np.unique(live_metadata["date"]))
    holdout_dates = unique_dates[-cross_validator.HOLDOUT_N_DATES :]
    fold_n_dates = len(unique_dates) - cross_validator.HOLDOUT_N_DATES
    assert fresh_payload["schema_version"] == 2
    assert fresh_payload["feature_names"] == list(feature_engineering.INDICATORS)
    assert fresh_payload["n_dates"] == fold_n_dates
    assert fresh_payload["holdout"]["n_dates"] == cross_validator.HOLDOUT_N_DATES
    assert fresh_payload["holdout"]["bounds"] == {
        "start": _date_string(holdout_dates[0]),
        "end": _date_string(holdout_dates[-1]),
    }

    # Every freshly built fold is held out from the reserved window.
    for fold_key in fold_keys:
        record = fresh_payload[fold_key]
        assert set(record) == FOLD_RECORD_KEYS, fold_key
        assert record["n_dates"] == fold_n_dates, fold_key
        assert "gap_days" not in record, fold_key
        assert "purge_test_count" not in record, fold_key
        for index_field in ("train_indices", "test_indices"):
            fold_dates_seen = live_metadata["date"][
                np.asarray(record[index_field])
            ]
            assert (fold_dates_seen < holdout_dates[0]).all(), fold_key

    # The regenerated stored payload matches the freshly computed one field for
    # field, including both membership sides.
    assert set(live_folds) == set(fresh_payload) == fold_keys | {
        "schema_version",
        "feature_names",
        "holdout",
        "n_dates",
    }
    for key in ("schema_version", "feature_names", "holdout", "n_dates"):
        assert live_folds[key] == fresh_payload[key], key
    for fold_key in fold_keys:
        stored_record = live_folds[fold_key]
        assert set(stored_record) == FOLD_RECORD_KEYS, fold_key
        for field in FOLD_RECORD_KEYS:
            assert stored_record[field] == fresh_payload[fold_key][field], (
                fold_key,
                field,
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
