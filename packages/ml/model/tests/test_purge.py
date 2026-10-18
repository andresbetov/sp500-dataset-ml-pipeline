"""Tests for the canonical train-only purge of the cross-validation split.

The split keeps every test block whole and purges only training rows whose
forward label window reaches into the test block: the last kept training date
sits five trading dates before the block start, the first kept test date is the
block start, and exactly the five dates in between are excluded. Every check
runs against a synthetic business-day calendar and temporary artifact roots, so
the live inputs and the stored fold artifacts are neither read nor written.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_purge.py
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.training import training

# Exactly divisible by the N_FOLDS + 1 = 6 date blocks, so every block boundary
# is an exact date index.
N_DATES = 60
TICKERS = ("AAA", "BBB", "CCC")
EXCLUDED_DATES = 5
FOLD_PAYLOAD_KEYS = frozenset(
    {
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
)


def _synthetic_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows over a synthetic trading calendar and their unique-date positions."""
    dates = np.repeat(
        pd.bdate_range("2000-01-03", periods=N_DATES).values, len(TICKERS)
    )
    unique_dates = np.sort(np.unique(dates))
    tpos = np.searchsorted(unique_dates, dates)
    return dates, unique_dates, tpos


def test_every_fold_keeps_the_test_block_whole_with_five_excluded_dates() -> None:
    """All folds: kept train ends at B-6, kept test starts at B, five dates between."""
    _, unique_dates, tpos = _synthetic_inputs()
    boundaries = cross_validator._fold_boundaries(len(unique_dates))

    observed = []
    for fold_idx in range(cross_validator.N_FOLDS):
        block_start = int(boundaries[fold_idx + 1])
        block_end = int(boundaries[fold_idx + 2])

        split_result = cross_validator._split_date_blocks(tpos, block_start, block_end)
        train_indices, test_indices, purge_train_count = split_result[:3]
        train_tpos = tpos[train_indices]
        test_tpos = tpos[test_indices]
        last_kept_train = int(train_tpos.max())
        first_kept_test = int(test_tpos.min())
        separation = first_kept_test - last_kept_train
        observed.append(
            (
                fold_idx,
                block_start,
                last_kept_train,
                first_kept_test,
                separation,
                purge_train_count,
            )
        )

        # Training keeps every date through B-6; the test set is the whole raw
        # block [B, block_end).
        assert np.array_equal(
            np.unique(train_tpos), np.arange(0, block_start - EXCLUDED_DATES)
        ), observed[-1]
        assert np.array_equal(
            test_indices,
            np.flatnonzero((tpos >= block_start) & (tpos < block_end)),
        ), observed[-1]
        assert np.array_equal(
            np.unique(test_tpos), np.arange(block_start, block_end)
        ), observed[-1]

        # Exactly the trading dates B-5 .. B-1 are excluded between the kept sets.
        kept_tpos = np.concatenate((train_tpos, test_tpos))
        for excluded_tpos in range(block_start - EXCLUDED_DATES, block_start):
            assert excluded_tpos not in kept_tpos, observed[-1]
        between_dates = unique_dates[last_kept_train + 1 : first_kept_test]
        assert np.array_equal(
            between_dates, unique_dates[block_start - EXCLUDED_DATES : block_start]
        ), observed[-1]
        assert len(between_dates) == EXCLUDED_DATES, observed[-1]

        # The train-side purge width is the branch-point value: the five dates
        # before the block start, one row per ticker per date.
        assert purge_train_count == EXCLUDED_DATES * len(TICKERS), observed[-1]

        # The contract returns train, test, and the train purge count only.
        assert len(split_result) == 3, observed[-1]

    expected = [
        (
            fold_idx,
            int(boundaries[fold_idx + 1]),
            int(boundaries[fold_idx + 1]) - 6,
            int(boundaries[fold_idx + 1]),
            6,
            EXCLUDED_DATES * len(TICKERS),
        )
        for fold_idx in range(cross_validator.N_FOLDS)
    ]
    assert observed == expected


def test_separation_predicate_uses_the_configured_minimum_and_train_purge() -> None:
    """The guard checks the configured minimum and the positive train purge."""
    train_tpos = np.array([0, 1, 2, 3, 4])
    test_tpos = np.array([10, 11])

    assert cross_validator.SEPARATION_MIN == 6
    assert cross_validator._verify_fold_separation(0, train_tpos, test_tpos, 1) == 6

    with pytest.raises(ValueError, match="embargo"):
        cross_validator._verify_fold_separation(0, train_tpos, np.array([9, 10]), 1)

    with pytest.raises(ValueError, match="purge"):
        cross_validator._verify_fold_separation(0, train_tpos, test_tpos, 0)


def test_fold_structure_and_persisted_payload_without_the_test_purge_key() -> None:
    """Blocks partition the remainder, windows expand, and the payload loses one key."""
    dates, unique_dates, tpos = _synthetic_inputs()
    n_rows = len(dates)
    metadata = {
        "date": dates,
        "ticker": np.tile(np.array(TICKERS), N_DATES),
        "regime": (np.arange(n_rows) % len(cross_validator.REGIME_NAMES)).astype(
            np.int8
        ),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # The exercise must never write into the repository's live artifact tree.
        assert not root.is_relative_to(REPO_ROOT)
        inputs = root / "inputs"
        inputs.mkdir()
        np.save(inputs / "X.npy", np.zeros((n_rows, 1)))
        np.save(inputs / "Y.npy", np.full(n_rows, 0.25))
        joblib.dump(metadata, inputs / "metadata.pkl")

        with mock.patch.object(cross_validator, "ARTIFACTS_DIR", root):
            folds = cross_validator.phase_3_cross_validation()

        persisted_path = root / "folds" / "folds_metadata.pkl"
        assert persisted_path.is_file()
        persisted = joblib.load(persisted_path)

    assert set(folds) == {f"fold_{idx}" for idx in range(cross_validator.N_FOLDS)}
    assert set(persisted) == set(folds)

    boundaries = cross_validator._fold_boundaries(len(unique_dates))
    test_blocks = []
    previous_kept_dates = None
    previous_train_n = None
    for fold_idx in range(cross_validator.N_FOLDS):
        fold = folds[f"fold_{fold_idx}"]
        stored = persisted[f"fold_{fold_idx}"]
        block_start = int(boundaries[fold_idx + 1])
        block_end = int(boundaries[fold_idx + 2])

        # The persisted payload keeps every key except the removed test purge.
        assert set(stored) == FOLD_PAYLOAD_KEYS, fold_idx
        assert FOLD_PAYLOAD_KEYS <= set(fold), fold_idx
        assert "purge_test_count" not in fold, fold_idx
        assert "purge_test_count" not in stored, fold_idx
        assert stored["gap_days"] == cross_validator.CV_EMBARGO_DAYS, fold_idx
        assert stored["n_train"] == len(stored["train_indices"]), fold_idx
        assert stored["n_test"] == len(stored["test_indices"]), fold_idx

        stored_train_tpos = tpos[np.asarray(stored["train_indices"])]
        stored_test_tpos = tpos[np.asarray(stored["test_indices"])]
        assert np.intersect1d(stored_train_tpos, stored_test_tpos).size == 0, fold_idx

        # Test blocks are kept whole and partition the remainder after the
        # train-only first block.
        assert np.array_equal(
            np.unique(stored_test_tpos), np.arange(block_start, block_end)
        ), fold_idx
        test_blocks.append(set(range(block_start, block_end)))

        # Training windows expand: kept dates strictly grow fold over fold.
        kept_dates = np.unique(stored_train_tpos)
        if previous_kept_dates is not None:
            assert stored["n_train"] > previous_train_n, fold_idx
            assert set(kept_dates) > set(previous_kept_dates), fold_idx
        previous_kept_dates = kept_dates
        previous_train_n = stored["n_train"]

        # Stored bounds echo the canonical one-sided purge.
        assert stored["first_test_idx"] == block_start, fold_idx
        assert stored["last_train_idx"] == block_start - 6, fold_idx

    remainder = set(range(int(boundaries[1]), len(unique_dates)))
    union = set()
    for block in test_blocks:
        union |= block
    assert union == remainder
    assert sum(len(block) for block in test_blocks) == len(union)


def test_index_gap_equals_trading_day_distance_on_the_synthetic_calendar() -> None:
    """The index-space gap is the trading-day distance under the calendar map."""
    _, unique_dates, _ = _synthetic_inputs()
    assert np.array_equal(
        unique_dates,
        pd.bdate_range(unique_dates[0], periods=len(unique_dates)).values,
    )

    boundaries = cross_validator._fold_boundaries(len(unique_dates))
    for fold_idx in range(cross_validator.N_FOLDS):
        block_start = int(boundaries[fold_idx + 1])
        last_kept_train_date = unique_dates[block_start - 6]
        first_kept_test_date = unique_dates[block_start]

        # One unique date per business day: six index steps == six trading days.
        assert (
            np.busday_count(
                last_kept_train_date.astype("datetime64[D]"),
                first_kept_test_date.astype("datetime64[D]"),
            )
            == 6
        )
        between = unique_dates[
            (unique_dates > last_kept_train_date) & (unique_dates < first_kept_test_date)
        ]
        assert len(between) == EXCLUDED_DATES


def test_train_side_purge_count_is_unchanged() -> None:
    """The one-sided change leaves the training-side purge width untouched."""
    _, _, tpos = _synthetic_inputs()
    boundaries = cross_validator._fold_boundaries(N_DATES)

    counts = []
    for fold_idx in range(cross_validator.N_FOLDS):
        block_start = int(boundaries[fold_idx + 1])
        block_end = int(boundaries[fold_idx + 2])
        _, _, purge_train_count = cross_validator._split_date_blocks(
            tpos, block_start, block_end
        )
        counts.append(purge_train_count)

    assert counts == [EXCLUDED_DATES * len(TICKERS)] * cross_validator.N_FOLDS


def test_model_training_indices_floor_semantics_unchanged() -> None:
    """The training-row floor filter still drops at-or-below-floor rows only."""
    _, _, tpos = _synthetic_inputs()
    boundaries = cross_validator._fold_boundaries(N_DATES)
    train_indices, _, _ = cross_validator._split_date_blocks(
        tpos, int(boundaries[1]), int(boundaries[2])
    )

    y = np.full(tpos.shape[0], 0.25)
    y[train_indices[:3]] = 0.0
    y[train_indices[3:5]] = training.RV_FLOOR
    y[train_indices[5]] = np.nextafter(training.RV_FLOOR, np.inf)

    kept = training._model_training_indices(
        {"train_indices": train_indices.tolist()}, y
    )

    assert kept.tolist() == train_indices[5:].tolist()
    assert len(kept) == len(train_indices) - 5


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
