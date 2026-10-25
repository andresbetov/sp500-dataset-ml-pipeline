"""Tests for the canonical train-only purge and the CV boundary guards.

The split keeps every test block whole and purges only training rows whose
forward label window reaches into the test block: the last kept training date
sits five trading dates before the block start, the first kept test date is the
block start, and exactly the five dates in between are excluded. Block edges
come from exact integer floors, undersized and degenerate calendars fail with
named errors, and every fold satisfies the label-window invariant. Every check
runs against a synthetic business-day calendar and temporary artifact roots, so
the live inputs and the stored fold artifacts are neither read nor written; the
one live non-regression check reads them behind the shared skip guard.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_purge.py
"""

from __future__ import annotations

import ast
import inspect
import re
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR, REPO_ROOT, requires_live_artifacts
from packages.ml.model import config as model_config
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.training import training

# Total synthetic calendar: the 500-date holdout plus a 36-date fold remainder,
# exactly divisible by the N_FOLDS + 1 = 6 date blocks, so every block boundary
# is an exact date index.
N_DATES = 536
HOLDOUT_N_DATES = 500
FOLD_N_DATES = N_DATES - HOLDOUT_N_DATES
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
PAYLOAD_METADATA_KEYS = frozenset(
    {"schema_version", "feature_names", "holdout", "n_dates"}
)

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"
CROSS_VALIDATOR_PATH = MODEL_ROOT / "cross_validation" / "cross_validator.py"
TRAINING_PATH = MODEL_ROOT / "training" / "training.py"
INLINE_FOLD_MINIMA = re.compile(
    r"len\(\s*\w*(?:train|test)_indices\s*\)\s*(?:==|<=|<)\s*0"
)

# The live calendar at the branch point: 6504 unique trading dates, the final
# 500 reserved, leaving 6004 fold dates whose six block edges are exact.
LIVE_N_DATES = 6504
LIVE_HOLDOUT_N_DATES = 500
LIVE_FOLD_EDGES = [0, 1000, 2001, 3002, 4002, 5003, 6004]

# HOLDOUT_N_DATES + (N_FOLDS + 1) * (PURGE_HORIZON_D + 1) with the current
# config; the test re-derives it from the module symbols.
MINIMUM_N_DATES = 536


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _synthetic_calendar(n_dates: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows over a synthetic trading calendar and their unique-date positions."""
    dates = np.repeat(
        pd.bdate_range("2000-01-03", periods=n_dates).values, len(TICKERS)
    )
    unique_dates = np.sort(np.unique(dates))
    tpos = np.searchsorted(unique_dates, dates)
    return dates, unique_dates, tpos


def _synthetic_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows over the synthetic calendar and their positions in the fold dates.

    The final HOLDOUT_N_DATES calendar dates are reserved before fold
    construction, so the positions are relative to the remainder and the
    reserved rows carry the out-of-range position len(fold_dates).
    """
    dates, unique_dates, _ = _synthetic_calendar(N_DATES)
    fold_dates = unique_dates[:FOLD_N_DATES]
    tpos = np.searchsorted(fold_dates, dates)
    return dates, fold_dates, tpos


def _write_phase_2_inputs(root: Path, n_dates: int) -> None:
    """Write minimal Phase 2 inputs for n_dates over the synthetic calendar."""
    dates, _, _ = _synthetic_calendar(n_dates)
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    np.save(inputs / "X.npy", np.zeros((len(dates), 1)))
    np.save(inputs / "Y.npy", np.full(len(dates), 0.25))
    joblib.dump(
        {
            "date": dates,
            "ticker": np.tile(np.array(TICKERS), n_dates),
            "regime": (
                np.arange(len(dates)) % len(cross_validator.REGIME_NAMES)
            ).astype(np.int8),
        },
        inputs / "metadata.pkl",
    )
    record_artifact(inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME)


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

    with pytest.raises(ValueError, match="separation"):
        cross_validator._verify_fold_separation(0, train_tpos, np.array([9, 10]), 1)

    with pytest.raises(ValueError, match="purge"):
        cross_validator._verify_fold_separation(0, train_tpos, test_tpos, 0)


def test_model_training_indices_floor_semantics_unchanged() -> None:
    """The training-row floor filter still drops at-or-below-floor rows only."""
    _, _, tpos = _synthetic_inputs()
    boundaries = cross_validator._fold_boundaries(FOLD_N_DATES)
    # Fold 1 keeps enough training rows to exercise the floor split.
    train_indices, _, _ = cross_validator._split_date_blocks(
        tpos, int(boundaries[2]), int(boundaries[3])
    )
    assert len(train_indices) >= 6

    y = np.full(tpos.shape[0], 0.25)
    y[train_indices[:3]] = 0.0
    y[train_indices[3:5]] = training.RV_FLOOR
    y[train_indices[5]] = np.nextafter(training.RV_FLOOR, np.inf)

    kept = training._model_training_indices(
        {"train_indices": train_indices.tolist()}, y
    )

    assert kept.tolist() == train_indices[5:].tolist()
    assert len(kept) == len(train_indices) - 5


def test_fold_boundaries_use_exact_integer_edges() -> None:
    """n = 100: integer edges, the final edge is n_dates, fold 4 is non-empty."""
    edges = cross_validator._fold_boundaries(100)

    assert edges.tolist() == [0, 16, 33, 50, 66, 83, 100]
    assert int(edges[-1]) == 100
    assert int(edges[-2]) < int(edges[-1])
    assert np.array_equal(
        edges,
        (np.arange(cross_validator.N_FOLDS + 2) * 100)
        // (cross_validator.N_FOLDS + 1),
    )
    # The float-step construction is gone from the boundary function.
    assert "linspace" not in inspect.getsource(cross_validator._fold_boundaries)


def test_block_guards() -> None:
    """Undersized calendars fail by name; the derived minimum builds every fold."""
    minimum = MINIMUM_N_DATES

    # A calendar one date short of the minimum, and a lone tiny one, name the
    # minimum instead of failing inside a numpy reduction.
    for undersized in (8, minimum - 1):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_phase_2_inputs(root, undersized)
            with mock.patch.object(cross_validator, "ARTIFACTS_DIR", root):
                with pytest.raises(
                    ValueError,
                    match=rf"at least {minimum} unique trading dates",
                ) as excinfo:
                    cross_validator.phase_3_cross_validation()

        message = str(excinfo.value)
        assert "zero-size" not in message, (undersized, message)
        assert str(undersized) in message, (undersized, message)

    # The derived minimum passes the guard and builds all five folds.
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_phase_2_inputs(root, minimum)
        with mock.patch.object(cross_validator, "ARTIFACTS_DIR", root):
            folds = cross_validator.phase_3_cross_validation()

    assert set(folds) == {f"fold_{idx}" for idx in range(cross_validator.N_FOLDS)}
    for fold in folds.values():
        assert fold["n_train"] > 0
        assert fold["n_test"] > 0
        assert (
            fold["first_test_idx"] - fold["last_train_idx"]
            == cross_validator.SEPARATION_MIN
        )

    # Degenerate calendars cannot form N_FOLDS + 1 non-empty blocks at all.
    for n_dates in (0, cross_validator.N_FOLDS):
        with pytest.raises(ValueError, match="degenerate calendar"):
            cross_validator._fold_boundaries(n_dates)
    cross_validator._fold_boundaries(cross_validator.N_FOLDS + 1)


def test_window_invariant_holds_across_the_calendar_sweep() -> None:
    """Valid calendars: the label window never crosses the train/test boundary."""
    for n_dates in (36, 60, 100, 500, 6504):
        _, unique_dates, tpos = _synthetic_calendar(n_dates)
        assert len(unique_dates) == n_dates
        edges = cross_validator._fold_boundaries(n_dates)
        assert int(edges[-1]) == n_dates

        train_sets: list[set[int]] = []
        test_sets: list[set[int]] = []
        for fold_idx in range(cross_validator.N_FOLDS):
            block_start = int(edges[fold_idx + 1])
            block_end = int(edges[fold_idx + 2])
            assert block_start < block_end

            train_indices, test_indices, purge_train_count = (
                cross_validator._split_date_blocks(tpos, block_start, block_end)
            )
            assert purge_train_count > 0
            train_tpos = tpos[train_indices]
            test_tpos = tpos[test_indices]
            assert train_tpos.size > 0 and test_tpos.size > 0

            # The label window of the last kept training date ends strictly
            # before the first kept test date, in unique-date index space.
            assert (
                int(train_tpos.max()) + cross_validator.LABEL_HORIZON_DAYS
                < int(test_tpos.min())
            )
            train_sets.append(set(np.unique(train_tpos).tolist()))
            test_sets.append(set(np.unique(test_tpos).tolist()))

        # Test blocks stay disjoint and partition the remainder.
        seen: set[int] = set()
        for test_set in test_sets:
            assert not (seen & test_set)
            seen |= test_set
        assert seen == set(range(int(edges[1]), n_dates))

        # Training windows expand strictly fold over fold.
        for earlier, later in zip(train_sets, train_sets[1:]):
            assert earlier < later


def test_window_invariant_raises_by_name_when_the_predicate_would_pass() -> None:
    """A 5-index separation passes a lowered predicate but still fails the invariant."""
    train_tpos = np.array([0, 1, 2, 3, 4])
    test_tpos = np.array([9, 10])

    with mock.patch.object(cross_validator, "SEPARATION_MIN", 5):
        with pytest.raises(ValueError, match="window invariant"):
            cross_validator._verify_fold_separation(0, train_tpos, test_tpos, 1)


def test_purge_span_constants_come_from_config() -> None:
    """PURGE_HORIZON_D and SEPARATION_MIN mirror the loaded config."""
    loaded = model_config.load_config()

    assert cross_validator.PURGE_HORIZON_D == loaded["folds"]["purge_horizon_days"] == 5
    assert cross_validator.SEPARATION_MIN == loaded["folds"]["separation_min"] == 6
    assert cross_validator.LABEL_HORIZON_DAYS == loaded["folds"]["label_horizon_days"]
    assert cross_validator.HOLDOUT_N_DATES == loaded["holdout"]["n_dates"] == 500
    assert (
        cross_validator.HOLDOUT_N_DATES
        + (cross_validator.N_FOLDS + 1) * (cross_validator.PURGE_HORIZON_D + 1)
        == MINIMUM_N_DATES
        == 536
    )
    # The legacy alias is gone: the purge-span name is the only one in play.
    assert not hasattr(cross_validator, "CV_EMBARGO_DAYS")


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
