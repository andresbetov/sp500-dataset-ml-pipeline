"""Tests for the validation-only tuning harness and its mechanical test lock.

Every trial's fold data comes from one locked accessor: fit and validation
views are built from the purged inner-validation split of floor-retained
fold-train rows (the split the training path already uses), and any view that
would contain a test-block or holdout index raises ``TestLockError``. A tiny
synthetic campaign (a stub trainer, temporary output roots) exercises the
coarse/refinement/seed sequence, the per-trial sidecars carrying each trial's
train/validation index fingerprints through the sidecar + run-manifest
machinery, and the selection artifact (``selected_on: validation_qlike``, zero
``test_*`` keys, and a schema that rejects them). The checks never read or
write live ``artifacts/**`` or ``data/model_outputs/**``.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_tuning.py
"""

from __future__ import annotations

import inspect
import json
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model import config as model_config
from packages.ml.model import run_log
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME
from packages.ml.model.training import model_trainer, training
from packages.ml.model.tuning import harness

# Tiny campaign space: two coarse combos x two gamma refinements x two seed
# finalists -> six trials, all on the synthetic folds below.
TINY_SEARCH_SPACE = {
    "coarse_grid": {
        "learning_rate": [0.03, 0.05],
        "max_depth": [4],
        "min_child_weight": [1],
    },
    "finalist_refinement": {"gamma": [0.0, 0.1]},
    "seed_finalists": [42, 43],
}

# Two tickers over 50 trading dates, ticker-major like the stored inputs: the
# first 30 dates are fold dates (three 10-date blocks), the last 20 are the
# reserved holdout window. Fold 0 trains on block 0 and tests block 1; fold 1
# trains on blocks 0-1 and tests block 2.
TICKERS = ("AAA", "BBB")
N_DATES = 50
HOLDOUT_DATES = 20
BLOCK_DATES = 10
N_FEATURES = 3
FOLD_KEYS = ("fold_0", "fold_1")


def _date_pos(dates: np.ndarray) -> np.ndarray:
    """Per-row position in the sorted unique trading dates."""
    return np.unique(dates, return_inverse=True)[1]


def _tiny_folds() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Synthetic Phase 2-3 shaped inputs with two folds and a holdout window.

    The first feature is the row index, so a trainer double can map the rows
    it receives back to the input rows.
    """
    rng = np.random.default_rng(101)
    dates = np.datetime64("2000-01-03") + np.arange(N_DATES)
    row_dates = np.tile(dates, len(TICKERS))  # ticker-major stored order
    n_rows = row_dates.size
    X = rng.normal(size=(n_rows, N_FEATURES))
    X[:, 0] = np.arange(n_rows, dtype=np.float64)
    y = rng.uniform(0.05, 0.6, size=n_rows)

    fold_dates = dates[: N_DATES - HOLDOUT_DATES]
    # Block edges over the fold dates plus the holdout start as the last
    # exclusive bound: block 0 is train-only, block 1 is fold 0's test, and
    # block 2 is fold 1's test.
    block_edges = [
        fold_dates[BLOCK_DATES * block] for block in range(3)
    ] + [dates[N_DATES - HOLDOUT_DATES]]
    folds_metadata = {
        "schema_version": 2,
        "feature_names": [f"feature_{index}" for index in range(N_FEATURES)],
        "holdout": {
            "bounds": {
                "start": str(fold_dates[-1] + np.timedelta64(1, "D")),
                "end": str(dates[-1]),
            },
            "n_dates": HOLDOUT_DATES,
        },
        "n_dates": len(fold_dates),
    }
    for fold_idx in range(len(FOLD_KEYS)):
        block_start = block_edges[fold_idx + 1]
        block_end = block_edges[fold_idx + 2]
        train_mask = row_dates < block_start
        test_mask = (row_dates >= block_start) & (row_dates < block_end)
        folds_metadata[f"fold_{fold_idx}"] = {
            "train_indices": np.flatnonzero(train_mask).tolist(),
            "test_indices": np.flatnonzero(test_mask).tolist(),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
    return X, y, row_dates, folds_metadata


def _temp_root(tmpdir: str) -> Path:
    """A temporary tuning output root, guarded to stay outside the live tree."""
    root = Path(tmpdir) / "tuning"
    assert not root.resolve().is_relative_to(REPO_ROOT), (
        "campaign tests must target temporary roots, never live artifacts"
    )
    return root


def _expected_views(
    X: np.ndarray, y: np.ndarray, row_dates: np.ndarray, folds_metadata: dict
) -> dict:
    """The locked views rebuilt from the training split, for comparisons."""
    return {
        fold_key: harness.build_locked_fold_views(
            fold_key, X=X, y=y, dates=row_dates, folds_metadata=folds_metadata
        )
        for fold_key in FOLD_KEYS
    }


class _StubModel:
    """Picklable stand-in model with the only surface the harness reads."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float64)


def _stub_trainer(X_train, y_train, X_val=None, y_val=None, params=None):
    return _StubModel(), {}


def _recording_trainer(record: dict):
    """Trainer double recording the rows and params each fit receives."""

    def trainer(X_train, y_train, X_val=None, y_val=None, params=None):
        record["fit"].append(np.asarray(X_train)[:, 0].astype(np.int64))
        record["validation"].append(np.asarray(X_val)[:, 0].astype(np.int64))
        record["params"].append(dict(params or {}))
        return _StubModel(), {}

    return trainer


def _run_campaign(
    X: np.ndarray,
    y: np.ndarray,
    row_dates: np.ndarray,
    folds_metadata: dict,
    output_dir: Path,
    *,
    trainer=None,
) -> dict:
    """Run a tiny campaign; the stub trainer keeps every trial fast."""
    return harness.run_tuning_campaign(
        X=X,
        y=y,
        dates=row_dates,
        folds_metadata=folds_metadata,
        output_dir=output_dir,
        search_space=TINY_SEARCH_SPACE,
        trainer=_stub_trainer if trainer is None else trainer,
    )


def _all_keys(node: object):
    """Every mapping key anywhere in a JSON-like document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _all_keys(value)


def test_locked_views_come_from_the_purged_inner_split_and_stay_in_fold_train() -> None:
    X, y, row_dates, folds_metadata = _tiny_folds()
    date_pos = _date_pos(row_dates)
    holdout = set(harness.holdout_row_indices(row_dates, folds_metadata).tolist())

    for fold_key in FOLD_KEYS:
        fold = folds_metadata[fold_key]
        fold_idx = int(fold_key.split("_", 1)[1])
        train = set(fold["train_indices"])
        test = set(fold["test_indices"])

        views = harness.build_locked_fold_views(
            fold_key, X=X, y=y, dates=row_dates, folds_metadata=folds_metadata
        )

        # The accessor reuses the training task's split: fit and validation
        # are exactly its outputs for the fold's floor-retained fold-train.
        train_indices = np.asarray(fold["train_indices"], dtype=np.int64)
        retained = train_indices[training.floor_retained_mask(y[train_indices])]
        expected_fit, expected_validation, expected_purge = (
            training.inner_validation_split(fold_idx, retained, date_pos, y)
        )
        assert views["fit"]["indices"].tolist() == expected_fit.tolist(), fold_key
        assert (
            views["validation"]["indices"].tolist()
            == expected_validation.tolist()
        ), fold_key
        assert views["n_purged"] == expected_purge, fold_key

        for role in ("fit", "validation"):
            view = views[role]
            indices = set(view["indices"].tolist())
            assert indices, (fold_key, role)
            assert indices <= train, (fold_key, role)
            assert not (indices & test), (fold_key, role)
            assert not (indices & holdout), (fold_key, role)
            assert view["fingerprint"] == harness.index_fingerprint(view["indices"])
            assert view["X"].shape[0] == len(indices), (fold_key, role)
            assert view["y"].shape[0] == len(indices), (fold_key, role)

        # Fit and validation are disjoint, the tail is strictly temporal, and
        # the fit ends at least the label horizon before the tail starts.
        fit_indices = views["fit"]["indices"]
        validation_indices = views["validation"]["indices"]
        assert not set(fit_indices.tolist()) & set(validation_indices.tolist())
        assert (
            date_pos[validation_indices].min() - date_pos[fit_indices].max()
            >= training.LABEL_HORIZON_DAYS + 1
        ), fold_key

    # The fingerprint is canonical over the index set, not its order.
    assert harness.index_fingerprint([3, 1, 2]) == harness.index_fingerprint(
        [1, 2, 3]
    )


def test_test_block_and_holdout_indices_raise_test_lock() -> None:
    X, y, row_dates, folds_metadata = _tiny_folds()
    fold = folds_metadata["fold_0"]
    train = np.asarray(fold["train_indices"], dtype=np.int64)
    test = np.asarray(fold["test_indices"], dtype=np.int64)
    holdout = harness.holdout_row_indices(row_dates, folds_metadata)
    test_index = int(test[0])
    holdout_index = int(holdout[0])
    y_log = np.log(np.maximum(y, training.RV_FLOOR))

    def locked(role, indices):
        return harness.locked_view(
            "fold_0",
            role,
            indices,
            X=X,
            y=y_log,
            fold_train_indices=train,
            test_indices=test,
            holdout_indices=holdout,
        )

    for role in ("fit", "validation"):
        # A deliberate test-block read is refused by name...
        with pytest.raises(harness.TestLockError, match="test-block"):
            locked(role, [test_index])
        # ...a holdout read raises the same named error...
        with pytest.raises(harness.TestLockError, match="holdout"):
            locked(role, [holdout_index])
        # ...and a mixed view is refused on the test index it carries.
        with pytest.raises(harness.TestLockError, match="test-block"):
            locked(role, [int(train[0]), test_index])
        # A fold-train subset passes the lock and keeps its fingerprint.
        view = locked(role, train[:2])
        assert view["indices"].tolist() == train[:2].tolist()
        assert view["fingerprint"] == harness.index_fingerprint(train[:2])

    # An index outside fold-train (another fold's test block, neither this
    # fold's test nor the holdout) is refused too.
    other_fold_test = int(folds_metadata["fold_1"]["test_indices"][0])
    assert other_fold_test not in set(test.tolist())
    assert other_fold_test not in set(holdout.tolist())
    with pytest.raises(harness.TestLockError, match="outside fold-train"):
        locked("fit", [other_fold_test])

    # A float index cannot slip through integer coercion.
    with pytest.raises(ValueError, match="integers"):
        locked("fit", [1.5])


def test_search_space_config_pins_the_coarse_refinement_seed_space() -> None:
    loaded = model_config.load_config()
    assert loaded["tuning"]["select_on"] == "validation_qlike"
    space = loaded["tuning"]["search_space"]
    assert space["coarse_grid"] == {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 5, 25],
    }
    assert space["finalist_refinement"] == {"gamma": [0.0, 0.1]}
    assert space["seed_finalists"] == [42, 43, 44]

    coarse_combos = 1
    for values in space["coarse_grid"].values():
        coarse_combos *= len(values)
    assert coarse_combos == 27  # the configured coarse budget cap

    # The harness reads the same selector and refuses anything else.
    assert harness.SELECT_ON_VALIDATION_QLIKE == "validation_qlike"


def test_configured_search_space_runs_end_to_end() -> None:
    """The configured coarse/refinement/seed space drives the whole sequence."""
    X, y, row_dates, folds_metadata = _tiny_folds()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = harness.run_tuning_campaign(
            X=X,
            y=y,
            dates=row_dates,
            folds_metadata=folds_metadata,
            output_dir=_temp_root(tmpdir),
            trainer=_stub_trainer,
        )
        selection = json.loads(
            result["selection_path"].read_text(encoding="utf-8")
        )

    assert result["n_trials"] == 27 + 2 + 3
    assert selection["stages"]["coarse"]["n_trials"] == 27
    assert selection["stages"]["refinement"]["n_trials"] == 2
    assert selection["stages"]["seed_finalists"]["n_trials"] == 3
    # Identical stub scores tie, and ties keep the earliest seed finalist.
    assert selection["selected_trial"]["seed"] == 42
    assert selection["selected_trial"]["stage"] == "seed_finalists"
    assert selection["selected_on"] == "validation_qlike"


def test_campaign_refuses_a_selector_other_than_validation_qlike() -> None:
    X, y, row_dates, folds_metadata = _tiny_folds()
    loaded = model_config.load_config()
    poisoned = {
        **loaded,
        "tuning": {**loaded["tuning"], "select_on": "test_qlike"},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch.object(
            model_config, "get_config", return_value=poisoned
        ):
            with pytest.raises(ValueError, match="validation_qlike"):
                _run_campaign(
                    X, y, row_dates, folds_metadata, _temp_root(tmpdir)
                )


def test_tiny_campaign_writes_a_validation_only_selection_artifact() -> None:
    X, y, row_dates, folds_metadata = _tiny_folds()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_campaign(
            X, y, row_dates, folds_metadata, _temp_root(tmpdir)
        )

        selection = json.loads(
            result["selection_path"].read_text(encoding="utf-8")
        )
        assert selection["selected_on"] == "validation_qlike"
        assert (
            selection["selected_on"]
            == model_config.get_config()["tuning"]["select_on"]
        )
        assert selection["search_space"] == TINY_SEARCH_SPACE
        assert selection["n_trials"] == result["n_trials"] == 6
        assert selection["stages"]["coarse"]["n_trials"] == 2
        assert selection["stages"]["refinement"]["n_trials"] == 2
        assert selection["stages"]["seed_finalists"]["n_trials"] == 2

        # The winner is the best of the seed finalists on validation QLIKE.
        selected = selection["selected_trial"]
        assert selected["stage"] == "seed_finalists"
        assert selected["seed"] == 42
        seed_scores = [
            trial["validation_qlike"]
            for trial in selection["trials"]
            if trial["stage"] == "seed_finalists"
        ]
        assert selected["validation_qlike"] == min(seed_scores)
        assert selected["params"]["random_state"] == selected["seed"]
        for fold_key in FOLD_KEYS:
            assert selected["folds"][fold_key]["n_validation"] > 0

        # Zero test_* keys anywhere in the artifact, and the schema accepts it.
        assert not [
            key for key in _all_keys(selection) if key.startswith("test_")
        ]
        harness.validate_selection_artifact(selection)

        # The trial list is ordered and carries every stage exactly once.
        assert [trial["trial_index"] for trial in selection["trials"]] == list(
            range(6)
        )
        assert {trial["stage"] for trial in selection["trials"]} == {
            "coarse",
            "refinement",
            "seed_finalists",
        }


def test_selection_schema_rejects_test_prefixed_keys() -> None:
    X, y, row_dates, folds_metadata = _tiny_folds()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_campaign(
            X, y, row_dates, folds_metadata, _temp_root(tmpdir)
        )
        selection = json.loads(
            result["selection_path"].read_text(encoding="utf-8")
        )

    top_level = deepcopy(selection)
    top_level["test_qlike"] = 0.0
    with pytest.raises(harness.TestLockError, match="test_qlike"):
        harness.validate_selection_artifact(top_level)

    nested = deepcopy(selection)
    nested["selected_trial"]["folds"]["fold_0"]["test_fingerprint"] = "0" * 64
    with pytest.raises(harness.TestLockError, match="test_fingerprint"):
        harness.validate_selection_artifact(nested)

    in_list = deepcopy(selection)
    in_list["trials"][0]["folds"]["fold_1"]["test_qlike"] = 1.0
    with pytest.raises(harness.TestLockError, match="test_qlike"):
        harness.validate_selection_artifact(in_list)


def test_no_view_construction_path_can_carry_test_or_holdout_indices() -> None:
    X, y, row_dates, folds_metadata = _tiny_folds()

    # Enumerate the module's public view-construction paths: the low-level
    # choke point and the fold accessor, and nothing else.
    view_functions = {
        name: function
        for name, function in vars(harness).items()
        if inspect.isfunction(function)
        and "view" in name.lower()
        and not name.startswith("_")
    }
    assert set(view_functions) == {"locked_view", "build_locked_fold_views"}

    # Poisoned fold metadata: a fold-train row is relabeled as a test row. The
    # accessor refuses to build any view (the same row would land in fit).
    poisoned = deepcopy(folds_metadata)
    sentinel = int(poisoned["fold_0"]["train_indices"][0])
    poisoned["fold_0"]["test_indices"] = poisoned["fold_0"]["test_indices"] + [
        sentinel
    ]
    with pytest.raises(harness.TestLockError, match="test-block"):
        harness.build_locked_fold_views(
            "fold_0", X=X, y=y, dates=row_dates, folds_metadata=poisoned
        )

    # Poisoned holdout bounds: every fold-train row is inside the window, so
    # no view can be built.
    poisoned_holdout = deepcopy(folds_metadata)
    poisoned_holdout["holdout"]["bounds"]["start"] = str(row_dates[0])
    with pytest.raises(harness.TestLockError, match="holdout"):
        harness.build_locked_fold_views(
            "fold_0", X=X, y=y, dates=row_dates, folds_metadata=poisoned_holdout
        )

    # The whole campaign path refuses the poisoned metadata too, and the run
    # is finalized as failed with the lock error recorded.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = _temp_root(tmpdir)
        with pytest.raises(harness.TestLockError, match="test-block"):
            _run_campaign(X, y, row_dates, poisoned, output_dir)
        failed_manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert failed_manifest["status"] == "failed"
        assert "TestLockError" in failed_manifest["error"]

    # End to end: every row the trainer receives is one of the locked views'
    # rows, and no received row is in a test block or the holdout window.
    views = _expected_views(X, y, row_dates, folds_metadata)
    record = {"fit": [], "validation": [], "params": []}
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_campaign(
            X,
            y,
            row_dates,
            folds_metadata,
            _temp_root(tmpdir),
            trainer=_recording_trainer(record),
        )
    assert len(record["fit"]) == len(record["validation"]) == 12  # 6 trials x 2 folds

    allowed_fit = {
        tuple(views[fold_key]["fit"]["indices"].tolist()) for fold_key in FOLD_KEYS
    }
    allowed_validation = {
        tuple(views[fold_key]["validation"]["indices"].tolist())
        for fold_key in FOLD_KEYS
    }
    assert all(
        tuple(rows.tolist()) in allowed_fit for rows in record["fit"]
    )
    assert all(
        tuple(rows.tolist()) in allowed_validation for rows in record["validation"]
    )
    # Per-fold disjointness: the campaign visits the folds in sorted order for
    # every trial, so each recorded array maps back to its fold. A fold's view
    # may legitimately include an earlier fold's test block (expanding
    # windows), but never its own test block or any holdout row.
    fold_key_list = sorted(FOLD_KEYS)
    all_holdout = set(
        harness.holdout_row_indices(row_dates, folds_metadata).tolist()
    )
    for role in ("fit", "validation"):
        for position, rows in enumerate(record[role]):
            fold_key = fold_key_list[position % len(fold_key_list)]
            received = set(rows.tolist())
            assert not (received & set(folds_metadata[fold_key]["test_indices"])), (
                role,
                fold_key,
            )
            assert not (received & all_holdout), (role, fold_key)


def test_per_trial_sidecars_record_index_fingerprints_and_manifest_hashes() -> None:
    X, y, row_dates, folds_metadata = _tiny_folds()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = _temp_root(tmpdir)
        result = _run_campaign(X, y, row_dates, folds_metadata, output_dir)

        sidecars = sorted(result["run_dir"].glob("trial_*_sidecar.json"))
        assert len(sidecars) == 6
        views = _expected_views(X, y, row_dates, folds_metadata)

        fingerprints = {fold_key: set() for fold_key in FOLD_KEYS}
        for trial_index, path in enumerate(sidecars):
            document = json.loads(path.read_text(encoding="utf-8"))
            assert document["trial_index"] == trial_index
            assert document["stage"] in {"coarse", "refinement", "seed_finalists"}
            assert document["seed"] in {42, 43}
            for fold_key in FOLD_KEYS:
                fold_record = document["folds"][fold_key]
                assert (
                    fold_record["fit_fingerprint"]
                    == views[fold_key]["fit"]["fingerprint"]
                )
                assert (
                    fold_record["validation_fingerprint"]
                    == views[fold_key]["validation"]["fingerprint"]
                )
                assert fold_record["n_fit"] == len(
                    views[fold_key]["fit"]["indices"]
                )
                assert fold_record["n_validation"] == len(
                    views[fold_key]["validation"]["indices"]
                )
                fingerprints[fold_key].add(
                    (
                        fold_record["fit_fingerprint"],
                        fold_record["validation_fingerprint"],
                    )
                )

        # The views are built once per fold, so every trial records the same
        # fingerprint pair for a fold.
        for fold_key in FOLD_KEYS:
            assert len(fingerprints[fold_key]) == 1

        # The sidecars and the selection artifact are recorded in the run
        # manifest and the artifact registry with their hashes.
        manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "completed"
        registry = json.loads(
            (output_dir / REGISTRY_FILENAME).read_text(encoding="utf-8")
        )
        for path in sidecars + [result["selection_path"]]:
            key = run_log.to_repo_relative(path)
            assert manifest["artifacts"][key] == run_log.sha256_file(path)
            assert key in registry

        # The selection artifact carries the winning trial's fingerprints too.
        selection = json.loads(
            result["selection_path"].read_text(encoding="utf-8")
        )
        selected = selection["selected_trial"]
        for fold_key in FOLD_KEYS:
            assert (
                selected["folds"][fold_key]["fit_fingerprint"]
                == views[fold_key]["fit"]["fingerprint"]
            )
            assert (
                selected["folds"][fold_key]["validation_fingerprint"]
                == views[fold_key]["validation"]["fingerprint"]
            )


def test_baseline_seed_42_params_reproduce_the_default_trainer() -> None:
    # The baseline recipe is the pre-harness inline configuration.
    baseline = model_trainer.DEFAULT_PARAMS
    assert baseline["random_state"] == 42
    assert model_config.get_config()["training"]["seed"] == 42
    assert baseline["max_depth"] == 6
    assert baseline["min_child_weight"] == 1.0
    assert baseline["gamma"] == 0.0
    assert baseline["learning_rate"] == 0.05
    assert baseline["n_estimators"] == 500
    assert baseline["reg_alpha"] == 0.1
    assert baseline["reg_lambda"] == 1.0
    assert baseline["subsample"] == 0.8
    assert baseline["colsample_bytree"] == 0.8
    assert baseline["colsample_bylevel"] == 0.8

    # The default call and the explicit baseline mapping train identically.
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, N_FEATURES))
    y = rng.uniform(-3.0, -1.0, size=24)
    X_val = rng.normal(size=(8, N_FEATURES))
    y_val = rng.uniform(-3.0, -1.0, size=8)
    default_model, _ = model_trainer.train_xgboost_model(
        X, y, X_val=X_val, y_val=y_val
    )
    baseline_model, _ = model_trainer.train_xgboost_model(
        X, y, X_val=X_val, y_val=y_val, params=dict(baseline)
    )
    np.testing.assert_array_equal(
        default_model.predict(X_val), baseline_model.predict(X_val)
    )
    assert default_model.get_params()["random_state"] == 42

    # The same non-regression holds through the locked views the harness
    # builds, and every campaign trial starts from the baseline recipe.
    X_folds, y_folds, row_dates, folds_metadata = _tiny_folds()
    views = harness.build_locked_fold_views(
        "fold_0",
        X=X_folds,
        y=y_folds,
        dates=row_dates,
        folds_metadata=folds_metadata,
    )
    from_views, _ = model_trainer.train_xgboost_model(
        views["fit"]["X"],
        views["fit"]["y"],
        X_val=views["validation"]["X"],
        y_val=views["validation"]["y"],
        params=dict(baseline),
    )
    default_from_views, _ = model_trainer.train_xgboost_model(
        views["fit"]["X"],
        views["fit"]["y"],
        X_val=views["validation"]["X"],
        y_val=views["validation"]["y"],
    )
    np.testing.assert_array_equal(
        from_views.predict(views["validation"]["X"]),
        default_from_views.predict(views["validation"]["X"]),
    )

    record = {"fit": [], "validation": [], "params": []}
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_campaign(
            X_folds,
            y_folds,
            row_dates,
            folds_metadata,
            _temp_root(tmpdir),
            trainer=_recording_trainer(record),
        )
    assert record["params"], "the campaign must pass a parameter mapping"
    for params in record["params"]:
        assert set(params) == set(baseline)
        assert params["n_estimators"] == baseline["n_estimators"]
        assert params["reg_alpha"] == baseline["reg_alpha"]
        assert params["objective"] is baseline["objective"]
        assert params["eval_metric"] is baseline["eval_metric"]


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
