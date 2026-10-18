"""Tests for the widened pre-training prerequisite guards.

The phase-4 prerequisite pass validates the full input quartet — X, y, metadata,
folds_metadata — and raises named ValueErrors for feature, row, width, regime,
and fold-minima pathologies. The per-fold minima check has a single module-level
implementation that the cross-validation boundary can import.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_training_guards.py
"""

from __future__ import annotations

import ast
import inspect
import re
import traceback

import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model.training import training

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"
TRAINING_PATH = MODEL_ROOT / "training" / "training.py"

SCALER_TOKENS = ("scaler", "standardscaler", "minmaxscaler", "robustscaler")
MINIMA_MESSAGE_TOKENS = ("empty train indices", "empty test indices")
INLINE_MINIMA = re.compile(
    r"len\(\s*\w*(?:train|test)_indices\s*\)\s*(?:==|<=|<)\s*0"
)


def _product_modules() -> list:
    """Model package modules outside the test tree."""
    return [
        path
        for path in sorted(MODEL_ROOT.rglob("*.py"))
        if TESTS_DIR not in path.parents
    ]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _valid_fixture() -> tuple:
    """Small synthetic input quartet mirroring the Phase 2-3 shapes.

    Five expanding folds over 60 rows; each (ticker, date) pair is unique and
    every regime bucket appears at least once.
    """
    rng = np.random.default_rng(11)
    n_rows = 60
    n_indicators, n_ticker_features = 3, 1

    X = rng.normal(size=(n_rows, n_indicators + n_ticker_features))
    y = rng.uniform(0.05, 0.6, size=n_rows)
    metadata = {
        "ticker": np.array([f"T{row % 3}" for row in range(n_rows)]),
        "date": np.datetime64("2000-01-03") + np.arange(n_rows),
        "regime": (np.arange(n_rows) % 7).astype(np.int8),
        "n_indicators": n_indicators,
        "n_ticker_features": n_ticker_features,
    }

    folds_metadata = {}
    for fold_idx in range(5):
        train_stop = 10 * (fold_idx + 1)
        test_stop = 10 * (fold_idx + 2)
        folds_metadata[f"fold_{fold_idx}"] = {
            "train_indices": list(range(train_stop)),
            "test_indices": list(range(train_stop, test_stop)),
            "n_train": train_stop,
            "n_test": test_stop - train_stop,
        }

    return X, y, metadata, folds_metadata


def _editable_fixture() -> tuple:
    """Fixture with metadata and fold dicts copied for per-test mutation."""
    X, y, metadata, folds_metadata = _valid_fixture()
    metadata = dict(metadata)
    folds_metadata = {key: dict(fold) for key, fold in folds_metadata.items()}
    return X, y, metadata, folds_metadata


def test_valid_fixture_passes_validated_quartet() -> None:
    X, y, metadata, folds_metadata = _valid_fixture()

    training._validate_training_prerequisites(X, y, metadata, folds_metadata)


def test_non_finite_features_raise_named_error() -> None:
    X, y, metadata, folds_metadata = _valid_fixture()
    X = X.copy()
    X[0, 0] = np.nan
    X[1, 1] = np.nan
    X[2, 2] = np.inf

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    message = str(excinfo.value)
    assert "Invalid feature values" in message
    assert "2 NaNs" in message
    assert "1 infs" in message


def test_duplicate_ticker_date_rows_raise_named_error() -> None:
    X, y, metadata, folds_metadata = _editable_fixture()
    ticker = metadata["ticker"].copy()
    date = metadata["date"].copy()
    for source, target in ((2, 3), (5, 6)):
        ticker[target] = ticker[source]
        date[target] = date[source]
    metadata["ticker"] = ticker
    metadata["date"] = date

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    message = str(excinfo.value)
    assert "Duplicate (ticker, date)" in message
    assert "2 duplicated" in message


def test_feature_width_mismatch_raises_named_error() -> None:
    X, y, metadata, folds_metadata = _valid_fixture()
    X = np.hstack([X, X[:, :1]])  # 5 columns against the metadata-implied 4

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    message = str(excinfo.value)
    assert "Feature width mismatch" in message
    assert "X has 5 columns" in message
    assert "implies 4" in message


def test_missing_regime_key_raises_named_error() -> None:
    X, y, metadata, folds_metadata = _editable_fixture()
    del metadata["regime"]

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    assert "missing the 'regime' key" in str(excinfo.value)


def test_non_1d_regime_raises_named_error() -> None:
    X, y, metadata, folds_metadata = _editable_fixture()
    metadata["regime"] = metadata["regime"].reshape(-1, 1)

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    message = str(excinfo.value)
    assert "1-D" in message
    assert "(60, 1)" in message


def test_regime_length_mismatch_raises_named_error() -> None:
    X, y, metadata, folds_metadata = _editable_fixture()
    metadata["regime"] = metadata["regime"][:-1]

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    message = str(excinfo.value)
    assert "59 entries" in message
    assert "60 rows" in message


def test_regime_values_outside_calendar_buckets_raise_named_error() -> None:
    for regime_dtype, invalid_value, token in (
        (np.int8, 7, "[7]"),
        (np.int8, -1, "[-1]"),
        (np.float64, 1.5, "[1.5]"),
    ):
        X, y, metadata, folds_metadata = _editable_fixture()
        regime = metadata["regime"].astype(regime_dtype)
        regime[7] = invalid_value
        metadata["regime"] = regime

        with pytest.raises(ValueError) as excinfo:
            training._validate_training_prerequisites(X, y, metadata, folds_metadata)

        message = str(excinfo.value)
        assert "integers in 0..6" in message
        assert token in message, f"invalid value {invalid_value} not named"


def test_empty_train_fold_raises_named_error() -> None:
    X, y, metadata, folds_metadata = _editable_fixture()
    folds_metadata["fold_2"]["train_indices"] = []

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    message = str(excinfo.value)
    assert "Fold 2" in message
    assert "empty train indices" in message


def test_empty_test_fold_raises_named_error() -> None:
    X, y, metadata, folds_metadata = _editable_fixture()
    folds_metadata["fold_3"]["test_indices"] = []

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    message = str(excinfo.value)
    assert "Fold 3" in message
    assert "empty test indices" in message


def test_fold_minima_helper_names_fold_and_side() -> None:
    with pytest.raises(ValueError) as excinfo:
        training.validate_fold_minima(4, np.array([], dtype=int), np.array([0, 1]))
    message = str(excinfo.value)
    assert "Fold 4" in message
    assert "empty train" in message

    with pytest.raises(ValueError) as excinfo:
        training.validate_fold_minima(1, np.array([0, 1]), np.array([], dtype=int))
    message = str(excinfo.value)
    assert "Fold 1" in message
    assert "empty test" in message

    training.validate_fold_minima(0, np.array([0]), np.array([1]))


def test_single_fold_minima_implementation_and_no_inline_check() -> None:
    source = TRAINING_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAINING_PATH))

    helper_definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "validate_fold_minima"
    ]
    assert len(helper_definitions) == 1, "expected exactly one fold-minima helper"
    helper = helper_definitions[0]
    assert list(inspect.signature(training.validate_fold_minima).parameters) == [
        "fold_idx",
        "train_indices",
        "test_indices",
    ]

    definitions = [
        str(path.relative_to(REPO_ROOT))
        for path in _product_modules()
        for node in ast.walk(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        if isinstance(node, ast.FunctionDef) and node.name == "validate_fold_minima"
    ]
    assert definitions == [str(TRAINING_PATH.relative_to(REPO_ROOT))]

    prerequisite_calls = [
        node
        for node in ast.walk(_function(tree, "_validate_training_prerequisites"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_fold_minima"
    ]
    assert prerequisite_calls, "the prerequisite pass must use validate_fold_minima"

    phase_4 = _function(tree, "phase_4_training")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_training_prerequisites"
        for node in ast.walk(phase_4)
    ), "phase_4_training must run the prerequisite pass"

    # No inline copy of the emptiness check outside the helper body.
    helper_lines = range(helper.lineno, helper.end_lineno + 1)
    inline_hits = []
    for path in _product_modules():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if INLINE_MINIMA.search(line) and not (
                path == TRAINING_PATH and line_number in helper_lines
            ):
                inline_hits.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")
    assert not inline_hits, "inline fold-minima checks found: " + ", ".join(
        inline_hits
    )

    # The minima guard message lives in exactly one product module.
    message_hits = [
        str(path.relative_to(REPO_ROOT))
        for path in _product_modules()
        if any(
            token in path.read_text(encoding="utf-8")
            for token in MINIMA_MESSAGE_TOKENS
        )
    ]
    assert message_hits == [str(TRAINING_PATH.relative_to(REPO_ROOT))]


def test_target_finiteness_guard_still_fires() -> None:
    X, y, metadata, folds_metadata = _valid_fixture()
    y = y.copy()
    y[0] = np.nan
    y[1] = np.inf

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    message = str(excinfo.value)
    assert "Invalid target values" in message
    assert "1 NaNs" in message
    assert "1 infs" in message


def test_fold_count_guard_still_fires() -> None:
    X, y, metadata, folds_metadata = _valid_fixture()
    del folds_metadata["fold_4"]

    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, folds_metadata)

    assert "Expected 5 folds, got 4" in str(excinfo.value)


def test_fold_count_guard_counts_fold_keys_not_schema_metadata() -> None:
    X, y, metadata, folds_metadata = _valid_fixture()
    payload = {
        "schema_version": 2,
        "feature_names": [f"indicator_{idx}" for idx in range(3)],
        "holdout": {
            "bounds": {"start": "2024-01-02", "end": "2025-12-31"},
            "n_dates": 500,
        },
        "n_dates": 50,
        **folds_metadata,
    }

    # Metadata keys alone must not make the guard count more than five folds.
    training._validate_training_prerequisites(X, y, metadata, payload)

    # Removing a fold still fails even though the metadata keys remain.
    del payload["fold_3"]
    with pytest.raises(ValueError) as excinfo:
        training._validate_training_prerequisites(X, y, metadata, payload)

    assert "Expected 5 folds, got 4" in str(excinfo.value)


def test_model_training_indices_keeps_only_rows_above_floor() -> None:
    floor = training.RV_FLOOR
    y = np.array([0.0, floor, floor + 1e-15, floor * 2.0, 1e-9])
    fold_data = {"train_indices": [0, 1, 2, 3, 4]}

    kept = training._model_training_indices(fold_data, y)

    assert kept.tolist() == [2, 3, 4]


def test_no_scaler_in_product_modules() -> None:
    hits = []
    for path in _product_modules():
        text = path.read_text(encoding="utf-8").lower()
        introduced = [token for token in SCALER_TOKENS if token in text]
        if introduced:
            hits.append(f"{path.relative_to(REPO_ROOT)}: {introduced}")
    assert not hits, "scaler vocabulary introduced: " + ", ".join(hits)


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
