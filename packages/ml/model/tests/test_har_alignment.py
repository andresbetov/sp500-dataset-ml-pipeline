"""Tests pinning the HAR-regressor (ticker, date) alignment and warm-up.

``load_har_regressors`` verifies that the dataset parquet and the model input
(ticker, date) pairs agree row by row, requires each ticker's dates to be
strictly increasing, and returns the first-21-dates-per-ticker warm-up mask
that the HAR fit excludes. These checks pin that representation-only date
differences still align after normalization, that a same-date ticker swap and
non-monotonic dates raise named errors, that the mask covers exactly the first
``HAR_WARMUP_DAYS`` dates per ticker without changing component values, that
the fold path excludes the masked rows from the fit, records the count, and
still scores warm-up test rows (late index entrants) unchanged, and that the
live alignment is unchanged. The live checks read
``data/processed/dataset.parquet`` read-only and skip when the generated
dataset is absent, as on a fresh clone or in CI.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_har_alignment.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import DATASET_PATH
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact

FOLD_PATH_N_DATES = 60
FOLD_PATH_TEST_DATES = 6
FOLD_PATH_ENTRANT_START = 36

_live_cache: dict[str, np.ndarray] = {}


def _live_metadata() -> tuple[np.ndarray, np.ndarray]:
    """The live extraction's (dates, tickers); skips without the dataset."""
    if "dates" not in _live_cache:
        if not DATASET_PATH.is_file():
            pytest.skip(
                "requires the generated dataset absent from this checkout: "
                f"{DATASET_PATH}"
            )
        from packages.ml.model.features import feature_engineering

        frame = pd.read_parquet(DATASET_PATH)
        _, _, metadata = feature_engineering._extract_inputs(frame)
        _live_cache["dates"] = metadata["date"]
        _live_cache["tickers"] = metadata["ticker"]
    return _live_cache["dates"], _live_cache["tickers"]


def _synthetic_ohlc(
    tickers: np.ndarray, log_returns: np.ndarray
) -> dict[str, np.ndarray]:
    """Deterministic adjusted OHLC bars carrying the given close-to-close returns.

    Each ticker's close path compounds its returns from a base level of 50, so
    the shipped ``log_return`` is exactly the adjusted close-to-close difference
    and the first row's implied predecessor level matches the returns that
    follow it. Fixed open/high/low offsets keep the bar ordering.
    """
    log_returns = np.asarray(log_returns, dtype=np.float64)
    level = pd.Series(np.log(50.0) + log_returns).groupby(
        tickers, sort=False
    ).cumsum()
    close = np.exp(level.to_numpy())
    adj_open = close * 1.005
    return {
        "_adj_open": adj_open,
        "_adj_high": np.maximum(adj_open, close) * 1.02,
        "_adj_low": np.minimum(adj_open, close) * 0.98,
        "adj_close": close,
    }


def _write_dataset(
    directory: Path,
    tickers: np.ndarray,
    dates: np.ndarray,
    log_returns: np.ndarray,
) -> Path:
    """Write a HAR-shaped dataset parquet inside ``directory``."""
    path = directory / "dataset.parquet"
    pd.DataFrame(
        {
            "ticker": tickers,
            "date": pd.to_datetime(dates),
            "log_return": log_returns,
            **_synthetic_ohlc(tickers, log_returns),
        }
    ).to_parquet(path)
    return path


class _ConstantModel:
    """Picklable stand-in fold model that predicts a constant log target."""

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), -2.5, dtype=np.float64)


def _run_synthetic_fold_path() -> dict:
    """Run ``phase_4_training`` on synthetic artifacts under a temporary root.

    The dataset parquet, model inputs, and fold metadata are synthetic and the
    real loader and HAR fit run; only the fold-model trainer is doubled. Two
    tickers span all 60 dates, and a third enters mid-timeline at the start of
    fold 1's test block, mirroring a late index entrant: its first 21 dates —
    warm-up rows — fall inside test blocks, while every fold's training prefix
    also reaches past the warm-up. The pipeline's own artifact locations are
    redirected, so nothing is written outside the temporary root.
    """
    from packages.ml.model.cross_validation import cv_utils
    from packages.ml.model.training import har_benchmark, training

    rng = np.random.default_rng(23)
    n_dates = FOLD_PATH_N_DATES
    tickers: list[str] = []
    date_indices: list[int] = []
    for date_index in range(n_dates):
        tickers.extend(("AAA", "BBB"))
        date_indices.extend((date_index, date_index))
        if date_index >= FOLD_PATH_ENTRANT_START:
            tickers.append("CCC")
            date_indices.append(date_index)
    tickers = np.array(tickers)
    date_indices = np.array(date_indices)
    dates = np.datetime64("2020-01-02") + date_indices
    n_rows = len(tickers)
    log_returns = rng.normal(scale=0.01, size=n_rows)
    targets = rng.uniform(0.05, 0.6, size=n_rows)
    X = rng.normal(size=(n_rows, 4))
    X[:, 0] = np.arange(n_rows, dtype=np.float64)

    metadata = {
        "ticker": tickers,
        "date": dates,
        "regime": (np.arange(n_rows) % 7).astype(np.int8),
        "n_indicators": 3,
        "n_ticker_features": 1,
    }
    folds_metadata = {}
    for fold_idx in range(5):
        train_end = n_dates // 2 + FOLD_PATH_TEST_DATES * fold_idx
        test_end = train_end + FOLD_PATH_TEST_DATES
        train_indices = np.flatnonzero(date_indices < train_end)
        test_indices = np.flatnonzero(
            (date_indices >= train_end) & (date_indices < test_end)
        )
        folds_metadata[f"fold_{fold_idx}"] = {
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "n_train": int(train_indices.size),
            "n_test": int(test_indices.size),
            "train_date_min": str(dates[train_indices[0]]),
            "train_date_max": str(dates[train_indices[-1]]),
            "test_date_min": str(dates[test_indices[0]]),
            "test_date_max": str(dates[test_indices[-1]]),
        }

    train_rows: list = []
    fit_calls: list = []
    fit_results: list = []

    def trainer(X_train, y_train, X_val=None, y_val=None):
        train_rows.append(np.asarray(X_train)[:, 0].astype(np.int64))
        return _ConstantModel(), {}

    real_fit = training.fit_har_benchmark

    def fit_spy(**kwargs):
        fit_calls.append(
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in kwargs.items()
            }
        )
        result = real_fit(**kwargs)
        fit_results.append(result)
        return result

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = root / "inputs"
        folds = root / "folds"
        inputs.mkdir(parents=True)
        folds.mkdir(parents=True)
        (root / "checkpoints").mkdir()
        (root / "results").mkdir()
        np.save(inputs / "X.npy", X)
        np.save(inputs / "Y.npy", targets)
        joblib.dump(metadata, inputs / "metadata.pkl")
        joblib.dump(folds_metadata, folds / "folds_metadata.pkl")
        # The live readers verify recorded hashes before loading; the synthetic
        # inputs must carry their registry entries like the real writers'.
        record_artifact(
            inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME
        )
        record_artifact(
            folds / "folds_metadata.pkl", registry_path=root / REGISTRY_FILENAME
        )
        dataset = _write_dataset(root, tickers, dates, log_returns)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
            stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
            stack.enter_context(
                mock.patch.object(har_benchmark, "DATASET_PATH", dataset)
            )
            stack.enter_context(
                mock.patch.object(training, "train_xgboost_model", trainer)
            )
            stack.enter_context(
                mock.patch.object(training, "fit_har_benchmark", fit_spy)
            )
            _, summary, _ = training.phase_4_training()
            har_report = json.loads(
                (root / "results" / "har_benchmark.json").read_text(encoding="utf-8")
            )

    return {
        "tickers": tickers,
        "date_indices": date_indices,
        "targets": targets,
        "folds_metadata": folds_metadata,
        "train_rows": train_rows,
        "fit_calls": fit_calls,
        "fit_results": fit_results,
        "summary": summary,
        "har_report": har_report,
    }


def test_date_dtype_normalization() -> None:
    from packages.ml.model.training import har_benchmark

    dates, tickers = _live_metadata()
    canonical = dates.astype("datetime64[ns]")

    # Unit-only differences must not trip the row alignment.
    unit_converted = canonical.astype("datetime64[s]")
    unit_components, unit_mask = har_benchmark.load_har_regressors(
        unit_converted, tickers
    )
    assert unit_components.shape == (
        len(canonical),
        len(har_benchmark.HAR_WINDOWS),
    )

    # A tz-aware object column carries the same instants in another
    # representation; it must not trip the row alignment either.
    tz_aware = pd.Series(canonical).dt.tz_localize("UTC").to_numpy()
    assert tz_aware.dtype == np.dtype("object")
    tz_components, tz_mask = har_benchmark.load_har_regressors(tz_aware, tickers)
    assert tz_components.shape == (len(canonical), len(har_benchmark.HAR_WINDOWS))
    assert np.isfinite(tz_components).all()
    assert np.array_equal(tz_mask, unit_mask)


def test_nat_raises_named_missing_date_error() -> None:
    from packages.ml.model.training import har_benchmark

    dates, tickers = _live_metadata()
    missing = dates.astype("datetime64[ns]").copy()
    missing[len(missing) // 2] = np.datetime64("NaT")

    with pytest.raises(ValueError) as excinfo:
        har_benchmark.load_har_regressors(missing, tickers)

    message = str(excinfo.value)
    assert "missing date" in message
    assert "NaT" in message
    assert "misaligned" not in message


def test_same_date_ticker_swap_raises_named_pair_mismatch() -> None:
    from packages.ml.model.training import har_benchmark

    tickers = np.array(["AAA", "BBB"])
    dates = np.array(["2020-01-02", "2020-01-02"], dtype="datetime64[ns]")

    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(
            Path(tmp), tickers, dates, np.array([0.01, -0.02])
        )
        with mock.patch.object(har_benchmark, "DATASET_PATH", dataset):
            # The date column alone still agrees row by row; only the swapped
            # ticker pairs make the metadata disagree with the parquet.
            with pytest.raises(ValueError) as excinfo:
                har_benchmark.load_har_regressors(dates, tickers[::-1])

    message = str(excinfo.value)
    assert "misaligned" in message
    assert "(ticker, date)" in message
    assert "row 0" in message
    assert "'AAA'" in message
    assert "'BBB'" in message


def test_non_monotonic_ticker_dates_raise_named_error() -> None:
    from packages.ml.model.training import har_benchmark

    tickers = np.array(["CCC", "CCC", "CCC"])
    dates = np.array(
        ["2020-01-02", "2020-01-06", "2020-01-03"], dtype="datetime64[ns]"
    )

    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(
            Path(tmp), tickers, dates, np.array([0.01, -0.02, 0.03])
        )
        with mock.patch.object(har_benchmark, "DATASET_PATH", dataset):
            with pytest.raises(ValueError) as excinfo:
                har_benchmark.load_har_regressors(dates, tickers)

    message = str(excinfo.value)
    assert "CCC" in message
    assert "strictly increasing" in message
    assert "position 2" in message


def test_repeated_ticker_date_raises_named_error() -> None:
    from packages.ml.model.training import har_benchmark

    tickers = np.array(["CCC", "CCC"])
    dates = np.array(["2020-01-02", "2020-01-02"], dtype="datetime64[ns]")

    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(
            Path(tmp), tickers, dates, np.array([0.01, -0.02])
        )
        with mock.patch.object(har_benchmark, "DATASET_PATH", dataset):
            # Strictly increasing means an exact repeat is a violation too.
            with pytest.raises(ValueError) as excinfo:
                har_benchmark.load_har_regressors(dates, tickers)

    message = str(excinfo.value)
    assert "CCC" in message
    assert "strictly increasing" in message
    assert "position 1" in message


def test_warmup_mask_covers_first_21_dates_per_ticker() -> None:
    from packages.ml.model.training import har_benchmark

    n_dates = 25
    tickers = np.array(["AAA", "BBB"] * n_dates)
    dates = np.repeat(np.datetime64("2020-01-02") + np.arange(n_dates), 2)
    returns = np.linspace(-0.04, 0.04, len(tickers))

    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(Path(tmp), tickers, dates, returns)
        with mock.patch.object(har_benchmark, "DATASET_PATH", dataset):
            components, warmup_mask = har_benchmark.load_har_regressors(
                dates, tickers
            )

    assert har_benchmark.HAR_WARMUP_DAYS == 21
    expected_mask = np.repeat(
        np.arange(n_dates) < har_benchmark.HAR_WARMUP_DAYS, 2
    )
    assert warmup_mask.dtype == bool
    assert np.array_equal(warmup_mask, expected_mask)
    assert int(warmup_mask.sum()) == 21 * 2

    # The mask leaves the components alone: every row keeps the trailing-only
    # same-estimator math, warm-up rows included. The n = 1 column is the
    # closed form overnight_t**2 + rs_t (the n -> 1 limit), which pins that the
    # rebuilt components are live on warm-up rows too. Earlier rows use the
    # shipped log_return for the missing predecessor close, exactly like the
    # loader; every other bar uses its predecessor directly.
    bars = _synthetic_ohlc(tickers, returns)
    previous = (
        pd.Series(bars["adj_close"])
        .groupby(tickers, sort=False)
        .shift(1)
        .to_numpy()
    )
    overnight = np.log(bars["_adj_open"] / previous)
    missing = np.isnan(previous)
    overnight[missing] = (
        np.log(bars["_adj_open"][missing] / bars["adj_close"][missing])
        + returns[missing]
    )
    rogers_satchell = (
        np.log(bars["_adj_high"] / bars["adj_close"])
        * np.log(bars["_adj_high"] / bars["_adj_open"])
        + np.log(bars["_adj_low"] / bars["adj_close"])
        * np.log(bars["_adj_low"] / bars["_adj_open"])
    )
    assert np.isfinite(components).all()
    np.testing.assert_allclose(
        components[:, 0], overnight**2 + rogers_satchell
    )


def test_fold_path_excludes_warmup_rows_and_reconciles_counts() -> None:
    from packages.ml.model.training import har_benchmark, training

    evidence = _run_synthetic_fold_path()
    tickers = evidence["tickers"]
    date_indices = evidence["date_indices"]
    dates = np.datetime64("2020-01-02") + date_indices
    date_pos = np.unique(dates, return_inverse=True)[1]
    warmup_mask = np.array(
        [
            date_index < (FOLD_PATH_ENTRANT_START + har_benchmark.HAR_WARMUP_DAYS)
            if ticker == "CCC"
            else date_index < har_benchmark.HAR_WARMUP_DAYS
            for ticker, date_index in zip(tickers, date_indices)
        ]
    )
    # Layout-pinned counts: warm-up rows in each fold's training prefix and in
    # each fold's test block (fold 0 has no entrant rows in its test block).
    expected_train_warmup = (42, 42, 48, 54, 60)
    expected_test_warmup = (0, 6, 6, 6, 3)

    y_log = np.log(evidence["targets"])
    for fold_idx in range(5):
        fold_key = f"fold_{fold_idx}"
        fold_data = evidence["folds_metadata"][fold_key]
        train_indices = np.asarray(fold_data["train_indices"])
        test_indices = np.asarray(fold_data["test_indices"])
        expected_fit = train_indices[~warmup_mask[train_indices]]
        expected_excluded = int(np.count_nonzero(warmup_mask[train_indices]))
        expected_warmup_test_rows = int(np.count_nonzero(warmup_mask[test_indices]))
        assert expected_excluded == expected_train_warmup[fold_idx]
        assert expected_warmup_test_rows == expected_test_warmup[fold_idx]
        assert len(expected_fit) < len(train_indices)

        call = evidence["fit_calls"][fold_idx]
        result = evidence["fit_results"][fold_idx]

        # The fit receives the fold's floor-retained training rows and the
        # loader's warm-up mask; it excludes the warm-up rows itself and the
        # count reconciles with the training-set delta.
        assert np.array_equal(call["train_indices"], train_indices)
        assert np.array_equal(call["warmup_mask"], warmup_mask)
        assert result["n_warmup_excluded"] == expected_excluded
        assert result["n_train"] == len(expected_fit)
        assert (
            result["n_warmup_excluded"]
            == len(call["train_indices"]) - result["n_train"]
        )
        assert result["warmup_days"] == har_benchmark.HAR_WARMUP_DAYS

        # Scoring rows are untouched: the mask never filters the test set, and
        # the entrant's warm-up rows inside a test block are scored unchanged.
        assert np.array_equal(call["test_indices"], test_indices)
        assert result["n_warmup_test_rows"] == expected_warmup_test_rows
        assert result["n_test"] == len(test_indices)

        # The fold model fits the inner-fit rows (fold-train minus the purged
        # inner-validation tail), still a subset of the floor-retained train.
        expected_inner_fit, _, _ = training.inner_validation_split(
            fold_idx, train_indices, date_pos, evidence["targets"]
        )
        assert np.array_equal(evidence["train_rows"][fold_idx], expected_inner_fit)

        # The fold summary carries the count next to the floor counts.
        assert evidence["summary"][fold_key]["n_warmup_excluded"] == expected_excluded

        # The fit actually used the post-warm-up rows: the coefficients match a
        # manual OLS on exactly those rows.
        regressors = call["regressors"]
        design = np.column_stack(
            [np.ones(len(expected_fit)), regressors[expected_fit]]
        )
        expected_coefficients = np.linalg.lstsq(
            design, np.exp(2.0 * y_log[expected_fit]), rcond=None
        )[0]
        assert np.allclose(result["coefficients"], expected_coefficients)

    assert evidence["har_report"]["warmup_days"] == har_benchmark.HAR_WARMUP_DAYS


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
