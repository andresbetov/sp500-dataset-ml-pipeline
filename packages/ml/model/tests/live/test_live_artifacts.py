from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import ARTIFACTS_DIR, DATASET_PATH, REPO_ROOT  # noqa: E402
from conftest import requires_live_artifacts  # noqa: E402

LIVE_SUMMARY = ARTIFACTS_DIR / "results" / "fold_training_summary.json"
LIVE_HAR_REPORT = ARTIFACTS_DIR / "results" / "har_benchmark.json"


def test_live_metrics_have_no_nan_today() -> None:
    """Read-only: the stored live metrics are strict JSON with finite values."""
    requires_live_artifacts(ARTIFACTS_DIR)
    if not (LIVE_SUMMARY.is_file() and LIVE_HAR_REPORT.is_file()):
        pytest.skip("requires the stored live fold summary and HAR report")

    summary = json.loads(LIVE_SUMMARY.read_text(encoding="utf-8"))
    report = json.loads(LIVE_HAR_REPORT.read_text(encoding="utf-8"))

    invalid = []
    for fold_key, row in summary.items():
        for key in ("test_r2", "test_mae", "test_rmse", "test_qlike"):
            value = row.get(key)
            if value is None or not math.isfinite(value):
                invalid.append(f"{fold_key}.{key}")
    for fold_key, row in report["folds"].items():
        for key in ("har_r2", "model_r2", "model_qlike"):
            value = row.get(key)
            if value is None or not math.isfinite(value):
                invalid.append(f"har_report.{fold_key}.{key}")
    assert not invalid, f"live metrics carry invalid values today: {invalid}"


def test_live_har_components_are_finite_and_windowed() -> None:
    """Live HAR components stay finite on the generated dataset."""
    from packages.ml.model.training import har_benchmark

    if not DATASET_PATH.is_file():
        pytest.skip(f"requires the generated dataset absent: {DATASET_PATH}")
    assert har_benchmark.HAR_WINDOWS == (1, 5, 22)

    columns = pd.read_parquet(DATASET_PATH, columns=["ticker", "date"])
    components, warmup_mask = har_benchmark.load_har_regressors(
        columns["date"].to_numpy(), columns["ticker"].to_numpy()
    )
    assert components.shape == (len(columns), len(har_benchmark.HAR_WINDOWS))
    assert np.isfinite(components).all()
    assert warmup_mask.shape == (len(columns),)


def test_live_alignment_covers_the_first_21_dates_per_ticker() -> None:
    """The warm-up covers exactly the first 21 dates of every ticker."""
    from packages.ml.model.features import feature_engineering
    from packages.ml.model.training import har_benchmark

    if not DATASET_PATH.is_file():
        pytest.skip(f"requires the generated dataset absent: {DATASET_PATH}")
    frame = pd.read_parquet(DATASET_PATH)
    _, _, metadata = feature_engineering._extract_inputs(frame)
    components, warmup_mask = har_benchmark.load_har_regressors(
        metadata["date"], metadata["ticker"]
    )
    assert components.shape == (
        len(metadata["date"]),
        len(har_benchmark.HAR_WINDOWS),
    )
    assert np.isfinite(components).all()

    frame_tickers = frame["ticker"].to_numpy()
    n_tickers = len(pd.unique(frame_tickers))
    positions = pd.Series(np.arange(len(frame))).groupby(
        frame_tickers, sort=False
    ).cumcount()
    expected_mask = positions.to_numpy() < har_benchmark.HAR_WARMUP_DAYS
    assert np.array_equal(warmup_mask, expected_mask)
    assert int(warmup_mask.sum()) == n_tickers * har_benchmark.HAR_WARMUP_DAYS


def test_live_holdout_reservation_is_disjoint_and_purged() -> None:
    """Read-only: the persisted window matches the fold schema."""
    from packages.ml.model.cross_validation import cross_validator
    from packages.ml.model.evaluation import holdout

    requires_live_artifacts(ARTIFACTS_DIR)
    metadata_path = ARTIFACTS_DIR / "inputs" / "metadata.pkl"
    folds_path = ARTIFACTS_DIR / "folds" / "folds_metadata.pkl"
    y_path = ARTIFACTS_DIR / "inputs" / "Y.npy"
    if not (metadata_path.is_file() and folds_path.is_file() and y_path.is_file()):
        pytest.skip("live holdout reservation inputs are absent from this checkout")

    reservation = holdout.validate_holdout_reservation(
        joblib.load(metadata_path)["date"],
        np.load(y_path),
        joblib.load(folds_path),
    )
    assert reservation["n_holdout_dates"] == 500
    assert reservation["separation"] >= cross_validator.SEPARATION_MIN


def test_live_summary_loads_in_fold_order() -> None:
    """The live fold summary loads with five folds in index order."""
    from packages.ml.model.visualization.data_loader import (
        load_fold_training_summary,
    )

    if not LIVE_SUMMARY.is_file():
        pytest.skip("requires the live fold training summary absent here")
    frame = load_fold_training_summary(LIVE_SUMMARY)
    assert list(frame["fold"]) == [f"fold_{index}" for index in range(len(frame))]
    assert len(frame) == 5


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
