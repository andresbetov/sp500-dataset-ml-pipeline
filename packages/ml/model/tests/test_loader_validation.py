"""Tests for fold-summary schema validation and deterministic loading.

The loader owns the visualization boundary: fold keys must be named
``fold_<n>``, every fold must carry the columns the plots read, and unexpected
metric keys are warned about but retained. Folds are ordered by index so a
summary written in a different key order renders in the same order.

Every exercised summary is a temporary JSON fixture; the live summary is only
read by the skip-guarded check.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR

from packages.ml.model.visualization.data_loader import load_fold_training_summary



def _fold(index: int, **overrides) -> dict:
    """One valid fold record with every required plot column."""
    fold = {
        "train_date_min": f"200{index}-01-03",
        "train_date_max": f"200{index}-06-29",
        "test_date_min": f"200{index}-07-01",
        "test_date_max": f"200{index}-12-31",
        "n_train_samples": 1_000 + index,
        "n_test_samples": 200 + index,
        "test_r2": 0.10 + index * 0.01,
        "test_mae": 0.20 + index * 0.01,
        "test_rmse": 0.30 + index * 0.01,
        "train_time_seconds": 10.0 + index,
        "inference_time_seconds": 1.0 + index,
        "har_time_seconds": 2.0 + index,
        "fold_time_seconds": 13.0 + index,
        "n_features_indicators": 37,
        "n_features_ticker": 0,
    }
    fold.update(overrides)
    return fold


def _summary() -> dict:
    return {f"fold_{index}": _fold(index) for index in range(5)}


def _write_summary(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "fold_training_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("stray_key", ["baseline", "fold_x", "fold_"])
def test_malformed_fold_key_raises_value_error_naming_the_key(
    tmp_path: Path, stray_key: str
) -> None:
    payload = _summary()
    payload[stray_key] = _fold(9)
    summary_path = _write_summary(tmp_path, payload)

    with pytest.raises(ValueError) as excinfo:
        load_fold_training_summary(summary_path)

    assert stray_key in str(excinfo.value), str(excinfo.value)


def test_missing_required_column_raises_value_error_naming_fold_and_column(
    tmp_path: Path,
) -> None:
    payload = _summary()
    del payload["fold_1"]["test_r2"]
    summary_path = _write_summary(tmp_path, payload)

    with pytest.raises(ValueError) as excinfo:
        load_fold_training_summary(summary_path)

    message = str(excinfo.value)
    assert "fold_1" in message, message
    assert "test_r2" in message, message


def test_unexpected_metric_key_warns_and_is_retained(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    payload = _summary()
    payload["fold_0"]["baseline_schema"] = "unknown-but-retained"
    summary_path = _write_summary(tmp_path, payload)

    with caplog.at_level(
        logging.WARNING, logger="packages.ml.model.visualization.data_loader"
    ):
        frame = load_fold_training_summary(summary_path)

    assert "baseline_schema" in frame.columns
    assert (
        frame.loc[frame["fold"] == "fold_0", "baseline_schema"].iloc[0]
        == "unknown-but-retained"
    )
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("baseline_schema" in record.getMessage() for record in warnings), caplog.text


def test_reversed_key_order_yields_the_same_fold_sorted_frame(tmp_path: Path) -> None:
    payload = _summary()
    forward_path = _write_summary(tmp_path, payload)
    reverse_dir = tmp_path / "reversed"
    reverse_dir.mkdir()
    reverse_path = _write_summary(
        reverse_dir, {key: payload[key] for key in reversed(list(payload))}
    )

    forward = load_fold_training_summary(forward_path)
    reversed_frame = load_fold_training_summary(reverse_path)

    assert list(forward["fold"]) == [f"fold_{index}" for index in range(5)]
    assert list(reversed_frame["fold"]) == [f"fold_{index}" for index in range(5)]
    pd.testing.assert_frame_equal(forward, reversed_frame)


