from __future__ import annotations

import conftest  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest

from packages.ml.model.evaluation import report
from packages.ml.model.training import training
from test_loss_vectors import _run_phase

FOLD_SIZES = {
    "fold_0": 4,
    "fold_1": 5,
    "fold_2": 6,
    "fold_3": 7,
    "fold_4": 8,
}

AGGREGATE_KEYS = {"model_qlike", "har_qlike", "edge"}


def _fold_keys(summary: dict) -> list[str]:
    return sorted(key for key in summary if key.startswith("fold_"))


def _five_fold_frame() -> tuple[dict, object]:
    records = []
    start = 0
    for fold_key, size in FOLD_SIZES.items():
        row_idx = np.arange(start, start + size, dtype=np.int64)
        y_true = -1.0 - 0.1 * row_idx.astype(np.float64)
        y_pred_model = y_true + 0.05 * np.sin(row_idx)
        y_pred_har = y_true + 0.20 * np.cos(row_idx)
        is_floor = np.zeros(size, dtype=bool)
        if fold_key == "fold_4":
            is_floor[0] = True
            y_true[0] = -5.0
            y_pred_model[0] = -1.0
            y_pred_har[0] = -1.0
        records.append(
            {
                "fold": fold_key,
                "row_idx": row_idx,
                "date": np.datetime64("2000-01-03") + row_idx,
                "ticker": np.array([f"T{index % 3}" for index in row_idx]),
                "y_true": y_true,
                "y_pred_model": y_pred_model,
                "y_pred_har": y_pred_har,
                "is_floor": is_floor,
            }
        )
        start += size

    vectors = report.loss_vector_frame(records)
    payload = report.build_evaluation_summary(
        vectors,
        dual_oos_r2={},
        regime=np.zeros(start, dtype=np.int8),
        headline_row_set=training.HEADLINE_ROW_SET,
        total_floor_rows=1,
    )
    return payload, vectors


def test_pooled_values_are_row_weighted_from_the_vectors() -> None:
    payload, vectors = _five_fold_frame()
    headline = payload["headline"]
    retained = vectors[~vectors["is_floor"].to_numpy(dtype=bool)]
    pooled = headline["pooled"]

    assert pooled["n_rows"] == int(len(retained))
    assert pooled["model_qlike"] == pytest.approx(
        float(retained["loss_model"].mean()), rel=0.0, abs=1e-12
    )
    assert pooled["har_qlike"] == pytest.approx(
        float(retained["loss_har"].mean()), rel=0.0, abs=1e-12
    )
    assert pooled["edge"] == pytest.approx(
        float(retained["loss_har"].mean())
        - float(retained["loss_model"].mean()),
        rel=0.0,
        abs=1e-12,
    )
    assert "not an average" in pooled["weighting"].lower()

    aggregate = headline["aggregate"]
    assert pooled["model_qlike"] != pytest.approx(
        aggregate["model_qlike"], rel=0.0, abs=1e-9
    )

    fold_means = [
        float(
            retained.loc[retained["fold"] == fold_key, "loss_model"].mean()
        )
        for fold_key in sorted(FOLD_SIZES)
    ]
    assert aggregate["model_qlike"] == pytest.approx(
        float(np.mean(fold_means)), rel=0.0, abs=1e-12
    )


def test_persisted_headline_matches_the_fold_metric_mean() -> None:
    evidence = _run_phase()
    payload = evidence["evaluation_summary"]
    assert payload is not None, (
        "phase_4_training did not write results/evaluation_summary.json"
    )
    assert "headline" in payload, (
        "the aggregate headline is absent from evaluation_summary.json; "
        f"keys: {sorted(payload)}"
    )
    headline = payload["headline"]
    summary = evidence["summary_json"]
    fold_keys = _fold_keys(summary)
    assert len(fold_keys) == 5
    assert set(headline["folds"]) == set(fold_keys)

    expected_model = float(
        np.mean([summary[fold_key]["test_qlike"] for fold_key in fold_keys])
    )
    expected_har = float(
        np.mean([summary[fold_key]["har_qlike"] for fold_key in fold_keys])
    )
    aggregate = headline["aggregate"]
    assert set(aggregate) >= AGGREGATE_KEYS
    assert aggregate["model_qlike"] == pytest.approx(
        expected_model, rel=0.0, abs=1e-6
    )
    assert aggregate["har_qlike"] == pytest.approx(
        expected_har, rel=0.0, abs=1e-6
    )
    assert aggregate["edge"] == pytest.approx(
        expected_har - expected_model, rel=0.0, abs=1e-6
    )

    assert headline["n_valid"] == 5
    assert headline["n_total"] == 5
    assert headline["excluded_folds"] == []


def test_fold_summary_keeps_exactly_five_rows_and_no_aggregate() -> None:
    evidence = _run_phase()
    summary = evidence["summary_json"]
    assert len(summary) == 5
    assert set(summary) == {f"fold_{index}" for index in range(5)}
    for forbidden in ("aggregate", "headline", "global", "GLOBAL"):
        assert forbidden not in summary
    for row in summary.values():
        assert "aggregate" not in row
        assert "headline" not in row
