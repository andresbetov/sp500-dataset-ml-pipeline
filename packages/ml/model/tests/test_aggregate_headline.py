"""Tests for the promoted aggregate QLIKE headline.

The pipeline used to promote a ``GLOBAL`` fold mean in the log without stating
its weighting convention, and never persisted the aggregate as the headline.
The headline is now the unweighted mean of the five fold metrics, emitted in
``results/evaluation_summary.json`` under ``headline`` (outside the five-row
``fold_training_summary.json`` contract):

- ``aggregate`` — the unweighted mean of the per-fold model QLIKE and HAR
  QLIKE, and their edge (``har_qlike - model_qlike``);
- ``folds`` — per-fold model/HAR QLIKE, edge, and ``n_test`` counts;
- ``pooled`` — the row-weighted mean over the persisted per-row loss vectors,
  kept distinct from the equal-weight aggregate;
- ``n_valid``/``n_total``, the convention string, and the single-seed caveat.

These checks pin:

- a real five-fold ``phase_4_training`` run over synthetic artifacts: the
  persisted aggregate equals the mean of the five fold metrics within ``1e-6``;
- every per-fold edge equals ``har_qlike - test_qlike`` and every ``n_test``
  matches the fold summary;
- the five-row fold summary keeps exactly its five fold keys and no aggregate
  row (the loader contract);
- a read-only live check of the regenerated evaluation summary when present.

No live artifact is written: the phase checks drive a temporary artifact root
through the shared synthetic fixture in ``test_loss_vectors``.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_aggregate_headline.py
"""

from __future__ import annotations

import json
import traceback

import conftest  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest

from conftest import ARTIFACTS_DIR
from test_loss_vectors import _run_phase

LIVE_EVALUATION = ARTIFACTS_DIR / "results" / "evaluation_summary.json"
LIVE_SUMMARY = ARTIFACTS_DIR / "results" / "fold_training_summary.json"

AGGREGATE_KEYS = {"model_qlike", "har_qlike", "edge"}


def _fold_keys(summary: dict) -> list[str]:
    return sorted(key for key in summary if key.startswith("fold_"))


def _live_payloads() -> tuple[dict, dict]:
    """The live evaluation summary and fold summary, read-only.

    The evaluation summary only exists after an artifact regeneration on the
    branch that emits the headline; the check skips on an older checkout
    instead of failing on missing generated output.
    """
    if not (LIVE_EVALUATION.is_file() and LIVE_SUMMARY.is_file()):
        pytest.skip(
            "requires the live evaluation summary and fold summary; "
            "regenerate the model artifacts to run this check"
        )
    evaluation = json.loads(LIVE_EVALUATION.read_text(encoding="utf-8"))
    summary = json.loads(LIVE_SUMMARY.read_text(encoding="utf-8"))
    return evaluation, summary


def test_persisted_headline_matches_the_fold_metric_mean() -> None:
    """The aggregate is the unweighted mean of the five fold metrics."""
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


def test_per_fold_edges_and_n_test_match_the_fold_summary() -> None:
    """Every fold edge is ``har_qlike - test_qlike``; ``n_test`` reconciles."""
    evidence = _run_phase()
    headline = evidence["evaluation_summary"]["headline"]
    summary = evidence["summary_json"]
    for fold_key in _fold_keys(summary):
        block = headline["folds"][fold_key]
        assert block["model_qlike"] == pytest.approx(
            summary[fold_key]["test_qlike"], rel=1e-12, abs=1e-12
        )
        assert block["har_qlike"] == pytest.approx(
            summary[fold_key]["har_qlike"], rel=1e-12, abs=1e-12
        )
        assert block["edge"] == pytest.approx(
            summary[fold_key]["har_qlike"] - summary[fold_key]["test_qlike"],
            rel=1e-12,
            abs=1e-12,
        )
        assert block["n_test"] == summary[fold_key]["n_test_samples"]
        assert block["n_test"] == summary[fold_key]["n_test_scored"]
        assert block["n_test_total"] == summary[fold_key]["n_test_total"]


def test_fold_summary_keeps_exactly_five_rows_and_no_aggregate() -> None:
    """The loader contract is exactly five ``fold_*`` rows, never a sixth."""
    evidence = _run_phase()
    summary = evidence["summary_json"]
    assert len(summary) == 5
    assert set(summary) == {f"fold_{index}" for index in range(5)}
    for forbidden in ("aggregate", "headline", "global", "GLOBAL"):
        assert forbidden not in summary
    for row in summary.values():
        assert "aggregate" not in row
        assert "headline" not in row

    if LIVE_SUMMARY.is_file():
        live = json.loads(LIVE_SUMMARY.read_text(encoding="utf-8"))
        assert len(live) == 5
        assert all(key.startswith("fold_") for key in live)


def test_live_headline_matches_the_live_fold_metrics() -> None:
    """Read-only: the regenerated live aggregate equals the fold-metric mean."""
    evaluation, summary = _live_payloads()
    assert "headline" in evaluation, (
        "the live evaluation summary predates the headline emission; "
        f"keys: {sorted(evaluation)}"
    )
    headline = evaluation["headline"]
    fold_keys = _fold_keys(summary)
    expected_model = float(
        np.mean([summary[fold_key]["test_qlike"] for fold_key in fold_keys])
    )
    expected_har = float(
        np.mean([summary[fold_key]["har_qlike"] for fold_key in fold_keys])
    )
    assert headline["aggregate"]["model_qlike"] == pytest.approx(
        expected_model, rel=0.0, abs=1e-6
    )
    assert headline["aggregate"]["har_qlike"] == pytest.approx(
        expected_har, rel=0.0, abs=1e-6
    )
    assert headline["aggregate"]["edge"] == pytest.approx(
        expected_har - expected_model, rel=0.0, abs=1e-6
    )
    for fold_key in fold_keys:
        block = headline["folds"][fold_key]
        assert block["edge"] == pytest.approx(
            summary[fold_key]["har_qlike"] - summary[fold_key]["test_qlike"],
            rel=1e-12,
            abs=1e-12,
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
