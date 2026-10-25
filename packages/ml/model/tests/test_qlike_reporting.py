"""Tests for the QLIKE headline reporting convention.

The promoted headline is the **unweighted mean of the 5 fold metrics**. The
reporting contract has to make that explicit instead of leaving a bare
``GLOBAL`` number in the log:

- ``headline.convention`` states the unweighted fold-mean convention and
  ``headline.weighting`` says each fold contributes equally regardless of
  ``n_test``;
- ``headline.edge_definition`` states ``har_qlike - model_qlike`` (positive =
  model lower loss);
- ``headline.caveat`` records the single-seed limitation;
- ``headline.folds[*].label`` marks each fold as one of five (fold_0 is not
  the headline);
- ``headline.pooled`` is the row-weighted mean over the persisted per-row
  loss vectors, never an average of per-fold ratios;
- the phase log promotes the aggregate and carries the convention and caveat.

These checks use the shared five-fold synthetic phase fixture and a direct
five-fold vector fixture with unequal fold sizes, so the equal-weight
aggregate and the row-weighted pooled value cannot coincide by accident. No
live artifact is written.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_qlike_reporting.py
"""

from __future__ import annotations

import logging
import traceback

import conftest  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest

from packages.ml.model.evaluation import report
from packages.ml.model.training import training
from test_loss_vectors import _run_phase

CONVENTION = "unweighted mean of the 5 fold metrics"
FOLD_SIZES = {
    "fold_0": 4,
    "fold_1": 5,
    "fold_2": 6,
    "fold_3": 7,
    "fold_4": 8,
}


def _five_fold_frame() -> tuple[dict, object]:
    """A strict five-fold summary with unequal fold sizes and one floor row.

    The row-weighted pooled mean and the equal-weight fold mean differ because
    the folds have different retained-row counts; the floor row would move the
    pooled mean if it leaked in, so the fixture also pins the floor exclusion.
    """
    records = []
    start = 0
    for fold_key, size in FOLD_SIZES.items():
        row_idx = np.arange(start, start + size, dtype=np.int64)
        y_true = -1.0 - 0.1 * row_idx.astype(np.float64)
        y_pred_model = y_true + 0.05 * np.sin(row_idx)
        y_pred_har = y_true + 0.20 * np.cos(row_idx)
        is_floor = np.zeros(size, dtype=bool)
        if fold_key == "fold_4":
            # One persisted at-or-below-floor row: its loss is huge relative to
            # the headline rows and must stay out of every pooled value.
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
    """Pooled values are means of the persisted vectors, not fold-mean ratios."""
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

    # The fixture's unequal fold sizes make the row-weighted pooled value
    # differ from the equal-weight aggregate. A ratio-averaging bug would
    # collapse them onto the same number.
    aggregate = headline["aggregate"]
    assert pooled["model_qlike"] != pytest.approx(
        aggregate["model_qlike"], rel=0.0, abs=1e-9
    )

    # The aggregate is the plain mean of the five fold means, each computed on
    # the floor-retained rows.
    fold_means = [
        float(
            retained.loc[retained["fold"] == fold_key, "loss_model"].mean()
        )
        for fold_key in sorted(FOLD_SIZES)
    ]
    assert aggregate["model_qlike"] == pytest.approx(
        float(np.mean(fold_means)), rel=0.0, abs=1e-12
    )


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                if "caplog" in _fn.__code__.co_varnames:
                    continue
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
