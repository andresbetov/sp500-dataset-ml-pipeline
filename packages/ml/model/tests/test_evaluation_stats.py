"""Tests for the evaluation statistics and the strict evaluation summary.

The evaluation report consumes the persisted per-row loss vectors and emits one
strict-JSON ``results/evaluation_summary.json`` carrying:

- the dual OOS-R² conventions the training phase already emitted (consumed,
  not recomputed);
- per-fold and pooled QLIKE distributions (mean, median, IQR, top-1% tail mean
  and loss share);
- the floor with/without decomposition, driven only by the persisted
  ``is_floor`` column (the reporting layer never re-derives floor membership);
- decile QLIKE on global edges, regime QLIKE on the persisted metadata labels;
- the date-clustered gap (descriptive mean ± SE, ``n_dates``, ``n_valid``),
  with quarantined folds deterministically excluded;
- DM/CW/GW on the model-vs-HAR loss differentials with Newey-West HAC lag 4;
- Mincer-Zarnowitz regressions per fold and pooled with ``n_valid``;
- expanding-origin loss curves per fold; and the calibration stub, which states
  explicitly that no calibration is applied.

The fixtures are hand-computed: the tests recompute each statistic explicitly
instead of calling the implementation they check. No live artifact is written;
the strict-JSON writer writes only under a temporary directory.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_evaluation_stats.py
"""

from __future__ import annotations

import json
import math
import tempfile
import traceback
from pathlib import Path

import conftest  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest

from packages.ml.model.training import training

HAC_LAG = 4


def _report():
    """Import the report module inside the test so the RED state stays visible."""
    from packages.ml.model.evaluation import report as module

    return module


def _record(
    fold: str,
    row_idx,
    dates,
    tickers,
    y_true,
    y_pred_model,
    y_pred_har,
    is_floor,
) -> dict:
    return {
        "fold": fold,
        "row_idx": np.asarray(row_idx, dtype=np.int64),
        "date": np.asarray(dates),
        "ticker": np.asarray(tickers),
        "y_true": np.asarray(y_true, dtype=np.float64),
        "y_pred_model": np.asarray(y_pred_model, dtype=np.float64),
        "y_pred_har": np.asarray(y_pred_har, dtype=np.float64),
        "is_floor": np.asarray(is_floor, dtype=bool),
    }


def _two_fold_frame(report):
    """Two folds with distinct date sets; fold_1 is the quarantine probe."""
    fold_0 = _record(
        "fold_0",
        row_idx=[0, 1, 2, 3],
        dates=[
            np.datetime64("2000-01-03"),
            np.datetime64("2000-01-03"),
            np.datetime64("2000-01-04"),
            np.datetime64("2000-01-04"),
        ],
        tickers=["AAA", "BBB", "AAA", "BBB"],
        y_true=[-1.0, -1.5, -2.0, -2.5],
        y_pred_model=[-1.1, -1.4, -2.2, -2.4],
        y_pred_har=[-1.05, -1.45, -2.1, -2.45],
        is_floor=[False, False, False, False],
    )
    fold_1 = _record(
        "fold_1",
        row_idx=[4, 5, 6, 7],
        dates=[
            np.datetime64("2000-01-05"),
            np.datetime64("2000-01-05"),
            np.datetime64("2000-01-06"),
            np.datetime64("2000-01-06"),
        ],
        tickers=["AAA", "BBB", "AAA", "BBB"],
        y_true=[-3.0, -3.5, -4.0, -4.5],
        y_pred_model=[-2.8, -3.6, -4.2, -4.4],
        y_pred_har=[-3.1, -3.4, -3.9, -4.6],
        is_floor=[False, False, False, False],
    )
    return report.loss_vector_frame([fold_0, fold_1])


def test_mz_regression_identical_forecasts_is_unbiased() -> None:
    """Identical forecasts give alpha = 0 and beta = 1."""
    report = _report()
    y = np.array([-3.0, -2.0, -1.0, -0.5, -2.5])

    block = report.mz_regression(y, y.copy())

    assert block["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert block["beta"] == pytest.approx(1.0, abs=1e-12)
    assert block["n_obs"] == len(y)
    assert block["hac_lag"] == HAC_LAG


def test_mz_regression_biased_forecast_is_hand_computed() -> None:
    """A shifted forecast moves alpha, not beta; a scaled one moves beta."""
    report = _report()
    y = np.array([-3.0, -2.0, -1.0, -0.5, -2.5])

    shifted = report.mz_regression(y, y + 0.7)
    assert shifted["alpha"] == pytest.approx(-0.7, abs=1e-12)
    assert shifted["beta"] == pytest.approx(1.0, abs=1e-12)

    driver = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    target = 2.0 + 0.5 * driver
    scaled = report.mz_regression(target, driver)
    assert scaled["alpha"] == pytest.approx(2.0, abs=1e-12)
    assert scaled["beta"] == pytest.approx(0.5, abs=1e-12)


def test_dm_zero_differential_is_zero_and_hac_lag_is_four() -> None:
    """Equal losses give exactly DM = 0; the reported lag is horizon - 1 = 4."""
    report = _report()
    assert report.HAC_LAG == HAC_LAG
    losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dates = np.array(
        [np.datetime64("2000-01-03") + offset for offset in range(len(losses))]
    )

    block = report.dm_test(losses, losses.copy(), dates)

    assert block["statistic"] == 0.0
    assert block["p_value"] == pytest.approx(1.0)
    assert block["hac_lag"] == HAC_LAG
    assert block["n_dates"] == len(losses)


def test_dm_hac_se_matches_the_bartlett_sandwich_hand_computation() -> None:
    """The HAC standard error equals the explicit Newey-West computation."""
    report = _report()
    differential = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -2.0])
    dates = np.array(
        [
            np.datetime64("2000-01-03") + offset
            for offset in range(len(differential))
        ]
    )

    # Hand computation: OLS on a constant has residual = differential - mean,
    # so the Bartlett-weighted meat is sum(r^2) + 2 * sum_k w_k * sum(r_t
    # r_{t-k}), with w_k = 1 - k / (lag + 1), and cov = meat / n**2.
    n = len(differential)
    residual = differential - differential.mean()
    meat = float(np.sum(residual**2))
    for offset in range(1, HAC_LAG + 1):
        weight = 1.0 - offset / (HAC_LAG + 1)
        products = residual[offset:] * residual[:-offset]
        meat += 2.0 * weight * float(np.sum(products))
    expected_se = math.sqrt(meat / n**2)
    expected_statistic = float(differential.mean()) / expected_se

    block = report.dm_test(
        differential, np.zeros_like(differential), dates
    )

    assert block["mean_differential"] == pytest.approx(
        float(differential.mean()), rel=1e-12, abs=1e-12
    )
    assert block["hac_se"] == pytest.approx(expected_se, rel=1e-12, abs=1e-12)
    assert block["statistic"] == pytest.approx(
        expected_statistic, rel=1e-12, abs=1e-12
    )


def test_cw_and_gw_on_a_zero_differential() -> None:
    """Zero differentials give a zero CW/GW statistic; nesting stays unclaimed."""
    report = _report()
    y = np.array([-2.0, -1.5, -1.0, -0.5, -2.5])
    dates = np.array(
        [np.datetime64("2000-01-03") + offset for offset in range(len(y))]
    )
    losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    cw = report.cw_test(y, y.copy(), y.copy(), dates)
    assert cw["statistic"] == 0.0
    assert cw["hac_lag"] == HAC_LAG
    assert cw["nesting_claimed"] is False

    gw = report.gw_test(losses, losses.copy(), dates)
    assert gw["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert gw["hac_lag"] == HAC_LAG


def test_loss_distribution_is_hand_computed() -> None:
    """Median, IQR, and the top-1% mean and share match the hand computation."""
    report = _report()
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])

    block = report.loss_distribution(losses)

    assert block["n"] == 10
    assert block["mean"] == pytest.approx(14.5, rel=1e-12)
    assert block["median"] == pytest.approx(5.5, rel=1e-12)
    assert block["q25"] == pytest.approx(3.25, rel=1e-12)
    assert block["q75"] == pytest.approx(7.75, rel=1e-12)
    assert block["iqr"] == pytest.approx(4.5, rel=1e-12)
    # k = max(1, ceil(0.01 * 10)) = 1: the single largest loss is the tail.
    assert block["top1_mean"] == pytest.approx(100.0, rel=1e-12)
    assert block["top1_share"] == pytest.approx(
        100.0 / float(losses.sum()), rel=1e-12
    )


def test_deciles_are_ten_of_ten_on_a_shaped_fixture() -> None:
    """A well-shaped target fills all ten global-edge bins on every fold."""
    report = _report()
    y_true = np.arange(100, dtype=np.float64)
    loss_model = y_true / 100.0
    loss_har = y_true / 200.0

    edges = report.decile_edges(y_true)
    assert len(edges) == 11
    bins = report.decile_table(y_true, loss_model, loss_har, edges)

    assert len(bins) == 10
    assert [row["n_rows"] for row in bins] == [10] * 10
    assert [row["decile"] for row in bins] == list(range(10))
    # Bin 0 holds y_true 0..9, so its mean log target and QLIKE are known.
    assert bins[0]["mean_y_true_log"] == pytest.approx(4.5, rel=1e-12)
    assert bins[0]["qlike_model"] == pytest.approx(0.045, rel=1e-12)
    assert bins[-1]["edge_high"] == pytest.approx(99.0, rel=1e-12)


def test_regime_groups_use_the_metadata_labels() -> None:
    """The regime table follows the persisted integer buckets 0..6."""
    report = _report()
    regime = np.array([0, 0, 1, 6])
    loss_model = np.array([1.0, 2.0, 3.0, 4.0])
    loss_har = np.array([0.5, 0.5, 0.5, 0.5])

    table = report.regime_table(regime, loss_model, loss_har)

    assert [row["regime"] for row in table] == list(range(7))
    assert table[0]["n_rows"] == 2
    assert table[0]["qlike_model"] == pytest.approx(1.5, rel=1e-12)
    assert table[1]["n_rows"] == 1
    assert table[6]["n_rows"] == 1
    assert table[2]["n_rows"] == 0
    assert table[2]["qlike_model"] is None


def test_floor_decomposition_follows_the_persisted_mask() -> None:
    """With/without tables follow ``is_floor`` even when it disagrees with y_true."""
    report = _report()
    floor_log = float(np.log(training.RV_FLOOR))
    # Row 2 is a genuine floor row; row 3 is flagged as floor while its target
    # is ordinary. A re-derivation from y_true would swap the two, so the
    # decomposition's counts prove the persisted mask is consumed.
    frame = report.loss_vector_frame(
        [
            _record(
                "fold_0",
                row_idx=[0, 1, 2, 3],
                dates=[np.datetime64("2000-01-03")] * 4,
                tickers=["AAA"] * 4,
                y_true=[-1.0, -2.0, floor_log, -1.0],
                y_pred_model=[-1.0, -1.0, -1.0, -1.0],
                y_pred_har=[-1.0, -1.0, -1.0, -1.0],
                is_floor=[False, False, True, True],
            )
        ]
    )

    block = report.floor_decomposition(frame, total_floor_rows=10)

    assert block["counts"]["n_floor_rows_total"] == 10
    assert block["counts"]["n_floor_rows_in_test_blocks"] == 2
    assert block["counts"]["n_floor_rows_never_evaluated"] == 8
    assert block["n_rows"] == 4
    assert block["n_floor_rows"] == 2
    assert block["with_floor"]["n_rows"] == 4
    assert block["without_floor"]["n_rows"] == 2

    losses_model = frame["loss_model"].to_numpy()
    losses_har = frame["loss_har"].to_numpy()
    retained = ~frame["is_floor"].to_numpy()
    assert block["without_floor"]["model"]["mean"] == pytest.approx(
        float(losses_model[retained].mean()), rel=1e-12
    )
    assert block["with_floor"]["model"]["mean"] == pytest.approx(
        float(losses_model.mean()), rel=1e-12
    )
    assert block["with_floor"]["har"]["mean"] == pytest.approx(
        float(losses_har.mean()), rel=1e-12
    )
    # k = max(1, ceil(0.01 * 4)) = 1 on both sides.
    assert block["with_floor"]["model"]["top1_mean"] == pytest.approx(
        float(losses_model.max()), rel=1e-12
    )
    assert block["with_floor"]["model"]["top1_share"] == pytest.approx(
        float(losses_model.max() / losses_model.sum()), rel=1e-12
    )
    assert block["without_floor"]["model"]["top1_mean"] == pytest.approx(
        float(losses_model[retained].max()), rel=1e-12
    )


def test_date_clustered_gap_is_hand_computed() -> None:
    """Equal-weight per-date means reproduce mean ± SE, n_dates, n_rows."""
    report = _report()
    dates = np.array(
        [
            np.datetime64("2000-01-03"),
            np.datetime64("2000-01-03"),
            np.datetime64("2000-01-04"),
            np.datetime64("2000-01-04"),
        ]
    )
    loss_model = np.array([1.0, 3.0, 4.0, 6.0])
    loss_har = np.zeros(4)

    block = report.date_clustered_gap(loss_model, loss_har, dates)

    # Per-date means are 2 and 5: mean 3.5, sample sd 3/sqrt(2), SE over two
    # dates = (3/sqrt(2))/sqrt(2) = 1.5.
    assert block["mean"] == pytest.approx(3.5, rel=1e-12)
    assert block["se"] == pytest.approx(1.5, rel=1e-12)
    assert block["n_dates"] == 2
    assert block["n_rows"] == 4

    constant = report.date_clustered_gap(
        np.array([2.0, 2.0, 2.0]),
        np.zeros(3),
        np.array(
            [
                np.datetime64("2000-01-03"),
                np.datetime64("2000-01-04"),
                np.datetime64("2000-01-05"),
            ]
        ),
    )
    assert constant["se"] == 0.0


def test_builder_excludes_quarantined_folds_from_gap_and_tests() -> None:
    """A quarantined fold is visibly excluded from the gap/SE/DM/MZ inputs."""
    report = _report()
    frame = _two_fold_frame(report)
    regime = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    dual_oos_r2 = {
        "folds": {
            "fold_0": {"oos_r2_prevailing_mean": 0.50, "oos_r2_vs_har": 0.25},
            "fold_1": {"oos_r2_prevailing_mean": 0.40, "oos_r2_vs_har": 0.20},
        },
        "pooled": {
            "oos_r2_prevailing_mean": 0.45,
            "oos_r2_vs_har": 0.22,
            "n_scored_rows": 8,
        },
    }

    payload = report.build_evaluation_summary(
        frame,
        dual_oos_r2=dual_oos_r2,
        regime=regime,
        headline_row_set=training.HEADLINE_ROW_SET,
        total_floor_rows=0,
        quarantined_folds={"fold_1": {"error": "ValueError: injected"}},
    )

    gap = payload["date_clustered_gap"]
    assert gap["excluded_folds"] == ["fold_1"]
    assert gap["pooled"]["n_valid"] == 1
    assert gap["pooled"]["n_total"] == 2
    assert set(gap["folds"]) == {"fold_0"}
    fold_0_gap = report.date_clustered_gap(
        frame.loc[frame["fold"] == "fold_0", "loss_model"].to_numpy(),
        frame.loc[frame["fold"] == "fold_0", "loss_har"].to_numpy(),
        frame.loc[frame["fold"] == "fold_0", "date"].to_numpy(),
    )
    assert gap["pooled"]["mean"] == pytest.approx(fold_0_gap["mean"], rel=1e-12)
    assert gap["pooled"]["se"] == pytest.approx(fold_0_gap["se"], rel=1e-12)

    assert set(payload["dm_cw_gw"]["folds"]) == {"fold_0"}
    assert set(payload["mz"]["folds"]) == {"fold_0"}
    assert payload["loss_distributions"]["folds"]["fold_1"]["quarantined"] is True
    assert payload["loss_distributions"]["pooled"]["n_valid_folds"] == 1

    # The dual OOS-R² conventions are consumed, not recomputed.
    assert payload["dual_oos_r2"]["folds"]["fold_0"] == {
        "oos_r2_prevailing_mean": 0.50,
        "oos_r2_vs_har": 0.25,
        "n_scored_rows": 4,
    }
    assert payload["dual_oos_r2"]["pooled"]["n_scored_rows"] == 8
    assert payload["dm_cw_gw"]["hac_lag"] == HAC_LAG
    assert payload["calibration"]["status"] == "not applied"
    assert payload["calibration"]["applied"] is False
    assert len(payload["decile_qlike"]["folds"]["fold_0"]["bins"]) == 10
    assert len(payload["regime_qlike"]["folds"]["fold_0"]["bins"]) == 7
    assert payload["expanding_origin"]["folds"]["fold_1"]["n_points"] == 4


def test_expanding_origin_curve_is_hand_computed() -> None:
    """The forward-origin curve is the expanding mean in date order."""
    report = _report()
    dates = np.array(
        [
            np.datetime64("2000-01-05"),
            np.datetime64("2000-01-03"),
            np.datetime64("2000-01-03"),
            np.datetime64("2000-01-04"),
        ]
    )
    loss_model = np.array([6.0, 1.0, 3.0, 4.0])
    loss_har = np.array([0.0, 0.0, 0.0, 0.0])

    curve = report.expanding_origin_curve(dates, loss_model, loss_har)

    # Sorted by date (stable within a date): [1, 3] then [4] then [6].
    assert curve["n"] == 4
    assert np.allclose(
        curve["loss_model"], [1.0, 2.0, 8.0 / 3.0, 3.5], rtol=0.0, atol=1e-12
    )
    assert np.allclose(
        curve["edge"], [1.0, 2.0, 8.0 / 3.0, 3.5], rtol=0.0, atol=1e-12
    )

    block = report.expanding_origin_block(dates, loss_model, loss_har)
    assert block["n_points"] == 4
    assert block["final"]["loss_model"] == pytest.approx(3.5, rel=1e-12)


def test_calibration_stub_declares_not_applied() -> None:
    """The stub is explicit: no calibration is applied, no fake factor ships."""
    report = _report()
    block = report.calibration_stub()

    assert report.CALIBRATION_STATUS == "not applied"
    assert block["applied"] is False
    assert block["status"] == "not applied"
    assert "not applied" in block["note"].lower()
    assert block["interface"]


def test_strict_summary_is_written_with_nulls_and_no_tmp_residue() -> None:
    """The summary writer sanitizes non-finite values and writes atomically."""
    report = _report()
    frame = _two_fold_frame(report)
    payload = report.build_evaluation_summary(
        frame,
        dual_oos_r2={
            "folds": {
                "fold_0": {
                    "oos_r2_prevailing_mean": float("nan"),
                    "oos_r2_vs_har": 0.25,
                }
            },
            "pooled": {"oos_r2_prevailing_mean": 0.45, "oos_r2_vs_har": 0.22},
        },
        regime=np.zeros(8, dtype=np.int8),
        headline_row_set=training.HEADLINE_ROW_SET,
        total_floor_rows=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results" / "evaluation_summary.json"
        report.write_evaluation_summary(payload, path)

        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text
        assert not list(Path(tmpdir).rglob("*.tmp"))

        def reject(token: str):
            raise AssertionError(f"non-strict JSON constant emitted: {token}")

        document = json.loads(text, parse_constant=reject)
        assert document["dual_oos_r2"]["folds"]["fold_0"][
            "oos_r2_prevailing_mean"
        ] is None
        assert document["calibration"]["applied"] is False


def test_report_module_does_not_rederive_floor_membership() -> None:
    """The reporting layer consumes ``is_floor``; it never compares to the floor."""
    report = _report()
    source = Path(report.__file__).read_text(encoding="utf-8")

    assert "RV_FLOOR" not in source
    assert "floor_retained_mask" not in source
    assert "log(max(" not in source


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
