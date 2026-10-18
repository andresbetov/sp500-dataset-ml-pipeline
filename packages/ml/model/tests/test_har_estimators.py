"""Tests pinning the HAR components to the target's Yang-Zhang estimator family.

The volatility target is a Yang-Zhang window variance over adjusted OHLC bars
plus the prior adjusted close. These checks pin the benchmark's 1/5/22
trailing components to the **same** bar-level quantities and combination
formula, so the benchmark can no longer silently mix close-to-close squared
returns with a Yang-Zhang target:

- a tiny recorded OHLC frame (two tickers, seven bars each, values chosen so
  the Yang-Zhang and close-to-close builds separate) with the expected
  component vector computed by hand and hardcoded here, within ``1e-12``;
- the rebuild flips the components away from the old squared-close-to-close
  build (by more than ``REBUILD_DIFFERENCE_TOLERANCE``), and that old build
  fails the recorded vector;
- prefix sensitivity: mutating a future bar never changes an earlier row's
  components;
- the warm-up mask/counter from the alignment task is consumed unchanged: the
  same rows stay excluded from the fit and counted, with no duplicate logic;
- non-regression: the sigma<->variance conversions stay exact (the target's
  ``exp(2*log)`` and the forecast's ``0.5*log`` with the floor clip), the
  components stay finite on live data, and ``HAR_WINDOWS`` is unchanged.

The live check reads ``data/processed/dataset.parquet`` read-only and skips
when the generated dataset is absent, as on a fresh clone or in CI.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_har_estimators.py
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from conftest import DATASET_PATH

# Parity tolerance for the recorded expected vector, per the brief.
PARITY_TOLERANCE = 1e-12

# The recorded frame separates the same-estimator build from the old
# squared-close-to-close build by 6.35e-3 at its widest; the flip must clear
# this stated margin, and the old build must miss the recorded vector by it.
REBUILD_DIFFERENCE_TOLERANCE = 1e-3

# ---------------------------------------------------------------------------
# Recorded fixture frame: two tickers, seven bars each, in frame order
# (AAA rows 0-6 then BBB rows 0-6). Each ticker's first row has no
# predecessor inside the frame - the shipped dataset drops each ticker's
# earliest rows upstream - so the prior close lives in ``log_return``
# (phantom predecessors: AAA 100.0, BBB 48.0). Values are chosen so the
# Yang-Zhang variance (overnight jumps + Rogers-Satchell ranges) differs
# clearly from the squared close-to-close returns.
# ---------------------------------------------------------------------------
FIXTURE_TICKERS = np.array(["AAA"] * 7 + ["BBB"] * 7)
FIXTURE_DATES = np.concatenate(
    [np.datetime64("2020-01-02") + np.arange(7)] * 2
)
FIXTURE_ADJ_OPEN = np.array(
    [104.0, 101.0, 104.0, 96.0, 99.0, 92.0, 95.0,
     52.0, 50.0, 52.5, 48.0, 50.5, 46.0, 48.0]
)
FIXTURE_ADJ_HIGH = np.array(
    [105.0, 104.5, 104.5, 99.5, 99.5, 95.0, 96.0,
     53.0, 52.5, 53.0, 50.0, 51.0, 48.0, 48.5]
)
FIXTURE_ADJ_LOW = np.array(
    [100.0, 99.5, 96.0, 94.5, 91.5, 90.5, 89.5,
     49.5, 49.5, 48.0, 47.5, 46.5, 45.5, 44.5]
)
FIXTURE_ADJ_CLOSE = np.array(
    [101.0, 103.0, 97.0, 98.5, 93.0, 94.5, 91.0,
     50.0, 51.5, 48.5, 49.5, 47.0, 47.5, 45.0]
)
# log(close_t) - log(close_{t-1}), with each ticker's phantom predecessor
# standing in for close_{-1} on its first row (the shipped parquet stores the
# same close-to-close differences computed before its dropna).
FIXTURE_LOG_RETURN = np.array(
    [0.00995033085316788, 0.01960847138837618, -0.06001800972625304,
     0.01534556967466028, -0.05745705502478682, 0.0160003413464409,
     -0.03774032798284743, 0.04082199452025481, 0.0295588022415445,
     -0.06001800972625304, 0.02040887163120741, -0.0518250678645864,
     0.01058210933053694, -0.05406722127027574]
)

# The recorded expected components (columns: n = 1, 5, 22), computed by hand
# from the contract:
#
#   overnight_t     = ln(adj_open_t / adj_close_{t-1})
#   open_to_close_t = ln(adj_close_t / adj_open_t)
#   rs_t            = ln(H_t/C_t)*ln(H_t/O_t) + ln(L_t/C_t)*ln(L_t/O_t)
#   component_n(t)  = var(overnight | trailing n) + k_n*var(open_to_close |
#                     trailing n) + (1-k_n)*mean(rs | trailing n)
#   k_n             = 0.34 / (1.34 + (n+1)/(n-1)),  n >= 2
#
# with the short-window conventions: a window of m observations uses the
# sample variance (ddof=1) of those m; m = 1 uses the squared value about
# zero; the nominal k_n is used; the n = 1 column is the n -> 1 limit
# (overnight_t^2 + rs_t; the open-to-close weight tends to 0).
#
# Row-0 derivations (single-bar rule plus the phantom predecessor):
#   AAA row 0: ln(104/100) = 0.0392207131532813; squared = 0.001538264...
#              rs = ln(105/101)*ln(105/104) + ln(100/101)*ln(100/104)
#                 = 0.00037157... + 0.00039027... = 0.00076193...
#              n = 1 column = 0.00230019929481977
#   BBB row 0: ln(52/48) = 0.0800427076735364; squared = 0.00640683...
#              rs = ln(53/50)*ln(53/52) + ln(99/100)*ln(99/104)
#                 = 0.00111... + 0.00049... = 0.00160510...
#              n = 1 column = 0.00801194316475866
# Short-window row (AAA row 2): the n = 5 window spans rows 0-2 (m = 3),
# a ddof=1 variance over the three available observations with the nominal
# k_5 = 0.34/(1.34 + 6/4) = 0.11971830985915495.
EXPECTED_COMPONENTS = np.array(
    [
        [0.00230019929481977, 0.00231155102392018, 0.00231343779938662],
        [0.00100982369888287, 0.00169196756246981, 0.00169811004502275],
        [0.00128001822690313, 0.00152496555336201, 0.00154512147204815],
        [0.00112197719979027, 0.00157132869345255, 0.00159161554138769],
        [0.00164699918517053, 0.00156958743261553, 0.00158696079926542],
        [0.00099718242310314, 0.00138300716479788, 0.0015497164875554],
        [0.00157918393289823, 0.00146267752940553, 0.00151999820007081],
        [0.00801194316475866, 0.00800394074125498, 0.00800261065522093],
        [0.00133638644638633, 0.00478126044653984, 0.00479906085616484],
        [0.00213950808732211, 0.00349075473384581, 0.00351977267586275],
        [0.00094952949650691, 0.00320898407128644, 0.00323968944521312],
        [0.00208732904969853, 0.00284448504529564, 0.00287248014819689],
        [0.00137831138351193, 0.0018927182085876, 0.00286165314753219],
        [0.00173178536927729, 0.00193668066076819, 0.00266072275744366],
    ]
)

# Long deterministic frame for the warm-up and prefix-sensitivity checks:
# two tickers over 27 dates, interleaved by date (AAA, BBB per date).
LONG_FRAME_DATES = 27
WARMUP_FRAME_DATES = 27
WARMUP_FIT_DATES = 24
WARMUP_COLUMNS = ("_adj_open", "_adj_high", "_adj_low", "adj_close")


def _fixture_frame() -> pd.DataFrame:
    """The recorded seven-bar, two-ticker OHLC frame."""
    return pd.DataFrame(
        {
            "ticker": FIXTURE_TICKERS,
            "date": FIXTURE_DATES,
            "_adj_open": FIXTURE_ADJ_OPEN,
            "_adj_high": FIXTURE_ADJ_HIGH,
            "_adj_low": FIXTURE_ADJ_LOW,
            "adj_close": FIXTURE_ADJ_CLOSE,
            "log_return": FIXTURE_LOG_RETURN,
        }
    )


def _write_dataset(path: Path, frame: pd.DataFrame) -> Path:
    """Write a HAR-shaped dataset parquet and return its path."""
    frame.to_parquet(path)
    return path


def _load_components(dataset: Path, dates, tickers):
    """Run the real loader against a dataset parquet."""
    from packages.ml.model.training import har_benchmark

    with mock.patch.object(har_benchmark, "DATASET_PATH", dataset):
        return har_benchmark.load_har_regressors(dates, tickers)


def _old_squared_close_components(
    log_returns: np.ndarray, tickers: np.ndarray
) -> np.ndarray:
    """The pre-rebuild build: per-ticker trailing means of squared log_return."""
    from packages.ml.model.training import har_benchmark

    squared = pd.Series(np.asarray(log_returns, dtype=np.float64) ** 2)
    components = np.empty((len(squared), len(har_benchmark.HAR_WINDOWS)))
    for column, window in enumerate(har_benchmark.HAR_WINDOWS):
        components[:, column] = squared.groupby(tickers, sort=False).transform(
            lambda values, window=window: values.rolling(
                window, min_periods=1
            ).mean()
        )
    return components


def _long_frame(n_dates: int = LONG_FRAME_DATES) -> pd.DataFrame:
    """Deterministic live-shaped OHLC frame: two tickers, interleaved by date."""
    date_index = np.arange(n_dates)
    tickers = np.array(["AAA", "BBB"] * n_dates)
    close = 100.0 + 6.0 * np.sin(0.7 * date_index) + 0.5 * date_index
    close = np.repeat(close, 2)
    frame = pd.DataFrame(
        {
            "ticker": tickers,
            "date": np.repeat(np.datetime64("2020-01-02") + date_index, 2),
            "_adj_open": close * 1.01,
            "_adj_high": close * 1.03,
            "_adj_low": close * 0.97,
            "adj_close": close,
        }
    )
    log_price = np.log(close)
    log_return = log_price - np.roll(log_price, 2)
    log_return[:2] = 0.01  # each ticker's first row: phantom predecessor
    frame["log_return"] = log_return
    return frame


def test_matches_recorded_expected_vector() -> None:
    """The same-estimator build reproduces the hand-recorded vector."""
    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(Path(tmp) / "dataset.parquet", _fixture_frame())
        components, _ = _load_components(
            dataset, FIXTURE_DATES, FIXTURE_TICKERS
        )

    deviation = np.abs(components - EXPECTED_COMPONENTS).max()
    assert deviation <= PARITY_TOLERANCE, (
        f"same-estimator components deviate from the recorded expected vector "
        f"by {deviation:.3e} (tolerance {PARITY_TOLERANCE:.0e})"
    )


def test_rebuild_flips_components_and_old_build_fails_the_vector() -> None:
    """Squared close-to-close components cannot pass; the rebuild shifts them."""
    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(Path(tmp) / "dataset.parquet", _fixture_frame())
        components, _ = _load_components(
            dataset, FIXTURE_DATES, FIXTURE_TICKERS
        )

    old = _old_squared_close_components(FIXTURE_LOG_RETURN, FIXTURE_TICKERS)

    flip = np.abs(components - old).max()
    assert flip > REBUILD_DIFFERENCE_TOLERANCE, (
        f"the same-estimator rebuild must move the components by more than "
        f"{REBUILD_DIFFERENCE_TOLERANCE:.0e}; largest change is {flip:.3e}"
    )

    old_deviation = np.abs(old - EXPECTED_COMPONENTS).max()
    assert old_deviation > REBUILD_DIFFERENCE_TOLERANCE, (
        f"the old squared-close-to-close build must fail the recorded vector "
        f"(deviation {old_deviation:.3e} must exceed "
        f"{REBUILD_DIFFERENCE_TOLERANCE:.0e})"
    )


def test_no_look_ahead_prefix_sensitivity() -> None:
    """A later bar's OHLC never changes an earlier row's components."""
    frame = _long_frame()
    dates = frame["date"].to_numpy()
    tickers = frame["ticker"].to_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(Path(tmp) / "dataset.parquet", frame)
        before, _ = _load_components(dataset, dates, tickers)

        mutated = frame.copy()
        row = 2 * 20 + 1  # date 20, ticker BBB (a middle row of the frame)
        mutated.loc[row, "_adj_high"] *= 1.02
        mutated.loc[row, "_adj_open"] *= 0.995
        mutated.loc[row, "adj_close"] *= 0.99
        mutated.loc[row, "_adj_low"] *= 0.98
        dataset = _write_dataset(dataset, mutated)
        after, _ = _load_components(dataset, dates, tickers)

    changed = np.abs(after[row] - before[row]).max()
    assert changed > 0.0, "the mutated future bar must move that row's components"
    assert np.array_equal(after[:row], before[:row]), (
        "changing a future row's OHLC changed an earlier row's components"
    )


def test_warmup_interface_unchanged_and_consumed_by_the_fit() -> None:
    """The alignment task's mask/counter still governs the fit, unchanged."""
    from packages.ml.model.training import har_benchmark

    frame = _long_frame(WARMUP_FRAME_DATES)
    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(Path(tmp) / "dataset.parquet", frame)
        components, warmup_mask = _load_components(
            dataset, frame["date"].to_numpy(), frame["ticker"].to_numpy()
        )

    assert har_benchmark.HAR_WARMUP_DAYS == 21
    position = pd.Series(np.arange(len(frame))).groupby(
        frame["ticker"], sort=False
    ).cumcount()
    assert np.array_equal(
        warmup_mask, (position.to_numpy() < har_benchmark.HAR_WARMUP_DAYS)
    )
    assert np.isfinite(components).all()

    n_per_date = 2
    train_indices = np.arange(n_per_date * WARMUP_FIT_DATES)
    test_indices = np.arange(
        n_per_date * WARMUP_FIT_DATES, len(frame)
    )
    y_log = np.log(0.08 + 0.001 * np.arange(len(frame)))
    result = har_benchmark.fit_har_benchmark(
        train_indices=train_indices,
        test_indices=test_indices,
        target_log_train=y_log[train_indices],
        target_log_test=y_log[test_indices],
        regressors=components,
        forecast_floor=1e-12,
        warmup_mask=warmup_mask,
    )

    excluded = int(np.count_nonzero(warmup_mask[train_indices]))
    assert excluded == 2 * har_benchmark.HAR_WARMUP_DAYS
    assert result["n_warmup_excluded"] == excluded
    assert result["n_train"] == len(train_indices) - excluded
    assert result["n_test"] == len(test_indices)
    assert result["n_warmup_test_rows"] == int(
        np.count_nonzero(warmup_mask[test_indices])
    )
    assert result["warmup_days"] == har_benchmark.HAR_WARMUP_DAYS


def test_sigma_variance_conversions_exact() -> None:
    """The documented sigma<->variance conversions remain bit-exact."""
    from packages.ml.model.training import har_benchmark

    assert har_benchmark.HAR_WINDOWS == (1, 5, 22)

    with tempfile.TemporaryDirectory() as tmp:
        dataset = _write_dataset(Path(tmp) / "dataset.parquet", _fixture_frame())
        components, _ = _load_components(
            dataset, FIXTURE_DATES, FIXTURE_TICKERS
        )

    n_rows = len(components)
    indices = np.arange(n_rows)
    y_log = np.log(np.linspace(0.08, 0.5, n_rows))
    floor = 1e-12
    result = har_benchmark.fit_har_benchmark(
        train_indices=indices,
        test_indices=indices,
        target_log_train=y_log,
        target_log_test=y_log,
        regressors=components,
        forecast_floor=floor,
        warmup_mask=np.zeros(n_rows, dtype=bool),
    )

    design = np.column_stack([np.ones(n_rows), components])
    coefficients = np.asarray(result["coefficients"])
    # Target side: the fit consumes exp(2 * log-sigma) variance levels.
    expected_coefficients = np.linalg.lstsq(
        design, np.exp(2.0 * y_log), rcond=None
    )[0]
    assert np.array_equal(coefficients, expected_coefficients)
    # Report side: forecast_log = 0.5 * ln(max(design @ coefficients, floor**2)).
    expected_forecast_log = 0.5 * np.log(
        np.maximum(design @ coefficients, floor**2)
    )
    assert np.array_equal(result["forecast_log"], expected_forecast_log)


def test_live_components_not_the_old_build() -> None:
    """Live components stay finite and are no longer squared close-to-close."""
    from packages.ml.model.training import har_benchmark

    if not DATASET_PATH.is_file():
        pytest.skip(
            "requires the generated dataset absent from this checkout: "
            f"{DATASET_PATH}"
        )
    assert har_benchmark.HAR_WINDOWS == (1, 5, 22)

    columns = pd.read_parquet(DATASET_PATH, columns=["ticker", "date"])
    components, warmup_mask = har_benchmark.load_har_regressors(
        columns["date"].to_numpy(), columns["ticker"].to_numpy()
    )
    assert components.shape == (len(columns), len(har_benchmark.HAR_WINDOWS))
    assert np.isfinite(components).all()
    assert warmup_mask.shape == (len(columns),)

    returns = pd.read_parquet(DATASET_PATH, columns=["log_return"])[
        "log_return"
    ].to_numpy()
    old = _old_squared_close_components(
        returns, columns["ticker"].to_numpy()
    )
    shift = np.abs(components - old).max()
    assert shift > REBUILD_DIFFERENCE_TOLERANCE, (
        f"live components still match the squared-close-to-close build "
        f"(largest change {shift:.3e})"
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
