"""Tests pinning the HAR-regressor date alignment.

``load_har_regressors`` checks that the dataset parquet and the model input
dates agree row by row. These checks pin that representation-only date
differences still align after normalization, that missing dates raise the
named missing-date error rather than the generic misalignment error, and that
the live alignment is unchanged. The live checks read
``data/processed/dataset.parquet`` read-only and skip when the generated
dataset is absent, as on a fresh clone or in CI.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_har_alignment.py
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import pytest

from conftest import DATASET_PATH

_live_cache: dict[str, np.ndarray] = {}


def _live_metadata_dates() -> np.ndarray:
    """The live extraction's date column; skips without the dataset."""
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
    return _live_cache["dates"]


def test_date_dtype_normalization() -> None:
    from packages.ml.model.training import har_benchmark

    canonical = _live_metadata_dates().astype("datetime64[ns]")

    # Unit-only differences must not trip the row alignment.
    unit_converted = canonical.astype("datetime64[s]")
    unit_components = har_benchmark.load_har_regressors(unit_converted)
    assert unit_components.shape == (
        len(canonical),
        len(har_benchmark.HAR_WINDOWS),
    )

    # A tz-aware object column carries the same instants in another
    # representation; it must not trip the row alignment either.
    tz_aware = pd.Series(canonical).dt.tz_localize("UTC").to_numpy()
    assert tz_aware.dtype == np.dtype("object")
    tz_components = har_benchmark.load_har_regressors(tz_aware)
    assert tz_components.shape == (len(canonical), len(har_benchmark.HAR_WINDOWS))
    assert np.isfinite(tz_components).all()


def test_nat_raises_named_missing_date_error() -> None:
    from packages.ml.model.training import har_benchmark

    missing = _live_metadata_dates().astype("datetime64[ns]").copy()
    missing[len(missing) // 2] = np.datetime64("NaT")

    with pytest.raises(ValueError) as excinfo:
        har_benchmark.load_har_regressors(missing)

    message = str(excinfo.value)
    assert "missing date" in message
    assert "NaT" in message
    assert "misaligned" not in message


def test_live_alignment_non_regression() -> None:
    from packages.ml.model.training import har_benchmark

    canonical = _live_metadata_dates().astype("datetime64[ns]")
    components = har_benchmark.load_har_regressors(canonical)
    assert components.shape == (len(canonical), len(har_benchmark.HAR_WINDOWS))
    assert np.isfinite(components).all()


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
