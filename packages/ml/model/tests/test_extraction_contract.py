"""Tests pinning the extraction contract: dataset identity and indicator order.

The live checks read ``data/processed/dataset.parquet`` (read-only, including a
streamed sha256 over the file) and skip when the generated dataset is absent,
as on a fresh clone or in CI. The contract, comment, and synthetic checks need
no generated data. This file is the shared home for the extraction-contract
tests.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_extraction_contract.py
"""

from __future__ import annotations

import ast
import hashlib
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from conftest import ARTIFACTS_DIR, DATASET_PATH, REPO_ROOT

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"
FEATURE_ENGINEERING_PATH = MODEL_ROOT / "features" / "feature_engineering.py"

# The two dataset columns computed before the dataset's dropna step; they are
# pinned in the metadata and exempt from model-side reconstruction checks.
NON_RECONSTRUCTIBLE_COLUMNS = ("market_abs_ret_5d", "volume_lag_5")

# The exact key set of metadata["dataset"].
PIN_KEYS = (
    "path",
    "sha256",
    "mtime",
    "rows",
    "unique_dates",
    "extra_columns_count",
    "extra_columns",
    "indicators_hash",
    "non_reconstructible_columns",
)

_live_cache: dict[str, tuple] = {}


def _feature_engineering():
    from packages.ml.model.features import feature_engineering

    return feature_engineering


def _require_dataset() -> None:
    if not DATASET_PATH.is_file():
        pytest.skip(
            "requires the generated dataset absent from this checkout: "
            f"{DATASET_PATH}"
        )


def _live_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Load the live dataset once per process and run the extraction."""
    if "value" not in _live_cache:
        _require_dataset()
        feature_engineering = _feature_engineering()
        frame = pd.read_parquet(DATASET_PATH)
        X, Y, metadata = feature_engineering._extract_inputs(frame)
        _live_cache["value"] = (frame, X, Y, metadata)
    return _live_cache["value"]


def _pin(metadata: dict) -> dict:
    assert "dataset" in metadata, "metadata carries no 'dataset' pin"
    return metadata["dataset"]


def _indicators_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rederivation_patterns(name: str) -> tuple[re.Pattern[str], ...]:
    quoted = re.escape(name)
    flags = re.MULTILINE
    return (
        # frame["name"] = ... / frame["name"] += ...
        re.compile(
            rf"""\[\s*["']{quoted}["']\s*\]\s*(?:[+\-*/%]|\*\*|//)?=(?!=)""",
            flags,
        ),
        # name = ... / name: annotation = ...
        re.compile(rf"^\s*{quoted}\s*(?::[^=\n]+)?=(?!=)", flags),
        # call(name=...) / call(..., name=...)
        re.compile(rf"""[,(]\s*{quoted}\s*=(?!=)""", flags),
    )


def _rederivation_hits() -> list[str]:
    hits = []
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        if TESTS_DIR in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for name in NON_RECONSTRUCTIBLE_COLUMNS:
            for pattern in _rederivation_patterns(name):
                for match in pattern.finditer(text):
                    line_no = text.count("\n", 0, match.start()) + 1
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {name}")
    return hits


def _synthetic_frame(rows: int = 4, drop: str | None = None) -> pd.DataFrame:
    """A minimal in-memory frame shaped like the extraction input."""
    feature_engineering = _feature_engineering()
    data = {
        "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
        "ticker": ["AAA"] * rows,
        "realized_volatility_5d": np.full(rows, 0.01, dtype=np.float64),
    }
    for name in feature_engineering.INDICATORS:
        if name != drop:
            data[name] = np.zeros(rows, dtype=np.float64)
    return pd.DataFrame(data)


def test_indicator_contract_count() -> None:
    feature_engineering = _feature_engineering()
    assert feature_engineering.EXPECTED_INDICATOR_COUNT == 37
    assert len(feature_engineering.INDICATORS) == 37
    assert len(set(feature_engineering.INDICATORS)) == 37


def test_extras_recorded() -> None:
    frame, _, _, metadata = _live_inputs()
    feature_engineering = _feature_engineering()
    pin = _pin(metadata)

    schema_names = pq.ParquetFile(DATASET_PATH).schema.names
    expected = [
        name for name in schema_names if name not in feature_engineering.INDICATORS
    ]
    assert pin["extra_columns"] == expected
    assert pin["extra_columns_count"] == len(expected)
    # The loaded frame and the file schema must agree on the extras.
    frame_extras = [
        name for name in frame.columns if name not in feature_engineering.INDICATORS
    ]
    assert expected == frame_extras


def test_dataset_pin_matches_source() -> None:
    frame, _, _, metadata = _live_inputs()
    pin = _pin(metadata)
    assert set(pin) == set(PIN_KEYS)

    resolved = Path(pin["path"])
    assert resolved == DATASET_PATH.resolve()
    assert resolved.is_file()

    assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
    assert pin["sha256"] == _stream_sha256(DATASET_PATH)

    assert pin["rows"] == pq.ParquetFile(DATASET_PATH).metadata.num_rows
    assert pin["rows"] == frame.shape[0]
    assert pin["unique_dates"] == int(frame["date"].nunique())
    # Both sides are stat reads of the same unchanged file.
    assert pin["mtime"] == DATASET_PATH.stat().st_mtime


def test_indicators_hash_tracks_order() -> None:
    _, _, _, metadata = _live_inputs()
    feature_engineering = _feature_engineering()
    recorded = _pin(metadata)["indicators_hash"]

    names = list(feature_engineering.INDICATORS)
    assert re.fullmatch(r"[0-9a-f]{64}", recorded)
    assert recorded == _indicators_hash(names)

    swapped = names.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert _indicators_hash(swapped) != recorded

    renamed = ["renamed_" + names[0], *names[1:]]
    assert _indicators_hash(renamed) != recorded


def test_non_reconstructible_columns_recorded() -> None:
    _, _, _, metadata = _live_inputs()
    recorded = _pin(metadata)["non_reconstructible_columns"]
    assert recorded == list(NON_RECONSTRUCTIBLE_COLUMNS)
    for name in NON_RECONSTRUCTIBLE_COLUMNS:
        assert recorded.count(name) == 1


def test_no_model_side_rederivation() -> None:
    hits = _rederivation_hits()
    assert not hits, (
        "the non-reconstructible dataset columns must not be re-derived in the "
        "model package (they are computed upstream, before the dataset's dropna "
        "step):\n" + "\n".join(hits)
    )


def test_contract_comment_is_current() -> None:
    text = FEATURE_ENGINEERING_PATH.read_text(encoding="utf-8")
    for stale in ("15 volatility inputs", "7 redundant"):
        assert stale not in text, f"stale contract-comment token remains: {stale!r}"
    assert re.search(r"15\s*[-\u2212\u2013]\s*7", text) is None


def test_missing_indicator_still_raises() -> None:
    feature_engineering = _feature_engineering()
    frame = _synthetic_frame(drop="rsi_14")
    with pytest.raises(ValueError) as excinfo:
        feature_engineering._extract_inputs(frame)
    assert "rsi_14" in str(excinfo.value)


def test_phase_2_builds_metadata_from_extraction_helper() -> None:
    tree = ast.parse(FEATURE_ENGINEERING_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "phase_2_feature_selection"
    )
    helper_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_extract_inputs"
    ]
    assert len(helper_calls) == 1, (
        "phase_2_feature_selection must build X/Y/metadata through _extract_inputs "
        "so the persisted metadata is the tested extraction output"
    )


def test_x_y_values_unchanged() -> None:
    frame, X, Y, _ = _live_inputs()
    feature_engineering = _feature_engineering()
    assert X.shape == (frame.shape[0], feature_engineering.EXPECTED_INDICATOR_COUNT)
    assert np.array_equal(X, frame[feature_engineering.INDICATORS].values)
    assert np.array_equal(Y, frame["realized_volatility_5d"].values)


def test_phase_2_arrays_match_snapshot() -> None:
    X_path = ARTIFACTS_DIR / "inputs" / "X.npy"
    Y_path = ARTIFACTS_DIR / "inputs" / "Y.npy"
    if not (X_path.is_file() and Y_path.is_file()):
        pytest.skip("requires generated model inputs absent from this checkout")

    _, X, Y, _ = _live_inputs()
    X_snapshot = np.load(X_path, mmap_mode="r")
    Y_snapshot = np.load(Y_path, mmap_mode="r")
    assert X.shape == X_snapshot.shape
    assert np.array_equal(X, X_snapshot)
    assert np.array_equal(Y, Y_snapshot)


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
