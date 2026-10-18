from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from packages.ml.model.config import get_config
from packages.ml.model.utils import ARTIFACTS_DIR, get_project_paths, normalize_dates

logger = logging.getLogger(__name__)

DATASET_PATH = get_project_paths()["dataset_path"]

INDICATORS = [
	"atr20",
	"atr5",
	"ema_12",
	"ema_26",
	"gap5",
	"gk5",
	"har22",
	"har5",
	"har63",
	"macd",
	"macd_hist",
	"market_abs_ret_5d",
	"park20",
	"park5",
	"price_vs_sma_10",
	"price_vs_sma_20",
	"price_vs_sma_50",
	"return_lag_1",
	"return_lag_10",
	"return_lag_2",
	"return_lag_3",
	"return_lag_5",
	"rolling_max_10",
	"rolling_mean_5",
	"rolling_min_10",
	"rsi_14",
	"rv22",
	"rv5_past",
	"rv63",
	"rs5",
	"simple_return",
	"volvol22",
	"volume_lag_1",
	"volume_lag_3",
	"volume_lag_5",
	"zscore_price_vs_sma_20",
	"zscore_volume_20",
]

# The pinned model-input contract: exactly the 37 INDICATORS entries below, in
# this order. Dataset columns outside the list (raw and adjusted OHLCV,
# log_price, log_return, macd_signal, market_regime, date, ticker, target) are
# recorded as extras at extraction and never fed to the model. Keep this list
# and the dataset columns in lockstep; feature-importance arrays are mapped
# onto INDICATORS by position.
EXPECTED_INDICATOR_COUNT = 37
if len(INDICATORS) != EXPECTED_INDICATOR_COUNT:
	raise ValueError(
		f"INDICATORS holds {len(INDICATORS)} entries; the pinned contract is "
		f"{EXPECTED_INDICATOR_COUNT}"
	)

# Convention: these two dataset columns are computed upstream on per-date
# cross-sections before the dataset's dropna step, so they cannot be re-derived
# from the shipped parquet. They stay in the pinned input list but are exempt
# from any model-side reconstruction check; the dataset-side fix is routed to
# the dataset pipeline, not this package. Model code must never re-derive them.
NON_RECONSTRUCTIBLE_COLUMNS = [
    "market_abs_ret_5d",
    "volume_lag_5",
]


def _stream_sha256(path: Path) -> str:
    """Streaming sha256 of a file, to pin the dataset identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indicators_order_hash() -> str:
    """Deterministic fingerprint of the pinned INDICATORS order."""
    return hashlib.sha256("\n".join(INDICATORS).encode("utf-8")).hexdigest()


def _dataset_pin(df: pd.DataFrame) -> dict:
    """Dataset identity recorded in the Phase-2 metadata."""
    extra_columns = [name for name in df.columns if name not in INDICATORS]
    return {
        "path": str(DATASET_PATH.resolve()),
        "sha256": _stream_sha256(DATASET_PATH),
        "mtime": DATASET_PATH.stat().st_mtime,
        "rows": int(df.shape[0]),
        "unique_dates": int(df["date"].nunique()),
        "extra_columns_count": len(extra_columns),
        "extra_columns": extra_columns,
        "indicators_hash": _indicators_order_hash(),
        "non_reconstructible_columns": list(NON_RECONSTRUCTIBLE_COLUMNS),
    }


def validate_extracted_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
) -> dict:
    """Validate the extracted model inputs and return the checks record.

    The record is attached to the Phase-2 metadata as ``input_checks`` and has
    exactly these keys: ``n_rows``, ``n_features``, ``x_nan_count``,
    ``x_inf_count``, ``x_nan_rates``, ``x_inf_rates``, ``duplicate_pairs``,
    ``zero_variance_columns``, ``y_nan_count``, ``y_inf_count``,
    ``y_negative_count``, ``rv_floor``, ``degenerate_count``,
    ``degenerate_rate``. Per-column rates are fractions keyed by indicator
    name.

    Non-finite X and non-finite or negative Y are contract violations and
    raise a named ``ValueError`` before any fit path sees the arrays. Duplicate
    ``(ticker, date)`` pairs and constant columns are recorded, not raised.
    Target mass at or below the configured volatility floor is a dataset
    property: it is recorded and warned about, never raised.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n_rows = int(X.shape[0])
    n_features = int(X.shape[1])
    rv_floor = float(get_config()["rv_floor"])

    finite_x = np.isfinite(X)
    if not finite_x.all():
        nonfinite = X[~finite_x]
        x_nan_count = int(np.isnan(nonfinite).sum())
        x_inf_count = int(np.isinf(nonfinite).sum())
        raise ValueError(
            f"Invalid feature values. Found {x_nan_count} NaNs and {x_inf_count} infs"
        )
    del finite_x

    feature_names = list(INDICATORS[:n_features])
    if len(feature_names) != n_features:
        feature_names = [f"feature_{index}" for index in range(n_features)]

    def _rate(count: int) -> float:
        return float(count) / n_rows if n_rows else 0.0

    x_nan_counts = np.isnan(X).sum(axis=0)
    x_inf_counts = np.isinf(X).sum(axis=0)
    x_nan_rates = {
        name: _rate(int(count)) for name, count in zip(feature_names, x_nan_counts)
    }
    x_inf_rates = {
        name: _rate(int(count)) for name, count in zip(feature_names, x_inf_counts)
    }

    duplicate_pairs = int(
        pd.MultiIndex.from_arrays([np.asarray(tickers), np.asarray(dates)])
        .duplicated()
        .sum()
    )

    column_spreads = X.max(axis=0) - X.min(axis=0)
    zero_variance_columns = [
        name for name, spread in zip(feature_names, column_spreads) if spread == 0
    ]

    y_nan_count = int(np.isnan(Y).sum())
    y_inf_count = int(np.isinf(Y).sum())
    y_negative_count = int(np.count_nonzero(Y < 0))
    if y_nan_count or y_inf_count or y_negative_count:
        raise ValueError(
            f"Invalid target values. Found {y_nan_count} NaNs, {y_inf_count} infs, "
            f"and {y_negative_count} negative values"
        )

    degenerate_count = int(np.count_nonzero(Y <= rv_floor))
    degenerate_rate = _rate(degenerate_count)
    if degenerate_count:
        logger.warning(
            f"Degenerate target mass: {degenerate_count} rows at or below rv_floor "
            f"{rv_floor:g} ({degenerate_rate:.6%} of {n_rows} rows)"
        )

    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "x_nan_count": int(x_nan_counts.sum()),
        "x_inf_count": int(x_inf_counts.sum()),
        "x_nan_rates": x_nan_rates,
        "x_inf_rates": x_inf_rates,
        "duplicate_pairs": duplicate_pairs,
        "zero_variance_columns": zero_variance_columns,
        "y_nan_count": y_nan_count,
        "y_inf_count": y_inf_count,
        "y_negative_count": y_negative_count,
        "rv_floor": rv_floor,
        "degenerate_count": degenerate_count,
        "degenerate_rate": degenerate_rate,
    }


def _extract_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract the pinned model inputs from a loaded dataset frame.

    Returns X (the 37 pinned indicators in contract order), Y (the realized
    volatility target), and the metadata later phases consume: the date and
    ticker columns, the row-aligned calendar regime labels, the feature counts,
    the pinned dataset record, and the extraction input checks.
    """
    # Extract metadata (original ticker and date columns for reference); the
    # date column is canonicalized so representation-only differences do not
    # reach the row-alignment comparisons downstream.
    date_column = normalize_dates(df["date"].values)
    ticker_column = df["ticker"].values

    missing_indicators = [name for name in INDICATORS if name not in df.columns]
    if missing_indicators:
        missing_list = ", ".join(missing_indicators)
        raise ValueError(f"Dataset is missing INDICATORS columns: {missing_list}")

    # Select the pinned 37 indicators (no ticker encoding)
    X = df[INDICATORS].values

    # Extract target: realized_volatility_5d (continuous volatility values)
    Y = df["realized_volatility_5d"].values

    # Fail loud on corrupt inputs and record the extraction diagnostics before
    # any metadata or array is written.
    input_checks = validate_extracted_inputs(X, Y, date_column, ticker_column)

    # Create metadata dict for cross-validation
    metadata = {
        "date": date_column,
        "ticker": ticker_column,
        "regime": df["market_regime"].values,
        "n_indicators": len(INDICATORS),
        "n_ticker_features": 0,
        "dataset": _dataset_pin(df),
        "input_checks": input_checks,
    }
    return X, Y, metadata


def phase_2_feature_selection() -> tuple[np.ndarray, np.ndarray, dict, None]:
    """
    Load dataset and select technical indicators only (no ticker encoding).

    Returns:
        X: Feature matrix (n_samples, n_features) = the 37 pinned indicators only
        Y: Target vector (n_samples,) with continuous volatility values
        metadata: Dict with 'date' and 'ticker' columns, the row-aligned
            'regime' labels, n_indicators, n_ticker_features, the pinned dataset
            record, and the extraction input checks
        encoder: None (no ticker encoding applied)
    """
    logger.info("Phase 2: Feature Selection (No Ticker Encoding)")

    # Load dataset (already validated and cleaned by dataset pipeline)
    logger.info(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Extract the pinned model inputs; the metadata carries the pinned dataset
    # record in addition to the date and ticker columns.
    logger.info(f"Selecting {len(INDICATORS)} technical indicators (no ticker encoding)")
    X, Y, metadata = _extract_inputs(df)
    logger.info(f"Feature matrix X: {X.shape}")
    logger.info(f"Target Y (realized volatility): shape {Y.shape}, range [{Y.min():.6f}, {Y.max():.6f}], mean {Y.mean():.6f}")

    # Create subdirectories
    (ARTIFACTS_DIR / "inputs").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "legacy").mkdir(parents=True, exist_ok=True)

    # Save X, Y, metadata to inputs
    X_path = ARTIFACTS_DIR / "inputs" / "X.npy"
    y_path = ARTIFACTS_DIR / "inputs" / "Y.npy"
    metadata_path = ARTIFACTS_DIR / "inputs" / "metadata.pkl"

    np.save(X_path, X)
    np.save(y_path, Y)
    joblib.dump(metadata, metadata_path)
    logger.info(f"Saved X to {X_path}")
    logger.info(f"Saved Y to {y_path}")
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info(f"✓ Phase 2 complete: X {X.shape}, Y {Y.shape}")

    return X, Y, metadata, None
