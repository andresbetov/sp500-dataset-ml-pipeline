from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from packages.ml.model import atomic
from packages.ml.model.artifact_integrity import (
    REGISTRY_FILENAME,
    load_verified,
    record_artifact,
)
from packages.ml.model.config import get_config
from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)
N_FOLDS = get_config()["folds"]["n_folds"]
# The persisted fold-payload schema version: version 2 adds the pinned feature
# names, the reserved holdout window, the remainder date count, and the typed
# gap keys (purge_horizon_days, separation_units) that replace gap_days.
SCHEMA_VERSION = 2
# The final N unique trading dates are reserved as holdout before fold
# construction; folds are derived only on the unique dates before that window.
HOLDOUT_N_DATES = get_config()["holdout"]["n_dates"]
# The purge span in unique-trading-date index space: complete trading days excluded
# between the last kept training date and the first kept test date; enforced as
# first_test_idx - last_train_idx >= SEPARATION_MIN.
PURGE_HORIZON_D = get_config()["folds"]["purge_horizon_days"]
# The target of row t aggregates the forward window [t+1, t+5], so a row's label window
# reaches five trading days past its own date.
LABEL_HORIZON_DAYS = get_config()["folds"]["label_horizon_days"]
# Minimum separation in unique-trading-date index space, sourced from config;
# a separation of S leaves S - 1 complete trading dates between the kept sets.
SEPARATION_MIN = get_config()["folds"]["separation_min"]
# Seven calendar buckets, mirroring the dataset's market_regime label.
REGIME_NAMES = {
    0: "pre-crisis (2000-2007)",
    1: "financial-crisis (2008-2010)",
    2: "post-crisis (2011-2019)",
    3: "covid (2020-2021)",
    4: "post-covid (2022)",
    5: "post-covid (2023-2025)",
    6: "post-covid (2026)",
}


def _fold_boundaries(n_dates: int, n_folds: int | None = None) -> np.ndarray:
    """Contiguous block edges over the sorted unique trading dates.

    N_FOLDS + 1 blocks: fold k tests on block k+1 and trains on blocks 0..k, so the
    first block is train-only and the training window expands with every fold. The
    edges are the exact integer floors (k * n_dates) // (N_FOLDS + 1), so the last
    edge is exactly n_dates, the per-edge arithmetic stays integer, and no block is
    empty. ``n_folds`` overrides the configured count for in-memory constructions
    (tests, sweep fixtures); production callers keep the sole config default.
    """
    folds = N_FOLDS if n_folds is None else int(n_folds)
    edges = (np.arange(folds + 2) * n_dates) // (folds + 1)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"degenerate calendar: empty block boundary set for {n_dates} unique "
            f"trading dates ({edges.tolist()} cannot form {folds + 1} non-empty "
            f"contiguous blocks)"
        )
    return edges


def _split_date_blocks(
    tpos: np.ndarray,
    block_start: int,
    block_end: int,
    label_horizon_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Row indices for one fold with the train-only purge of the 5-day label overlap.

    tpos holds each row's index into the sorted unique trading dates. Only train rows
    whose label window reaches into the test block (tpos + 5 >= block_start) are
    removed; every row of the test block is kept whole. The last kept train row sits
    on block_start - 6, so exactly five complete trading dates (block_start - 5 ..
    block_start - 1) are excluded between the kept train and test dates.

    Returns:
        train_indices, test_indices, purge_train_count
    """
    horizon = (
        LABEL_HORIZON_DAYS if label_horizon_days is None else int(label_horizon_days)
    )
    train_mask = tpos < block_start
    test_mask = (tpos >= block_start) & (tpos < block_end)
    purge_train_mask = train_mask & (tpos + horizon >= block_start)
    train_indices = np.flatnonzero(train_mask & ~purge_train_mask)
    test_indices = np.flatnonzero(test_mask)
    return train_indices, test_indices, int(purge_train_mask.sum())


def _verify_fold_separation(
    fold_idx: int,
    train_tpos: np.ndarray,
    test_tpos: np.ndarray,
    purge_train_count: int,
    separation_min: int | None = None,
    label_horizon_days: int | None = None,
) -> int:
    """Enforce the separation predicate and the window invariant for one fold.

    The single separation predicate lives in unique-trading-date index space:
    first_test_idx - last_train_idx >= SEPARATION_MIN (at least five complete
    excluded trading days); this module applies no AFML-style embargo — the
    only exclusion is the training-side purge of the label horizon. It subsumes
    train/test date disjointness. The window invariant
    max_kept_train + LABEL_HORIZON_DAYS < min_kept_test restates the
    no-overlap condition it guarantees, so a config drift that lowers the minimum
    below the label horizon still fails by name.

    Returns:
        The observed date-index separation.
    """
    minimum = SEPARATION_MIN if separation_min is None else int(separation_min)
    horizon = (
        LABEL_HORIZON_DAYS if label_horizon_days is None else int(label_horizon_days)
    )
    first_test_idx = int(test_tpos.min())
    last_train_idx = int(train_tpos.max())
    separation = first_test_idx - last_train_idx
    if separation < minimum:
        raise ValueError(
            f"Fold {fold_idx}: separation predicate violated — first test date index "
            f"{first_test_idx} minus last train date index {last_train_idx} = "
            f"{separation} < {minimum}"
        )
    if last_train_idx + horizon >= first_test_idx:
        raise ValueError(
            f"Fold {fold_idx}: window invariant violated — last kept train index "
            f"{last_train_idx} + label horizon {horizon} >= first kept "
            f"test index {first_test_idx}"
        )
    if purge_train_count <= 0:
        raise ValueError(
            f"Fold {fold_idx}: training-side purge count must be positive, got "
            f"{purge_train_count}"
        )
    return separation


def _canonical_json(value: object) -> object:
    """Project a fold payload onto deterministic strict-JSON data.

    Numpy scalars/arrays become native values and mappings are keyed by string so
    two constructions from the same metadata serialize identically regardless of
    container type. An unsupported object raises rather than being stringified
    with a memory-dependent repr.
    """
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_canonical_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"fold payload carries an unsupported value {type(value).__name__}"
    )


def fold_manifest_hash(folds_metadata: Mapping) -> str:
    """Canonical SHA-256 of a fold payload (metadata and every fold record).

    The digest is over sorted keys and compact JSON, so two in-memory
    constructions from the same metadata hash equal while any membership, date,
    regime, or gap change alters it. It is stable across runs and processes
    because no object identity, wall-clock value, or container order enters the
    serialization.
    """
    payload = json.dumps(
        _canonical_json(folds_metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fold_payload(
    dates: np.ndarray,
    regime: np.ndarray | None = None,
    *,
    n_folds: int | None = None,
    holdout_n_dates: int | None = None,
    feature_names: Sequence[str] | None = None,
    schema_version: int | None = None,
) -> dict:
    """Build the schema-versioned fold payload entirely in memory.

    The pure construction behind Phase 3: it reads only the row dates, the
    optional regime labels, and the configured fold constants, and returns the
    same payload Phase 3 persists (no artifact loads or writes). ``n_folds`` and
    ``holdout_n_dates`` override the configured counts for tiny in-memory
    fixtures; production leaves them unset. Two calls with the same metadata
    return equal payloads and therefore equal :func:`fold_manifest_hash` digests.
    """
    values = np.asarray(dates)
    folds = int(N_FOLDS if n_folds is None else n_folds)
    holdout_count = int(
        HOLDOUT_N_DATES if holdout_n_dates is None else holdout_n_dates
    )
    if folds < 1:
        raise ValueError(f"n_folds must be positive, got {folds}")
    if holdout_count < 0:
        raise ValueError(f"holdout_n_dates must be non-negative, got {holdout_count}")
    regime_values = None if regime is None else np.asarray(regime)
    if regime_values is not None and len(regime_values) != len(values):
        raise ValueError(
            f"regime has {len(regime_values)} entries but dates has {len(values)}"
        )

    unique_dates = np.sort(np.unique(values))
    if holdout_count:
        holdout_dates = unique_dates[-holdout_count:]
        fold_dates = unique_dates[:-holdout_count]
    else:
        holdout_dates = unique_dates[:0]
        fold_dates = unique_dates

    # Every one of the folds + 1 fold blocks needs PURGE_HORIZON_D + 1 trading
    # dates so the earliest fold still keeps a training date after the
    # train-side purge. The minimum is derived from the split arithmetic, on
    # top of the reserved holdout window.
    min_fold_dates = (folds + 1) * (PURGE_HORIZON_D + 1)
    min_unique_dates = holdout_count + min_fold_dates
    if len(unique_dates) < min_unique_dates:
        raise ValueError(
            f"Need at least {min_unique_dates} unique trading dates "
            f"(holdout {holdout_count} + {min_fold_dates} for folds: "
            f"(folds + 1) * (PURGE_HORIZON_D + 1)), got {len(unique_dates)}"
        )

    boundaries = _fold_boundaries(len(fold_dates), folds)
    # Reserved-window rows sort past the last fold date, so their out-of-range
    # position can never match a fold's train or test mask.
    tpos = np.searchsorted(fold_dates, values)
    fold_n_dates = int(len(fold_dates))

    # Function-local import keeps the heavy training stack off this module's
    # import path; validate_fold_minima is the single per-fold minima
    # implementation.
    from packages.ml.model.training.training import validate_fold_minima

    fold_records = {}
    for fold_idx in range(folds):
        block_start = int(boundaries[fold_idx + 1])
        block_end = int(boundaries[fold_idx + 2])

        train_indices, test_indices, purge_train_count = _split_date_blocks(
            tpos, block_start, block_end
        )
        validate_fold_minima(fold_idx, train_indices, test_indices)

        train_tpos = tpos[train_indices]
        test_tpos = tpos[test_indices]
        separation = _verify_fold_separation(
            fold_idx, train_tpos, test_tpos, purge_train_count
        )

        train_dates = values[train_indices]
        test_dates = values[test_indices]
        train_date_min = pd.Timestamp(train_dates.min()).strftime("%Y-%m-%d")
        train_date_max = pd.Timestamp(train_dates.max()).strftime("%Y-%m-%d")
        test_date_min = pd.Timestamp(test_dates.min()).strftime("%Y-%m-%d")
        test_date_max = pd.Timestamp(test_dates.max()).strftime("%Y-%m-%d")

        regime_dist_train = {}
        regime_dist_test = {}
        if regime_values is not None:
            regimes_train = regime_values[train_indices]
            regimes_test = regime_values[test_indices]
            for regime_id in range(len(REGIME_NAMES)):
                regime_dist_train[REGIME_NAMES[regime_id]] = int(
                    np.sum(regimes_train == regime_id)
                )
                regime_dist_test[REGIME_NAMES[regime_id]] = int(
                    np.sum(regimes_test == regime_id)
                )

        fold_records[f"fold_{fold_idx}"] = {
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "train_date_min": train_date_min,
            "train_date_max": train_date_max,
            "test_date_min": test_date_min,
            "test_date_max": test_date_max,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "first_test_idx": int(test_tpos.min()),
            "last_train_idx": int(train_tpos.max()),
            "purge_train_count": purge_train_count,
            "regime_distribution_train": regime_dist_train,
            "regime_distribution_test": regime_dist_test,
            "n_dates": fold_n_dates,
            "purge_horizon_days": PURGE_HORIZON_D,
            "separation_units": SEPARATION_MIN,
        }

    holdout_bounds = {
        "start": (
            pd.Timestamp(holdout_dates[0]).strftime("%Y-%m-%d")
            if holdout_dates.size
            else None
        ),
        "end": (
            pd.Timestamp(holdout_dates[-1]).strftime("%Y-%m-%d")
            if holdout_dates.size
            else None
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION if schema_version is None else int(schema_version),
        "feature_names": (
            list(INDICATORS) if feature_names is None else list(feature_names)
        ),
        "holdout": {
            "bounds": holdout_bounds,
            "n_dates": int(holdout_dates.size),
        },
        "n_dates": fold_n_dates,
        **fold_records,
    }


def write_folds_metadata_atomically(payload: dict, path: Path) -> None:
    """Persist the fold payload through a sibling ``*.tmp`` + rename.

    A failure while pickling or renaming removes the temporary file and
    leaves any previous fold metadata byte-identical; no ``*.tmp`` residue
    remains. Phase 3 is the only caller.
    """

    def _write(tmp_path: Path) -> None:
        joblib.dump(payload, tmp_path)

    atomic.write_via_tmp(path, _write)


def phase_3_cross_validation() -> dict:
    """
    Date-group cross-validation with a train-only purge and a holdout reserve.

    The final HOLDOUT_N_DATES unique trading dates are reserved as the holdout
    window before any fold construction, so no fold train or test row can fall
    inside it. The remaining sorted unique trading dates are split into
    contiguous blocks; fold k trains on blocks 0..k and tests on block k+1
    (expanding windows). Each fold drops only the training rows whose 5-day
    forward target reaches into the test block, so five complete trading dates
    (the block start minus 5 through minus 1) are excluded between the last kept
    training date and the first kept test date; the test block itself is kept
    whole. Row indices are positions in the Phase 2 inputs; first_test_idx /
    last_train_idx index the fold remainder's sorted unique-date array, and both
    kept sets are positions in that same array, so under the shared
    trading-calendar assumption an index-space gap equals the trading-day
    distance.

    The persisted payload is the schema-versioned superset: the schema version,
    the pinned feature names, the recorded holdout window, the remainder date
    count, and the configured fold records (which carry typed gap keys). The
    return value stays the fold dict the orchestrator consumes.

    Returns:
        folds: Dict with the configured folds containing train/test indices and
        date bounds
    """
    logger.info("Phase 3: Date-Group Cross-Validation Setup")

    # Load Phase 2 outputs
    logger.info("Loading Phase 2 outputs...")
    X = np.load(ARTIFACTS_DIR / "inputs" / "X.npy", mmap_mode="r")
    y = np.load(ARTIFACTS_DIR / "inputs" / "Y.npy", mmap_mode="r")
    metadata = load_verified(
        ARTIFACTS_DIR / "inputs" / "metadata.pkl",
        registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME,
    )

    logger.info(f"Loaded X: {X.shape}, y: {y.shape}")

    # Extract dates from metadata
    dates = metadata["date"]
    logger.info(f"Date range (original order): {dates.min()} to {dates.max()}")

    # Extract regime from metadata (for regime distribution analysis)
    regime = metadata.get("regime", None)
    if regime is None:
        logger.warning("regime not found in metadata, regime distribution will not be calculated")

    # Rows are organized by ticker, so splits are built on the sorted unique trading
    # dates and every row carries its own date's position in that array.
    unique_dates = np.sort(np.unique(dates))
    logger.info(f"Unique trading dates: {len(unique_dates):,}")

    # The pure in-memory builder is the single construction path: it reserves
    # the holdout window, derives the expanding blocks, applies the train-only
    # purge, and validates every fold before this phase persists the payload.
    payload = build_fold_payload(dates, regime)
    fold_keys = sorted(
        (key for key in payload if key.startswith("fold_")),
        key=lambda key: int(key.split("_", 1)[1]),
    )
    folds = {fold_key: payload[fold_key] for fold_key in fold_keys}
    holdout = payload["holdout"]
    fold_n_dates = int(payload["n_dates"])

    logger.info(
        f"Date-group folds: {N_FOLDS} expanding folds over {N_FOLDS + 1} contiguous blocks "
        f"of the {fold_n_dates:,} unique dates before the {holdout['n_dates']:,}-date "
        f"holdout, purge span {PURGE_HORIZON_D} trading days"
    )

    # Function-local import keeps the heavy training stack off this module's import
    # path; validate_fold_minima is the single per-fold minima implementation. The
    # loop re-checks the builder's output before the payload is persisted.
    from packages.ml.model.training.training import validate_fold_minima

    for fold_key in fold_keys:
        fold_idx = int(fold_key.split("_", 1)[1])
        fold_data = folds[fold_key]
        validate_fold_minima(
            fold_idx,
            np.asarray(fold_data["train_indices"]),
            np.asarray(fold_data["test_indices"]),
        )
        separation = (
            int(fold_data["first_test_idx"]) - int(fold_data["last_train_idx"])
        )
        logger.info(
            f"Fold {fold_idx}: Train {fold_data['n_train']:,} | Test {fold_data['n_test']:,} | "
            f"{fold_data['train_date_min']} to {fold_data['test_date_max']} | train-only purge "
            f"{fold_data['purge_train_count']:,} rows | {separation - 1} excluded dates "
            f"between kept train and test | separation {separation} dates"
        )

    # Save the schema-versioned fold payload to joblib pickle
    (ARTIFACTS_DIR / "folds").mkdir(parents=True, exist_ok=True)
    folds_metadata_path = ARTIFACTS_DIR / "folds" / "folds_metadata.pkl"

    holdout_bounds = holdout["bounds"]
    write_folds_metadata_atomically(payload, folds_metadata_path)
    # Record the hash at write time so the Phase 4 reader (and any later
    # reader) can verify this artifact before loading it.
    record_artifact(
        folds_metadata_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME
    )

    logger.info(f"Saved fold metadata to {folds_metadata_path}")
    logger.info(
        f"Reserved holdout window: {holdout_bounds['start']} to {holdout_bounds['end']} "
        f"({int(holdout['n_dates']):,} dates); folds derived on {fold_n_dates:,} dates"
    )
    logger.info("Phase 3 complete: date-group cross-validation with a train-only purge")

    return folds
