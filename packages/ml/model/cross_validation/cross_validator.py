from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
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


def _fold_boundaries(n_dates: int) -> np.ndarray:
    """Contiguous block edges over the sorted unique trading dates.

    N_FOLDS + 1 blocks: fold k tests on block k+1 and trains on blocks 0..k, so the
    first block is train-only and the training window expands with every fold. The
    edges are the exact integer floors (k * n_dates) // (N_FOLDS + 1), so the last
    edge is exactly n_dates, the per-edge arithmetic stays integer, and no block is
    empty.
    """
    edges = (np.arange(N_FOLDS + 2) * n_dates) // (N_FOLDS + 1)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"degenerate calendar: empty block boundary set for {n_dates} unique "
            f"trading dates ({edges.tolist()} cannot form {N_FOLDS + 1} non-empty "
            f"contiguous blocks)"
        )
    return edges


def _split_date_blocks(
    tpos: np.ndarray,
    block_start: int,
    block_end: int,
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
    train_mask = tpos < block_start
    test_mask = (tpos >= block_start) & (tpos < block_end)
    purge_train_mask = train_mask & (tpos + LABEL_HORIZON_DAYS >= block_start)
    train_indices = np.flatnonzero(train_mask & ~purge_train_mask)
    test_indices = np.flatnonzero(test_mask)
    return train_indices, test_indices, int(purge_train_mask.sum())


def _verify_fold_separation(
    fold_idx: int,
    train_tpos: np.ndarray,
    test_tpos: np.ndarray,
    purge_train_count: int,
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
    first_test_idx = int(test_tpos.min())
    last_train_idx = int(train_tpos.max())
    separation = first_test_idx - last_train_idx
    if separation < SEPARATION_MIN:
        raise ValueError(
            f"Fold {fold_idx}: separation predicate violated — first test date index "
            f"{first_test_idx} minus last train date index {last_train_idx} = "
            f"{separation} < {SEPARATION_MIN}"
        )
    if last_train_idx + LABEL_HORIZON_DAYS >= first_test_idx:
        raise ValueError(
            f"Fold {fold_idx}: window invariant violated — last kept train index "
            f"{last_train_idx} + label horizon {LABEL_HORIZON_DAYS} >= first kept "
            f"test index {first_test_idx}"
        )
    if purge_train_count <= 0:
        raise ValueError(
            f"Fold {fold_idx}: training-side purge count must be positive, got "
            f"{purge_train_count}"
        )
    return separation


def phase_3_cross_validation() -> dict:
    """
    Date-group 5-fold cross-validation with a train-only purge and a holdout reserve.

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
    count, and the five fold records (which carry typed gap keys). The return
    value stays the five-fold dict the orchestrator consumes.

    Returns:
        folds: Dict with 5 folds containing train/test indices and date bounds
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

    # Reserve the final holdout window before any fold construction: folds are
    # derived only on the unique dates before it, so no fold train or test row
    # can reach into the reserved dates by construction.
    if HOLDOUT_N_DATES:
        holdout_dates = unique_dates[-HOLDOUT_N_DATES:]
        fold_dates = unique_dates[:-HOLDOUT_N_DATES]
    else:
        holdout_dates = unique_dates[:0]
        fold_dates = unique_dates

    # Every one of the N_FOLDS + 1 fold blocks needs PURGE_HORIZON_D + 1 trading
    # dates so the earliest fold still keeps a training date after the train-side
    # purge; 36 with the current config. The minimum is derived from the split
    # arithmetic, on top of the reserved holdout window.
    min_fold_dates = (N_FOLDS + 1) * (PURGE_HORIZON_D + 1)
    min_unique_dates = HOLDOUT_N_DATES + min_fold_dates
    if len(unique_dates) < min_unique_dates:
        raise ValueError(
            f"Need at least {min_unique_dates} unique trading dates "
            f"(holdout {HOLDOUT_N_DATES} + {min_fold_dates} for folds: "
            f"(N_FOLDS + 1) * (PURGE_HORIZON_D + 1)), got {len(unique_dates)}"
        )

    boundaries = _fold_boundaries(len(fold_dates))
    logger.info(
        f"Date-group folds: {N_FOLDS} expanding folds over {N_FOLDS + 1} contiguous blocks "
        f"of the {len(fold_dates):,} unique dates before the {len(holdout_dates):,}-date "
        f"holdout, purge span {PURGE_HORIZON_D} trading days"
    )

    # Reserved-window rows sort past the last fold date, so their out-of-range
    # position can never match a fold's train or test mask.
    tpos = np.searchsorted(fold_dates, dates)
    fold_n_dates = int(len(fold_dates))

    folds = {}

    # Function-local import keeps the heavy training stack off this module's import
    # path; validate_fold_minima is the single per-fold minima implementation.
    from packages.ml.model.training.training import validate_fold_minima

    for fold_idx in range(N_FOLDS):
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

        # Get date bounds
        train_dates = dates[train_indices]
        test_dates = dates[test_indices]
        train_date_min = pd.Timestamp(train_dates.min()).strftime("%Y-%m-%d")
        train_date_max = pd.Timestamp(train_dates.max()).strftime("%Y-%m-%d")
        test_date_min = pd.Timestamp(test_dates.min()).strftime("%Y-%m-%d")
        test_date_max = pd.Timestamp(test_dates.max()).strftime("%Y-%m-%d")

        # Calculate regime distribution if regime metadata is available
        regime_dist_train = {}
        regime_dist_test = {}
        if regime is not None:
            regimes_train = regime[train_indices]
            regimes_test = regime[test_indices]
            for regime_id in range(len(REGIME_NAMES)):
                regime_dist_train[REGIME_NAMES[regime_id]] = int(np.sum(regimes_train == regime_id))
                regime_dist_test[REGIME_NAMES[regime_id]] = int(np.sum(regimes_test == regime_id))

        fold_key = f"fold_{fold_idx}"
        folds[fold_key] = {
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

        logger.info(
            f"Fold {fold_idx}: Train {len(train_indices):,} | Test {len(test_indices):,} | "
            f"{train_date_min} to {test_date_max} | train-only purge {purge_train_count:,} rows | "
            f"{separation - 1} excluded dates between kept train and test | separation {separation} dates"
        )

    # Save the schema-versioned fold payload to joblib pickle
    (ARTIFACTS_DIR / "folds").mkdir(parents=True, exist_ok=True)
    folds_metadata_path = ARTIFACTS_DIR / "folds" / "folds_metadata.pkl"

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
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(INDICATORS),
        "holdout": {
            "bounds": holdout_bounds,
            "n_dates": int(holdout_dates.size),
        },
        "n_dates": fold_n_dates,
        **folds,
    }
    joblib.dump(payload, folds_metadata_path)
    # Record the hash at write time so the Phase 4 reader (and any later
    # reader) can verify this artifact before loading it.
    record_artifact(
        folds_metadata_path, registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME
    )

    logger.info(f"Saved fold metadata to {folds_metadata_path}")
    logger.info(
        f"Reserved holdout window: {holdout_bounds['start']} to {holdout_bounds['end']} "
        f"({int(holdout_dates.size):,} dates); folds derived on {fold_n_dates:,} dates"
    )
    logger.info("Phase 3 complete: date-group cross-validation with a train-only purge")

    return folds
