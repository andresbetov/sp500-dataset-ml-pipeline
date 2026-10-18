from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from packages.ml.model.config import get_config
from packages.ml.model.utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)
N_FOLDS = get_config()["folds"]["n_folds"]
# The purge span in unique-trading-date index space: complete trading days excluded
# between the last kept training date and the first kept test date; enforced as
# first_test_idx - last_train_idx >= SEPARATION_MIN.
PURGE_HORIZON_D = get_config()["folds"]["purge_horizon_days"]
# Legacy embargo-named alias kept for existing imports; the purge-span constant
# above is the real one.
CV_EMBARGO_DAYS = PURGE_HORIZON_D
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

    The single embargo predicate lives in unique-trading-date index space:
    first_test_idx - last_train_idx >= SEPARATION_MIN (at least five complete
    excluded trading days). It subsumes train/test date disjointness. The window
    invariant max_kept_train + LABEL_HORIZON_DAYS < min_kept_test restates the
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
            f"Fold {fold_idx}: embargo violated — first test date index {first_test_idx} "
            f"minus last train date index {last_train_idx} = {separation} < {SEPARATION_MIN}"
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
    Date-group 5-fold cross-validation with a train-only purge.

    The sorted unique trading dates are split into contiguous blocks; fold k trains on
    blocks 0..k and tests on block k+1 (expanding windows). Each fold drops only the
    training rows whose 5-day forward target reaches into the test block, so five
    complete trading dates (the block start minus 5 through minus 1) are excluded
    between the last kept training date and the first kept test date; the test block
    itself is kept whole. Row indices are positions in the Phase 2 inputs;
    first_test_idx / last_train_idx index the sorted unique-date array, and both kept
    sets are positions in that same array, so under the shared trading-calendar
    assumption an index-space gap equals the trading-day distance.

    Returns:
        folds: Dict with 5 folds containing train/test indices and date bounds
    """
    logger.info("Phase 3: Date-Group Cross-Validation Setup")

    # Load Phase 2 outputs
    logger.info("Loading Phase 2 outputs...")
    X = np.load(ARTIFACTS_DIR / "inputs" / "X.npy", mmap_mode="r")
    y = np.load(ARTIFACTS_DIR / "inputs" / "Y.npy", mmap_mode="r")
    metadata = joblib.load(ARTIFACTS_DIR / "inputs" / "metadata.pkl")

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
    tpos = np.searchsorted(unique_dates, dates)
    logger.info(f"Unique trading dates: {len(unique_dates):,}")

    # Every one of the N_FOLDS + 1 blocks needs PURGE_HORIZON_D + 1 trading dates so
    # the earliest fold still keeps a training date after the train-side purge; 36
    # with the current config. The minimum is derived from the split arithmetic.
    min_unique_dates = (N_FOLDS + 1) * (PURGE_HORIZON_D + 1)
    if len(unique_dates) < min_unique_dates:
        raise ValueError(
            f"Need at least {min_unique_dates} unique trading dates "
            f"((N_FOLDS + 1) * (PURGE_HORIZON_D + 1)), got {len(unique_dates)}"
        )

    boundaries = _fold_boundaries(len(unique_dates))
    logger.info(
        f"Date-group folds: {N_FOLDS} expanding folds over {N_FOLDS + 1} contiguous blocks "
        f"of the unique dates, purge span {PURGE_HORIZON_D} trading days"
    )

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
            "gap_days": PURGE_HORIZON_D,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "first_test_idx": int(test_tpos.min()),
            "last_train_idx": int(train_tpos.max()),
            "purge_train_count": purge_train_count,
            "regime_distribution_train": regime_dist_train,
            "regime_distribution_test": regime_dist_test,
        }

        logger.info(
            f"Fold {fold_idx}: Train {len(train_indices):,} | Test {len(test_indices):,} | "
            f"{train_date_min} to {test_date_max} | train-only purge {purge_train_count:,} rows | "
            f"{separation - 1} excluded dates between kept train and test | separation {separation} dates"
        )

    # Save fold metadata to joblib pickle
    (ARTIFACTS_DIR / "folds").mkdir(parents=True, exist_ok=True)
    folds_metadata_path = ARTIFACTS_DIR / "folds" / "folds_metadata.pkl"

    # Convert to joblib-serializable format
    folds_json = {
        fold_key: {
            "train_indices": fold_data["train_indices"],
            "test_indices": fold_data["test_indices"],
            "train_date_min": fold_data["train_date_min"],
            "train_date_max": fold_data["train_date_max"],
            "test_date_min": fold_data["test_date_min"],
            "test_date_max": fold_data["test_date_max"],
            "gap_days": fold_data["gap_days"],
            "n_train": fold_data["n_train"],
            "n_test": fold_data["n_test"],
            "first_test_idx": fold_data["first_test_idx"],
            "last_train_idx": fold_data["last_train_idx"],
            "purge_train_count": fold_data["purge_train_count"],
        }
        for fold_key, fold_data in folds.items()
    }

    joblib.dump(folds_json, folds_metadata_path)

    logger.info(f"Saved fold metadata to {folds_metadata_path}")
    logger.info("Phase 3 complete: date-group cross-validation with a train-only purge")

    return folds
