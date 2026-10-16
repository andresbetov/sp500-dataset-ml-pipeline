from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)
N_FOLDS = 5
# Complete trading days that must be excluded between the last kept training date and
# the first kept test date; enforced as first_test_idx - last_train_idx >= CV_EMBARGO_DAYS + 1
# in unique-trading-date index space.
CV_EMBARGO_DAYS = 5
# The target of row t aggregates the forward window [t+1, t+5], so a row's label window
# reaches five trading days past its own date.
LABEL_HORIZON_DAYS = 5
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
    first block is train-only and the training window expands with every fold.
    """
    return np.linspace(0, n_dates, N_FOLDS + 2).astype(int)


def _split_date_blocks(
    tpos: np.ndarray,
    block_start: int,
    block_end: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Row indices for one fold with the two-sided purge of the 5-day label overlap.

    tpos holds each row's index into the sorted unique trading dates. Train rows whose
    label window reaches into the test block (tpos + 5 >= block_start) and test rows
    whose label window overlaps the last training row's window
    (tpos + 1 <= (block_start - 1) + 5) are both removed, which leaves at least five
    complete trading dates between the kept train and test dates.

    Returns:
        train_indices, test_indices, purge_train_count, purge_test_count
    """
    train_mask = tpos < block_start
    test_mask = (tpos >= block_start) & (tpos < block_end)
    last_raw_train_idx = block_start - 1
    purge_train_mask = train_mask & (tpos + LABEL_HORIZON_DAYS >= block_start)
    purge_test_mask = test_mask & (tpos + 1 <= last_raw_train_idx + LABEL_HORIZON_DAYS)
    train_indices = np.flatnonzero(train_mask & ~purge_train_mask)
    test_indices = np.flatnonzero(test_mask & ~purge_test_mask)
    return train_indices, test_indices, int(purge_train_mask.sum()), int(purge_test_mask.sum())


def _verify_fold_separation(
    fold_idx: int,
    train_tpos: np.ndarray,
    test_tpos: np.ndarray,
    purge_train_count: int,
    purge_test_count: int,
) -> int:
    """Enforce the embargo predicate and the non-empty purges for one fold.

    The single embargo predicate lives in unique-trading-date index space:
    first_test_idx - last_train_idx >= CV_EMBARGO_DAYS + 1 (at least five complete
    excluded trading days). It subsumes train/test date disjointness.

    Returns:
        The observed date-index separation.
    """
    first_test_idx = int(test_tpos.min())
    last_train_idx = int(train_tpos.max())
    separation = first_test_idx - last_train_idx
    if separation < CV_EMBARGO_DAYS + 1:
        raise ValueError(
            f"Fold {fold_idx}: embargo violated — first test date index {first_test_idx} "
            f"minus last train date index {last_train_idx} = {separation} < {CV_EMBARGO_DAYS + 1}"
        )
    if purge_train_count <= 0 or purge_test_count <= 0:
        raise ValueError(
            f"Fold {fold_idx}: purge counts must be positive, got "
            f"train={purge_train_count}, test={purge_test_count}"
        )
    return separation


def phase_3_cross_validation() -> dict:
    """
    Date-group 5-fold cross-validation with embargo and purge.

    The sorted unique trading dates are split into contiguous blocks; fold k trains on
    blocks 0..k and tests on block k+1 (expanding windows). Each fold then drops the
    two-sided label overlap of the 5-day forward target, so same-date overlap across
    the train/test boundary is impossible. Row indices are positions in the Phase 2
    inputs; first_test_idx / last_train_idx index the sorted unique-date array.

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

    if len(unique_dates) < N_FOLDS + 1:
        raise ValueError(f"Need at least {N_FOLDS + 1} unique trading dates, got {len(unique_dates)}")

    boundaries = _fold_boundaries(len(unique_dates))
    logger.info(
        f"Date-group folds: {N_FOLDS} expanding folds over {N_FOLDS + 1} contiguous blocks "
        f"of the unique dates, embargo={CV_EMBARGO_DAYS} trading days"
    )

    folds = {}

    for fold_idx in range(N_FOLDS):
        block_start = int(boundaries[fold_idx + 1])
        block_end = int(boundaries[fold_idx + 2])

        train_indices, test_indices, purge_train_count, purge_test_count = _split_date_blocks(
            tpos, block_start, block_end
        )

        train_tpos = tpos[train_indices]
        test_tpos = tpos[test_indices]
        separation = _verify_fold_separation(
            fold_idx, train_tpos, test_tpos, purge_train_count, purge_test_count
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
            "gap_days": CV_EMBARGO_DAYS,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "first_test_idx": int(test_tpos.min()),
            "last_train_idx": int(train_tpos.max()),
            "purge_train_count": purge_train_count,
            "purge_test_count": purge_test_count,
            "regime_distribution_train": regime_dist_train,
            "regime_distribution_test": regime_dist_test,
        }

        logger.info(
            f"Fold {fold_idx}: Train {len(train_indices):,} | Test {len(test_indices):,} | "
            f"{train_date_min} to {test_date_max} | purged train/test {purge_train_count:,}/{purge_test_count:,} | "
            f"embargo {separation} dates"
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
            "purge_test_count": fold_data["purge_test_count"],
        }
        for fold_key, fold_data in folds.items()
    }

    joblib.dump(folds_json, folds_metadata_path)

    logger.info(f"Saved fold metadata to {folds_metadata_path}")
    logger.info("Phase 3 complete: date-group cross-validation with embargo and purge")

    return folds
