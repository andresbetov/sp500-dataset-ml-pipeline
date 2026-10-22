from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"


# Configure logging for all modules
def setup_logging() -> logging.Logger:
    """Setup logging with consistent format."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def get_project_paths() -> dict[str, Path]:
    """Get all project paths."""
    project_root = SCRIPT_DIR.parent.parent.parent
    
    return {
        "project_root": project_root,
        "model_dir": SCRIPT_DIR,
        "artifacts_dir": ARTIFACTS_DIR,
        "dataset_path": project_root / "data" / "processed" / "dataset.parquet",
        "outputs_dir": project_root / "data" / "model_outputs",
    }


def normalize_dates(values: np.ndarray) -> np.ndarray:
    """Canonicalize a date sequence for row-aligned comparisons.

    Returns a ``datetime64[ns]`` array of naive UTC instants with the input
    order and length preserved. Datetime64 input keeps its instants through a
    unit cast; tz-aware objects, strings, and other datetime-like input go
    through ``pd.to_datetime(..., utc=True)`` and drop the timezone. A missing
    date is a contract violation: ``NaT`` anywhere raises a named ``ValueError``
    counting the missing dates, never a silent drop or reordering.
    """
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        canonical = array.astype("datetime64[ns]")
    else:
        canonical = (
            pd.to_datetime(array, utc=True)
            .tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )

    missing = int(np.isnat(canonical).sum())
    if missing:
        raise ValueError(
            f"dates contain {missing} missing date(s) (NaT); "
            "every row must carry a date for row alignment"
        )
    return canonical
