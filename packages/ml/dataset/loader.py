from __future__ import annotations

import functools
from pathlib import Path

import kagglehub
import pandas as pd

DATASET_SLUG = "jacksaleeby/s-and-p500-historical-data"


def _pick_csv_file(dataset_path: Path, file_name: str | None) -> Path:
    if file_name:
        csv_path = dataset_path / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        return csv_path

    csv_files = sorted(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_path}")
    return csv_files[0]


@functools.lru_cache(maxsize=1)
def load_dataframe(file_name: str | None = None) -> pd.DataFrame:
    """Load S&P 500 dataset with automatic caching.
    First call downloads and reads CSV from Kaggle (via kagglehub).
    Subsequent calls return cached dataframe from memory.
    """
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    csv_path = _pick_csv_file(dataset_path, file_name)
    return pd.read_csv(csv_path)

