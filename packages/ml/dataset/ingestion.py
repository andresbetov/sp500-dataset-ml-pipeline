"""Stage 1 — ingestion: load the pinned Kaggle CSV and write its manifest."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import kagglehub
import pandas as pd

from .utils import INGESTION_MANIFEST_PATH

DATASET_HANDLE = "jacksaleeby/s-and-p500-historical-data/versions/1"
DATASET_CSV_NAME = "SP500_Historical_Data.csv"

logger = logging.getLogger(__name__)


def _pick_csv_file(dataset_path: Path) -> Path:
    """Resolve the pinned CSV file inside a dataset bundle."""
    csv_files = sorted(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_path}")
    if len(csv_files) > 1:
        listed = ", ".join(str(path) for path in csv_files)
        raise ValueError(
            f"Ambiguous CSV selection: {len(csv_files)} files found in {dataset_path}: {listed}"
        )
    if csv_files[0].name != DATASET_CSV_NAME:
        raise ValueError(
            f"Unexpected CSV file {csv_files[0].name!r}; expected {DATASET_CSV_NAME!r}"
        )
    return csv_files[0]


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_ingestion_manifest(csv_path: Path, df: pd.DataFrame) -> None:
    missing = {"Ticker", "Date"} - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Cannot build the ingestion manifest; missing columns: {missing_list}")

    dates = pd.to_datetime(df["Date"], errors="raise")
    manifest = {
        "source": DATASET_HANDLE,
        "file_name": csv_path.name,
        "source_sha256": _sha256_of_file(csv_path),
        "rows": len(df),
        "tickers": int(df["Ticker"].nunique()),
        "date_min": dates.min().strftime("%Y-%m-%d"),
        "date_max": dates.max().strftime("%Y-%m-%d"),
    }
    INGESTION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    INGESTION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Ingestion manifest written to %s", INGESTION_MANIFEST_PATH)


def load_dataframe() -> pd.DataFrame:
    """Load the pinned S&P 500 dataset and write its ingestion manifest.

    The Kaggle download is cached on disk by kagglehub; every call re-reads the
    pinned CSV and returns a fresh frame, so callers never share mutable state.
    """
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_path = _pick_csv_file(dataset_path)
    df = pd.read_csv(csv_path, keep_default_na=False)
    _write_ingestion_manifest(csv_path, df)
    return df
