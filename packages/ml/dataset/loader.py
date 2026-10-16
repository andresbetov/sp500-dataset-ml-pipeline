from __future__ import annotations

import functools
import hashlib
import json
import logging
from pathlib import Path

import kagglehub
import pandas as pd

DATASET_HANDLE = "jacksaleeby/s-and-p500-historical-data/versions/1"
DATASET_CSV_NAME = "SP500_Historical_Data.csv"
INGESTION_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "ingestion_manifest.json"
)

logger = logging.getLogger(__name__)


def _pick_csv_file(dataset_path: Path, file_name: str | None) -> Path:
    """Resolve the pinned CSV file inside a dataset bundle."""
    if file_name:
        csv_path = dataset_path / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        return csv_path

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


@functools.lru_cache(maxsize=1)
def load_dataframe(file_name: str | None = None) -> pd.DataFrame:
    """Load S&P 500 dataset with automatic caching.
    First call downloads and reads CSV from Kaggle (via kagglehub).
    Subsequent calls return cached dataframe from memory.
    """
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_path = _pick_csv_file(dataset_path, file_name)
    df = pd.read_csv(csv_path, keep_default_na=False)
    _write_ingestion_manifest(csv_path, df)
    return df
