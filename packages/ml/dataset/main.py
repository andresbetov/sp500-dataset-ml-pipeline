from __future__ import annotations

from pathlib import Path

import pandas as pd

from loader import load_dataframe
from preparation import prepare_dataframe
from features import build_features_dataframe


DEFAULT_PARQUET_PATH = Path("data/processed/sp500_features.parquet")


def _save_parquet(df: pd.DataFrame, output_path: str | Path) -> Path:
    parquet_path = Path(output_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def get_dataframe(
    file_name: str | None = None,
    parquet_path: str | Path = DEFAULT_PARQUET_PATH,
) -> pd.DataFrame:
    loaded = load_dataframe(file_name=file_name)
    prepared = prepare_dataframe(loaded)
    featured = build_features_dataframe(prepared)
    _save_parquet(featured, output_path=parquet_path)
    return featured

