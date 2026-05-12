from __future__ import annotations

from pathlib import Path

import pandas as pd

from loader import load_dataframe
from preparation import prepare_dataframe
from features import build_features_dataframe

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_FILE_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def _save_parquet(df: pd.DataFrame, file_name: str) -> None:
    file_path = Path(DEFAULT_FILE_DIRECTORY / (file_name + ".parquet"))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)


def generate_featured_dataset(
    file_name: str | None = None,
) -> None:
    raw_df = load_dataframe(file_name=file_name)
    prepared_df = prepare_dataframe(raw_df)
    featured_df = build_features_dataframe(prepared_df)
    featured_df = featured_df.dropna()
    _save_parquet(featured_df, "featured")