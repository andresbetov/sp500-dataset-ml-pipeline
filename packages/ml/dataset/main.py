from __future__ import annotations

from pathlib import Path

import pandas as pd

from loader import load_dataframe
from preparation import prepare_dataframe
from features import build_features_dataframe

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_FILE_DIRECTORY = PROJECT_ROOT / "data" / "processed"

def _save_file(df: pd.DataFrame, file_name: str, file_type: str) -> None:
    file_path = Path(DEFAULT_FILE_DIRECTORY / (file_name + (".parquet" if file_type == "parquet" else ".csv")))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_type == "parquet":
        df.to_parquet(file_path, index=False)
    elif file_type == "csv":
        df.to_csv(file_path, index=False)

# def _save_parquet(df: pd.DataFrame, file_name: str) -> None:
#     file_path = Path(DEFAULT_FILE_DIRECTORY / (file_name + ".parquet"))
#     file_path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_parquet(file_path, index=False)


def get_dataframe(
    file_name: str | None = None,
) -> pd.DataFrame:
    loaded = load_dataframe(file_name=file_name)
    prepared = prepare_dataframe(loaded)
    featured = build_features_dataframe(prepared)
    _save_file(featured, "featured", file_type="csv")
    return featured