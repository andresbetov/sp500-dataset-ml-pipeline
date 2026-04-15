from __future__ import annotations

import pandas as pd

from .loader import load_dataframe
from .preparation import prepare_dataframe


def get_dataframe(file_name: str | None = None) -> pd.DataFrame:
    loaded = load_dataframe(file_name=file_name)
    return prepare_dataframe(loaded)


if __name__ == "__main__":
    print(get_dataframe().head())
