"""Single source of truth for the model package's runtime constants.

Edit ``base.yaml`` next to this module to change the numeric training/CV
constants; consumers read them through :func:`get_config`.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "base.yaml"


def get_config_path() -> Path:
    """Resolve the config file path.

    ``MODEL_CONFIG_PATH`` wins when set; otherwise ``base.yaml`` next to this
    file (never the current working directory).
    """
    env_path = os.environ.get("MODEL_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config and return it as a fresh dict."""
    config_path = Path(path) if path is not None else get_config_path()
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=None)
def get_config() -> dict:
    """Cached default load of the config file."""
    return load_config()
