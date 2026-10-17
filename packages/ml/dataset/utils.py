"""Shared paths and logging setup for the Phase 1 dataset build."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = DATA_PROCESSED_DIR / "dataset.parquet"
INGESTION_MANIFEST_PATH = DATA_PROCESSED_DIR / "ingestion_manifest.json"
# Phase 1 writes into the model package's tree (AGENTS invariant); relocating
# model artifacts requires editing the dataset phase.
MODEL_RESULTS_DIR = PROJECT_ROOT / "packages" / "ml" / "model" / "artifacts" / "results"
REGIME_CHARACTERIZATION_PATH = MODEL_RESULTS_DIR / "regime_characterization.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
