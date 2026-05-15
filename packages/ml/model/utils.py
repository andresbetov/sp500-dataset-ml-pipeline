from __future__ import annotations

import logging
import sys
from pathlib import Path

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
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    
    return {
        "project_root": project_root,
        "model_dir": script_dir,
        "models_dir": script_dir / "models",
        "dataset_path": project_root / "data" / "processed" / "dataset.parquet",
        "outputs_dir": project_root / "data" / "model_outputs",
    }
