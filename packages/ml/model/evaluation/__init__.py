"""Evaluation statistics and the shared metric error policy.

The package is the shared home for the strict-JSON sanitizer every results
writer pairs with ``allow_nan=False`` (non-finite floats become ``null``,
unencodable objects raise), the ``spread`` dispersion helper, and the
quarantine record shape for failed folds (``{"error": ...}``), plus the named
error for an all-invalid metric set.
"""

from __future__ import annotations

from packages.ml.model.evaluation.stats import (
    AllFoldsInvalidError,
    UnserializableError,
    quarantine_record,
    sanitize_json,
    spread,
)

__all__ = [
    "AllFoldsInvalidError",
    "UnserializableError",
    "quarantine_record",
    "sanitize_json",
    "spread",
]
