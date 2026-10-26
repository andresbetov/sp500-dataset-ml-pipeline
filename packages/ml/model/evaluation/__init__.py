"""Evaluation statistics, the shared metric error policy, and the report.

The package is the shared home for the strict-JSON sanitizer every results
writer pairs with ``allow_nan=False`` (non-finite floats become ``null``,
unencodable objects raise), the ``spread`` dispersion helper, and the
quarantine record shape for failed folds (``{"error": ...}``), plus the named
error for an all-invalid metric set. ``evaluation.report`` builds the per-row
loss vectors and the evaluation summary from them.
"""

from __future__ import annotations

from packages.ml.model.evaluation.stats import (
    AllFoldsInvalidError,
    UnserializableError,
    quarantine_record,
    sanitize_json,
    spread,
)
from packages.ml.model.evaluation.report import (
    HAC_LAG,
    build_evaluation_summary,
    cw_test,
    date_clustered_gap,
    decile_edges,
    decile_table,
    dm_test,
    expanding_origin_block,
    expanding_origin_curve,
    floor_decomposition,
    gw_test,
    loss_distribution,
    loss_vector_frame,
    mz_regression,
    regime_table,
    write_evaluation_summary,
    write_loss_vectors,
)

__all__ = [
    "AllFoldsInvalidError",
    "HAC_LAG",
    "UnserializableError",
    "build_evaluation_summary",
    "cw_test",
    "date_clustered_gap",
    "decile_edges",
    "decile_table",
    "dm_test",
    "expanding_origin_block",
    "expanding_origin_curve",
    "floor_decomposition",
    "gw_test",
    "loss_distribution",
    "loss_vector_frame",
    "mz_regression",
    "quarantine_record",
    "regime_table",
    "sanitize_json",
    "spread",
    "write_evaluation_summary",
    "write_loss_vectors",
]
