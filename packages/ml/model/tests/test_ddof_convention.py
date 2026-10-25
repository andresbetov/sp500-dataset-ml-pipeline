"""Tests pinning one standard-deviation convention for every reported spread.

Every "±" value in the training summary and the visualization loaders is the
**sample** standard deviation (``ddof=1``) produced by
``evaluation.stats.spread``. Population standard deviation (``ddof=0``) stays
available only when a caller passes ``ddof=0`` explicitly, which reproduces
the old bare ``np.std`` value.

The known fixture is the five fold R² values at the branch-point snapshot
(``0.589254``, ``0.133031``, ``0.059251``, ``0.294215``, ``0.420765``):
population std ``0.191897`` vs sample std ``0.214547``, ratio
``sqrt(5/4) = 1.118034``.

The live check reads ``artifacts/results/fold_training_summary.json``
read-only and skips when the generated summary is absent. No live artifact is
written.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_ddof_convention.py
"""

from __future__ import annotations

import ast
import json
import math
import traceback
from pathlib import Path
from unittest import mock

import conftest  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR, REPO_ROOT
from packages.ml.model.evaluation.stats import spread
from packages.ml.model.training import training
from packages.ml.model.visualization import data_loader

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TRAINING_PATH = MODEL_ROOT / "training" / "training.py"
DATA_LOADER_PATH = MODEL_ROOT / "visualization" / "data_loader.py"
LIVE_SUMMARY = ARTIFACTS_DIR / "results" / "fold_training_summary.json"

# The five fold R² values at the branch-point snapshot (the known fixture).
BRANCH_POINT_R2 = np.array(
    [0.589254, 0.133031, 0.059251, 0.294215, 0.420765], dtype=np.float64
)
BRANCH_POINT_POPULATION_STD = 0.191897
BRANCH_POINT_SAMPLE_STD = 0.214547

# Current live sample spread of the five fold ``test_r2`` values. Interim pin:
# re-record it when the live artifacts regenerate.
LIVE_R2_SPREAD_DDOF1 = 0.12527045323920308


def _fold_summary(values: np.ndarray, key: str = "test_r2") -> dict:
    """Minimal per-fold records carrying one metric key."""
    return {
        f"fold_{index}": {key: float(value)}
        for index, value in enumerate(values)
    }


def _clean_summary() -> dict:
    """Five finite folds with every globally aggregated metric."""
    return {
        f"fold_{index}": {
            key: float(10.0 * index + offset)
            for offset, key in enumerate(training.GLOBAL_METRIC_KEYS, start=1)
        }
        for index in range(5)
    }


def _attribute_calls(path: Path, function_name: str) -> list[str]:
    """Attribute calls named ``std``/``var`` inside one function's body."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    hits = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"std", "var"}
        ):
            hits.append(f"{path.name}:{node.lineno}: .{node.func.attr}()")
    return hits


def test_helper_is_the_sample_standard_deviation() -> None:
    """``spread`` defaults to the sample std and still honours ``ddof``."""
    assert spread([1, 2, 3, 4, 5]) == pytest.approx(1.5811388300841898, abs=1e-12)
    assert spread([1, 2, 3, 4, 5], ddof=0) == pytest.approx(
        1.4142135623730951, abs=1e-12
    )

    # The known fixture: the two conventions differ by sqrt(5/4).
    assert spread(BRANCH_POINT_R2) == pytest.approx(BRANCH_POINT_SAMPLE_STD, abs=1e-6)
    assert spread(BRANCH_POINT_R2, ddof=0) == pytest.approx(
        BRANCH_POINT_POPULATION_STD, abs=1e-6
    )
    ratio = spread(BRANCH_POINT_R2) / spread(BRANCH_POINT_R2, ddof=0)
    assert ratio == pytest.approx(math.sqrt(5.0 / 4.0), abs=1e-12)


def test_training_aggregation_uses_the_sample_spread() -> None:
    """The global metric std is ``spread(values)`` (ddof=1), never ``np.std``."""
    summary = _fold_summary(BRANCH_POINT_R2)
    for index, fold in enumerate(summary):
        summary[fold]["test_qlike"] = BRANCH_POINT_R2[index] / 10.0
    calls = []
    real_spread = training.spread

    def spy(values, ddof: int = 1):
        calls.append(float(ddof))
        return real_spread(values, ddof=ddof)

    with mock.patch.object(training, "spread", spy):
        metrics = training._global_metrics(summary)["metrics"]

    assert calls, "the training aggregation never called the shared spread helper"
    assert all(ddof == 1 for ddof in calls), calls
    assert metrics["test_r2"]["std"] == pytest.approx(
        float(np.std(BRANCH_POINT_R2, ddof=1)), rel=1e-12, abs=1e-12
    )
    assert metrics["test_r2"]["std"] == pytest.approx(
        BRANCH_POINT_SAMPLE_STD, abs=1e-6
    )
    assert metrics["test_r2"]["std"] != pytest.approx(
        BRANCH_POINT_POPULATION_STD, abs=1e-6
    )


def test_loader_metric_stats_uses_the_sample_spread() -> None:
    """The loader metric std is the same helper; NaN folds are filtered first."""
    frame = pd.DataFrame(
        {
            "test_r2": [*BRANCH_POINT_R2, np.nan],
            "test_mae": [0.1, 0.2, 0.3, 0.4, 0.5, np.nan],
            "test_rmse": [0.2, 0.3, 0.4, 0.5, 0.6, np.nan],
        }
    )
    calls = []
    real_spread = data_loader.spread

    def spy(values, ddof: int = 1):
        calls.append(float(ddof))
        return real_spread(values, ddof=ddof)

    with mock.patch.object(data_loader, "spread", spy):
        stats = data_loader.extract_metric_stats(frame)

    assert calls, "the loader metric stats never called the shared spread helper"
    assert all(ddof == 1 for ddof in calls), calls
    assert stats["r2_std"] == pytest.approx(
        float(np.std(BRANCH_POINT_R2, ddof=1)), rel=1e-12, abs=1e-12
    )
    assert stats["mae_std"] == pytest.approx(
        float(np.std(frame["test_mae"].dropna(), ddof=1)), rel=1e-12, abs=1e-12
    )
    assert stats["rmse_std"] == pytest.approx(
        float(np.std(frame["test_rmse"].dropna(), ddof=1)), rel=1e-12, abs=1e-12
    )


def test_single_valid_fold_has_no_sample_spread() -> None:
    """One finite fold has an undefined sample spread; the run must not abort."""
    summary = _fold_summary(BRANCH_POINT_R2)
    summary["fold_0"]["test_qlike"] = 0.25
    metrics = training._global_metrics(summary)["metrics"]

    assert metrics["test_r2"]["n_valid"] == 5
    assert metrics["test_r2"]["std"] == pytest.approx(
        BRANCH_POINT_SAMPLE_STD, abs=1e-6
    )
    assert metrics["test_qlike"]["n_valid"] == 1
    assert metrics["test_qlike"]["mean"] == pytest.approx(0.25, abs=1e-12)
    assert metrics["test_qlike"]["std"] is None
    assert training._format_metric(metrics["test_qlike"]) == "0.2500"


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except pytest.skip.Exception as exc:
                skipped += 1
                print(f"SKIP {_name}: {exc}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    suffix = f" ({skipped} skipped)" if skipped else ""
    print(f"All tests passed{suffix}")
