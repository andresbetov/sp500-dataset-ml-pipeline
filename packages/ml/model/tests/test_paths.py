"""Tests pinning the single construction site of the dataset path.

Every dataset consumer resolves ``dataset.parquet`` through
``packages.ml.model.utils.get_project_paths``; the registry in ``utils.py`` is
the only place left in the package that builds the path. The consumer checks
read the parquet read-only and skip when the generated dataset is absent, as on
a fresh clone or in CI; the scan and registry checks need no generated data.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_paths.py
"""

from __future__ import annotations

import ast
import re
import traceback
from pathlib import Path

import pytest

from conftest import DATASET_PATH, REPO_ROOT

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"
UTILS_PATH = MODEL_ROOT / "utils.py"
FEATURE_ENGINEERING_PATH = MODEL_ROOT / "features" / "feature_engineering.py"
HAR_BENCHMARK_PATH = MODEL_ROOT / "training" / "har_benchmark.py"

CONSUMER_PATHS = {
    "feature_engineering": FEATURE_ENGINEERING_PATH,
    "har_benchmark": HAR_BENCHMARK_PATH,
}

# The registry's exact contract: the five keys and the value each one resolves
# to, spelled out here so a registry edit cannot pass unnoticed.
EXPECTED_PATHS = {
    "project_root": REPO_ROOT,
    "model_dir": MODEL_ROOT,
    "artifacts_dir": MODEL_ROOT / "artifacts",
    "dataset_path": REPO_ROOT / "data" / "processed" / "dataset.parquet",
    "outputs_dir": REPO_ROOT / "data" / "model_outputs",
}

# The dataset-path construction idioms the scan counts: "data" / "processed"
# joins, literal dataset.parquet path segments, and parents[N] arithmetic
# combined with a "data" segment.
CONSTRUCTION_IDIOMS = (
    re.compile(r"""["']data["']\s*/\s*["']processed["']"""),
    re.compile(r"""["'][^"']*dataset\.parquet[^"']*["']"""),
    re.compile(r"""parents\s*\[\s*\d+\s*\][^\n]*["']data["']"""),
)

# Calls that would read a file at import time if they ran at module level.
IMPORT_TIME_READS = ("read_parquet", "read_csv", "read_json", "open")


def _construction_sites() -> list[str]:
    """Dataset-path construction sites in the package, tests excluded."""
    sites: list[str] = []
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        if TESTS_DIR in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if any(idiom.search(line) for idiom in CONSTRUCTION_IDIOMS):
                sites.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")
    return sites


def _module_level_calls(path: Path) -> list[str]:
    """Calls in module-level statements, descending into no function bodies."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in IMPORT_TIME_READS:
                calls.append(f"{path.relative_to(REPO_ROOT)}:{child.lineno}: {name}()")
    return calls


def _har_read_parquet_arguments() -> list[ast.expr]:
    """Arguments of every pd.read_parquet call inside load_har_regressors."""
    tree = ast.parse(HAR_BENCHMARK_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_har_regressors"
    )
    arguments: list[ast.expr] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_parquet"
            and node.args
        ):
            arguments.append(node.args[0])
    return arguments


def test_exactly_one_dataset_path_construction_site() -> None:
    sites = _construction_sites()
    assert len(sites) == 1, (
        "expected exactly one dataset-path construction site (the registry in "
        "utils.py); found:\n" + "\n".join(sites)
    )
    assert sites[0].startswith("packages/ml/model/utils.py:"), sites[0]


def test_consumers_resolve_the_registry_dataset_path() -> None:
    from packages.ml.model.features import feature_engineering
    from packages.ml.model.training import har_benchmark
    from packages.ml.model.utils import get_project_paths

    registry_dataset_path = get_project_paths()["dataset_path"]
    assert feature_engineering.DATASET_PATH == registry_dataset_path
    assert har_benchmark.DATASET_PATH == registry_dataset_path
    assert registry_dataset_path == DATASET_PATH


def test_registry_mapping_is_unchanged() -> None:
    from packages.ml.model.utils import get_project_paths

    assert get_project_paths() == EXPECTED_PATHS


def test_resolved_dataset_file_exists() -> None:
    from packages.ml.model.features import feature_engineering
    from packages.ml.model.training import har_benchmark

    for module in (feature_engineering, har_benchmark):
        if not module.DATASET_PATH.is_file():
            pytest.skip(
                "requires the generated dataset absent from this checkout: "
                f"{module.DATASET_PATH}"
            )
    assert feature_engineering.DATASET_PATH.resolve() == DATASET_PATH.resolve()
    assert har_benchmark.DATASET_PATH.resolve() == DATASET_PATH.resolve()


def test_consumer_modules_have_no_import_time_file_reads() -> None:
    calls = [
        call
        for path in CONSUMER_PATHS.values()
        for call in _module_level_calls(path)
    ]
    assert not calls, (
        "importing the consumer modules must not read files beyond module "
        "loading; module-level reads found:\n" + "\n".join(calls)
    )


def test_har_benchmark_reads_through_the_imported_constant() -> None:
    arguments = _har_read_parquet_arguments()
    assert arguments, "load_har_regressors no longer calls pd.read_parquet"
    assert all(
        isinstance(argument, ast.Name) and argument.id == "DATASET_PATH"
        for argument in arguments
    ), "load_har_regressors must read through the module's DATASET_PATH"


def test_load_har_regressors_reads_the_dataset() -> None:
    if not DATASET_PATH.is_file():
        pytest.skip(
            "requires the generated dataset absent from this checkout: "
            f"{DATASET_PATH}"
        )

    import numpy as np
    import pandas as pd

    from packages.ml.model.training import har_benchmark

    columns = pd.read_parquet(DATASET_PATH, columns=["ticker", "date"])
    components, warmup_mask = har_benchmark.load_har_regressors(
        columns["date"].to_numpy(), columns["ticker"].to_numpy()
    )
    assert components.shape == (len(columns), len(har_benchmark.HAR_WINDOWS))
    assert np.isfinite(components).all()
    assert warmup_mask.dtype == bool
    assert warmup_mask.shape == (len(columns),)


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
