"""Tests for the model package's package-absolute imports and entry points.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_imports.py
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

ENTRY_TIMEOUT_SECONDS = 30

Snapshot = dict[str, tuple[int, int]]


def _snapshot(
    repo_root: Path, artifacts_dir: Path, outputs_dir: Path
) -> Snapshot:
    """Size and mtime of every generated artifact and output file.

    Absent trees are treated as empty so a fresh clone still runs the checks.
    """
    state: Snapshot = {}
    for root in (artifacts_dir, outputs_dir):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                state[str(path.relative_to(repo_root))] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                )
    return state


def _snapshot_changes(before: Snapshot, after: Snapshot) -> list[str]:
    return sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )


def _run(
    repo_root: Path,
    artifacts_dir: Path,
    outputs_dir: Path,
    argv: list[str],
) -> tuple[subprocess.CompletedProcess[str], Snapshot, Snapshot]:
    """Run an entry-point command from the repo root and report artifact changes."""
    before = _snapshot(repo_root, artifacts_dir, outputs_dir)
    result = subprocess.run(
        argv,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=ENTRY_TIMEOUT_SECONDS,
    )
    return result, before, _snapshot(repo_root, artifacts_dir, outputs_dir)


def test_module_entry_dry_run(
    repo_root: Path, artifacts_dir: Path, outputs_dir: Path
) -> None:
    result, before, after = _run(
        repo_root,
        artifacts_dir,
        outputs_dir,
        [sys.executable, "-B", "-m", "packages.ml.model.main", "--dry-run"],
    )
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout.lower(), result.stdout
    assert after == before, _snapshot_changes(before, after)


def test_path_entry_dry_run(
    repo_root: Path, artifacts_dir: Path, outputs_dir: Path
) -> None:
    result, before, after = _run(
        repo_root,
        artifacts_dir,
        outputs_dir,
        [sys.executable, "-B", "packages/ml/model/main.py", "--dry-run"],
    )
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout.lower(), result.stdout
    assert after == before, _snapshot_changes(before, after)


def test_import_has_no_side_effects(
    repo_root: Path, artifacts_dir: Path, outputs_dir: Path
) -> None:
    before = _snapshot(repo_root, artifacts_dir, outputs_dir)
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import packages.ml.model.main"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=ENTRY_TIMEOUT_SECONDS,
    )
    after = _snapshot(repo_root, artifacts_dir, outputs_dir)
    assert result.returncode == 0, result.stderr
    assert after == before, _snapshot_changes(before, after)


if __name__ == "__main__":
    import inspect

    import pytest

    from conftest import ARTIFACTS_DIR, DATASET_PATH, OUTPUTS_DIR, REPO_ROOT

    fixtures = {
        "repo_root": REPO_ROOT,
        "artifacts_dir": ARTIFACTS_DIR,
        "outputs_dir": OUTPUTS_DIR,
        "dataset_path": DATASET_PATH,
    }

    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            kwargs = {
                parameter: fixtures[parameter]
                for parameter in inspect.signature(_fn).parameters
            }
            try:
                _fn(**kwargs)
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
