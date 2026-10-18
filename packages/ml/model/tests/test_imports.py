"""Tests for the model package's package-absolute imports and entry points.

pytest is not installed yet, so this file also runs standalone from the
repository root:

    .venv/bin/python -B packages/ml/model/tests/test_imports.py
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS_DIR = REPO_ROOT / "packages" / "ml" / "model" / "artifacts"
OUTPUTS_DIR = REPO_ROOT / "data" / "model_outputs"
INIT_FILES = [
    REPO_ROOT / "packages" / "ml" / "model" / "__init__.py",
    REPO_ROOT / "packages" / "ml" / "model" / "cross_validation" / "__init__.py",
    REPO_ROOT / "packages" / "ml" / "model" / "features" / "__init__.py",
    REPO_ROOT / "packages" / "ml" / "model" / "training" / "__init__.py",
]
ENTRY_TIMEOUT_SECONDS = 30

Snapshot = dict[str, tuple[int, int]]


def _snapshot() -> Snapshot:
    """Size and mtime of every generated artifact and output file."""
    state: Snapshot = {}
    for root in (ARTIFACTS_DIR, OUTPUTS_DIR):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                state[str(path.relative_to(REPO_ROOT))] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                )
    return state


def _snapshot_changes(before: Snapshot, after: Snapshot) -> list[str]:
    return sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )


def _run(argv: list[str]) -> tuple[subprocess.CompletedProcess[str], Snapshot, Snapshot]:
    """Run an entry-point command from the repo root and report artifact changes."""
    before = _snapshot()
    result = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=ENTRY_TIMEOUT_SECONDS,
    )
    return result, before, _snapshot()


def test_absolute_import() -> None:
    from packages.ml.model import main as main_module

    assert callable(main_module.main)


def test_module_entry_dry_run() -> None:
    result, before, after = _run(
        [sys.executable, "-B", "-m", "packages.ml.model.main", "--dry-run"]
    )
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout.lower(), result.stdout
    assert after == before, _snapshot_changes(before, after)


def test_path_entry_dry_run() -> None:
    result, before, after = _run(
        [sys.executable, "-B", "packages/ml/model/main.py", "--dry-run"]
    )
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout.lower(), result.stdout
    assert after == before, _snapshot_changes(before, after)


def test_unknown_argument_rejected() -> None:
    result, before, after = _run(
        [sys.executable, "-B", "-m", "packages.ml.model.main", "--nope"]
    )
    assert result.returncode != 0, result.stdout
    assert "usage" in result.stderr.lower(), result.stderr
    assert after == before, _snapshot_changes(before, after)


def test_import_has_no_side_effects() -> None:
    before = _snapshot()
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import packages.ml.model.main"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=ENTRY_TIMEOUT_SECONDS,
    )
    after = _snapshot()
    assert result.returncode == 0, result.stderr
    assert after == before, _snapshot_changes(before, after)


def test_init_parity() -> None:
    for path in INIT_FILES:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("from __future__ import annotations\n"), path


if __name__ == "__main__":
    failures = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All tests passed")
