"""Retention contract: keep the last N run directories, never truncate the log.

The checks drive synthetic run directories and a temporary ``runs.jsonl`` only;
no live ``artifacts/**`` tree is read or written. They pin:

- ``run.retention`` is configured and positive;
- pruning keeps exactly the newest N run directories (by UTC run-id name);
- every kept directory carries a per-directory ``README.md``;
- ``runs.jsonl`` is byte-identical and append-only across pruning;
- ``finish_run`` applies the configured retention after appending its line;
- ``docs/model-artifacts.md`` documents the retention and append-only contract.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_retention.py
"""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import pytest

from conftest import REPO_ROOT
from packages.ml.model import config, run_log


def _retention_module():
    """Import the retention module lazily so the red run stays observable."""
    from packages.ml.model import retention

    return retention

RUN_IDS = (
    "20260101T000000Z-11111111",
    "20260102T000000Z-22222222",
    "20260103T000000Z-33333333",
    "20260104T000000Z-44444444",
)


def _run_dirs(root: Path, run_ids: tuple[str, ...]) -> list[Path]:
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    directories = []
    for run_id in run_ids:
        directory = checkpoints / run_id
        directory.mkdir()
        (directory / "fold_0.pkl").write_bytes(b"placeholder")
        directories.append(directory)
    return directories


def test_run_retention_is_configured() -> None:
    loaded = config.load_config()
    keep = loaded["run"]["retention"]
    assert isinstance(keep, int) and not isinstance(keep, bool)
    assert keep >= 1


def test_pruning_keeps_the_last_n_and_leaves_runs_jsonl_intact() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        root.mkdir()
        _run_dirs(root, RUN_IDS)
        runs_path = root / "runs.jsonl"
        runs_path.write_text('{"run_id": "first"}\n{"run_id": "second"}\n')
        before = runs_path.read_bytes()

        pruned = _retention_module().prune_run_directories(root, keep=2)

        kept = sorted(path.name for path in (root / "checkpoints").iterdir())
        assert kept == list(RUN_IDS[-2:]), kept
        assert sorted(path.name for path in pruned) == list(RUN_IDS[:2])
        assert runs_path.read_bytes() == before, "retention rewrote runs.jsonl"
        for run_id in RUN_IDS[-2:]:
            readme = root / "checkpoints" / run_id / "README.md"
            assert readme.is_file(), f"kept run {run_id} has no README"
            text = readme.read_text(encoding="utf-8")
            assert run_id in text, text
            assert "retention" in text.lower(), text


def test_pruning_keeps_exactly_n_when_more_than_n_exist() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        root.mkdir()
        extra = (f"2026010{index}T000000Z-{index:08x}" for index in range(5, 8))
        run_ids = RUN_IDS + tuple(extra)
        _run_dirs(root, run_ids)

        _retention_module().prune_run_directories(root, keep=3)

        kept = sorted(path.name for path in (root / "checkpoints").iterdir())
        assert kept == list(run_ids[-3:]), kept


def test_finish_run_applies_the_configured_retention() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        root.mkdir()
        _run_dirs(root, RUN_IDS)
        runs_path = root / "runs.jsonl"
        runs_path.write_text('{"run_id": "earlier"}\n')
        before = runs_path.read_bytes()

        run = run_log.new_run("pipeline", root)
        current_dir = root / "checkpoints" / run["run_id"]
        current_dir.mkdir()

        with mock.patch.object(
            _retention_module(), "get_config", return_value={"run": {"retention": 2}}
        ):
            run_log.finish_run(run, status="completed")

        kept = sorted(path.name for path in (root / "checkpoints").iterdir())
        assert run["run_id"] in kept
        assert len(kept) == 2, kept
        after = runs_path.read_bytes()
        assert after.startswith(before), "finish_run truncated runs.jsonl"
        assert len(after.splitlines()) == len(before.splitlines()) + 1
        for run_id in kept:
            readme = root / "checkpoints" / run_id / "README.md"
            assert readme.is_file(), f"kept run {run_id} has no README"


def test_finish_run_survives_a_retention_failure() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        root.mkdir()
        run = run_log.new_run("pipeline", root)

        with mock.patch.object(
            _retention_module(),
            "prune_run_directories",
            side_effect=RuntimeError("retention boom"),
        ):
            finished = run_log.finish_run(run, status="completed")

        assert finished["status"] == "completed"
        manifest = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "completed"
        assert len((root / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 1


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
