"""Tests pinning GitHub Actions workflow dependencies to immutable commits.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_workflow_pins.py
"""

from __future__ import annotations

import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from conftest import REPO_ROOT

WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
CHECK_WORKFLOW = WORKFLOW_DIR / "checks.yml"
SMOKE_WORKFLOW = WORKFLOW_DIR / "model-smoke.yml"
DEPENDABOT_PATH = REPO_ROOT / ".github" / "dependabot.yml"

WORKFLOWS = (CHECK_WORKFLOW, SMOKE_WORKFLOW)

# Action -> (commit, release tag, expected number of uses).
EXPECTED_PINS = {
    "actions/checkout": ("11d5960a326750d5838078e36cf38b85af677262", "v4.4.0", 3),
    "astral-sh/setup-uv": ("d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86", "v5.4.2", 3),
    "actions/cache": ("0057852bfaa89a56745cba8c7296529d2fc39830", "v4.3.0", 2),
}
EXPECTED_USES_COUNT = 8

# Workflow structure at the branch point: jobs and their run commands must not
# change while the action refs are pinned.
EXPECTED_JOBS = {
    "checks.yml": ("static", "dataset-smoke"),
    "model-smoke.yml": ("model",),
}
EXPECTED_RUNS = {
    "checks.yml": (
        "uv sync --frozen",
        "uv run --frozen python -m compileall -q packages",
        "uv run --frozen python -m pytest packages/ml/model/tests -q",
        "uv sync --frozen",
        "uv run --frozen python packages/ml/dataset/main.py 2>&1 | tail -c 4000",
        "test -f data/processed/dataset.parquet\n"
        "uv run --frozen python -c \"import os, pyarrow.parquet as pq; "
        "p='data/processed/dataset.parquet'; m=pq.ParquetFile(p).metadata; "
        "print(f'dataset.parquet: rows={m.num_rows} cols={m.num_columns} "
        "bytes={os.path.getsize(p)}')\" | tee -a \"$GITHUB_STEP_SUMMARY\"\n",
    ),
    "model-smoke.yml": (
        "uv sync --frozen",
        "uv run --frozen python packages/ml/dataset/main.py 2>&1 | tail -c 2000",
        "uv run --frozen python packages/ml/model/main.py 2>&1 | tail -c 4000",
        "test -f packages/ml/model/artifacts/results/fold_training_summary.json\n"
        "echo \"model phase OK: fold_training_summary.json written\" "
        "| tee -a \"$GITHUB_STEP_SUMMARY\"\n",
    ),
}

_USES_RE = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<target>\S+)(?P<rest>.*)$")
_PIN_RE = re.compile(r"^(?P<action>[^@\s]+)@(?P<sha>[0-9a-f]{40})$")
_COMMENT_RE = re.compile(r"^\s+#\s+v\d+\.\d+\.\d+\s*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _uses_entries() -> list[tuple[Path, int, str, re.Match[str]]]:
    """Return (path, line number, stripped line, match) for every uses: line."""
    entries = []
    for path in WORKFLOWS:
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = _USES_RE.match(line)
            if match is not None:
                entries.append((path, lineno, line.strip(), match))
    return entries


def _pin_parts(entry: tuple[Path, int, str, re.Match[str]]) -> re.Match[str] | None:
    return _PIN_RE.match(entry[3].group("target"))


def _runs(path: Path) -> tuple[str, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    commands = []
    for job in data["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                commands.append(step["run"])
    return tuple(commands)


def test_every_action_is_pinned_with_tag_comment() -> None:
    entries = _uses_entries()

    assert len(entries) == EXPECTED_USES_COUNT, (
        f"expected {EXPECTED_USES_COUNT} uses: lines across {len(WORKFLOWS)} "
        f"workflows, found {len(entries)}"
    )

    problems = []
    for path, lineno, line, match in entries:
        pin = _pin_parts((path, lineno, line, match))
        if pin is None:
            problems.append(f"{path.name}:{lineno}: not a commit pin: {line}")
        elif _COMMENT_RE.match(match.group("rest")) is None:
            problems.append(
                f"{path.name}:{lineno}: missing '# vX.Y.Z' tag comment: {line}"
            )

    assert not problems, (
        "every action must be pinned to a 40-character commit SHA with a "
        "trailing '# vX.Y.Z' tag comment:\n" + "\n".join(problems)
    )


def test_exact_pins() -> None:
    observed: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in _uses_entries():
        pin = _pin_parts(entry)
        if pin is not None:
            observed[pin.group("action")][pin.group("sha")] += 1

    problems = []
    for action, (sha, tag, count) in EXPECTED_PINS.items():
        expected = Counter({sha: count})
        actual = observed.get(action, Counter())
        if actual != expected:
            problems.append(
                f"{action} must be {sha} ({tag}) exactly {count} time(s); "
                f"found {dict(actual) or 'no pinned uses'}"
            )

    unexpected = sorted(set(observed) - set(EXPECTED_PINS))
    if unexpected:
        problems.append(f"unexpected pinned action(s): {unexpected}")

    assert not problems, "\n".join(problems)


def test_no_floating_action_refs() -> None:
    floating = [
        f"{path.name}:{lineno}: {line}"
        for path, lineno, line, match in _uses_entries()
        if _SHA_RE.match(match.group("target").rsplit("@", 1)[-1]) is None
    ]

    assert not floating, (
        "uses: refs must be immutable commit SHAs, not tags or branches:\n"
        + "\n".join(floating)
    )


def test_dependabot_enables_weekly_action_updates() -> None:
    assert DEPENDABOT_PATH.is_file(), f"missing {DEPENDABOT_PATH}"

    data = yaml.safe_load(DEPENDABOT_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "dependabot.yml must contain a YAML mapping"
    assert data.get("version") == 2, f"expected version: 2, got {data.get('version')!r}"

    updates = data.get("updates")
    assert isinstance(updates, list) and updates, "dependabot.yml has no updates entries"

    action_updates = [
        update
        for update in updates
        if isinstance(update, dict)
        and update.get("package-ecosystem") == "github-actions"
    ]
    assert action_updates, "no package-ecosystem: github-actions update entry"

    assert any(
        update.get("directory") == "/"
        and update.get("schedule", {}).get("interval") == "weekly"
        for update in action_updates
    ), "github-actions updates must target '/' on a weekly schedule"


def test_workflow_logic_unchanged() -> None:
    for path in WORKFLOWS:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        job_names = tuple(data["jobs"])
        assert job_names == EXPECTED_JOBS[path.name], (
            f"{path.name}: job names changed: {job_names}"
        )
        assert _runs(path) == EXPECTED_RUNS[path.name], (
            f"{path.name}: run commands changed from the branch point"
        )

    checks_text = CHECK_WORKFLOW.read_text(encoding="utf-8")
    assert (
        "        run: uv run --frozen python -m compileall -q packages"
        in checks_text
    ), "byte-compile command changed"
    assert (
        "        run: uv run --frozen python -m pytest packages/ml/model/tests -q"
        in checks_text
    ), "pytest step command changed"


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
