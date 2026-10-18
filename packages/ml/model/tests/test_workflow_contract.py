"""Contract tests for the manual model-smoke workflow.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_workflow_contract.py
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import traceback
from pathlib import Path

import yaml

from conftest import REPO_ROOT

SMOKE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "model-smoke.yml"
MODEL_STEP_NAME = "Run Phases 2-5 (model)"
PIPELINE_COMMAND = (
    "uv run --frozen python packages/ml/model/main.py 2>&1 | tail -c 4000"
)
PNG_GLOB = "data/model_outputs/*.png"
PNG_MINIMUM = 14
MASKING_PIPELINE = "false | tail -c 100; echo survived"


def _workflow() -> dict:
    return yaml.safe_load(SMOKE_WORKFLOW.read_text(encoding="utf-8"))


def _steps() -> list[dict]:
    return _workflow()["jobs"]["model"]["steps"]


def _step_by_name(name: str) -> dict:
    for step in _steps():
        if step.get("name") == name:
            return step
    raise AssertionError(f"workflow step not found: {name!r}")


def _post_check_script() -> str:
    for step in _steps():
        script = step.get("run", "")
        if "fold_training_summary.json" in script:
            return script
    raise AssertionError(
        "no workflow step checks packages/ml/model/artifacts/results/"
        "fold_training_summary.json"
    )


def _png_check_script() -> str:
    """Return the post-check script from the PNG glob onward."""
    lines = _post_check_script().splitlines()
    for index, line in enumerate(lines):
        if "shopt -s nullglob" in line:
            return "\n".join(lines[index:]).strip()
    raise AssertionError("post-check step has no nullglob PNG count")


def _run_png_check(script: str, png_count: int) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix=f"png-check-{png_count}-") as tmp:
        root = Path(tmp)
        outputs = root / "data" / "model_outputs"
        outputs.mkdir(parents=True)
        for index in range(png_count):
            (outputs / f"chart_{index:02d}.png").write_bytes(b"")
        env = {
            **os.environ,
            "GITHUB_STEP_SUMMARY": str(root / "step-summary.md"),
        }
        return subprocess.run(
            ["bash", "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


def _run_bash(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_model_step_uses_pipefail_and_capped_tail() -> None:
    step = _step_by_name(MODEL_STEP_NAME)
    script = step.get("run", "")

    assert step.get("shell") == "bash", (
        "the model pipeline step must declare 'shell: bash'"
    )
    assert "set -o pipefail" in script, (
        "the model pipeline step must enable pipefail before the pipeline"
    )
    assert PIPELINE_COMMAND in script, (
        f"the capped pipeline command changed: {PIPELINE_COMMAND!r}"
    )
    pipeline_line = [line for line in script.splitlines() if PIPELINE_COMMAND in line]
    assert pipeline_line and pipeline_line[-1].rstrip().endswith("| tail -c 4000"), (
        "the pipeline command must still end with the output-capping tail"
    )


def test_pipefail_is_required_to_signal_pipeline_failure() -> None:
    without_pipefail = _run_bash("-e", "-c", MASKING_PIPELINE)
    assert without_pipefail.returncode == 0, (
        "expected 'bash -e' to mask the failing pipeline, got "
        f"exit {without_pipefail.returncode}: {without_pipefail.stdout!r}"
    )
    assert "survived" in without_pipefail.stdout

    with_pipefail = _run_bash("-eo", "pipefail", "-c", MASKING_PIPELINE)
    assert with_pipefail.returncode != 0, (
        "expected 'bash -eo pipefail' to surface the failing pipeline, got "
        f"exit {with_pipefail.returncode}: {with_pipefail.stdout!r}"
    )
    assert "survived" not in with_pipefail.stdout


def test_post_check_requires_fold_summary_and_phase_5_pngs() -> None:
    script = _post_check_script()

    assert (
        "test -f packages/ml/model/artifacts/results/fold_training_summary.json"
        in script
    ), "post-check must still require the fold training summary"
    assert "fold_training_summary.json written" in script
    assert PNG_GLOB in script, f"post-check must count {PNG_GLOB}"
    assert f'test "$png_count" -ge {PNG_MINIMUM}' in script, (
        f"post-check must require at least {PNG_MINIMUM} Phase-5 PNG charts"
    )

    png_script = _png_check_script()

    too_few = _run_png_check(png_script, 3)
    assert too_few.returncode != 0, (
        "the PNG post-check must fail with 3 charts: "
        f"{too_few.stdout!r} {too_few.stderr!r}"
    )
    assert "3 PNG" in too_few.stdout

    enough = _run_png_check(png_script, PNG_MINIMUM)
    assert enough.returncode == 0, (
        "the PNG post-check must pass with 14 charts: "
        f"{enough.stdout!r} {enough.stderr!r}"
    )
    assert "14 PNG" in enough.stdout


def test_workflow_trigger_permissions_and_job_unchanged() -> None:
    data = _workflow()

    triggers = data.get("on", data.get(True))
    assert isinstance(triggers, dict), f"workflow triggers changed: {triggers!r}"
    assert set(triggers) == {"workflow_dispatch"}, (
        "the manual smoke workflow must not gain push or pull_request triggers: "
        f"{sorted(triggers)}"
    )
    assert triggers["workflow_dispatch"] is None
    assert data.get("permissions") == {"contents": "read"}, (
        f"workflow permissions changed: {data.get('permissions')!r}"
    )
    assert tuple(data["jobs"]) == ("model",), (
        f"workflow job names changed: {tuple(data['jobs'])}"
    )


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
