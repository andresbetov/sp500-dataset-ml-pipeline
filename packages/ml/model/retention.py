"""Run-directory retention and per-directory provenance READMEs.

The run-scoped checkpoint directories live under
``artifacts/checkpoints/<run_id>/``. Retention keeps the newest ``N`` run
directories, where ``N`` is ``run.retention`` in the model config, and removes
the older ones. The append-only run log (``artifacts/runs.jsonl``) is never
touched: retention prunes directories, not history.

Every kept run directory carries a ``README.md`` naming the run id, its
config/fold-manifest identity, its role, and the retention rule. A directory
created by the pipeline gets a README at run start; synthetic or legacy
directories kept by a prune get one written on demand.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from packages.ml.model import atomic
from packages.ml.model.config import get_config

logger = logging.getLogger(__name__)

README_FILENAME = "README.md"
CHECKPOINTS_DIRNAME = "checkpoints"

# The run-id form: UTC stamp plus the config hash, e.g.
# ``20260101T000000Z-1a2b3c4d``. Only directories matching this form are
# retention candidates; anything else under ``checkpoints/`` is left alone.
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")

DEFAULT_RETENTION = 5


def configured_retention() -> int:
    """The configured ``run.retention`` count (default 5 when absent)."""
    loaded = get_config()
    run_section = loaded.get("run") or {}
    value = run_section.get("retention", DEFAULT_RETENTION)
    try:
        keep = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"run.retention must be an integer, got {value!r}") from exc
    if keep < 1:
        raise ValueError(f"run.retention must be positive, got {keep}")
    return keep


def run_readme_text(
    run_id: str, run: Mapping[str, Any] | None = None
) -> str:
    """The per-directory README for one run-scoped checkpoint directory."""
    lines = [
        f"# Run {run_id}",
        "",
        "This directory holds the run-scoped fold checkpoints and sidecars.",
        "",
        f"- Run id: `{run_id}`",
    ]
    if run is not None:
        if run.get("config_hash"):
            lines.append(f"- Config hash: `{run['config_hash']}`")
        if run.get("fold_manifest_hash"):
            lines.append(f"- Fold manifest hash: `{run['fold_manifest_hash']}`")
        if run.get("status"):
            lines.append(f"- Status: `{run['status']}`")
        selection = run.get("fold_selection")
        if isinstance(selection, Mapping):
            lines.append(
                "- Fold selection: "
                f"requested {selection.get('requested')}, "
                f"trained {selection.get('trained')}, "
                f"resumed {selection.get('resumed')}"
            )
        if run.get("resumed"):
            lines.append("- Resumed from a previous invocation: yes")
    lines.extend(
        [
            "",
            "## Contents",
            "",
            "- `fold_<i>.pkl`: trained model checkpoint",
            "- `fold_<i>_params.json`: full training parameter sidecar",
            "- `fold_<i>_summary.json`: fold metric/identity record",
            "- `fold_<i>_predictions.pkl`: fold predictions used by resumes",
            "",
            "## Retention",
            "",
            "Run directories are pruned to the newest `run.retention` entries "
            "(model `config/base.yaml`). `artifacts/runs.jsonl` is append-only "
            "and is never pruned, so run history survives directory retention.",
            "",
        ]
    )
    return "\n".join(lines)


def ensure_run_readme(
    run_dir: Path | str,
    *,
    run: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the run directory's README (atomically) when it is absent.

    ``overwrite`` refreshes a README for the current run so its recorded
    selection/status stays current; a generic prune never overwrites an
    existing README.
    """
    directory = Path(run_dir)
    readme = directory / README_FILENAME
    if readme.exists() and not overwrite:
        return readme
    directory.mkdir(parents=True, exist_ok=True)
    run_id = run.get("run_id") if run is not None else None
    atomic.write_text_atomically(
        readme, run_readme_text(run_id or directory.name, run)
    )
    return readme


def _run_directories(checkpoints_dir: Path) -> list[Path]:
    if not checkpoints_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and RUN_ID_RE.fullmatch(path.name)
        ),
        key=lambda path: path.name,
    )


def prune_run_directories(
    artifacts_dir: Path | str,
    keep: int | None = None,
    *,
    current_run: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Keep the newest ``keep`` run directories and remove the rest.

    Candidates are the run-id-shaped directories under
    ``artifacts/checkpoints/``; the current run (when given) is always kept,
    even if a clock skew would sort it outside the last ``keep`` names. Every
    kept directory gets a README if it has none. ``runs.jsonl`` and every
    non-candidate entry are untouched. Returns the removed directories.
    """
    limit = configured_retention() if keep is None else int(keep)
    if limit < 1:
        raise ValueError(f"retention must keep at least one run directory, got {limit}")

    checkpoints_dir = Path(artifacts_dir) / CHECKPOINTS_DIRNAME
    directories = _run_directories(checkpoints_dir)
    kept_names = {path.name for path in directories[-limit:]}
    current_id = current_run.get("run_id") if current_run is not None else None
    if current_id:
        kept_names.add(str(current_id))

    pruned: list[Path] = []
    for directory in directories:
        if directory.name in kept_names:
            continue
        shutil.rmtree(directory)
        pruned.append(directory)
        logger.info("Retention pruned run directory %s", directory)

    for directory in directories:
        if directory.name in kept_names and directory.is_dir():
            run = current_run if directory.name == current_id else None
            ensure_run_readme(directory, run=run, overwrite=run is not None)
    if pruned:
        logger.info(
            "Retention kept %d run directories, pruned %d",
            len(kept_names),
            len(pruned),
        )
    return pruned
