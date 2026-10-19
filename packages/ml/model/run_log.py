"""Run provenance: an append-only JSONL run log and a run manifest.

Every model run records one JSON object with its identity and outcome:

    {
      "run_id": "YYYYMMDDTHHMMSSZ-<config-hash8>",  # UTC stamp + 8-hex config hash
      "phase": "pipeline" | "phase_4_training",
      "status": "running" | "completed" | "failed",
      "git_sha": "<40-hex>" | null,                 # best-effort, null outside a repo
      "config_hash": "<8 hex>",
      "environment": {
        "python": "...", "<library>": "..." | null, ...,  # versions
        "PYTHONHASHSEED": "<value>" | null,              # process environment
        "threads": {"policy": ..., "n_jobs": ..., "omp_num_threads": ...},
        "seed": ..., "random_state": ...,
        "git_sha": "<40-hex>" | null,
        "lock_hash": "<64-hex>" | null,                  # sha256 of uv.lock
      },
      "started_at": "<UTC ISO-8601>",
      "finished_at": "<UTC ISO-8601>" | null,
      "duration_seconds": <float> | null,
      "phases": [{"phase": ..., "status": ..., "duration_seconds": ...}, ...],
      "folds": [{"fold": "fold_0", "status": ..., "duration_seconds": ...}, ...],
      "artifacts": {"<repo-relative path>": "<sha256 hex>", ...},
      "error": "<type>: <message>" | null,
      "selection": {"run_id": ..., "trial_index": ..., "config_hash": ...,  # absent without one
                    "artifact": "<repo-relative selection.json>"},
    }

A phase-4 run records the frozen tuning selection under ``selection`` when
the config names one (``tuning.selected``); the context identifies the
campaign and the selection artifact (the source of truth for the selected
parameters, frozen under ``training.hyperparameters``) without carrying the
parameters themselves.

``finish_run`` writes the object twice: pretty-printed to
``artifacts/run_manifest.json`` (the latest run) and as one strict-JSON line
appended to ``artifacts/runs.jsonl`` (one line per run, never truncated). Both
writers use ``allow_nan=False``, so a NaN fails the write instead of leaking a
non-standard token into the log.

Persisted paths are repo-relative (``to_repo_relative``); readers resolve them
back to absolute with ``resolve_stored_path``, which also accepts the legacy
absolute form stored before this module existed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from packages.ml.model import config as model_config
from packages.ml.model.utils import get_project_paths

RUNS_FILENAME = "runs.jsonl"
MANIFEST_FILENAME = "run_manifest.json"

# Key libraries whose versions are recorded with every run.
_KEY_LIBRARIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "joblib",
    "pyarrow",
    "PyYAML",
)

_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def config_hash(config: Mapping | None = None) -> str:
    """First 8 hex digits of the canonical config's SHA-256.

    ``None`` hashes the loaded default config; callers may pass an explicit
    mapping (e.g. a test's changed config) to get its digest instead.
    """
    loaded = model_config.get_config() if config is None else config
    payload = json.dumps(
        loaded, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def make_run_id(config_digest: str, now: datetime | None = None) -> str:
    """UTC timestamp plus the 8-hex config hash, e.g. ``20260101T000000Z-1a2b3c4d``."""
    stamp = (now if now is not None else _utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_digest}"


def selection_context() -> dict | None:
    """The frozen tuning selection named in the config, or ``None`` when absent.

    A phase-4 run records this context in its manifest so the run's recipe is
    traceable to the selection artifact -- the source of truth for the
    parameters frozen under ``training.hyperparameters``. The context names
    the campaign run id, the winning trial index, the campaign's config hash,
    and the artifact's repo-relative path; it never carries the parameters
    themselves. A config without ``tuning.selected`` returns ``None``; a
    present but malformed section fails by name rather than emitting a partial
    context.
    """
    loaded = model_config.get_config()
    selected = (loaded.get("tuning") or {}).get("selected")
    if selected is None:
        return None
    if not isinstance(selected, Mapping):
        raise ValueError(
            f"tuning.selected must be a mapping, got {type(selected).__name__}"
        )
    missing = [
        key for key in ("run_id", "trial_index", "config_hash") if key not in selected
    ]
    if missing:
        raise ValueError(f"tuning.selected is missing keys: {missing}")
    context = {
        "run_id": selected["run_id"],
        "trial_index": selected["trial_index"],
        "config_hash": selected["config_hash"],
    }
    artifacts = (loaded.get("paths") or {}).get("artifacts")
    if artifacts:
        context["artifact"] = (
            f"{artifacts}/tuning/{selected['run_id']}/selection.json"
        )
    return context


def git_sha(project_root: Path | str | None = None) -> str | None:
    """Best-effort ``HEAD`` sha; ``None`` when git or the repository is absent."""
    root = (
        Path(project_root)
        if project_root is not None
        else get_project_paths()["project_root"]
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = completed.stdout.strip()
    return sha if _GIT_SHA_RE.fullmatch(sha) else None


def environment_versions() -> dict[str, str | None]:
    """Python and key library versions; ``None`` when a distribution is absent."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in _KEY_LIBRARIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def thread_policy(config: Mapping | None = None) -> dict[str, Any]:
    """The declared bit-reproducibility policy, read from the config.

    The policy keeps one trainer thread (the top-level ``training.n_jobs`` pin;
    older configs without it fall back to the frozen
    ``training.hyperparameters.n_jobs``) and one OpenMP thread
    (``training.omp_num_threads``). The baseline seed and the frozen trainer
    seed are reported alongside, so a run record states the whole recipe's
    determinism inputs rather than only its thread count.
    """
    loaded = model_config.get_config() if config is None else config
    training = loaded.get("training") or {}
    hyperparameters = training.get("hyperparameters") or {}
    # The top-level pin wins over the frozen selection's historical value; the
    # hyperparameter fallback keeps legacy configs (no top-level key) working.
    n_jobs = training.get("n_jobs", hyperparameters.get("n_jobs", 1))
    return {
        "policy": str(training.get("determinism_policy", "bit-reproducible")),
        "seed": int(training.get("seed", 42)),
        "random_state": int(
            hyperparameters.get("random_state", training.get("seed", 42))
        ),
        "n_jobs": int(n_jobs),
        "omp_num_threads": int(training.get("omp_num_threads", 1)),
    }


def apply_thread_policy(config: Mapping | None = None) -> dict[str, Any]:
    """Enforce the declared OMP thread count and return the effective policy.

    ``OMP_NUM_THREADS`` is pinned to the configured value before any fit: the
    policy chooses bit-reproducibility over speed, so an inherited multi-thread
    setting is replaced rather than trusted. The returned mapping is the
    effective policy (the environment value as a string, plus the process's
    ``PYTHONHASHSEED``) and is what the run record persists.
    """
    policy = thread_policy(config)
    os.environ["OMP_NUM_THREADS"] = str(policy["omp_num_threads"])
    return {
        **policy,
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def lock_hash(project_root: Path | str | None = None) -> str | None:
    """SHA-256 of ``uv.lock``; ``None`` when the lockfile is absent."""
    root = (
        Path(project_root)
        if project_root is not None
        else get_project_paths()["project_root"]
    )
    try:
        return sha256_file(root / "uv.lock")
    except FileNotFoundError:
        return None


def environment_manifest(
    project_root: Path | str | None = None,
    config: Mapping | None = None,
) -> dict[str, Any]:
    """The per-run environment block: versions, hash seed, threads, provenance.

    Captured at run start after the thread policy is applied, so the recorded
    ``OMP_NUM_THREADS`` is the effective process value. ``git_sha`` and
    ``lock_hash`` are best-effort: both stay ``None`` outside a repository or
    without a lockfile, while the rest of the block is always present.
    """
    root = (
        Path(project_root)
        if project_root is not None
        else get_project_paths()["project_root"]
    )
    policy = apply_thread_policy(config)
    return {
        **environment_versions(),
        "PYTHONHASHSEED": policy["PYTHONHASHSEED"],
        "threads": {
            "policy": policy["policy"],
            "n_jobs": policy["n_jobs"],
            "omp_num_threads": policy["omp_num_threads"],
        },
        "seed": policy["seed"],
        "random_state": policy["random_state"],
        "git_sha": git_sha(root),
        "lock_hash": lock_hash(root),
    }


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file's bytes, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_repo_relative(
    path: Path | str, project_root: Path | str | None = None
) -> str:
    """Render an absolute path relative to the repository root (POSIX form).

    Paths already stored relative are returned normalized as-is. A path outside
    the repository (e.g. a test's temporary artifact root) becomes a
    ``..``-relative string that ``resolve_stored_path`` still resolves.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    root = (
        Path(project_root)
        if project_root is not None
        else get_project_paths()["project_root"]
    )
    return Path(os.path.relpath(candidate, root)).as_posix()


def resolve_stored_path(
    stored: Path | str, project_root: Path | str | None = None
) -> Path:
    """Resolve a stored path: repo-relative against the root, absolute unchanged."""
    candidate = Path(stored)
    if candidate.is_absolute():
        return candidate
    root = (
        Path(project_root)
        if project_root is not None
        else get_project_paths()["project_root"]
    )
    return Path(os.path.normpath(root / candidate))


def new_run(
    phase: str,
    artifacts_dir: Path | str | None = None,
    config: Mapping | None = None,
) -> dict[str, Any]:
    """Start a run record: identity, config hash, environment, and start time.

    The returned dict is the mutable record handed to the ``record_*`` helpers
    and finalized by ``finish_run``. Keys prefixed with ``_`` are internal state
    and never reach the manifest.
    """
    paths = get_project_paths()
    loaded = model_config.get_config() if config is None else config
    digest = config_hash(loaded)
    environment = environment_manifest(paths["project_root"], config=loaded)
    return {
        "run_id": make_run_id(digest),
        "phase": phase,
        "status": "running",
        "git_sha": environment["git_sha"],
        "config_hash": digest,
        "environment": environment,
        "started_at": _utc_now().isoformat(),
        "finished_at": None,
        "duration_seconds": None,
        "phases": [],
        "folds": [],
        "artifacts": {},
        "error": None,
        "_artifacts_dir": (
            Path(artifacts_dir) if artifacts_dir is not None else paths["artifacts_dir"]
        ),
        "_project_root": paths["project_root"],
        "_started_monotonic": time.monotonic(),
        "_pending_artifacts": [],
    }


def record_phase(
    run: dict[str, Any], phase: str, status: str, duration_seconds: float
) -> None:
    """Append one phase outcome (name, status, duration) to the run record."""
    run["phases"].append(
        {
            "phase": phase,
            "status": status,
            "duration_seconds": float(duration_seconds),
        }
    )


def record_fold(
    run: dict[str, Any], fold: str, status: str, duration_seconds: float
) -> None:
    """Append one fold outcome to the run record."""
    run["folds"].append(
        {
            "fold": fold,
            "status": status,
            "duration_seconds": float(duration_seconds),
        }
    )


def record_artifacts(run: dict[str, Any], paths: Iterable[Path | str]) -> None:
    """Queue artifact files whose SHA-256 lands in the finalized record."""
    run["_pending_artifacts"].extend(Path(path) for path in paths)


def write_manifest(
    record: Mapping[str, Any], manifest_path: Path | str | None = None
) -> Path:
    """Write the pretty-printed manifest (strict JSON) and return its path."""
    target = (
        Path(manifest_path)
        if manifest_path is not None
        else get_project_paths()["artifacts_dir"] / MANIFEST_FILENAME
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")
    return target


def append_run(
    record: Mapping[str, Any], runs_path: Path | str | None = None
) -> Path:
    """Append one strict-JSON line to the run log and return its path."""
    target = (
        Path(runs_path)
        if runs_path is not None
        else get_project_paths()["artifacts_dir"] / RUNS_FILENAME
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, allow_nan=False, default=str)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return target


def finish_run(
    run: dict[str, Any], status: str = "completed", error: str | None = None
) -> dict[str, Any]:
    """Finalize a run: hash queued artifacts, write the manifest, append the log line."""
    run["status"] = status
    run["error"] = error
    run["finished_at"] = _utc_now().isoformat()
    run["duration_seconds"] = time.monotonic() - run.pop("_started_monotonic")
    artifacts_dir = run.pop("_artifacts_dir")
    project_root = run.pop("_project_root")
    pending = run.pop("_pending_artifacts")
    run["artifacts"] = {
        to_repo_relative(path, project_root): sha256_file(path) for path in pending
    }
    write_manifest(run, artifacts_dir / MANIFEST_FILENAME)
    append_run(run, artifacts_dir / RUNS_FILENAME)
    return run
