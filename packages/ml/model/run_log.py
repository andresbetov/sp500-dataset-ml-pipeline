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

The manifest is also written at run start and after each fold (``new_run`` /
``write_run_state``), without appending a log line, so a hard interruption
still leaves the identity, fold hash, and fold selection a ``--resume`` needs.
Manifest writes are atomic (sibling ``*.tmp`` + ``os.replace``); the JSONL log
is append-only. After appending the finalized line, ``finish_run`` applies
run-directory retention (``run.retention``) without touching the log.

Persisted paths are repo-relative (``to_repo_relative``); readers resolve them
back to absolute with ``resolve_stored_path``, which also accepts the legacy
absolute form stored before this module existed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from packages.ml.model import atomic
from packages.ml.model import config as model_config
from packages.ml.model.utils import get_project_paths

RUNS_FILENAME = "runs.jsonl"
MANIFEST_FILENAME = "run_manifest.json"

logger = logging.getLogger(__name__)


class ResumeError(RuntimeError):
    """A resume request lacks the run identity or sidecar prerequisites."""


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
    mapping (e.g. a test's changed config) to get its digest instead. The
    operational ``run`` section (retention count, log path) is excluded: those
    keys change where/how run history is kept, not what a fold trains under,
    so they must not invalidate a resume of the same training recipe.
    """
    loaded = model_config.get_config() if config is None else config
    training_config = {
        key: value for key, value in loaded.items() if key != "run"
    }
    payload = json.dumps(
        training_config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def make_run_id(config_digest: str, now: datetime | None = None) -> str:
    """UTC timestamp plus the 8-hex config hash, e.g. ``20260101T000000Z-1a2b3c4d``."""
    stamp = (now if now is not None else _utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_digest}"


def load_manifest(
    artifacts_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load the latest run manifest, refusing a missing or malformed document.

    ``--resume`` reads this manifest for the run id and identity hashes; a
    missing manifest is a named resume failure, never a silent fresh run.
    """
    if manifest_path is not None:
        target = Path(manifest_path)
    else:
        root = (
            Path(artifacts_dir)
            if artifacts_dir is not None
            else get_project_paths()["artifacts_dir"]
        )
        target = root / MANIFEST_FILENAME
    if not target.is_file():
        raise ResumeError(
            f"cannot resume: run manifest {target} is missing; a resumable run "
            "must have completed at least its run-start manifest write"
        )
    try:
        with open(target, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(
            f"cannot resume: run manifest {target} is unreadable: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ResumeError(
            f"cannot resume: run manifest {target} is not a JSON object"
        )
    return manifest


def resume_identity(
    artifacts_dir: Path | str | None = None,
    *,
    config_hash_value: str | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """The manifest identity a resume depends on, validated by name.

    A resumable manifest must name its ``run_id``, the ``config_hash`` the run
    trained under, and the ``fold_manifest_hash`` of the fold construction. The
    current config hash must equal the recorded one. Any missing key or
    mismatch raises :class:`ResumeError` naming the offending identity field.
    """
    manifest = load_manifest(artifacts_dir, manifest_path)
    required = ("run_id", "config_hash", "fold_manifest_hash")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        target = (
            Path(manifest_path)
            if manifest_path is not None
            else (
                Path(artifacts_dir)
                if artifacts_dir is not None
                else get_project_paths()["artifacts_dir"]
            )
            / MANIFEST_FILENAME
        )
        raise ResumeError(
            f"cannot resume: run manifest {target} is missing required "
            f"identity key(s): {', '.join(missing)}"
        )

    current = config_hash() if config_hash_value is None else config_hash_value
    if manifest["config_hash"] != current:
        raise ResumeError(
            "cannot resume: config hash mismatch — the run manifest records "
            f"{manifest['config_hash']}, the current config is {current}; "
            "resume only a run trained under the same config"
        )

    if artifacts_dir is not None:
        root = Path(artifacts_dir)
    elif manifest_path is not None:
        root = Path(manifest_path).parent
    else:
        root = get_project_paths()["artifacts_dir"]
    _verify_resume_inputs(manifest, root)
    return manifest


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


INPUT_IDENTITY_KEYS = (
    "dataset_sha256",
    "dataset_path",
    "x_sha256",
    "y_sha256",
)


def input_identity(
    artifacts_dir: Path | str | None = None,
    metadata: Mapping | None = None,
) -> dict[str, Any]:
    """The Phase-2 data identity a resume must match.

    Records the dataset content digest pinned in the Phase-2 metadata plus the
    byte hashes of the persisted ``X.npy``/``Y.npy`` inputs. ``dataset_path``
    and ``dataset_sha256`` are ``None`` for synthetic/legacy callers without a
    dataset pin; production Phase 2 always records both.
    """
    root = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else get_project_paths()["artifacts_dir"]
    )
    pin = (metadata or {}).get("dataset") or {}
    dataset_path = pin.get("path") if isinstance(pin, Mapping) else None
    dataset_sha = pin.get("sha256") if isinstance(pin, Mapping) else None
    if dataset_sha is None and dataset_path:
        candidate = Path(dataset_path)
        if candidate.is_file():
            dataset_sha = sha256_file(candidate)
    return {
        "dataset_sha256": dataset_sha,
        # Persist the same repo-relative path contract as every other stored
        # manifest path; absolute machine paths never enter runs.jsonl.
        "dataset_path": (
            to_repo_relative(dataset_path) if dataset_path else None
        ),
        "x_sha256": sha256_file(root / "inputs" / "X.npy"),
        "y_sha256": sha256_file(root / "inputs" / "Y.npy"),
    }


def _verify_resume_inputs(manifest: Mapping[str, Any], artifacts_dir: Path) -> None:
    """Verify persisted inputs and fold construction before any rewrite.

    ``--resume`` must not let Phase 2/3 overwrite the data/identity a run is
    being resumed against before the mismatch is known. This checks the
    manifest's ``input_identity`` against the on-disk dataset and input arrays,
    then loads the persisted fold payload through the artifact registry and
    recomputes its manifest hash. Every failure is a named
    :class:`ResumeError`.
    """
    expected = manifest.get("input_identity")
    if not isinstance(expected, Mapping):
        raise ResumeError(
            "cannot resume: run manifest is missing input_identity "
            "(dataset digest + X/Y hashes); start a fresh run to record it"
        )
    missing = [key for key in INPUT_IDENTITY_KEYS if key not in expected]
    if missing:
        raise ResumeError(
            "cannot resume: run manifest input_identity is missing key(s): "
            + ", ".join(missing)
        )

    inputs_dir = artifacts_dir / "inputs"
    for key, filename in (("x_sha256", "X.npy"), ("y_sha256", "Y.npy")):
        path = inputs_dir / filename
        if not path.is_file():
            raise ResumeError(
                f"cannot resume: persisted input {path} is missing; the run's "
                "Phase-2 inputs are required to reuse its folds"
            )
        actual = sha256_file(path)
        if actual != expected[key]:
            raise ResumeError(
                f"cannot resume: input digest mismatch for {path} — manifest "
                f"records {expected[key]}, file has {actual}"
            )

    dataset_sha = expected.get("dataset_sha256")
    if dataset_sha:
        dataset_path = expected.get("dataset_path")
        candidate = (
            resolve_stored_path(dataset_path)
            if dataset_path
            else get_project_paths()["dataset_path"]
        )
        if not candidate.is_file():
            raise ResumeError(
                f"cannot resume: dataset {candidate} is missing; the run's "
                "dataset identity cannot be verified"
            )
        actual = sha256_file(candidate)
        if actual != dataset_sha:
            raise ResumeError(
                f"cannot resume: dataset digest mismatch for {candidate} — "
                f"manifest records {dataset_sha}, file has {actual}; the "
                "dataset changed since the run's folds were built"
            )

    folds_path = artifacts_dir / "folds" / "folds_metadata.pkl"
    if not folds_path.is_file():
        raise ResumeError(
            f"cannot resume: persisted fold metadata {folds_path} is missing; "
            "the run's fold construction cannot be verified"
        )
    from packages.ml.model.artifact_integrity import load_verified
    from packages.ml.model.cross_validation.cross_validator import (
        fold_manifest_hash,
    )

    registry_path = artifacts_dir / "artifact_registry.json"
    try:
        folds_metadata = load_verified(folds_path, registry_path=registry_path)
        actual_hash = fold_manifest_hash(folds_metadata)
    except Exception as exc:
        raise ResumeError(
            f"cannot resume: persisted fold metadata {folds_path} failed "
            f"verification: {type(exc).__name__}: {exc}"
        ) from exc
    expected_hash = manifest["fold_manifest_hash"]
    if actual_hash != expected_hash:
        raise ResumeError(
            "cannot resume: fold manifest mismatch — the run manifest records "
            f"{expected_hash}, the persisted folds hash to {actual_hash}; "
            "refusing before Phase 2/3 rewrites the source artifacts"
        )


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
    run_id: str | None = None,
) -> dict[str, Any]:
    """Start a run record: identity, config hash, environment, and start time.

    ``run_id`` reuses a prior run's identity for ``--resume``; omitting it
    mints a new UTC-stamped id. The returned dict is the mutable record handed
    to the ``record_*`` helpers and finalized by ``finish_run``. Keys prefixed
    with ``_`` are internal state and never reach the manifest.
    """
    paths = get_project_paths()
    loaded = model_config.get_config() if config is None else config
    digest = config_hash(loaded)
    environment = environment_manifest(paths["project_root"], config=loaded)
    record = {
        "run_id": run_id if run_id is not None else make_run_id(digest),
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
    # Persist a run-start manifest for a fresh run so a hard interruption
    # before finish_run still leaves the identity a later --resume needs. A
    # resumed run keeps the existing manifest until its state write, so the
    # identity it is being resumed from is never overwritten with a start
    # record that lacks the fold hash.
    if run_id is None:
        write_run_state(record)
    return record


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
    """Write the pretty-printed manifest (strict JSON) atomically.

    The document goes through a sibling ``*.tmp`` file and ``os.replace``, so
    a mid-write failure leaves the previous manifest byte-identical and no
    ``*.tmp`` residue behind.
    """
    target = (
        Path(manifest_path)
        if manifest_path is not None
        else get_project_paths()["artifacts_dir"] / MANIFEST_FILENAME
    )

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(
                record, handle, indent=2, sort_keys=True, allow_nan=False, default=str
            )
            handle.write("\n")

    atomic.write_via_tmp(target, _write)
    return target


def write_run_state(run: Mapping[str, Any]) -> Path:
    """Persist the current public run record without finalizing it.

    Used at run start and after each fold so a hard interruption still leaves
    the identity, fold hash, and fold selection a later ``--resume`` needs.
    Internal ``_`` state and pending-artifact hashes stay in memory until
    :func:`finish_run`.
    """
    artifacts_dir = run.get("_artifacts_dir")
    if artifacts_dir is None:
        artifacts_dir = get_project_paths()["artifacts_dir"]
    public = {key: value for key, value in run.items() if not key.startswith("_")}
    return write_manifest(public, Path(artifacts_dir) / MANIFEST_FILENAME)


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
    # Run-directory retention is a directory policy only: runs.jsonl has just
    # been appended and is never rewritten or truncated. A retention failure
    # must not mask the run's own outcome (or the exception being handled).
    from packages.ml.model.retention import prune_run_directories

    try:
        prune_run_directories(artifacts_dir, current_run=run)
    except Exception:
        logger.exception(
            "Retention failed after run %s; the run record and runs.jsonl "
            "line are preserved",
            run.get("run_id"),
        )
    return run
