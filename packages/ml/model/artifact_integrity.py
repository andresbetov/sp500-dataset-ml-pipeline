"""Size- and sha256-verified artifact loads (integrity, not authenticity).

Every live ``joblib.load`` in the model package goes through ``load_verified``:
the file's byte size and SHA-256 are compared against the entry recorded for it
in the artifact registry before the pickle is deserialized. A missing entry, a
missing file, a size mismatch, or a hash mismatch raises
``ArtifactIntegrityError`` and ``joblib.load`` is never called, so a tampered
or swapped artifact cannot execute.

The registry is ``artifact_registry.json`` in the artifacts root. Each writer
site calls ``record_artifact`` immediately after the artifact lands on disk;
the update is atomic (sibling ``*.tmp`` + ``os.replace``), so a reader later in
the same run finds the entry the moment the artifact exists. The run manifest's
artifact list remains the finalized-run record; this registry is the
incremental hash store that backs same-run verification. Entries are keyed by
the repo-relative path (``run_log.to_repo_relative``) and carry both ``size``
and ``sha256``.

A matching hash proves integrity, not authenticity: an attacker who can rewrite
the artifact can also rewrite the registry entry, so this guards against
corruption and accidental substitution, not against a writer with access to the
registry.

The module also hosts the one sanctioned xgboost native-format decode,
``load_native_model``: it reads the source bytes first and refuses empty or
unrecognized input with a named error before any native call, because
``Booster.load_model`` on a zero-length buffer aborts the interpreter
(SIGABRT, uncatchable). The native formats stay prediction-only: the pkl and
its params sidecar carry the training recipe.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import xgboost as xgb

from packages.ml.model.run_log import (
    resolve_stored_path,
    sha256_file,
    to_repo_relative,
)
from packages.ml.model.utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# The registry lives beside the run manifest in the artifacts root.
REGISTRY_FILENAME = "artifact_registry.json"


class ArtifactIntegrityError(RuntimeError):
    """An artifact failed size/sha256 verification; the load was refused."""


class EmptyModelBufferError(ArtifactIntegrityError):
    """A native model source is empty; the decoder would abort the process."""


class UnrecognizedModelFormatError(ArtifactIntegrityError):
    """A native model source does not start like a JSON or UBJSON model."""


def _default_registry_path() -> Path:
    return ARTIFACTS_DIR / REGISTRY_FILENAME


def _read_registry(registry_path: Path) -> dict[str, Any]:
    """The registry document, or ``{}`` when it does not exist yet."""
    if not registry_path.is_file():
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(
            f"cannot read artifact registry {registry_path}: {error}"
        ) from error
    if not isinstance(registry, dict):
        raise ArtifactIntegrityError(
            f"artifact registry {registry_path} is not a JSON object"
        )
    return registry


def _write_registry(registry_path: Path, registry: dict[str, Any]) -> None:
    """Write the registry via a sibling ``*.tmp`` + ``os.replace``.

    A reader never observes a partially written registry; any failure removes
    the temporary file, so no ``*.tmp`` residue is left behind.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_path.with_name(registry_path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(registry, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(tmp_path, registry_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def record_artifact(
    path: Path | str,
    *,
    registry_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Record ``path``'s byte size and SHA-256 in the artifact registry.

    Called by each writer site right after the artifact is written; the
    registry update is atomic, so a same-run reader that needs the entry later
    always finds it. Returns the recorded entry (``{"size", "sha256"}``).
    """
    artifact = Path(path)
    if not artifact.is_absolute():
        artifact = resolve_stored_path(artifact, project_root)
    size = artifact.stat().st_size
    digest = sha256_file(artifact)

    target = Path(registry_path) if registry_path is not None else _default_registry_path()
    registry = _read_registry(target)
    stored = to_repo_relative(artifact, project_root)
    entry = {"size": size, "sha256": digest}
    registry[stored] = entry
    _write_registry(target, registry)
    logger.debug(
        "Recorded artifact %s (size %d, sha256 %s...) in %s",
        stored,
        size,
        digest[:12],
        target,
    )
    return entry


def load_verified(
    path: Path | str,
    *,
    registry_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> Any:
    """Verify ``path`` against its registry entry, then ``joblib.load`` it.

    The registry entry must exist, the file must be present, and its byte size
    and SHA-256 must both match before the pickle is deserialized; any
    deviation raises ``ArtifactIntegrityError`` and no load call happens. The
    hashing time is logged.
    """
    resolved = resolve_stored_path(path, project_root)
    target = Path(registry_path) if registry_path is not None else _default_registry_path()
    stored = to_repo_relative(resolved, project_root)
    registry = _read_registry(target)

    entry = registry.get(stored)
    if entry is None:
        raise ArtifactIntegrityError(
            f"no integrity entry for {resolved} (registry key {stored!r}) in "
            f"{target}; refusing to load an unrecorded artifact"
        )
    if not isinstance(entry, dict):
        raise ArtifactIntegrityError(
            f"malformed integrity entry for {stored!r} in {target}: {entry!r}"
        )
    if not resolved.is_file():
        raise ArtifactIntegrityError(
            f"{resolved} is recorded in {target} but the file is missing"
        )

    actual_size = resolved.stat().st_size
    if actual_size != entry.get("size"):
        raise ArtifactIntegrityError(
            f"size mismatch for {resolved}: registry records "
            f"{entry.get('size')} bytes, file has {actual_size}"
        )

    started = time.perf_counter()
    actual_digest = sha256_file(resolved)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if actual_digest != entry.get("sha256"):
        raise ArtifactIntegrityError(
            f"sha256 mismatch for {resolved}: registry records "
            f"{entry.get('sha256')}, file has {actual_digest}"
        )

    logger.info(
        "Verified %s (size %d, sha256 %s..., hashed in %.1f ms)",
        resolved,
        actual_size,
        actual_digest[:12],
        elapsed_ms,
    )
    return joblib.load(resolved)


# --- guarded native-format loads -------------------------------------------

# Both supported native formats serialize the model as an object: JSON models
# are text that starts with "{", UBJSON (and ``Booster.save_raw()``) models are
# binary that starts with the object marker "{". The sniff is a prefix check,
# not a parse.
_ASCII_WHITESPACE = b" \t\r\n"


def _looks_like_native_model(data: bytes) -> bool:
    """Whether ``data`` starts like a JSON or UBJSON native model."""
    return data.lstrip(_ASCII_WHITESPACE)[:1] == b"{"


def load_native_model(
    source: Path | str | bytes | bytearray | memoryview,
) -> xgb.Booster:
    """Load an xgboost native model (JSON/UBJSON) from a path or buffer.

    ``source`` is a filesystem path (``str``/``Path``) or the model bytes
    (``bytes``/``bytearray``/``memoryview``); the bytes are read (or
    normalized) and checked before any native call: a missing path raises
    ``FileNotFoundError``, an empty file or buffer raises
    ``EmptyModelBufferError``, and a source that does not start like a
    JSON/UBJSON model raises ``UnrecognizedModelFormatError``. Only a
    non-empty, model-shaped buffer reaches ``Booster.load_model``.

    The guard must run first: ``Booster.load_model`` on a zero-length buffer
    aborts the interpreter (SIGABRT, uncatchable) instead of raising, so a
    torn or truncated artifact handed straight to the decoder would kill the
    process. The returned booster is a prediction artifact only — the native
    formats do not carry the training recipe (the pkl remains the training
    artifact and the params sidecar carries ``get_params()``).
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        origin = "in-memory buffer"
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"native model file not found: {path}")
        data = path.read_bytes()
        origin = str(path)

    if not data:
        raise EmptyModelBufferError(
            f"empty native model buffer ({origin}): the xgboost decoder "
            f"aborts the process (SIGABRT) on a zero-length buffer, so the "
            f"load is refused"
        )
    if not _looks_like_native_model(data):
        raise UnrecognizedModelFormatError(
            f"unrecognized native model format ({origin}): expected a "
            f"JSON or UBJSON object (leading '{{'), got {data[:8]!r}"
        )

    booster = xgb.Booster()
    booster.load_model(bytearray(data))
    logger.info("Loaded native model from %s (%d bytes)", origin, len(data))
    return booster
