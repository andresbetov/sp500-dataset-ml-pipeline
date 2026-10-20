"""Locked holdout evaluation and the final holdout report.

Phase 3 reserves the final ``holdout.n_dates`` unique trading dates before any
fold is built. This module freezes the evaluation protocol over that reserved
window and evaluates it once:

* The lock payload is the canonical SHA-256 over the frozen selection recipe
  (training hyperparameters plus the seed/thread policy), the Phase-2 dataset
  pin, the fold-manifest hash, and the recorded holdout bounds. The lock is
  written to ``results/holdout_lock.json`` and recorded in the run manifest
  before any holdout metric is computed.
* The one protocol is a single final refit on the purged remainder: every
  floor-retained row before the holdout entry whose five-day label window ends
  before that entry, with no inner-validation tail. The canonical HAR-RV
  benchmark is fitted on the same purged remainder (minus its per-ticker
  warm-up), and both predict the holdout block in one pass. The holdout rows
  are scored under the same strict floor policy as every other evaluation.
* A completed evaluation can never be repeated silently: a second call raises
  :class:`HoldoutLockError`, and a resume with the flag simply records a
  skipped holdout instead of re-scoring. Only an explicit
  ``--force-holdout <reason>`` appends an override record and re-evaluates the
  same locked identity; a changed config, dataset pin, fold hash, or bounds is
  a lock mismatch and always raises.

The final report (``results/holdout_report.json``) carries the holdout metrics,
the lock identity, the config hash, git sha, environment reference, and
artifact hashes. ``results/SHA256SUMS`` lists the report and its sibling
holdout artifacts in ``sha256sum`` format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from packages.ml.model import atomic
from packages.ml.model import config as model_config
from packages.ml.model.artifact_integrity import (
    REGISTRY_FILENAME,
    load_verified,
    record_artifact,
)
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.evaluation.report import (
    cw_test,
    date_clustered_gap,
    decile_edges,
    decile_table,
    dm_test,
    gw_test,
    loss_distribution,
    mz_regression,
    post_hoc_adjustment,
    regime_table,
)
from packages.ml.model.evaluation.stats import sanitize_json
from packages.ml.model.run_log import (
    MANIFEST_FILENAME,
    RUNS_FILENAME,
    ResumeError,
    config_hash,
    environment_versions,
    finish_run,
    git_sha,
    load_manifest,
    new_run,
    record_artifacts,
    record_holdout,
    record_phase,
    resolve_stored_path,
    sha256_file,
    thread_policy,
    to_repo_relative,
    write_manifest,
    write_run_state,
)
from packages.ml.model.training.har_benchmark import (
    COMPARISON_NOTE,
    FIT_SCALE,
    HAC_LAG,
    REPORT_SCALE,
    fit_har_benchmark,
    har_forecast_log,
    load_har_regressors,
)
from packages.ml.model.training.model_trainer import train_xgboost_model
from packages.ml.model.training.qlike_objective import qlike_losses
from packages.ml.model.training.training import (
    HEADLINE_ROW_SET,
    LABEL_HORIZON_DAYS,
    RV_FLOOR,
    floor_retained_mask,
    oos_r2_prevailing_mean,
    oos_r2_vs_har,
    qlike_loss,
)
from packages.ml.model.utils import ARTIFACTS_DIR, normalize_dates

logger = logging.getLogger(__name__)

HOLDOUT_SCHEMA_VERSION = 1
LOCK_FILENAME = "holdout_lock.json"
REPORT_FILENAME = "holdout_report.json"
SUMS_FILENAME = "SHA256SUMS"
PREDICTIONS_FILENAME = "holdout_predictions.pkl"
MODEL_FILENAME = "holdout_model.pkl"
OVERRIDES_FILENAME = "holdout_overrides.jsonl"

SCALE_NOTE = (
    "log target (log daily sigma); QLIKE in per-day sigma units over the "
    "floor-retained holdout rows"
)
SINGLE_SEED_CAVEAT = (
    "single seed: the final refit and every selection decision behind it come "
    "from one training run (one seed); seed-to-seed uncertainty is not "
    "estimated, so the holdout edge is a run-level result, not a seed-level "
    "confidence statement"
)
FLOOR_POLICY_CAVEAT = (
    "floor policy: at-or-below-floor targets (raw RV <= the configured floor) "
    "are excluded from fitting and scoring under the single strict floor rule; "
    f"the headline row set is {HEADLINE_ROW_SET}"
)
HOLDOUT_CAVEAT = (
    "holdout: the final {n_dates} trading dates ({start} to {end}) were "
    "reserved before fold construction, never used by tuning or folds, and "
    "evaluated exactly once after the lock was recorded; the result is a "
    "single point estimate of that window"
)
HOLDOUT_OVERRIDE_CAVEAT = (
    "holdout: the final {n_dates} trading dates ({start} to {end}) were "
    "reserved before fold construction and never used by tuning or folds; "
    "this report is an explicit override re-evaluation of the same locked "
    "window, and the override record is part of the report"
)
ADJUSTMENT_CAVEAT = post_hoc_adjustment()["note"]
ONE_SHOT_NOTE = (
    "the lock is recorded in the run manifest before any holdout metric is "
    "computed; a second evaluation raises HoldoutLockError; a post-lock resume "
    "records a skipped holdout instead of re-scoring; only an explicit "
    "--force-holdout <reason> appends an override record and re-evaluates"
)


class HoldoutLockError(RuntimeError):
    """The locked holdout was already evaluated, or its identity changed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime | None = None) -> str:
    return (moment if moment is not None else _utc_now()).isoformat()


def _canonical_json(value: object) -> str:
    """Compact, sorted, strict JSON used for every digest in this module."""
    return json.dumps(
        sanitize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def lock_sha256(payload: Mapping) -> str:
    """The canonical SHA-256 of a lock payload."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _dataset_pin(metadata: Mapping) -> dict:
    """The verified Phase-2 dataset pin (content hash plus stored path)."""
    pin = metadata.get("dataset")
    if not isinstance(pin, Mapping):
        raise HoldoutLockError(
            "the holdout lock requires the Phase-2 dataset pin "
            "(metadata['dataset']); a synthetic or legacy input set cannot "
            "bind the final report to a dataset snapshot"
        )
    digest = pin.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise HoldoutLockError(
            "the Phase-2 dataset pin is missing a 64-hex sha256; the holdout "
            "lock refuses an unbound dataset"
        )
    stored_path = pin.get("path")
    if not stored_path:
        raise HoldoutLockError(
            "the Phase-2 dataset pin is missing its path; the holdout lock "
            "cannot verify the dataset bytes"
        )
    resolved = resolve_stored_path(stored_path)
    if not resolved.is_file():
        raise HoldoutLockError(
            f"the pinned dataset {resolved} is missing; the holdout lock "
            "cannot verify the dataset identity"
        )
    actual = sha256_file(resolved)
    if actual != digest:
        raise HoldoutLockError(
            "dataset pin mismatch: the Phase-2 metadata records "
            f"{digest}, the dataset bytes hash to {actual}"
        )
    return {"sha256": digest, "path": to_repo_relative(resolved)}


def build_lock_payload(
    *,
    config: Mapping,
    metadata: Mapping,
    folds_metadata: Mapping,
    input_hashes: Mapping,
) -> dict:
    """Build the canonical lock payload from the frozen inputs.

    The payload binds the whole claim: the selected recipe and determinism
    policy, the dataset pin, the derived ``X``/``Y`` hashes, the
    fold-construction digest, the full configured config hash, and the reserved
    holdout bounds plus the purge constants that define the final-refit entry.
    """
    policy = thread_policy(config)
    training_config = config.get("training") or {}
    hyperparameters = training_config.get("hyperparameters") or {}
    if not isinstance(hyperparameters, Mapping) or not hyperparameters:
        raise HoldoutLockError(
            "the frozen selection has an empty training.hyperparameters "
            "mapping; the holdout lock refuses to bind an unspecified recipe"
        )
    missing_hashes = [
        key
        for key in ("x_sha256", "y_sha256")
        if not isinstance(input_hashes.get(key), str)
        or len(input_hashes[key]) != 64
    ]
    if missing_hashes:
        raise HoldoutLockError(
            "the holdout lock requires the derived input hashes: missing or "
            f"malformed {', '.join(missing_hashes)}"
        )
    holdout = folds_metadata.get("holdout")
    if not isinstance(holdout, Mapping):
        raise HoldoutLockError(
            "the fold payload carries no reserved holdout window "
            "(holdout.bounds); the holdout lock refuses to bind an unspecified "
            "window"
        )
    bounds = holdout.get("bounds")
    if not isinstance(bounds, Mapping) or not bounds.get("start") or not bounds.get(
        "end"
    ):
        raise HoldoutLockError(
            "the reserved holdout window is missing its bounds; the holdout "
            "lock requires both the start and the end date"
        )
    configured = config.get("holdout") or {}
    return {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "selection": {
            "hyperparameters": {
                str(key): sanitize_json(value)
                for key, value in hyperparameters.items()
            },
            "seed": int(policy["seed"]),
            "random_state": int(policy["random_state"]),
            "threads": {
                "policy": str(policy["policy"]),
                "n_jobs": int(policy["n_jobs"]),
                "omp_num_threads": int(policy["omp_num_threads"]),
            },
        },
        "dataset": _dataset_pin(metadata),
        "derived_inputs": {
            "x_sha256": str(input_hashes["x_sha256"]),
            "y_sha256": str(input_hashes["y_sha256"]),
        },
        "config_hash": config_hash(config),
        "fold_manifest_hash": cross_validator.fold_manifest_hash(folds_metadata),
        "holdout": {
            "bounds": {"start": str(bounds["start"]), "end": str(bounds["end"])},
            "n_dates": int(holdout.get("n_dates") or 0),
            "configured_n_dates": int(configured.get("n_dates") or 0),
            "purge_horizon_days": int(cross_validator.PURGE_HORIZON_D),
            "label_horizon_days": int(LABEL_HORIZON_DAYS),
            "separation_min": int(cross_validator.SEPARATION_MIN),
        },
    }


def _fold_keys(folds_metadata: Mapping) -> list[str]:
    """Fold keys in index order; only ``fold_*`` keys are fold records."""
    keys = [key for key in folds_metadata if str(key).startswith("fold_")]
    try:
        return sorted(keys, key=lambda key: int(str(key).split("_", 1)[1]))
    except ValueError as exc:
        raise HoldoutLockError(
            f"fold keys must be fold_<index>; got {sorted(keys)}"
        ) from exc


def validate_holdout_reservation(
    dates,
    y,
    folds_metadata: Mapping,
    config: Mapping | None = None,
) -> dict:
    """Re-check the reserved window and derive the purged final-refit rows.

    The recorded window must cover exactly the configured final N unique
    trading dates, every fold train/test row must fall before it, and the
    floor-retained remainder rows whose five-day label window ends before the
    holdout start must be separated from the first holdout date by at least
    the configured separation (five complete excluded trading dates). Any
    violation raises :class:`HoldoutLockError` naming the reservation field.
    """
    loaded = model_config.get_config() if config is None else config
    configured = int((loaded.get("holdout") or {}).get("n_dates") or 0)
    holdout = folds_metadata.get("holdout")
    if not isinstance(holdout, Mapping):
        raise HoldoutLockError(
            "fold metadata has no reserved holdout window; the holdout "
            "evaluation refuses to derive an implicit one"
        )
    recorded_n = int(holdout.get("n_dates") or 0)
    if recorded_n != configured:
        raise HoldoutLockError(
            f"holdout reservation mismatch: the fold payload reserves "
            f"{recorded_n} dates, the config requires {configured}"
        )
    bounds = holdout.get("bounds") or {}
    start = bounds.get("start")
    end = bounds.get("end")
    if start is None or end is None:
        raise HoldoutLockError(
            "holdout reservation has n_dates but no complete bounds"
        )

    dates = normalize_dates(np.asarray(dates))
    values = np.asarray(y, dtype=np.float64)
    if values.shape != dates.shape:
        raise HoldoutLockError(
            f"holdout reservation received {dates.size} dates and {values.size} "
            "targets; the arrays must be row-aligned"
        )
    unique_dates = np.sort(np.unique(dates))
    if configured < 1 or configured > unique_dates.size:
        raise HoldoutLockError(
            f"holdout reservation of {configured} dates does not fit the "
            f"{unique_dates.size}-date calendar"
        )
    expected_holdout = unique_dates[-configured:]
    expected_start = pd.Timestamp(expected_holdout[0]).strftime("%Y-%m-%d")
    expected_end = pd.Timestamp(expected_holdout[-1]).strftime("%Y-%m-%d")
    if str(start) != expected_start or str(end) != expected_end:
        raise HoldoutLockError(
            "holdout bounds mismatch: the recorded window is "
            f"{start}..{end}, the final {configured} trading dates are "
            f"{expected_start}..{expected_end}"
        )

    entry_pos = int(unique_dates.size - configured)
    date_pos = np.searchsorted(unique_dates, dates)
    holdout_indices = np.flatnonzero(dates >= expected_holdout[0])
    observed_dates = np.unique(dates[holdout_indices])
    if observed_dates.size != configured:
        raise HoldoutLockError(
            f"holdout reservation covers {observed_dates.size} of the "
            f"configured {configured} unique trading dates; the rows and the "
            "recorded window disagree"
        )
    holdout_rows = set(holdout_indices.tolist())

    last_fold_pos = -1
    fold_key_list = _fold_keys(folds_metadata)
    if not fold_key_list:
        raise HoldoutLockError(
            "fold metadata has no fold_* records; the holdout reservation "
            "cannot be re-checked against the fold construction"
        )
    for fold_key in fold_key_list:
        fold = folds_metadata[fold_key]
        for role in ("train_indices", "test_indices"):
            raw = fold.get(role)
            if raw is None:
                raise HoldoutLockError(
                    f"{fold_key} has no {role}; the holdout reservation cannot "
                    "be re-checked"
                )
            indices = np.asarray(raw, dtype=np.int64).reshape(-1)
            hits = np.intersect1d(indices, holdout_indices)
            if hits.size:
                raise HoldoutLockError(
                    f"{fold_key} {role} intersects the reserved holdout at "
                    f"{hits[:8].tolist()}; the reservation is not disjoint"
                )
            if indices.size and int(date_pos[indices].max()) >= entry_pos:
                raise HoldoutLockError(
                    f"{fold_key} {role} reaches the reserved holdout entry; "
                    "folds must be derived on the remainder only"
                )
            if indices.size:
                last_fold_pos = max(last_fold_pos, int(date_pos[indices].max()))

    retained = floor_retained_mask(values)
    remainder = (date_pos < entry_pos) & retained
    fit_mask = remainder & (date_pos + LABEL_HORIZON_DAYS < entry_pos)
    fit_indices = np.flatnonzero(fit_mask)
    if fit_indices.size == 0:
        raise HoldoutLockError(
            "the purged remainder holds no floor-retained fit rows; the final "
            "refit cannot proceed"
        )
    last_fit_pos = int(date_pos[fit_indices].max())
    separation = entry_pos - last_fit_pos
    if separation < int(cross_validator.SEPARATION_MIN):
        raise HoldoutLockError(
            "holdout separation violated: the last final-refit date index "
            f"{last_fit_pos} is {separation} dates before the holdout entry, "
            f"below the configured minimum {cross_validator.SEPARATION_MIN}"
        )
    purged = remainder & ~fit_mask
    holdout_floor = retained[holdout_indices]
    scored_indices = holdout_indices[holdout_floor]
    if scored_indices.size == 0:
        raise HoldoutLockError(
            "every holdout row is at or below the target floor; there is "
            "nothing to score"
        )
    return {
        "holdout_indices": holdout_indices,
        "holdout_dates": expected_holdout,
        "scored_indices": scored_indices,
        "fit_indices": np.unique(fit_indices),
        "holdout_floor": holdout_floor,
        "bounds": {"start": str(start), "end": str(end)},
        "n_holdout_dates": int(configured),
        "n_holdout_rows": int(holdout_indices.size),
        "n_scored_rows": int(scored_indices.size),
        "n_floor_excluded": int(holdout_indices.size - scored_indices.size),
        "n_purged_fit_rows": int(np.count_nonzero(purged)),
        "first_holdout_pos": int(entry_pos),
        "last_fold_pos": int(last_fold_pos),
        "last_fit_pos": int(last_fit_pos),
        "separation": int(separation),
        "unique_dates": unique_dates,
        "holdout_rows": holdout_rows,
    }


def _load_inputs(root: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Load the verified Phase-2/3 inputs for the holdout evaluation."""
    registry_path = root / REGISTRY_FILENAME
    X = np.load(root / "inputs" / "X.npy", mmap_mode="r")
    y = np.load(root / "inputs" / "Y.npy")
    metadata = load_verified(
        root / "inputs" / "metadata.pkl", registry_path=registry_path
    )
    folds_metadata = load_verified(
        root / "folds" / "folds_metadata.pkl", registry_path=registry_path
    )
    return X, y, metadata, folds_metadata


def _read_lock(path: Path) -> dict | None:
    """Read and self-verify the durable lock document, or ``None``."""
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutLockError(
            f"holdout lock {path} is unreadable: {exc}"
        ) from exc
    if not isinstance(document, dict) or not isinstance(
        document.get("payload"), Mapping
    ):
        raise HoldoutLockError(
            f"holdout lock {path} is not a lock document with a payload"
        )
    recorded = document.get("sha256")
    recomputed = lock_sha256(document["payload"])
    if recorded != recomputed:
        raise HoldoutLockError(
            f"holdout lock {path} is corrupt: it records {recorded}, its "
            f"payload hashes to {recomputed}"
        )
    return document


def _first_difference(recorded: object, current: object, path: str = "") -> str:
    """The first differing payload path, for a named lock-mismatch error."""
    if isinstance(recorded, Mapping) and isinstance(current, Mapping):
        recorded_keys = set(recorded)
        current_keys = set(current)
        for key in sorted(recorded_keys | current_keys):
            if key not in recorded:
                return f"{path}{key} (new value)"
            if key not in current:
                return f"{path}{key} (missing)"
            if recorded[key] != current[key]:
                return _first_difference(
                    recorded[key], current[key], f"{path}{key}."
                )
        return path.rstrip(".") or "<root>"
    if recorded != current:
        return path.rstrip(".") or "<root>"
    return path.rstrip(".") or "<root>"


def _lock_field_label(path: str) -> str:
    if path.startswith("selection") or path.startswith("config_hash"):
        return "config"
    if path.startswith("dataset"):
        return "dataset pin"
    if path.startswith("derived_inputs"):
        return "derived input hashes"
    if path.startswith("fold_manifest_hash"):
        return "fold manifest"
    if path.startswith("holdout"):
        return "holdout bounds"
    return "lock payload"


def _verify_lock(document: Mapping, current_payload: Mapping) -> None:
    """Refuse a lock whose recorded identity differs from the live inputs."""
    recorded = document["payload"]
    if recorded == current_payload:
        return
    path = _first_difference(recorded, current_payload)
    label = _lock_field_label(path)
    raise HoldoutLockError(
        f"holdout lock mismatch on {label} ({path}): the locked evaluation "
        "binds the frozen selection recipe, dataset pin, fold-manifest hash, "
        "and holdout bounds; a changed input cannot reuse the lock"
    )


def _write_lock_document(path: Path, document: Mapping) -> dict:
    """Write one lock document atomically; the caller owns its identity."""
    atomic.write_json_atomically(
        path,
        document,
        trailing_newline=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return dict(document)


def _write_lock(
    path: Path, payload: Mapping, *, run_id: str, when: datetime
) -> dict:
    document = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "sha256": lock_sha256(payload),
        "payload": dict(payload),
        "run_id": run_id,
        "recorded_at": _iso(when),
    }
    _write_lock_document(path, document)
    return document


def _manifest_holdout_record(root: Path) -> dict | None:
    """The latest manifest's holdout block, if any.

    The results lock is the primary artifact, but the manifest's holdout block
    is a second durable record of the same lock. It is read even when the
    caller supplies a fresh run record, so losing the results lock cannot make
    a completed/overridden holdout look unevaluated.
    """
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = load_manifest(root)
    except ResumeError as exc:
        raise HoldoutLockError(
            f"cannot read run manifest {manifest_path} while checking the "
            f"holdout lock: {exc}"
        ) from exc
    record = manifest.get("holdout")
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise HoldoutLockError(
            f"run manifest {manifest_path} carries a non-object holdout record"
        )
    return dict(record)


def _runs_holdout_record(root: Path) -> dict | None:
    """The last holdout-bearing record in the append-only run log.

    ``run_manifest.json`` is overwritten by every run; ``runs.jsonl`` is
    append-only. If an ordinary run replaced the manifest without carrying the
    holdout block (historical state or a lost manifest), the last recorded
    holdout block is still authoritative: it is verified against the live
    inputs and used to restore the results lock. A malformed/torn log line is
    logged and skipped so one interrupted append cannot hide every earlier
    record.
    """
    runs_path = root / RUNS_FILENAME
    if not runs_path.is_file():
        return None
    text = runs_path.read_text(encoding="utf-8")
    last: dict | None = None
    for position, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(
                "Ignoring malformed line %d in %s while searching for the "
                "last holdout lock record",
                position,
                runs_path,
            )
            continue
        if not isinstance(record, Mapping):
            continue
        holdout = record.get("holdout")
        if isinstance(holdout, Mapping) and holdout.get("lock_sha256"):
            last = dict(holdout)
    if last is not None:
        logger.warning(
            "No manifest holdout block; using the last holdout-bearing "
            "runs.jsonl record (lock %s)",
            str(last.get("lock_sha256"))[:12],
        )
    return last


def _lock_from_manifest(record: Mapping, current_payload: Mapping) -> dict:
    """Reconstruct and verify the lock document from the manifest record.

    New manifest records carry the full lock payload, so a lost results lock
    can be restored without weakening identity checks. A record without the
    payload cannot be verified and is refused by name.
    """
    digest = record.get("lock_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise HoldoutLockError(
            "the run manifest records a holdout lock without a 64-hex "
            "lock_sha256"
        )
    payload = record.get("lock_payload")
    if not isinstance(payload, Mapping):
        raise HoldoutLockError(
            "the run manifest records a holdout lock but not its lock_payload; "
            "the missing results lock cannot be verified or restored"
        )
    recomputed = lock_sha256(payload)
    if recomputed != digest:
        raise HoldoutLockError(
            "the run manifest holdout record is inconsistent: it records "
            f"{digest}, its lock payload hashes to {recomputed}"
        )
    if payload != current_payload:
        path = _first_difference(payload, current_payload)
        label = _lock_field_label(path)
        raise HoldoutLockError(
            f"holdout lock mismatch on {label} ({path}): the manifest's "
            "recorded lock no longer matches the live selection, dataset, "
            "fold construction, or holdout bounds"
        )
    recorded_bounds = record.get("bounds")
    if recorded_bounds and recorded_bounds != payload["holdout"]["bounds"]:
        raise HoldoutLockError(
            "the run manifest holdout record's bounds disagree with its lock "
            "payload"
        )
    return {
        "schema_version": payload.get(
            "schema_version", HOLDOUT_SCHEMA_VERSION
        ),
        "sha256": digest,
        "payload": dict(payload),
        "run_id": record.get("lock_run_id") or record.get("evaluated_at_run"),
        "recorded_at": record.get("locked_at") or _iso(),
    }


def _restore_lock_file(
    path: Path, document: Mapping, *, registry_path: Path
) -> None:
    """Restore a results lock from a verified manifest record."""
    _write_lock_document(path, document)
    record_artifact(path, registry_path=registry_path)
    logger.warning(
        "Restored the missing holdout lock %s from the run manifest record "
        "(lock %s)",
        path,
        document["sha256"][:12],
    )


def _read_overrides(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    records = []
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            is_torn_tail = position == len(lines) - 1 and not text.endswith("\n")
            if is_torn_tail:
                logger.warning(
                    "Ignoring a torn final override line in %s", path
                )
                continue
            raise HoldoutLockError(
                f"holdout override log {path} carries malformed JSON: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise HoldoutLockError(
                f"holdout override log {path} carries a non-object record"
            )
        records.append(record)
    return records


def _append_override(
    path: Path,
    *,
    reason: str,
    lock_document: Mapping,
    run_id: str,
    when: datetime,
) -> dict:
    record = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "recorded_at": _iso(when),
        "run_id": run_id,
        "reason": reason,
        "previous_lock_sha256": lock_document["sha256"],
        "bounds": lock_document["payload"]["holdout"]["bounds"],
        "n_dates": lock_document["payload"]["holdout"]["n_dates"],
    }
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing and not existing.endswith("\n"):
        # Drop a torn final line from an interrupted append before writing the
        # complete log through the atomic writer.
        existing = existing[: existing.rfind("\n") + 1]
    line = (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    atomic.write_text_atomically(path, existing + line)
    return record


def _skip_record(
    lock_document: Mapping,
    *,
    run_id: str | None,
    report_path: Path,
) -> dict:
    payload = lock_document["payload"]
    return {
        "status": "skipped",
        "lock_sha256": lock_document["sha256"],
        "lock_payload": dict(payload),
        "lock_run_id": lock_document.get("run_id"),
        "locked_at": lock_document.get("recorded_at"),
        "config_hash": payload["config_hash"],
        "bounds": payload["holdout"]["bounds"],
        "n_dates": payload["holdout"]["n_dates"],
        "evaluated_at_run": lock_document.get("run_id"),
        "skipped_at_run": run_id,
        "recorded_at": _iso(),
        "reason": (
            "a locked holdout already exists; resume records the skip instead "
            "of re-evaluating the window"
        ),
        "report_path": (
            to_repo_relative(report_path) if report_path.is_file() else None
        ),
        "report_sha256": (
            sha256_file(report_path) if report_path.is_file() else None
        ),
    }


def compute_holdout_metrics(
    *,
    y_true,
    y_pred_model,
    y_pred_har,
    dates,
    regime,
    n_floor_excluded: int = 0,
    n_holdout_rows: int | None = None,
) -> dict:
    """The holdout metric block from the scored rows only.

    Every statistic uses the same floor-retained holdout rows the refit was
    scored on. The dual OOS-R² conventions, the model/HAR QLIKE comparison,
    the DM/CW/GW tests (Newey-West HAC lag 4), the Mincer-Zarnowitz
    regression, the date-clustered descriptive gap, the loss distribution,
    and the decile/regime tables reuse the evaluation layer's single
    implementations.

    ``n_floor_excluded`` and ``n_holdout_rows`` are the count inputs the
    scored-row arrays cannot carry; direct callers state them explicitly, and
    the evaluation path passes the reservation counts. When
    ``n_holdout_rows`` is omitted it is derived as scored + excluded rows.
    """
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    model = np.asarray(y_pred_model, dtype=np.float64).reshape(-1)
    har = np.asarray(y_pred_har, dtype=np.float64).reshape(-1)
    date_values = normalize_dates(np.asarray(dates)).reshape(-1)
    regime_values = np.asarray(regime).reshape(-1)
    if not (target.size == model.size == har.size == date_values.size == regime_values.size):
        raise HoldoutLockError(
            "holdout metric arrays are misaligned: "
            f"{target.size} targets, {model.size} model forecasts, "
            f"{har.size} HAR forecasts, {date_values.size} dates, "
            f"{regime_values.size} regime labels"
        )
    if target.size == 0:
        raise HoldoutLockError("holdout metrics received no scored rows")
    if int(n_floor_excluded) < 0:
        raise HoldoutLockError(
            f"n_floor_excluded must be non-negative, got {n_floor_excluded}"
        )
    excluded = int(n_floor_excluded)
    total = (
        int(target.size) + excluded
        if n_holdout_rows is None
        else int(n_holdout_rows)
    )
    if total != int(target.size) + excluded:
        raise HoldoutLockError(
            "holdout row counts are inconsistent: "
            f"{target.size} scored + {excluded} excluded rows != {total} total"
        )
    if not (
        np.isfinite(target).all()
        and np.isfinite(model).all()
        and np.isfinite(har).all()
    ):
        raise HoldoutLockError("holdout metrics received non-finite values")

    loss_model = qlike_losses(target - model)
    loss_har = qlike_losses(target - har)
    model_qlike = float(qlike_loss(target, model))
    har_qlike = float(qlike_loss(target, har))
    edges = decile_edges(target)
    return {
        "n_scored_rows": int(target.size),
        "n_floor_excluded": excluded,
        "n_holdout_rows": total,
        "scale": SCALE_NOTE,
        "row_set": HEADLINE_ROW_SET,
        "model_qlike": model_qlike,
        "har_qlike": har_qlike,
        "edge": float(har_qlike - model_qlike),
        "model_r2": float(r2_score(target, model)),
        "model_mae": float(mean_absolute_error(target, model)),
        "model_rmse": float(np.sqrt(mean_squared_error(target, model))),
        "har_r2": float(r2_score(target, har)),
        "oos_r2_prevailing_mean": float(
            oos_r2_prevailing_mean(target, model)
        ),
        "oos_r2_vs_har": float(oos_r2_vs_har(target, model, har)),
        "loss_distribution": {
            "model": loss_distribution(loss_model),
            "har": loss_distribution(loss_har),
        },
        "dm": dm_test(loss_model, loss_har, date_values, lag=HAC_LAG),
        "cw": cw_test(target, model, har, date_values, lag=HAC_LAG),
        "gw": gw_test(loss_model, loss_har, date_values, lag=HAC_LAG),
        "mz": mz_regression(target, model, lag=HAC_LAG),
        "date_clustered_gap": date_clustered_gap(
            loss_model, loss_har, date_values
        ),
        "decile_qlike": {
            "edges": edges,
            "bins": decile_table(target, loss_model, loss_har, edges),
        },
        "regime_qlike": regime_table(
            regime_values, loss_model, loss_har
        ),
    }


def _refit_and_score(
    *,
    X: np.ndarray,
    y: np.ndarray,
    metadata: Mapping,
    reservation: Mapping,
    trainer: Callable,
    har_regressors: np.ndarray | None,
    har_warmup_mask: np.ndarray | None,
) -> dict:
    """The one final refit, the HAR fit on the same rows, and one prediction pass."""
    y_log = np.log(np.maximum(np.asarray(y, dtype=np.float64), RV_FLOOR))
    fit_indices = np.asarray(reservation["fit_indices"], dtype=np.int64)
    holdout_indices = np.asarray(
        reservation["holdout_indices"], dtype=np.int64
    )
    scored_indices = np.asarray(reservation["scored_indices"], dtype=np.int64)
    holdout_floor = np.asarray(reservation["holdout_floor"], dtype=bool)

    fit_started = time.time()
    model, _label_mapping = trainer(
        np.asarray(X)[fit_indices], y_log[fit_indices]
    )
    fit_seconds = time.time() - fit_started

    inference_started = time.time()
    y_pred_all = np.asarray(
        model.predict(np.asarray(X)[holdout_indices]), dtype=np.float64
    ).reshape(-1)
    inference_seconds = time.time() - inference_started
    if y_pred_all.size != holdout_indices.size:
        raise HoldoutLockError(
            f"the final refit predicted {y_pred_all.size} holdout rows for "
            f"{holdout_indices.size} rows"
        )

    if har_regressors is None or har_warmup_mask is None:
        regressors, warmup_mask = load_har_regressors(
            metadata["date"], metadata["ticker"]
        )
    else:
        regressors = np.asarray(har_regressors, dtype=np.float64)
        warmup_mask = np.asarray(har_warmup_mask, dtype=bool)

    scored_mask = floor_retained_mask(np.asarray(y)[holdout_indices])
    if not np.array_equal(scored_mask, holdout_floor):
        raise HoldoutLockError(
            "holdout floor membership changed between the reservation check "
            "and scoring"
        )
    y_true = y_log[holdout_indices][scored_mask]
    y_pred_model = y_pred_all[scored_mask]

    har_started = time.time()
    har_result = fit_har_benchmark(
        train_indices=fit_indices,
        test_indices=scored_indices,
        target_log_train=y_log[fit_indices],
        target_log_test=y_true,
        regressors=regressors,
        forecast_floor=RV_FLOOR,
        warmup_mask=warmup_mask,
        fold_key="holdout",
        coefficient_interpretation=False,
    )
    har_seconds = time.time() - har_started
    har_forecast = np.asarray(har_result.pop("forecast_log"), dtype=np.float64)
    if har_forecast.size != scored_indices.size:
        raise HoldoutLockError(
            f"the HAR benchmark forecast {har_forecast.size} rows for "
            f"{scored_indices.size} scored holdout rows"
        )

    floor_rows = ~scored_mask
    if floor_rows.any():
        har_floor, _clip_count = har_forecast_log(
            har_result["coefficients"],
            regressors,
            holdout_indices[floor_rows],
            RV_FLOOR,
        )
    else:
        har_floor = np.empty(0, dtype=np.float64)
    har_all = np.empty(holdout_indices.size, dtype=np.float64)
    har_all[scored_mask] = har_forecast
    har_all[floor_rows] = har_floor

    metrics = compute_holdout_metrics(
        y_true=y_true,
        y_pred_model=y_pred_model,
        y_pred_har=har_forecast,
        dates=np.asarray(metadata["date"])[scored_indices],
        regime=np.asarray(metadata["regime"])[scored_indices],
        n_floor_excluded=int(reservation["n_floor_excluded"]),
        n_holdout_rows=int(holdout_indices.size),
    )
    return {
        "model": model,
        "metrics": metrics,
        "har_result": har_result,
        "y_log": y_log,
        "y_pred_all": y_pred_all,
        "har_all": har_all,
        "scored_mask": scored_mask,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "har_seconds": har_seconds,
    }


def _artifact_hashes(
    *,
    root: Path,
    lock_path: Path,
    model_path: Path,
    predictions_path: Path,
    metadata: Mapping,
    input_hashes: Mapping,
) -> dict[str, str]:
    artifacts = {
        to_repo_relative(root / "inputs" / "X.npy"): str(
            input_hashes["x_sha256"]
        ),
        to_repo_relative(root / "inputs" / "Y.npy"): str(
            input_hashes["y_sha256"]
        ),
        to_repo_relative(root / "inputs" / "metadata.pkl"): sha256_file(
            root / "inputs" / "metadata.pkl"
        ),
        to_repo_relative(root / "folds" / "folds_metadata.pkl"): sha256_file(
            root / "folds" / "folds_metadata.pkl"
        ),
        to_repo_relative(lock_path): sha256_file(lock_path),
        to_repo_relative(model_path): sha256_file(model_path),
        to_repo_relative(predictions_path): sha256_file(predictions_path),
    }
    dataset_pin = metadata.get("dataset") or {}
    dataset_path = dataset_pin.get("path")
    if dataset_path:
        resolved = resolve_stored_path(dataset_path)
        if resolved.is_file():
            artifacts[to_repo_relative(resolved)] = sha256_file(resolved)
    return artifacts


def _write_sums(path: Path, entries: Mapping[str, str]) -> None:
    text = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(entries.items())
    )
    atomic.write_text_atomically(path, text)


def evaluate_holdout(
    *,
    run: dict | None = None,
    artifacts_dir: Path | str | None = None,
    resume: bool = False,
    force_reason: str | None = None,
    trainer: Callable | None = None,
    har_regressors: np.ndarray | None = None,
    har_warmup_mask: np.ndarray | None = None,
    config: Mapping | None = None,
) -> dict:
    """Evaluate the locked holdout once and write the final report.

    On the first call the lock payload is written to
    ``results/holdout_lock.json`` and recorded in the run manifest before the
    refit, the HAR fit, or any metric. A second call raises
    :class:`HoldoutLockError`; ``resume=True`` records a skipped holdout in the
    manifest instead of re-scoring. ``force_reason`` is the explicit
    ``--force-holdout <reason>`` override: it appends an override record and
    re-evaluates the same locked identity, but never accepts a changed config,
    dataset pin, fold hash, or bounds.

    Args:
        run: The owning run record; omitted for a standalone holdout run.
        artifacts_dir: Artifact root; defaults to the package's live root.
        resume: Record a skip instead of re-evaluating an existing lock.
        force_reason: The non-empty override reason (implies re-evaluation).
        trainer: Trainer callable with the ``train_xgboost_model`` signature;
            injectable for tests.
        har_regressors: Optional prebuilt HAR components (tests).
        har_warmup_mask: Optional prebuilt HAR warm-up mask (tests).
        config: Optional config mapping (tests); defaults to the live config.

    Returns:
        A record with ``status`` (``completed``, ``overridden``, or
        ``skipped``), the lock document, the report path, and the report.
    """
    if force_reason is not None:
        if not isinstance(force_reason, str) or not force_reason.strip():
            raise ValueError(
                "--force-holdout requires a non-empty reason; an empty reason "
                "cannot justify overriding the one-shot lock"
            )
        force_reason = force_reason.strip()

    root = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else ARTIFACTS_DIR
    )
    results_dir = root / "results"
    lock_path = results_dir / LOCK_FILENAME
    report_path = results_dir / REPORT_FILENAME
    sums_path = results_dir / SUMS_FILENAME
    predictions_path = results_dir / PREDICTIONS_FILENAME
    model_path = results_dir / MODEL_FILENAME
    overrides_path = results_dir / OVERRIDES_FILENAME
    registry_path = root / REGISTRY_FILENAME

    loaded_config = model_config.get_config() if config is None else config
    input_hashes = {
        "x_sha256": sha256_file(root / "inputs" / "X.npy"),
        "y_sha256": sha256_file(root / "inputs" / "Y.npy"),
    }
    X, y, metadata, folds_metadata = _load_inputs(root)
    current_payload = build_lock_payload(
        config=loaded_config,
        metadata=metadata,
        folds_metadata=folds_metadata,
        input_hashes=input_hashes,
    )
    reservation = validate_holdout_reservation(
        metadata["date"], y, folds_metadata, config=loaded_config
    )

    # The manifest's holdout block is a second durable record. It is read even
    # when the caller supplies a fresh run: deleting the results lock must not
    # make a completed/overridden holdout look unevaluated.
    run_holdout = (run or {}).get("holdout")
    manifest_record = _manifest_holdout_record(root)
    if (
        manifest_record is None
        and isinstance(run_holdout, Mapping)
        and run_holdout.get("lock_sha256")
    ):
        manifest_record = dict(run_holdout)
    if manifest_record is None:
        manifest_record = _runs_holdout_record(root)

    existing_lock = _read_lock(lock_path)
    manifest_lock = None
    if manifest_record is not None and manifest_record.get("lock_sha256"):
        if existing_lock is not None:
            if manifest_record["lock_sha256"] != existing_lock["sha256"]:
                raise HoldoutLockError(
                    "the run manifest holdout lock "
                    f"{manifest_record['lock_sha256']} disagrees with the "
                    f"results lock {existing_lock['sha256']}; the two records "
                    "cannot both describe the same one-shot evaluation"
                )
        else:
            manifest_lock = _lock_from_manifest(
                manifest_record, current_payload
            )
    if existing_lock is None and manifest_lock is not None:
        existing_lock = manifest_lock
        _restore_lock_file(
            lock_path, existing_lock, registry_path=registry_path
        )

    if existing_lock is not None:
        _verify_lock(existing_lock, current_payload)
    if (
        existing_lock is not None
        and force_reason is not None
        and (
            manifest_record is None
            or manifest_record.get("lock_sha256") != existing_lock["sha256"]
        )
    ):
        raise HoldoutLockError(
            "--force-holdout requires the run manifest's holdout lock record "
            "for the current lock; no matching record is available, so the "
            "override cannot be recorded against a verified evaluation"
        )

    if existing_lock is not None:
        if force_reason is None:
            if resume:
                skip = _skip_record(
                    existing_lock,
                    run_id=(run or {}).get("run_id"),
                    report_path=report_path,
                )
                if run is None:
                    manifest = load_manifest(root)
                    record_holdout(manifest, skip)
                    write_manifest(manifest, root / MANIFEST_FILENAME)
                else:
                    record_holdout(run, skip)
                    write_run_state(run)
                    record_artifacts(run, [lock_path])
                logger.warning(
                    "Holdout already locked (%s); resume records a skip instead "
                    "of re-evaluating",
                    existing_lock["sha256"][:12],
                )
                return {
                    "status": "skipped",
                    "lock": existing_lock,
                    "record": skip,
                    "report_path": (
                        str(report_path) if report_path.is_file() else None
                    ),
                }
            raise HoldoutLockError(
                "the holdout is already locked and evaluated "
                f"(lock {existing_lock['sha256'][:12]}); a second evaluation is "
                "refused. Re-run with --force-holdout <reason> only for an "
                "explicit, recorded override"
            )
    elif force_reason is not None:
        raise ValueError(
            "--force-holdout requires an existing holdout lock; there is "
            "nothing to override"
        )

    standalone = run is None
    if standalone:
        run = new_run("holdout", root)
    run_id = run["run_id"]
    if (
        run.get("config_hash")
        and run["config_hash"] != current_payload["config_hash"]
    ):
        raise HoldoutLockError(
            "the owning run's config hash "
            f"{run['config_hash']} does not match the locked config "
            f"{current_payload['config_hash']}; the report cannot claim a "
            "recipe the run did not train under"
        )
    started = time.monotonic()
    when = _utc_now()

    override_records = _read_overrides(overrides_path)
    override_record = None
    if force_reason is not None:
        override_record = _append_override(
            overrides_path,
            reason=force_reason,
            lock_document=existing_lock,
            run_id=run_id,
            when=when,
        )
        override_records = override_records + [override_record]
        record_artifact(overrides_path, registry_path=registry_path)
        record_artifacts(run, [overrides_path])

    if existing_lock is None:
        lock_document = _write_lock(
            lock_path, current_payload, run_id=run_id, when=when
        )
        record_artifact(lock_path, registry_path=registry_path)
    else:
        lock_document = existing_lock

    holdout_record = {
        "status": "locked",
        "lock_sha256": lock_document["sha256"],
        "lock_payload": dict(current_payload),
        "lock_run_id": lock_document.get("run_id"),
        "config_hash": run.get("config_hash") or current_payload["config_hash"],
        "bounds": current_payload["holdout"]["bounds"],
        "n_dates": current_payload["holdout"]["n_dates"],
        "lock_path": to_repo_relative(lock_path),
        "evaluated_at_run": run_id,
        "locked_at": lock_document["recorded_at"],
        "protocol": ONE_SHOT_NOTE,
        "override": (
            None
            if override_record is None
            else {
                "reason": override_record["reason"],
                "recorded_at": override_record["recorded_at"],
                "path": to_repo_relative(overrides_path),
            }
        ),
    }
    # The lock reaches the manifest before any metric exists: this state write
    # is what a post-lock resume or a second call reads.
    record_holdout(run, holdout_record)
    write_run_state(run)
    record_artifacts(run, [lock_path])

    active_trainer = trainer if trainer is not None else train_xgboost_model
    try:
        evaluation = _refit_and_score(
            X=X,
            y=y,
            metadata=metadata,
            reservation=reservation,
            trainer=active_trainer,
            har_regressors=har_regressors,
            har_warmup_mask=har_warmup_mask,
        )

        atomic.write_pickle_atomically(
            model_path, evaluation["model"]
        )
        record_artifact(model_path, registry_path=registry_path)
        predictions_payload = {
            "schema_version": HOLDOUT_SCHEMA_VERSION,
            "run_id": run_id,
            "lock_sha256": lock_document["sha256"],
            "holdout_bounds": current_payload["holdout"]["bounds"],
            "holdout_indices": np.asarray(
                reservation["holdout_indices"], dtype=np.int64
            ),
            "scored_indices": np.asarray(
                reservation["scored_indices"], dtype=np.int64
            ),
            "fit_indices": np.asarray(
                reservation["fit_indices"], dtype=np.int64
            ),
            "is_floor": ~np.asarray(evaluation["scored_mask"], dtype=bool),
            "y_true_log": evaluation["y_log"][
                np.asarray(reservation["holdout_indices"], dtype=np.int64)
            ],
            "y_pred_model_log": evaluation["y_pred_all"],
            "y_pred_har_log": evaluation["har_all"],
            "har_coefficients": evaluation["har_result"]["coefficients"],
        }
        atomic.write_pickle_atomically(
            predictions_path, predictions_payload
        )
        record_artifact(predictions_path, registry_path=registry_path)

        metrics = evaluation["metrics"]
        artifacts = _artifact_hashes(
            root=root,
            lock_path=lock_path,
            model_path=model_path,
            predictions_path=predictions_path,
            metadata=metadata,
            input_hashes=input_hashes,
        )
        bounds = current_payload["holdout"]["bounds"]
        caveats = {
            "single_seed": SINGLE_SEED_CAVEAT,
            "floor_policy": FLOOR_POLICY_CAVEAT,
            "holdout": (
                HOLDOUT_OVERRIDE_CAVEAT if override_record is not None
                else HOLDOUT_CAVEAT
            ).format(
                n_dates=current_payload["holdout"]["n_dates"],
                start=bounds["start"],
                end=bounds["end"],
            ),
            "adjustment_not_applied": ADJUSTMENT_CAVEAT,
        }
        environment = dict(run.get("environment") or environment_versions())
        protocol = {
            "name": (
                "single final refit on the purged remainder + canonical HAR-RV "
                "on the same rows, predicting the holdout block once"
            ),
            "refit": {
                "estimator": (
                    "one XGBoost model on the frozen selection recipe"
                ),
                "fit_rows": (
                    "every floor-retained row before the holdout entry whose "
                    "five-day label window ends before that entry"
                ),
                "rows": int(np.asarray(reservation["fit_indices"]).size),
                "n_purged_rows": int(reservation["n_purged_fit_rows"]),
                "validation_tail": False,
                "early_stopping": "not used in the final refit",
            },
            "benchmark": {
                "model": (
                    "canonical HAR-RV OLS on the same purged remainder, with "
                    "the per-ticker warm-up excluded from the fit"
                ),
                "fit_rows": int(evaluation["har_result"]["n_train"]),
                "warmup_rows_excluded": int(
                    evaluation["har_result"]["n_warmup_excluded"]
                ),
                "warmup_days": int(evaluation["har_result"]["warmup_days"]),
                "fit_scale": FIT_SCALE,
                "report_scale": REPORT_SCALE,
                "comparison_note": COMPARISON_NOTE,
            },
            "purge": {
                "label_horizon_days": int(LABEL_HORIZON_DAYS),
                "purge_horizon_days": int(cross_validator.PURGE_HORIZON_D),
                "separation_min": int(cross_validator.SEPARATION_MIN),
                "last_fit_date": pd.Timestamp(
                    reservation["unique_dates"][reservation["last_fit_pos"]]
                ).strftime("%Y-%m-%d"),
                "first_holdout_date": bounds["start"],
                "separation_dates": int(reservation["separation"]),
            },
            "holdout": {
                "bounds": bounds,
                "n_dates": int(reservation["n_holdout_dates"]),
                "n_rows": int(reservation["n_holdout_rows"]),
                "n_scored_rows": int(reservation["n_scored_rows"]),
                "n_floor_excluded": int(reservation["n_floor_excluded"]),
            },
            "one_shot": {
                "lock_before_metrics": True,
                "durable_record": (
                    "the run manifest carries the full lock payload; a lost "
                    "results lock is restored from it and remains binding"
                ),
                "second_evaluation": (
                    "refused with HoldoutLockError"
                ),
                "resume": (
                    "records a skipped holdout in the run manifest; never "
                    "silently re-evaluates"
                ),
                "override": (
                    "explicit --force-holdout <reason> appends an override "
                    "record and re-evaluates the same locked identity"
                ),
            },
            "selection": current_payload["selection"],
        }
        report = {
            "schema_version": HOLDOUT_SCHEMA_VERSION,
            "report_type": "locked holdout final report",
            "generated_at": _iso(),
            "run_id": run_id,
            "config_hash": run.get("config_hash")
            or current_payload["config_hash"],
            "git_sha": git_sha(),
            "environment_reference": (
                f"{to_repo_relative(root / MANIFEST_FILENAME)}#environment"
            ),
            "environment": environment,
            "lock": {
                "sha256": lock_document["sha256"],
                "payload": current_payload,
                "recorded_at": lock_document["recorded_at"],
            },
            "protocol": protocol,
            "holdout": {
                **protocol["holdout"],
                "bounds": bounds,
            },
            "headline": {
                "metric": "QLIKE",
                "convention": (
                    "single holdout block; equal-weight over its floor-retained "
                    "rows"
                ),
                "weighting": (
                    "one observation per scored row in the reserved window; "
                    "the final refit is not reweighted by fold"
                ),
                "edge_definition": (
                    "edge = har_qlike - model_qlike over the scored holdout "
                    "rows; positive means the model has the lower QLIKE"
                ),
                "row_set": HEADLINE_ROW_SET,
                "scale": SCALE_NOTE,
                "n_scored_rows": int(metrics["n_scored_rows"]),
                "aggregate": {
                    "model_qlike": metrics["model_qlike"],
                    "har_qlike": metrics["har_qlike"],
                    "edge": metrics["edge"],
                },
                "caveats": caveats,
            },
            "metrics": metrics,
            "holdout_metrics": metrics,
            "benchmark": {
                **evaluation["har_result"],
                "har_qlike": metrics["har_qlike"],
                "har_r2": metrics["har_r2"],
            },
            "post_hoc_adjustment": {
                "status": post_hoc_adjustment()["status"],
                "applied": False,
                "note": ADJUSTMENT_CAVEAT,
            },
            "overrides": override_records,
            "artifacts": artifacts,
            "artifact_hashes": artifacts,
            "report_artifacts": {
                MODEL_FILENAME: sha256_file(model_path),
                PREDICTIONS_FILENAME: sha256_file(predictions_path),
                LOCK_FILENAME: sha256_file(lock_path),
            },
            "timing": {
                "refit_seconds": float(evaluation["fit_seconds"]),
                "inference_seconds": float(evaluation["inference_seconds"]),
                "har_seconds": float(evaluation["har_seconds"]),
                "total_seconds": float(time.monotonic() - started),
            },
        }
        atomic.write_json_atomically(
            report_path,
            sanitize_json(report),
            trailing_newline=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        record_artifact(report_path, registry_path=registry_path)
        sums_entries = {
            REPORT_FILENAME: sha256_file(report_path),
            MODEL_FILENAME: sha256_file(model_path),
            PREDICTIONS_FILENAME: sha256_file(predictions_path),
            LOCK_FILENAME: sha256_file(lock_path),
        }
        _write_sums(sums_path, sums_entries)
        record_artifact(sums_path, registry_path=registry_path)
    except Exception as exc:
        holdout_record["status"] = "failed"
        holdout_record["error"] = f"{type(exc).__name__}: {exc}"
        record_holdout(run, holdout_record)
        write_run_state(run)
        if standalone:
            finish_run(
                run, status="failed", error=f"{type(exc).__name__}: {exc}"
            )
        raise

    status = "overridden" if override_record is not None else "completed"
    holdout_record.update(
        {
            "status": status,
            "completed_at": _iso(),
            "duration_seconds": float(time.monotonic() - started),
            "report_path": to_repo_relative(report_path),
            "report_sha256": sums_entries[REPORT_FILENAME],
            "sha256sums_path": to_repo_relative(sums_path),
            "sha256sums_sha256": sha256_file(sums_path),
            "predictions_path": to_repo_relative(predictions_path),
            "predictions_sha256": sums_entries[PREDICTIONS_FILENAME],
            "model_path": to_repo_relative(model_path),
            "model_sha256": sums_entries[MODEL_FILENAME],
            "n_scored_rows": int(metrics["n_scored_rows"]),
            "n_floor_excluded": int(metrics["n_floor_excluded"]),
            "holdout_metrics": {
                "model_qlike": metrics["model_qlike"],
                "har_qlike": metrics["har_qlike"],
                "edge": metrics["edge"],
                "model_r2": metrics["model_r2"],
                "har_r2": metrics["har_r2"],
            },
        }
    )
    if override_record is not None:
        holdout_record["override"] = {
            "reason": override_record["reason"],
            "recorded_at": override_record["recorded_at"],
            "path": to_repo_relative(overrides_path),
        }
    record_holdout(run, holdout_record)
    write_run_state(run)
    record_artifacts(
        run,
        [lock_path, model_path, predictions_path, report_path, sums_path],
    )
    if standalone:
        record_phase(
            run,
            "holdout_evaluation",
            "completed",
            time.monotonic() - started,
        )
        finish_run(run, status="completed")
    logger.info(
        "Holdout %s: model QLIKE=%.6f, HAR QLIKE=%.6f, edge=%+.6f on %d "
        "scored rows; report %s",
        status,
        metrics["model_qlike"],
        metrics["har_qlike"],
        metrics["edge"],
        metrics["n_scored_rows"],
        report_path,
    )
    return {
        "status": status,
        "lock": lock_document,
        "report": report,
        "report_path": str(report_path),
        "sums_path": str(sums_path),
        "overrides": override_records,
    }
