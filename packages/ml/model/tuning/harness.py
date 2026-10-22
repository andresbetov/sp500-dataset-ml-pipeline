"""Validation-only tuning harness with a mechanical test lock.

The harness tunes the fold-model hyperparameters on the purged inner-validation
tail and records the selection on validation QLIKE; the test block and the
reserved holdout window stay untouched. The lock is mechanical:

* One accessor, :func:`build_locked_fold_views`, builds every trial's fold
  data from the persisted fold metadata and the training path's
  ``inner_validation_split`` (the same split the fold models early-stop on,
  never a copy). It returns a fit view and a validation view.
* Every view is constructed through :func:`locked_view`, which refuses any
  index inside the fold's test block or the reserved holdout window (or
  outside fold-train at all) with :class:`TestLockError` before a view exists.
* The selection artifact schema, :func:`validate_selection_artifact`, rejects
  any ``test_``-prefixed key, so a selection record cannot carry test data.

A campaign runs the configured coarse grid, re-runs the best coarse
combination per gamma, and re-runs the best refined combination per seed; each
trial writes a strict-JSON sidecar with its parameters, per-fold validation
QLIKE, and its fit/validation index fingerprints through the shared atomic
sidecar writer, and the run manifest records every sidecar and the selection
artifact by hash. The campaign is opt-in: nothing in the pipeline calls it,
and the tests drive tiny synthetic campaigns under temporary roots.

Public surface:

* :class:`TestLockError` -- raised by any view or artifact that would carry
  test-block or holdout data.
* :func:`index_fingerprint` -- canonical SHA-256 of an index list.
* :func:`holdout_row_indices` -- rows inside the recorded holdout window.
* :func:`locked_view` -- the single view constructor (the lock's choke point).
* :func:`build_locked_fold_views` -- the single fold-data accessor.
* :func:`validate_selection_artifact` -- the selection schema (rejects
  ``test_``-prefixed keys).
* :func:`run_tuning_campaign` -- the grid/refinement/seed sequence.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from packages.ml.model import config as model_config
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.run_log import (
    environment_versions,
    finish_run,
    new_run,
    record_artifacts,
    record_phase,
)
from packages.ml.model.training import model_trainer, training

logger = logging.getLogger(__name__)

# The one selection metric the harness records; any other configured selector
# is refused before a run starts.
SELECT_ON_VALIDATION_QLIKE = "validation_qlike"

# The coarse-stage budget cap: the configured grid may not exceed this many
# combinations (the recorded tuning budget).
MAX_COARSE_TRIALS = 27

# The tuning knobs the coarse grid sweeps; the remaining recipe stays at the
# trainer's baseline defaults.
COARSE_KNOBS = ("learning_rate", "max_depth", "min_child_weight")


class TestLockError(RuntimeError):
    """A tuning view or artifact would carry test-block or holdout data."""


def index_fingerprint(indices: Sequence[int] | np.ndarray) -> str:
    """Canonical SHA-256 over an index list (order-insensitive).

    The indices are sorted and hashed as int64 bytes, so a view's fingerprint
    identifies its index set exactly. Every trial records one fingerprint per
    fit/validation view, and a reader can recompute the digest from the locked
    accessor to verify that no trial saw different rows.
    """
    canonical = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def holdout_row_indices(dates: np.ndarray, folds_metadata: Mapping) -> np.ndarray:
    """Row indices inside the reserved holdout window, from the recorded bounds.

    The persisted payload records the holdout window's date bounds; every row
    whose date falls on or after the recorded start is a holdout row. The lock
    refuses to build any view without the recorded window: absent or malformed
    holdout metadata raises by name rather than silently skipping the check.
    """
    holdout = folds_metadata.get("holdout")
    if not isinstance(holdout, Mapping) or "bounds" not in holdout:
        raise ValueError(
            "fold metadata has no reserved holdout window (holdout.bounds); "
            "the tuning lock refuses to build views without it"
        )
    n_dates = int(holdout.get("n_dates") or 0)
    start = (holdout.get("bounds") or {}).get("start")
    if n_dates == 0:
        return np.empty(0, dtype=np.int64)
    if start is None:
        raise ValueError("holdout window records n_dates > 0 but no bounds.start")
    return np.flatnonzero(np.asarray(dates) >= np.datetime64(start))


def _assert_locked(
    fold_key: str,
    role: str,
    requested: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    holdout_indices: np.ndarray,
) -> None:
    """Refuse any view index that is test-block, holdout, or outside fold-train."""
    test_hits = np.intersect1d(requested, test_indices)
    if test_hits.size:
        raise TestLockError(
            f"{fold_key} {role} view contains test-block indices "
            f"{test_hits[:8].tolist()}; the tuning lock refuses test data"
        )
    holdout_hits = np.intersect1d(requested, holdout_indices)
    if holdout_hits.size:
        raise TestLockError(
            f"{fold_key} {role} view contains holdout indices "
            f"{holdout_hits[:8].tolist()}; the tuning lock refuses the reserved "
            f"holdout window"
        )
    outside = np.setdiff1d(requested, train_indices)
    if outside.size:
        raise TestLockError(
            f"{fold_key} {role} view contains indices outside fold-train "
            f"{outside[:8].tolist()}; trials may only see fold-train rows"
        )


def locked_view(
    fold_key: str,
    role: str,
    indices: Sequence[int] | np.ndarray,
    *,
    X: np.ndarray,
    y: np.ndarray,
    fold_train_indices: np.ndarray,
    test_indices: np.ndarray,
    holdout_indices: np.ndarray,
) -> dict:
    """The single view constructor: every harness view goes through this lock.

    ``indices`` must be integers inside ``fold_train_indices`` and disjoint
    from the fold's test block and the reserved holdout window; any violation
    raises :class:`TestLockError` before a view exists. The returned view
    carries the requested rows, their fingerprint, and the role/fold identity.
    """
    raw = np.asarray(indices)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(
            f"{fold_key} {role} view indices must be integers, got dtype "
            f"{raw.dtype}"
        )
    requested = raw.astype(np.int64, copy=False)
    _assert_locked(
        fold_key,
        role,
        requested,
        train_indices=fold_train_indices,
        test_indices=test_indices,
        holdout_indices=holdout_indices,
    )
    return {
        "fold_key": fold_key,
        "role": role,
        "indices": requested,
        "fingerprint": index_fingerprint(requested),
        "X": X[requested],
        "y": y[requested],
    }


def build_locked_fold_views(
    fold_key: str,
    *,
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    folds_metadata: Mapping,
    tail_fraction: float | None = None,
) -> dict:
    """The single fold-data accessor: locked fit + purged validation views.

    The views are the training task's split, not a copy: floor-retained
    fold-train rows (the shared ``floor_retained_mask`` policy) feed
    ``training.inner_validation_split``, so the validation view is the purged
    inner-validation tail and the fit view is its remainder. Both views are
    constructed through :func:`locked_view`, so no view can carry a test-block
    or holdout index. ``y`` is the raw target (the floor policy and the split
    read it); the views carry ``log(max(y, RV_FLOOR))``, the modeled target.

    Args:
        fold_key: The fold record's key in ``folds_metadata``.
        X: Phase-2 feature matrix, row-aligned with ``y`` and ``dates``.
        y: Raw Phase-2 target array.
        dates: Per-row trading date array (``metadata["date"]``).
        folds_metadata: The persisted fold payload (fold_* records + holdout).
        tail_fraction: Inner-validation tail fraction; defaults to the
            configured ``training.validation_tail_fraction``.

    Returns:
        ``{"fold_key", "fit", "validation", "n_purged"}`` where the two views
        are :func:`locked_view` results.
    """
    fold = folds_metadata.get(fold_key)
    if fold is None:
        raise ValueError(f"fold metadata has no {fold_key!r} record")
    y_values = np.asarray(y)
    dates_values = np.asarray(dates)
    train_indices = np.asarray(fold["train_indices"], dtype=np.int64)
    test_indices = np.asarray(fold["test_indices"], dtype=np.int64)
    holdout = holdout_row_indices(dates_values, folds_metadata)
    date_pos = np.unique(dates_values, return_inverse=True)[1]
    y_log = np.log(np.maximum(y_values, training.RV_FLOOR))

    # Floor-retained fold-train rows under the shared strict floor predicate;
    # the split itself is the training path's inner_validation_split.
    retained = train_indices[training.floor_retained_mask(y_values[train_indices])]
    fold_idx = int(fold_key.split("_", 1)[1])
    fit_indices, validation_indices, purge_count = training.inner_validation_split(
        fold_idx,
        retained,
        date_pos,
        y_values,
        training.VALIDATION_TAIL_FRACTION
        if tail_fraction is None
        else tail_fraction,
    )
    fit = locked_view(
        fold_key,
        "fit",
        fit_indices,
        X=X,
        y=y_log,
        fold_train_indices=train_indices,
        test_indices=test_indices,
        holdout_indices=holdout,
    )
    validation = locked_view(
        fold_key,
        "validation",
        validation_indices,
        X=X,
        y=y_log,
        fold_train_indices=train_indices,
        test_indices=test_indices,
        holdout_indices=holdout,
    )
    return {
        "fold_key": fold_key,
        "fit": fit,
        "validation": validation,
        "n_purged": int(purge_count),
    }


def validate_selection_artifact(payload: Mapping) -> None:
    """Reject any ``test_``-prefixed key anywhere in the selection artifact.

    The selection record is validation-only: it may name validation metrics
    and index fingerprints, never test-block data. The scan is recursive, so a
    ``test_*`` key nested inside a trial or fold record fails by name before
    the artifact is written.
    """

    def _walk(node: object, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if isinstance(key, str) and key.startswith("test_"):
                    raise TestLockError(
                        f"selection artifact carries a test_-prefixed key "
                        f"{path}{key}; selection is recorded on validation "
                        f"QLIKE only"
                    )
                _walk(value, f"{path}{key}.")
        elif isinstance(node, (list, tuple)):
            for position, value in enumerate(node):
                _walk(value, f"{path}[{position}].")

    _walk(payload, "")


def _validate_search_space(space: Mapping) -> None:
    """Check the configured search space shape and its coarse budget cap."""
    if not isinstance(space, Mapping):
        raise ValueError(
            f"tuning search space must be a mapping, got {type(space).__name__}"
        )
    grid = space.get("coarse_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("tuning search space is missing coarse_grid")
    coarse_combos = 1
    for knob in COARSE_KNOBS:
        values = grid.get(knob)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(
                f"tuning search space coarse_grid.{knob} must be a non-empty list"
            )
        coarse_combos *= len(values)
    if coarse_combos > MAX_COARSE_TRIALS:
        raise ValueError(
            f"tuning search space has {coarse_combos} coarse combinations; "
            f"the budget cap is {MAX_COARSE_TRIALS}"
        )
    refinement = space.get("finalist_refinement")
    if not isinstance(refinement, Mapping):
        raise ValueError("tuning search space is missing finalist_refinement")
    gammas = refinement.get("gamma")
    if not isinstance(gammas, (list, tuple)) or not gammas:
        raise ValueError(
            "tuning search space finalist_refinement.gamma must be a non-empty list"
        )
    seeds = space.get("seed_finalists")
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("tuning search space seed_finalists must be a non-empty list")
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ValueError(
            f"tuning search space seed_finalists must be integers, got {seeds!r}"
        )


def _best_trial(trials: list[dict]) -> dict:
    """The trial with the minimum validation QLIKE (earliest wins ties)."""
    return min(
        trials,
        key=lambda trial: (trial["validation_qlike"], trial["trial_index"]),
    )


def _run_trial(
    *,
    trial_index: int,
    stage: str,
    seed: int,
    params: dict,
    fold_views: Mapping,
    trainer: Callable,
    run: dict,
    run_dir: Path,
    registry_path: Path,
    environment: dict,
) -> dict:
    """Run one trial across every fold and write its strict-JSON sidecar.

    The trainer only ever receives the locked fit/validation views. The trial
    score is the mean validation QLIKE across folds; the sidecar records the
    effective parameters, the per-fold QLIKE, and each fold's fit/validation
    index fingerprints. The sidecar is recorded in the artifact registry and
    queued for the run manifest immediately after the atomic write.
    """
    started = time.time()
    fold_records = {}
    for fold_key, views in fold_views.items():
        model, _label_mapping = trainer(
            X_train=views["fit"]["X"],
            y_train=views["fit"]["y"],
            X_val=views["validation"]["X"],
            y_val=views["validation"]["y"],
            params=params,
        )
        predictions = np.asarray(
            model.predict(views["validation"]["X"]), dtype=np.float64
        )
        best_iteration = getattr(model, "best_iteration", None)
        fold_records[fold_key] = {
            "n_fit": int(len(views["fit"]["indices"])),
            "n_validation": int(len(views["validation"]["indices"])),
            "fit_fingerprint": views["fit"]["fingerprint"],
            "validation_fingerprint": views["validation"]["fingerprint"],
            "validation_qlike": float(
                training.qlike_loss(views["validation"]["y"], predictions)
            ),
            "best_iteration": (
                int(best_iteration) if best_iteration is not None else None
            ),
        }

    trial = {
        "trial_index": trial_index,
        "stage": stage,
        "seed": seed,
        "params": {
            key: training.sidecar_value(value) for key, value in params.items()
        },
        "validation_qlike": float(
            np.mean(
                [record["validation_qlike"] for record in fold_records.values()]
            )
        ),
        "folds": fold_records,
        "duration_seconds": time.time() - started,
        "environment": environment,
        "git_sha": run["git_sha"],
    }
    sidecar_path = run_dir / f"trial_{trial_index:04d}_sidecar.json"
    training.write_json_atomically(
        sidecar_path,
        trial,
        trailing_newline=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    record_artifact(sidecar_path, registry_path=registry_path)
    record_artifacts(run, [sidecar_path])
    logger.info(
        f"Trial {trial_index} ({stage}, seed {seed}): mean validation QLIKE "
        f"{trial['validation_qlike']:.6f} across {len(fold_records)} fold(s)"
    )
    return trial


def _compact_trial(trial: Mapping) -> dict:
    """A trial's selection-record projection (fingerprints, no index lists)."""
    return {
        "trial_index": trial["trial_index"],
        "stage": trial["stage"],
        "seed": trial["seed"],
        "params": dict(trial["params"]),
        "validation_qlike": trial["validation_qlike"],
        "folds": {
            fold_key: {
                "n_fit": record["n_fit"],
                "n_validation": record["n_validation"],
                "fit_fingerprint": record["fit_fingerprint"],
                "validation_fingerprint": record["validation_fingerprint"],
                "validation_qlike": record["validation_qlike"],
                "best_iteration": record["best_iteration"],
            }
            for fold_key, record in trial["folds"].items()
        },
    }


def run_tuning_campaign(
    *,
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    folds_metadata: Mapping,
    output_dir: Path | str,
    search_space: Mapping | None = None,
    trainer: Callable | None = None,
) -> dict:
    """Run the coarse grid -> gamma refinement -> seed finalists sequence.

    Every trial trains one model per fold through the locked accessor's fit
    view and scores the validation view's QLIKE; the trial score is the mean
    across folds. The coarse stage sweeps the configured grid at the baseline
    seed, the refinement stage re-runs the best coarse combination per gamma,
    and the seed-finalist stage re-runs the best refined combination per seed;
    the selection is the best seed finalist on validation QLIKE (ties keep the
    earliest trial). Each trial writes a sidecar under ``output_dir/<run_id>/``
    recording its parameters, per-fold validation QLIKE, and fit/validation
    index fingerprints; every sidecar and the selection artifact are recorded
    in the artifact registry and the run manifest. The selection artifact
    carries ``selected_on: validation_qlike`` and zero ``test_``-prefixed keys;
    :func:`validate_selection_artifact` refuses it otherwise. A failure
    finalizes the run as failed and re-raises.

    Args:
        X: Phase-2 feature matrix, row-aligned with ``y`` and ``dates``.
        y: Raw Phase-2 target array.
        dates: Per-row trading date array (``metadata["date"]``).
        folds_metadata: The persisted fold payload (fold_* records + holdout).
        output_dir: Campaign output root. The run-scoped files land in
            ``output_dir/<run_id>/``; ``run_manifest.json``, ``runs.jsonl``,
            and ``artifact_registry.json`` land at the root.
        search_space: Search-space mapping; defaults to the configured
            ``tuning.search_space``.
        trainer: Trainer callable with the ``train_xgboost_model`` signature
            (``X_train, y_train, X_val, y_val, params=...``); defaults to
            ``train_xgboost_model``. Injectable for tests.

    Returns:
        The campaign record: run identity and paths, the selection payload,
        and the per-trial records.
    """
    loaded = model_config.get_config()
    select_on = loaded["tuning"]["select_on"]
    if select_on != SELECT_ON_VALIDATION_QLIKE:
        raise ValueError(
            f"tuning.select_on must be {SELECT_ON_VALIDATION_QLIKE!r}, got "
            f"{select_on!r}; selection is recorded on validation QLIKE only"
        )
    if search_space is None:
        search_space = loaded["tuning"]["search_space"]
    _validate_search_space(search_space)
    train = trainer if trainer is not None else model_trainer.train_xgboost_model
    base_seed = int(loaded["training"]["seed"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / REGISTRY_FILENAME
    run = new_run("tuning", artifacts_dir=output_dir)
    run_dir = output_dir / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_versions()

    trials: list[dict] = []
    stages: dict[str, dict] = {}

    try:
        # The one accessor builds every fold's locked views; a metadata
        # poisoning that would put test/holdout rows into a view fails here,
        # before any trial runs.
        fold_views = {
            fold_key: build_locked_fold_views(
                fold_key, X=X, y=y, dates=dates, folds_metadata=folds_metadata
            )
            for fold_key in training.fold_keys(folds_metadata)
        }

        def execute_stage(stage: str, candidates: list[tuple[dict, int]]) -> dict:
            stage_started = time.monotonic()
            stage_trials: list[dict] = []
            for knobs, seed in candidates:
                trial = _run_trial(
                    trial_index=len(trials),
                    stage=stage,
                    seed=seed,
                    params={
                        **model_trainer.DEFAULT_PARAMS,
                        **knobs,
                        "random_state": seed,
                    },
                    fold_views=fold_views,
                    trainer=train,
                    run=run,
                    run_dir=run_dir,
                    registry_path=registry_path,
                    environment=environment,
                )
                trials.append(trial)
                stage_trials.append(trial)
            best = _best_trial(stage_trials)
            stages[stage] = {
                "n_trials": len(stage_trials),
                "best_trial_index": best["trial_index"],
                "best_validation_qlike": best["validation_qlike"],
            }
            record_phase(
                run,
                f"tuning_{stage}",
                "completed",
                time.monotonic() - stage_started,
            )
            return best

        grid = search_space["coarse_grid"]
        best_coarse = execute_stage(
            "coarse",
            [
                (
                    {
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "min_child_weight": min_child_weight,
                        "gamma": model_trainer.DEFAULT_PARAMS["gamma"],
                    },
                    base_seed,
                )
                for learning_rate, max_depth, min_child_weight in itertools.product(
                    grid["learning_rate"],
                    grid["max_depth"],
                    grid["min_child_weight"],
                )
            ],
        )

        refined_knobs = {
            knob: best_coarse["params"][knob] for knob in COARSE_KNOBS
        }
        best_refined = execute_stage(
            "refinement",
            [
                ({**refined_knobs, "gamma": gamma}, base_seed)
                for gamma in search_space["finalist_refinement"]["gamma"]
            ],
        )

        finalist_knobs = {**refined_knobs, "gamma": best_refined["params"]["gamma"]}
        selected = execute_stage(
            "seed_finalists",
            [
                (dict(finalist_knobs), seed)
                for seed in search_space["seed_finalists"]
            ],
        )

        selection_payload = {
            "selected_on": SELECT_ON_VALIDATION_QLIKE,
            "selection_rule": (
                "minimum mean validation QLIKE across folds (lower is better)"
            ),
            "run_id": run["run_id"],
            "config_hash": run["config_hash"],
            "git_sha": run["git_sha"],
            "n_trials": len(trials),
            "search_space": search_space,
            "stages": stages,
            "selected_trial": _compact_trial(selected),
            "trials": [_compact_trial(trial) for trial in trials],
        }
        validate_selection_artifact(selection_payload)
        selection_path = run_dir / "selection.json"
        training.write_json_atomically(
            selection_path,
            selection_payload,
            trailing_newline=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        record_artifact(selection_path, registry_path=registry_path)
        record_artifacts(run, [selection_path])
    except Exception as exc:
        finish_run(run, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise

    finish_run(run, status="completed")
    return {
        "run_id": run["run_id"],
        "output_dir": output_dir,
        "run_dir": run_dir,
        "manifest_path": output_dir / "run_manifest.json",
        "selection_path": selection_path,
        "selected_on": SELECT_ON_VALIDATION_QLIKE,
        "selected_trial": selection_payload["selected_trial"],
        "n_trials": len(trials),
        "trials": trials,
    }
