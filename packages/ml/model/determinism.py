"""Determinism entry: the seed sweep and the equality fingerprints.

The pipeline's deterministic contract is declared in ``config/base.yaml`` and
recorded per run by :mod:`packages.ml.model.run_log`: one trainer thread, a
pinned ``OMP_NUM_THREADS``, the baseline/trainer seeds, ``PYTHONHASHSEED``,
library versions, the git sha, and the ``uv.lock`` hash. This module adds the
two-run evidence that the contract actually holds:

* :func:`run_seed_sweep` trains the configured seeds (at least three) on every
  fold through the validation-only tuning lock (fit and purged inner-validation
  views, never a test block or the holdout window) and writes
  ``seed_sweep.json`` with per-seed/per-fold validation QLIKE, index
  fingerprints, and model/prediction hashes, together with the policy and
  environment that produced it.
* :func:`packages.ml.model.cross_validation.cross_validator.fold_manifest_hash`
  gives the fold-construction digest; two in-memory constructions from the
  same metadata are hash-equal.

The default sweep configuration is intentionally small (a handful of trees) so
the entry is cheap evidence, not a retrain. Nothing in the pipeline calls it;
run it explicitly:

    python -m packages.ml.model.determinism

It writes ``packages/ml/model/artifacts/results/seed_sweep.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np

if __name__ == "__main__" and __package__ is None:
    # Direct execution: python packages/ml/model/determinism.py — normalize to
    # the canonical package identity so the package-absolute imports resolve
    # without an installation (PEP 366).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "packages.ml.model"

from packages.ml.model import config as model_config
from packages.ml.model import run_log
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, load_verified
from packages.ml.model.cross_validation import cross_validator
from packages.ml.model.cross_validation.cv_utils import load_folds_metadata
from packages.ml.model.training import model_trainer, training
from packages.ml.model.tuning import harness
from packages.ml.model.utils import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

SEED_SWEEP_FILENAME = "seed_sweep.json"

# The sweep stays cheap: this many boosting rounds per tiny fit, not the full
# configured round count. Three seeds are the minimum the report requires.
DEFAULT_SWEEP_TREES = 25
DEFAULT_SWEEP_SEEDS = (42, 43, 44)
MINIMUM_SWEEP_SEEDS = 3


def _sha256_json(payload: object) -> str:
    """SHA-256 over canonical strict JSON."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    """SHA-256 over the contiguous little-endian float64 bytes of an array."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _model_sha256(model: object) -> str:
    """SHA-256 of the joblib-serialized model bytes."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _sweep_params(seed: int, n_estimators: int) -> dict:
    """The frozen recipe with a tiny round count, the seed, and one thread."""
    params = model_trainer.default_params()
    params.update(
        {
            "n_estimators": int(n_estimators),
            "random_state": int(seed),
            "n_jobs": 1,
        }
    )
    return params


def _serializable_params(params: Mapping) -> dict:
    """Project trainer parameters onto strict-JSON data (callables by name)."""
    return {key: training.sidecar_value(value) for key, value in params.items()}


def default_seed_sweep_seeds() -> list[int]:
    """The configured seed finalists, or the built-in three-seed fallback."""
    loaded = model_config.get_config()
    search_space = (loaded.get("tuning") or {}).get("search_space") or {}
    configured = search_space.get("seed_finalists")
    if configured and len({int(seed) for seed in configured}) >= MINIMUM_SWEEP_SEEDS:
        return [int(seed) for seed in configured]
    return list(DEFAULT_SWEEP_SEEDS)


def run_seed_sweep(
    *,
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    folds_metadata: Mapping,
    output_path: Path | str,
    seeds: Sequence[int] | None = None,
    n_estimators: int = DEFAULT_SWEEP_TREES,
    trainer: Callable | None = None,
) -> dict:
    """Train every seed on every fold's validation view and write the sweep.

    The fold data comes from :func:`harness.build_locked_fold_views`, the same
    validation-only lock the tuning campaign uses: a fit view of floor-retained
    fold-train rows and the purged inner-validation tail, never a test-block or
    holdout index. For each seed the trainer receives the tiny recipe and the
    pinned one-thread setting; each fold records its validation QLIKE, index
    fingerprints, best iteration, and model/prediction hashes. The document
    also carries the thread/seed policy, the environment manifest, and the
    fold-manifest digest, so the sweep is self-describing evidence.

    Args:
        X: Phase-2 feature matrix, row-aligned with ``y`` and ``dates``.
        y: Raw Phase-2 target array.
        dates: Per-row trading date array (``metadata["date"]``).
        folds_metadata: The fold payload (fold_* records plus holdout).
        output_path: Destination of the strict-JSON ``seed_sweep.json``.
        seeds: Override seeds; defaults to the configured seed finalists.
        n_estimators: Tiny boosting-round count for the sweep fits.
        trainer: Trainer callable with the ``train_xgboost_model`` signature;
            defaults to the real trainer and is injectable for tests.

    Returns:
        The written sweep document.
    """
    seed_values = (
        default_seed_sweep_seeds() if seeds is None else [int(seed) for seed in seeds]
    )
    distinct_seeds = list(dict.fromkeys(seed_values))
    if len(distinct_seeds) < MINIMUM_SWEEP_SEEDS:
        raise ValueError(
            "a seed sweep requires at least "
            f"{MINIMUM_SWEEP_SEEDS} distinct seeds, got {distinct_seeds}"
        )
    if int(n_estimators) < 1:
        raise ValueError(f"n_estimators must be positive, got {n_estimators}")

    train = trainer if trainer is not None else model_trainer.train_xgboost_model
    effective_policy = run_log.apply_thread_policy()
    environment = run_log.environment_manifest()

    fold_views = {
        fold_key: harness.build_locked_fold_views(
            fold_key,
            X=np.asarray(X),
            y=np.asarray(y),
            dates=np.asarray(dates),
            folds_metadata=folds_metadata,
        )
        for fold_key in training.fold_keys(folds_metadata)
    }

    seed_records = []
    for seed in distinct_seeds:
        params = _sweep_params(seed, n_estimators)
        params_payload = _serializable_params(params)
        fold_records = {}
        for fold_key, views in fold_views.items():
            model, _label_mapping = train(
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
                "predictions_sha256": _array_sha256(predictions),
                "model_sha256": _model_sha256(model),
            }
        seed_records.append(
            {
                "seed": seed,
                "params": params_payload,
                "params_sha256": _sha256_json(params_payload),
                "mean_validation_qlike": float(
                    np.mean(
                        [
                            record["validation_qlike"]
                            for record in fold_records.values()
                        ]
                    )
                ),
                "folds": fold_records,
            }
        )
        logger.info(
            f"Seed sweep {seed}: mean validation QLIKE "
            f"{seed_records[-1]['mean_validation_qlike']:.6f} across "
            f"{len(fold_records)} fold(s)"
        )

    document = {
        "config_hash": run_log.config_hash(),
        "policy": {
            "policy": effective_policy["policy"],
            "seed": effective_policy["seed"],
            "random_state": effective_policy["random_state"],
            "n_jobs": effective_policy["n_jobs"],
            "omp_num_threads": effective_policy["omp_num_threads"],
            "PYTHONHASHSEED": effective_policy["PYTHONHASHSEED"],
            "seeds": distinct_seeds,
            "n_estimators": int(n_estimators),
        },
        "environment": environment,
        "fold_manifest_hash": cross_validator.fold_manifest_hash(folds_metadata),
        "folds": list(fold_views),
        "seeds": seed_records,
    }
    output_path = Path(output_path)
    training.write_json_atomically(
        output_path,
        document,
        trailing_newline=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    logger.info(f"Seed sweep written to {output_path}")
    return document


def main(argv: Sequence[str] | None = None) -> dict:
    """Run the sweep against the live Phase-2/3 inputs and write the artifact."""
    parser = argparse.ArgumentParser(
        description=(
            "Train at least three seeds on every fold's validation view and "
            "write seed_sweep.json with the determinism policy."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "results" / SEED_SWEEP_FILENAME,
        help=f"destination (default: results/{SEED_SWEEP_FILENAME})",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=DEFAULT_SWEEP_TREES,
        help=f"boosting rounds per tiny fit (default: {DEFAULT_SWEEP_TREES})",
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        help="seed to sweep (repeatable; default: the configured seed finalists)",
    )
    args = parser.parse_args(argv)

    X = np.load(ARTIFACTS_DIR / "inputs" / "X.npy", mmap_mode="r")
    y = np.load(ARTIFACTS_DIR / "inputs" / "Y.npy", mmap_mode="r")
    metadata = load_verified(
        ARTIFACTS_DIR / "inputs" / "metadata.pkl",
        registry_path=ARTIFACTS_DIR / REGISTRY_FILENAME,
    )
    folds_metadata = load_folds_metadata()
    return run_seed_sweep(
        X=X,
        y=y,
        dates=metadata["date"],
        folds_metadata=folds_metadata,
        output_path=args.output,
        seeds=args.seeds,
        n_estimators=args.trees,
    )


if __name__ == "__main__":
    main()
