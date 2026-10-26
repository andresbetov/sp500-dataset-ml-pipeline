# Operations

This is the canonical runbook for the model pipeline. It covers the run lifecycle, resume, retention, the artifact registry, and the locked holdout. Fold geometry and metric definitions live in [pipelines/model.md](pipelines/model.md); how to read the numbers lives in [results.md](results.md).

## Scope and mental model

The pipeline writes to two output roots, both relative to the repository root and both gitignored:

- `packages/ml/model/artifacts/` — pipeline state and results. This is the source of truth.
- `data/model_outputs/` — presentation charts produced by Phase 5.

When a number disagrees between a chart and an artifact, trust the artifact: charts are renderings of the results and are not authoritative, and they are produced for humans while the JSON and parquet artifacts are the machine-readable record. Phase 1 also writes into the model package's tree (`packages/ml/model/artifacts/results/regime_characterization.json`), so the two roots are not strictly phase-separated.

## Run state files

`run_manifest.json` is the latest run's state: identity, config hash, fold selection, environment versions, phase and fold status, durations, and artifact hashes. Every run overwrites it, so it describes only the most recent run.

`runs.jsonl` is the append-only run log: one strict-JSON line per run, never pruned. Strict JSON means no NaN or Infinity extensions, not that every number is exact. Retention removes directories, not history, so this file is the durable record of every run, including ones whose checkpoint directories have been pruned.

`artifact_registry.json` records byte size and SHA-256 for each recorded artifact and is updated atomically at write time. Every live `joblib.load` verifies an artifact against the registry before loading it, so a corrupt or stale file is detected rather than silently consumed.

All three are written through a temporary sibling file followed by an atomic replace, so a torn write leaves the previous file intact.

The three files have different lifetimes, and reading them together tells you the whole story: `run_manifest.json` tells you the latest run, `runs.jsonl` tells you every run, and `artifact_registry.json` tells you whether a given artifact is exactly the bytes the pipeline wrote.

## Normal run

All commands in this runbook run from the repository root after `uv sync`; activate `.venv` (`source .venv/bin/activate`) or prefix commands with `uv run`. The entry points resolve their own paths, but the root is the documented working directory.

Phase 1 must run first: it writes `data/processed/dataset.parquet`, and Phase 2 fails fast without it. See the [quickstart](../README.md).

```bash
python packages/ml/dataset/main.py   # Phase 1
python -m packages.ml.model.main     # Phases 2-5
```

The model entry runs Phase 2 (feature selection), Phase 3 (cross-validation), Phase 4 (training and evaluation), and Phase 5 (visualization) in order. A full run writes the inputs, the folds, one run-scoped checkpoint directory per fold, the results files, the charts, and the run state files above. Phase 4 is the training and evaluation protocol described in [pipelines/model.md](pipelines/model.md); it trains one model per fold and fits the HAR benchmark on training rows only. The full flag contract is in [reference/cli.md](reference/cli.md).

At the current checkout with default retention 5, the generated tree is a `~864 MiB` dataset parquet, `~2.2 GiB` of retained model artifacts, and `~5 MiB` of charts (dated snapshot, 2026-10-25). Runtime is hardware- and data-dependent; the run manifest records each phase duration, and the report records the holdout refit/inference timings. Plan for minutes, not seconds.

Each run owns one run record. Its phase statuses, durations, per-fold outcomes, fold selection, and artifact hashes land in `run_manifest.json` plus one appended line in `runs.jsonl`. A run id is a UTC stamp plus the config hash, for example `20261020T030822Z-407a2ad5`.

If a phase raises, the run is recorded as failed with the error and the exception is re-raised. A failed run stays resumable: the entry point preserves the identity keys so the next invocation can continue rather than start over. A fold whose metric computation fails is quarantined — recorded as an error and excluded from pooled aggregates — and the pipeline raises only when the headline QLIKE has no valid fold left.

## Partial runs

`--folds 0,2` trains only the named folds. The run is explicitly partial: `run_manifest.json` records `partial: true` plus the selection, and `packages/ml/model/artifacts/results/run_selection.json` restates the requested, trained, and resumed folds.

`run_selection.json` is written on every run, not just partial ones: a full run records `partial: false`. A full run also writes one summary row per configured fold, while a partial run's summaries, loss vectors, and charts contain only the selected folds.

## Resume

`--resume` continues the latest run. It reads `run_manifest.json` and requires its `run_id`, `config_hash`, `fold_manifest_hash`, and `input_identity` (the dataset SHA-256 plus the persisted `X.npy` and `Y.npy` hashes).

A fold is reused only when all four of its sidecars exist — the `fold_<i>.pkl` checkpoint, `fold_<i>_params.json`, `fold_<i>_summary.json`, and `fold_<i>_predictions.pkl` — and match their `artifact_registry.json` size and SHA-256 entries. The summary must also record the predictions file's hash, so a summary left beside a newer predictions file is detected and the fold recomputed.

An incomplete fold is retrained. A missing or mismatched identity refuses by name; the pipeline never falls back to a fresh run silently. The persisted inputs and fold payload are verified before Phase 2 or Phase 3 can rewrite them.

Each kept run directory carries a `README.md` naming the run id, config hash, fold-manifest hash, status, fold selection, resumed flag, contents, and retention rule. It does not record the input identity.

## Retention

After each run, retention keeps the newest `run.retention` (default 5) run-scoped checkpoint directories under `checkpoints/` and removes the older ones. The current run is always kept, even if a clock skew would sort it outside the newest names. Every kept directory carries a `README.md`. `runs.jsonl` is append-only and is never pruned, so run history survives directory retention.

Only directories whose name matches the run-id shape are retention candidates; anything else under `checkpoints/` is left alone.

## Locked holdout

The holdout is the most sensitive part of the pipeline. Treat this section as the authority.

The holdout is **opt-in and one-shot**. The default no-argument run never reads the reserved window. Only `--evaluate-holdout` reads it, after the normal phases complete.

The first `--evaluate-holdout` writes `packages/ml/model/artifacts/results/holdout_lock.json` and records the full lock payload in `run_manifest.json` **before** the final refit or any holdout metric. The lock is a SHA-256 over the frozen selection recipe (hyperparameters plus seed/thread policy), the Phase-2 dataset pin and derived `X`/`Y` hashes, the fold-manifest hash, and the reserved holdout bounds.

The lock is durable across ordinary runs. Every new run manifest carries the latest holdout block forward, and the append-only `runs.jsonl` is the fallback when a manifest has lost it. If the results lock file is deleted, it is restored from the manifest or the last holdout-bearing log record and remains binding; an ordinary run cannot erase the lock.

The holdout cannot be re-scored silently. A second `--evaluate-holdout` raises `HoldoutLockError`. A post-lock `--resume --evaluate-holdout` records a skip instead of re-scoring.

`--force-holdout <reason>` is the only override. The reason must be non-empty and it implies `--evaluate-holdout`. It requires a recorded lock, appends the reason to `packages/ml/model/artifacts/results/holdout_overrides.jsonl`, and still refuses a changed config, dataset pin, input hash, fold hash, or bounds. `holdout_overrides.jsonl` exists only after a force.

The holdout writes `holdout_model.pkl`, `holdout_predictions.pkl`, `holdout_report.json`, and `SHA256SUMS`, which lists the report and its sibling holdout artifacts in `sha256sum` format.

**Do not use the real holdout to audit or test documentation.** On a fresh checkout with no lock, the first `--evaluate-holdout` creates the lock and scores the holdout, and it cannot be repeated. Exercise the holdout only through the synthetic, temporary-root tests.

## Artifacts map

| Path | Phase | Contents |
|---|---|---|
| `data/processed/dataset.parquet` | 1 | Cleaned survivors-only panel plus engineered features and the target |
| `packages/ml/model/artifacts/inputs/` | 2 | `X.npy`, `Y.npy`, `metadata.pkl` — exactly what training consumes |
| `packages/ml/model/artifacts/folds/` | 3 | `folds_metadata.pkl` — fold indices, date ranges, and the reserved holdout |
| `packages/ml/model/artifacts/checkpoints/<run_id>/` | 4 | One `fold_<i>.pkl` per fold plus its params, summary, and predictions sidecars |
| `packages/ml/model/artifacts/results/` | 1, 4, holdout | Fold summaries, benchmark, `validation_predictions.pkl`, loss vectors, `evaluation_summary.json`, `run_selection.json`, `regime_characterization.json` (Phase 1), holdout lock/report, `holdout_overrides.jsonl` only after `--force-holdout` |
| `packages/ml/model/artifacts/run_manifest.json`, `runs.jsonl`, `artifact_registry.json` | 2-5 | Run state, run history, artifact integrity |
| `data/model_outputs/` | 5 | 14 PNG charts plus `manifest.json` |

Deep schema for the inputs, folds, and results lives in [pipelines/model.md](pipelines/model.md) and [results.md](results.md).

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| Phase 2 fails immediately | `data/processed/dataset.parquet` is missing | Run Phase 1 first |
| `--resume` refuses by name | Manifest identity or sidecar hash mismatch | Let the pipeline rebuild inputs/folds, or start a fresh run |
| Holdout refuses with `HoldoutLockError` | Already evaluated, or a changed identity | Do not force; confirm nothing changed |
| `holdout_lock.json` is missing but the holdout was evaluated | The results lock was deleted | The pipeline restores it from the manifest or `runs.jsonl`; it remains binding |
| Results hold fewer than the full fold set | A `--folds` partial run | Check `packages/ml/model/artifacts/results/run_selection.json` for the selection |
| A chart is missing | Phase 5 did not run, or a partial run | Re-run Phases 2-5; partial runs render only emitted folds |
| Tests skip on a fresh clone | Generated artifacts are absent | Expected; checks that need generated data are skipped |

## Configuration

`packages/ml/model/config/base.yaml` owns the runtime constants. Set the `MODEL_CONFIG_PATH` environment variable to point the pipeline at a different config file.

The config sections are `rv_floor` (the target floor), `folds` (fold count and purge constants), `training` (determinism policy, seed, threads, validation tail, and frozen hyperparameters), `paths` (dataset, artifacts, outputs), `run` (log path and retention), `tuning` (selection provenance and search space), and `holdout` (reserved date count).

Structural constants remain literals in code, not config keys: early stopping rounds (30), HAC lag (4), HAR warm-up days (21), the target window (5), and the minimum trading days for a ticker (500).

## Verification

```bash
uv run --frozen python -m pytest packages/ml/model/tests -q --ignore=packages/ml/model/tests/live
uv run --frozen python -m pytest packages/ml/model/tests -q
python -m packages.ml.model.main --dry-run
```

The hermetic unit suite collects 270 tests and is the CI gate. The full suite collects 275: the five extra checks live in `packages/ml/model/tests/live/` and skip when generated data is absent on a fresh clone. The `--dry-run` logs the planned phases and exits without writing anything.

The determinism evidence is opt-in and cheap, and it is safe to run because it uses validation-only fold views:

```bash
python -m packages.ml.model.determinism   # writes packages/ml/model/artifacts/results/seed_sweep.json
```

Never run `--evaluate-holdout` or `--force-holdout` to verify documentation. The full contributor workflow is in [AGENTS.md](../AGENTS.md).
