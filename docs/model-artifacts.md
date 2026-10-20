# Model artifacts

## Run directories and retention

`packages/ml/model/artifacts/checkpoints/<run_id>/` holds one run-scoped
directory per run: `fold_<i>.pkl` checkpoints, `fold_<i>_params.json`
parameter sidecars, `fold_<i>_summary.json` metric/identity records, and
`fold_<i>_predictions.pkl` resume records. Each file is written individually
through a sibling `*.tmp` + `os.replace`, so a torn write leaves the previous
file intact and no temporary residue. The summary also records the predictions
file's size and SHA-256; a summary left beside a newer predictions file is
rejected and the fold is recomputed.

After each run, retention keeps the last N run directories (N is
`run.retention` in `packages/ml/model/config/base.yaml`) and removes older
ones.
Each kept directory carries a per-directory `README.md` naming the run id, its
config/fold-manifest/input identity, the fold selection, the artifact contents,
and the retention rule. A run directory is resumable only when all four
per-fold files exist and match their artifact-registry size/SHA-256 entries,
the summary matches the predictions file, the manifest's input identity
(dataset SHA-256 + `X.npy`/`Y.npy` hashes) matches the persisted data, and
`--resume` also requires the manifest's `config_hash` and
`fold_manifest_hash` to match. Inputs and folds are verified before Phase 2/3
can rewrite them.

A full run's `results/fold_training_summary.json` keeps exactly five fold
rows. A `--folds` run is explicitly partial: it emits only the selected rows,
sets `partial: true` in `run_manifest.json`, and writes
`results/run_selection.json` with the requested/trained/resumed folds. Phase 5
diagnostics render the emitted folds and record normal per-plot statuses.

`packages/ml/model/artifacts/runs.jsonl` is append-only and kept forever:
retention prunes directories, never log lines. `run_manifest.json` records the
latest run (identity, config hash, fold-manifest hash, fold selection,
phase/fold status, durations, artifact hashes) and is written atomically.

## Locked holdout

The holdout evaluation is opt-in (`--evaluate-holdout`) and one-shot. The
first evaluation writes `results/holdout_lock.json` and records its SHA-256,
bounds, and full lock payload in the run manifest before the final refit or any
holdout metric. The lock hashes the frozen selection recipe, the Phase-2
dataset pin and derived `X`/`Y` hashes, the fold-manifest hash, and the reserved
bounds. The manifest record is binding: every new run manifest carries the
latest holdout block forward, and the append-only `runs.jsonl` is the fallback
when a manifest has lost it. A lost results lock is restored from either source
and remains a refusal/skip, never a silent re-evaluation. A second evaluation
raises `HoldoutLockError`; a post-lock resume records a skipped holdout instead
of re-scoring; `--force-holdout <reason>` requires a recorded lock, appends an
override record to `results/holdout_overrides.jsonl`, and still refuses a
changed lock identity. The evaluation writes `results/holdout_model.pkl`,
`results/holdout_predictions.pkl`, `results/holdout_report.json` (protocol,
holdout metrics, config hash, git sha, environment reference, artifact
hashes), and `results/SHA256SUMS` over the report and its sibling holdout
artifacts.

## Baseline provenance

`packages/ml/model/artifacts/baseline_v1/` is retained with a provenance
`README.md` and hash `manifest.json`; removal requires an explicit owner
override, and the override is appended to `runs.jsonl`.

## Stale directories

`packages/ml/model/artifacts/legacy/` is created empty by the entry point and
Phase 2. No pipeline phase writes files to or reads files from it; it is
retained in place (no deletion path) and is not an output root. The model
package README (`packages/ml/model/README.md`) documents both this directory
and the retained baseline snapshot above.
