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

## Baseline provenance

`packages/ml/model/artifacts/baseline_v1/` is retained with a provenance
`README.md` and hash `manifest.json`; removal requires an explicit owner
override, and the override is appended to `runs.jsonl`.
