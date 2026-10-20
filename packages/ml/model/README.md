# Model package

Phases 2-5 of the S&P 500 volatility pipeline: feature selection, temporal
cross-validation, fold training, and visualization. Phase 1 (code under
`packages/ml/dataset/`) builds `data/processed/dataset.parquet` and writes
`packages/ml/model/artifacts/results/regime_characterization.json`; it must
run before this package.

## Output roots

The package writes to two gitignored roots with different purposes:

| Root | Written by | Contents |
|---|---|---|
| `packages/ml/model/artifacts/` | Phase 1; Phases 2-4; model entry point | Source-of-truth pipeline state and results: `results/regime_characterization.json` (Phase 1); `inputs/`, `folds/`, `checkpoints/<run_id>/`, `results/`, and opt-in `tuning/<run_id>/`; plus `run_manifest.json`, `runs.jsonl`, and `artifact_registry.json`; on a real run the entry point writes run state before the phases and finalizes it after Phase 5 |
| `data/model_outputs/` | Phase 5 only | Presentation charts: the PNG figures and the plot `manifest.json` |

`artifacts/` is the pipeline's working tree and the source of truth for run
state and results: it holds the verified inputs, fold records, checkpoints, and
the per-run metric JSON and pickle payloads that the repository README points
to for the numbers. `data/model_outputs/` is the presentation layer, regenerated
from those artifacts on every visualization pass and never feeding a training
or validation step. See the tracked
[`docs/model-artifacts.md`](../../../docs/model-artifacts.md) for the retention
rules, resume identity, and evaluation-artifact contract.

## Pipeline phases

1. **Phase 2 — feature selection.** Reads the Phase 1 parquet and writes
   `artifacts/inputs/X.npy`, `Y.npy`, and `metadata.pkl` through the atomic
   writer, recording the metadata hash in the artifact registry.
2. **Phase 3 — temporal cross-validation.** Reserves the configured holdout
   window, builds the date-group folds with a train-only purge, and writes
   `artifacts/folds/folds_metadata.pkl`.
3. **Phase 4 — training.** Trains one QLIKE model per fold into
   `artifacts/checkpoints/<run_id>/` with parameter, summary, and predictions
   sidecars; writes the fold summary, HAR benchmark, loss vectors, and
   evaluation summary under `artifacts/results/`, and updates the run manifest
   at the artifacts root.
4. **Phase 5 — visualization.** Renders the figures and plot manifest into
   `data/model_outputs/`; the entry point then finalizes the run manifest,
   appends `runs.jsonl`, and applies checkpoint retention.

## Running and resuming

Run from the repository root (both forms are equivalent):

```bash
python -m packages.ml.model.main
python packages/ml/model/main.py
```

The entry point accepts:

```bash
# Log the planned phases and exit without writing anything
python -m packages.ml.model.main --dry-run

# Train a subset of the configured folds; the run is explicitly partial
python -m packages.ml.model.main --folds 0,2

# Continue the latest run manifest, reusing complete verified fold records
python -m packages.ml.model.main --resume
```

`--resume` requires the manifest's run id, config hash, fold-manifest hash, and
input identity to match the persisted data. A fold is reused only when its
checkpoint, parameter sidecar, summary, and predictions record all exist and
match their artifact-registry size/SHA-256 entries. A mismatch refuses by name
instead of falling back to a fresh run. Retention keeps the newest
`run.retention` run directories (configured in `config/base.yaml`);
`runs.jsonl` is append-only and is never pruned. The full command and flag
contract is in the repository [`README.md`](../../../README.md) and
[`AGENTS.md`](../../../AGENTS.md).

## Legacy and stale directories

Two directories in the artifacts root are retained on purpose; neither is an
output root, and no pipeline phase reads or writes files in either:

- `artifacts/legacy/` is empty and unused. The entry point and Phase 2 create
  it, but no file is written to it and no code reads it; it is retained in
  place and no deletion path exists.
- `artifacts/baseline_v1/` is the pre-remediation baseline snapshot
  (`2,668,192 × 39` legacy versus `2,669,959 × 53` current). It is kept for
  provenance with its own README and hash manifest, not used by any pipeline
  phase. Removal requires an explicit owner override appended to
  `artifacts/runs.jsonl`; the full disposition is recorded in
  [`docs/model-artifacts.md`](../../../docs/model-artifacts.md).

Untracked reproduction scripts may read the baseline snapshot deliberately;
they are not part of the pipeline.
