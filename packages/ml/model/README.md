# Model package

Phases 2-5 of the S&P 500 volatility pipeline: feature selection, temporal cross-validation, fold training and evaluation, and visualization. Phase 1 (under `packages/ml/dataset/`) must run first; it writes `data/processed/dataset.parquet` and `packages/ml/model/artifacts/results/regime_characterization.json`.

## Output roots

| Root | Contents |
|---|---|
| `packages/ml/model/artifacts/` | Pipeline state and results — the source of truth: inputs, folds, per-run checkpoints, results JSON/pickle payloads, and run state files. |
| `data/model_outputs/` | Presentation charts written by Phase 5 — rendered from the artifacts, never an input to training or validation. |

Both roots are gitignored.

## Phases

1. **Phase 2 — feature selection.** Reads the parquet, selects the 37 positional `INDICATORS`, validates finite X/Y, writes `packages/ml/model/artifacts/inputs/X.npy`, `Y.npy`, `metadata.pkl`.
2. **Phase 3 — temporal cross-validation.** Reserves the holdout, builds date-group folds with a train-only purge, writes `packages/ml/model/artifacts/folds/folds_metadata.pkl`.
3. **Phase 4 — training and evaluation.** Floors the target, trains one QLIKE model per fold, fits the HAR-RV benchmark on training rows, writes fold summaries, predictions, loss vectors, and the evaluation summary.
4. **Phase 5 — visualization.** Renders the figures and plot manifest into `data/model_outputs/`.

## Running

Run from the repository root; both forms are equivalent:

```bash
python -m packages.ml.model.main
python packages/ml/model/main.py
```

To check the entry point without writing anything:

```bash
python -m packages.ml.model.main --dry-run
```

The remaining flags (`--resume`, `--folds`, `--evaluate-holdout`, `--force-holdout`), the run lifecycle, resume, retention, and the holdout gate are in [`docs/operations.md`](../../../docs/operations.md). The canonical flag reference is [`docs/reference/cli.md`](../../../docs/reference/cli.md); the fold geometry, training, and evaluation protocol are in [`docs/pipelines/model.md`](../../../docs/pipelines/model.md). The dataset columns are documented in [`docs/reference/dataset-schema.md`](../../../docs/reference/dataset-schema.md), and the model card is [`docs/model-card.md`](../../../docs/model-card.md).

## Legacy and stale directories

- `packages/ml/model/artifacts/legacy/` is empty and unused; no phase reads or writes it.
- `packages/ml/model/artifacts/baseline_v1/` is a pre-remediation snapshot kept for provenance only; no phase reads it.
