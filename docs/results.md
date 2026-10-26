# Results

This is a reader's guide to the generated result artifacts. It is not a second report. For formulas, tables, uncertainty, and falsification tests, read [reports/model-performance-and-rigor.md](reports/model-performance-and-rigor.md).

## Two kinds of results

- **Live artifacts** under `packages/ml/model/artifacts/results/` describe the latest run and change with each retrain and each refresh of the source data. Read them when you want the current numbers.
- **The dated report** records one evaluation (2026-10-20) and never changes. Read it when you want the full analysis and the standing caveats.

When in doubt, prefer the live artifacts for current state and the report for interpretation.

## Artifact map

| Artifact | Contents |
|---|---|
| `evaluation_summary.json` | CV headline QLIKE and edge, DM/CW/GW, MZ, deciles, regimes, floor decomposition |
| `fold_training_summary.json` | Per-fold metrics, row counts, and best iteration |
| `har_benchmark.json` | The HAR-RV benchmark fit and per-fold comparison |
| `validation_predictions.pkl` | Per-fold validation predictions consumed by Phase 5 diagnostics |
| `run_selection.json` | Which folds this run emitted; `partial: true` means missing folds/summaries/charts are by selection, not data loss |
| `per_row_losses.parquet` | One loss-vector row per test-block row, with `is_floor` |
| `holdout_report.json` | The locked holdout metrics and protocol |
| `holdout_lock.json` | The lock binding the evaluation identity |
| `run_manifest.json` | Run identity, config hash, and artifact hashes |

The `is_floor` flag in `per_row_losses.parquet` is read from the persisted mask, never re-derived from the target.

## Conventions

- QLIKE is lower-is-better and is computed on per-day sigma, not annualized.
- The headline edge is `HAR − model`: positive means the model is better.
- The DM differential is `model − HAR`: negative means the model is better.
- R²/MAE/RMSE are computed on the log target and are diagnostics, not the headline loss.
- **Equal-weight** gives each fold one vote; **pooled** gives each row one vote.
- All scoring uses floor-retained rows; at-or-below-floor rows are excluded.

The sign flip between the edge and the DM differential is deliberate and easy to misread: a positive edge and a negative DM mean both say the model is better.

## Reading the headline

`evaluation_summary.json` has two headline forms. `headline.aggregate` is fold-equal: the mean of the per-fold QLIKEs (five with the default config). `headline.pooled` is row-weighted: one QLIKE over all scored test rows. The two agree closely but need not be identical, because later folds are larger and QLIKE varies by fold.

The dual out-of-sample R² is `1 − SSE_model/SSE_mean` and `1 − SSE_model/SSE_HAR`. `test_r2` keeps sklearn's fold-own-mean convention, so it can disagree with the prevailing-mean form. The HAR benchmark is scored on the same floor-retained test rows as the model, so the edge is a like-for-like comparison rather than a different test set.

## Where the evaluated headline lives

This guide deliberately does not copy the 2026-10-20 numbers. The canonical narrative table and its caveats are in [the report](reports/model-performance-and-rigor.md); the live values are in the artifacts named above. Keeping one copy avoids the failure mode where a README summary drifts from the evaluated report after a retrain.

The report is historical: it describes the evaluation of model commit `df5aa5a` and does not change with later runs. The live artifacts describe the latest run. Read the report for interpretation and the artifacts for current state.

## Safe inspection

Reading artifacts does not retrain anything and does not touch the holdout. Run these from the repository root:

```python
import json, pandas as pd

eval = json.load(open("packages/ml/model/artifacts/results/evaluation_summary.json"))
print(eval["headline"]["aggregate"]["model_qlike"])

folds = json.load(open("packages/ml/model/artifacts/results/fold_training_summary.json"))

har = json.load(open("packages/ml/model/artifacts/results/har_benchmark.json"))

losses = pd.read_parquet("packages/ml/model/artifacts/results/per_row_losses.parquet")
print(losses["is_floor"].sum())
```

Reading `holdout_report.json` and `holdout_lock.json` is safe; they are already written. Do not run `--evaluate-holdout` or `--force-holdout` just to inspect the numbers.

## Reproduction and audit

Safe checks:

```bash
uv run --frozen python -m pytest packages/ml/model/tests --collect-only -q -p no:cacheprovider
python -m packages.ml.model.main --dry-run
python -c "import json; print(json.load(open('packages/ml/model/artifacts/results/holdout_lock.json'))['sha256'])"
python -c "import json; print(json.load(open('packages/ml/model/artifacts/run_manifest.json'))['config_hash'])"
```

A full reproduction needs Phase 1 followed by Phases 2-5. The holdout is not an audit tool: it is one-shot and locked, so it cannot be re-run to check a claim. See [operations.md](operations.md).

## Claims ladder

The report's conclusion maps each claim to a status. In brief:

| Claim | Status |
|---|---|
| Average QLIKE edge over HAR in CV | Supported |
| Average QLIKE edge in the locked holdout | Supported, conditional on the window |
| Calibration | Rejected by MZ; no calibration layer applied |
| Tail safety | Not established |
| Seed robustness | Not established |
| Crisis robustness | Not established |
| Economic value | Not established |
| Superiority to non-HAR models | Not established |

For the evidence behind each row and the tests that would weaken or resolve it, read the [report](reports/model-performance-and-rigor.md).
