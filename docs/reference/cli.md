# CLI reference

This is the canonical command reference. The run lifecycle behind these commands is in [operations.md](../operations.md); the phase protocol is in [model.md](../pipelines/model.md). Run every command from the repository root.

## Environment

`uv sync` creates `.venv`. Activate it with `source .venv/bin/activate`, or prefix each command with `uv run`. The locked form used by CI is `uv run --frozen python …`.

## Entry points

| Entry point | Command | Needs |
|---|---|---|
| Phase 1 — dataset | `python packages/ml/dataset/main.py` | Network and Kaggle credentials (`~/.kaggle/kaggle.json`) |
| Phases 2-5 — model, module form | `python -m packages.ml.model.main` | `data/processed/dataset.parquet` |
| Phases 2-5 — model, by-path form | `python packages/ml/model/main.py` | Same as above |

The dataset entry takes no arguments. The model entry accepts the five flags below; with no flags it runs Phases 2-5 unchanged and never reads the holdout.

## Model flags

| Flag | Effect | Notes |
|---|---|---|
| `--dry-run` | Logs the planned phases and exits without writing. | Safe entry-point check; no artifact writes. |
| `--resume` | Continues the latest run manifest and reuses verified folds. | Requires matching `run_id`, `config_hash`, `fold_manifest_hash`, and `input_identity`; mismatches refuse by name. |
| `--folds 0,2` | Trains only the named fold indices. | The run is explicitly partial: `partial: true` and a selection record. |
| `--evaluate-holdout` | After Phases 2-5, evaluates the reserved holdout once under the written lock. | One-shot; the lock is recorded before any holdout metric. |
| `--force-holdout "reason"` | Re-evaluates an already locked holdout and appends the reason to the override record. | Implies `--evaluate-holdout`; the reason must be non-empty; a changed identity is still refused. |

The full holdout protocol, including recovery and refusal behavior, is in [operations.md](../operations.md#locked-holdout). Do not run the holdout as a verification step: on a fresh checkout, the first evaluation creates the lock and scores the holdout.

## Examples

```bash
# Full local run: Phase 1, then Phases 2-5
python packages/ml/dataset/main.py
python -m packages.ml.model.main

# Safe planning check
python -m packages.ml.model.main --dry-run

# Train only folds 0 and 2
python -m packages.ml.model.main --folds 0,2

# Continue the latest run, reusing complete verified folds
python -m packages.ml.model.main --resume

# Deliberate gate action only, after the normal pipeline is healthy
python -m packages.ml.model.main --evaluate-holdout
```

## Tests

The hermetic unit suite is the CI gate; the full suite adds the live artifact checks:

```bash
# Hermetic: 270 tests, no generated artifacts required
uv run --frozen python -m pytest packages/ml/model/tests -q --ignore=packages/ml/model/tests/live

# Full: 275 tests; the five live checks skip on a fresh clone
uv run --frozen python -m pytest packages/ml/model/tests -q
```

The live checks live under `packages/ml/model/tests/live/`; they read generated artifacts and skip when those are absent. See [AGENTS.md](../../AGENTS.md) for the contributor workflow. The technical report records `421` collected tests at the evaluated tree; after test-only pruning (PR #65), the current tree collects `270` hermetic and `275` full tests. The evaluated metrics are unchanged.

## Auxiliary entry points

```bash
# Validation-only seed sweep -> packages/ml/model/artifacts/results/seed_sweep.json
python -m packages.ml.model.determinism

# Re-render Phase 5 charts into data/model_outputs/
python -m packages.ml.model.visualization.visualize
```

The determinism sweep never touches a test block or the holdout window; its policy and output are described in [operations.md](../operations.md#verification). The visualization entry point is presentation-only and reads the artifacts; the phase protocol is in [model.md](../pipelines/model.md).

## Outputs

- Phase 1 writes `data/processed/dataset.parquet` and `data/processed/ingestion_manifest.json`; it also writes `packages/ml/model/artifacts/results/regime_characterization.json`.
- Phases 2-5 write `packages/ml/model/artifacts/` (state and results) and `data/model_outputs/` (charts).
- The optional holdout additionally writes `holdout_lock.json`, `holdout_report.json`, `holdout_model.pkl`, `holdout_predictions.pkl`, `SHA256SUMS`, and, after a force, `holdout_overrides.jsonl`.

See [operations.md](../operations.md#artifacts-map) for the full artifact map and [results.md](../results.md) for how to read the result files.
