# AGENTS.md

Working guide for AI coding agents in this repository. Read it before editing code.

## Environment

- Python >= 3.12, dependencies managed exclusively with `uv` (`.venv/` is uv-managed, gitignored).
- Kaggle credentials (`~/.kaggle/kaggle.json`) are required for Phase 1 only. Never print, copy, or commit their contents.
- There is no build system and no console entry points. `uv` manages dependencies and the pytest dev group; the model package has a pytest suite (see Verification workflow). There is no lint/typecheck configuration — do not reach for `ruff`, `mypy`, or `make`.

## Commands

```bash
uv sync                             # create or update .venv

python packages/ml/dataset/main.py  # Phase 1 — needs network + Kaggle creds
python packages/ml/model/main.py    # Phases 2-5 — local only
python -m packages.ml.model.main    # Phases 2-5 — same run, module form
```

Run both phases from the repository root. The dataset entry takes no arguments; the model entry takes `--dry-run` (log the planned phases and exit), `--resume` (continue the latest run manifest, skipping verified-complete folds), and `--folds` (comma-separated fold indices). With no arguments it runs the full pipeline unchanged. Both are cwd-independent (paths resolve from `__file__`), but running from the repo root is the documented path. `visualization/visualize.py` is not a documented entry point: run by path it cannot resolve its `packages.ml.model.*` imports. Run it as `python -m packages.ml.model.visualization.visualize` from the repo root instead.

## Hard rules

1. **Run these as scripts.** Both entries normalize their package identity (PEP 366) when run by path and work from any cwd; the model entry uses package-absolute imports and the dataset entry uses package-relative imports, and `import packages.ml.dataset` / `import packages.ml.model.main` both succeed from the repo root. Entry calls stay behind `if __name__ == "__main__"`.
2. **Phase 1 before Phases 2-5.** Phase 2 reads `data/processed/dataset.parquet` and fails immediately if it is missing. All generated data is gitignored, so a fresh clone has none.
3. **Never commit generated output.** `packages/ml/model/artifacts/`, `data/model_outputs/`, `data/processed/dataset.parquet`, and `data/processed/ingestion_manifest.json` are gitignored (as are `*.pkl` and `*.joblib` anywhere in the tree, `data/` included); no other `data/` path is ignored, so check before adding anything there. Markdown is no longer ignored — the old blanket `*.md` rule was removed, so `AGENTS.md` and any new `.md` commit normally.
4. **No absolute or machine-specific paths.** Everything resolves from `__file__`; `model/utils.py:get_project_paths` is the model path registry and `dataset/utils.py` holds the Phase 1 path constants. Keep it that way.
5. **No evidence, no "done".** Verification means running the model pytest suite for model-package changes and running the phase you touched; quote the output you saw.

## Invariants — breaking these corrupts artifacts or silently mislabels results

- **Fold count is config-sourced.** `config/base.yaml` owns `folds.n_folds`; `cross_validation/cross_validator.py` re-exports it as `N_FOLDS` and builds that many date-group folds. `model/training/training.py` compares the persisted `fold_*` count against it and fails by name on a mismatch, and `visualization/training_diagnostics.py` derives `FOLD_NAMES`, `FOLD_COLORS`, its loops, and its panel grid from it. Change the count in config, not as a literal.
- **Numeric training/CV constants come from config.** The model package's runtime constants are sourced from `packages/ml/model/config/base.yaml`; change them there, not as literals in code. The trainer's baseline recipe is the frozen selected configuration under `training.hyperparameters` (read at call time by `model_trainer.default_params()`); phase-4 runs record the selection provenance from `tuning.selected` in their run manifest.
- **Determinism is pinned and recorded.** `training.n_jobs` (effective, over the frozen selection's historical `-1`) and `training.omp_num_threads` pin the trainer and `OMP_NUM_THREADS` to one thread before every fit; `run_log.new_run` records the effective values, the baseline/trainer seeds, `PYTHONHASHSEED`, library versions, git sha, and `uv.lock` hash in the run manifest. `determinism.run_seed_sweep` (or `python -m packages.ml.model.determinism`) writes `results/seed_sweep.json` from validation-only fold views; `cross_validator.fold_manifest_hash` gives the fold-construction digest. Do not reintroduce multi-thread defaults silently.
- **Resume trusts only complete, verified fold records.** A fold is resumable only when `checkpoints/<run_id>/fold_<i>.pkl`, `fold_<i>_params.json`, `fold_<i>_summary.json`, and `fold_<i>_predictions.pkl` all exist and match their artifact-registry size/SHA-256 entries; the summary must also record the predictions file's size/hash and repeat its summary block. The identity block matches the manifest's `run_id`, `config_hash`, `fold_manifest_hash`, `input_identity` (dataset SHA-256 plus `X.npy`/`Y.npy` hashes), and coefficient setting. `--resume` verifies the persisted inputs and fold payload before Phase 2/3 can rewrite them, and refuses by name when any identity is missing, mismatched, or unreadable; never fall back to a fresh run silently. Each sidecar file is individually atomic; a stale summary/predictions pair is detected by the recorded hash and recomputed.
- **Retention prunes directories, never history.** After each run, `run.retention` (config) keeps the newest run-scoped `checkpoints/<run_id>/` directories and removes older ones; every kept directory carries a `README.md`. `artifacts/runs.jsonl` is append-only and is never truncated or pruned by retention. A retention failure is logged without masking the run's original outcome.
- **Subset runs are explicitly partial.** A full run keeps exactly five fold rows in `results/fold_training_summary.json`. A `--folds` run emits only the selected rows, sets `partial: true` in the run manifest, writes `results/run_selection.json`, and Phase 5 diagnostics render only the emitted folds.
- **`INDICATORS` order is positional.** `X = df[INDICATORS].values` fixes column order, and feature-importance arrays are mapped onto `INDICATORS` by index. Reorder the list and every importance chart silently mislabels. Renaming an entry means updating the dataset columns it reads — Phase 2 asserts every name exists in the parquet before training, and the module-level count assertion fails on list drift. The legacy alias columns were removed from the dataset, so no alias-pair bookkeeping remains.
- **`INDICATORS` holds 37 entries** — the pinned model-input list after dedup: the two legacy alias copies, both volume SMAs, `macd_signal`, `log_return` and `market_regime` left the list, and the volatility family joined it. `macd_signal`, `log_return` and `market_regime` stay in the dataset as columns (the first is re-anchored, the second feeds the target, the third is analysis metadata). A module-level count assertion fails fast on drift.
- **`market_regime` is a hardcoded calendar label**, not data-derived: seven calendar buckets cover 2000-2007 through 2026 and a date past 2026 raises. Extend `packages/ml/dataset/features/regimes.py` before ingesting data past 2026.
- **Folds are date-group blocks with a train-only purge.** `cross_validation/cross_validator.py` splits the sorted unique trading dates into six contiguous blocks; fold k tests on block k+1 and trains on blocks 0..k minus the training rows whose 5-day label window reaches into the test block. Test blocks are kept whole, so the last kept training date sits exactly five trading dates before the first kept test date. The separation predicate `first_test_idx - last_train_idx >= 6` (unique-date index space) is asserted per fold, so same-date, different-ticker overlap across the train/test boundary cannot occur. Each fold also holds out the temporally last 15% of its floor-retained training rows (`training.validation_tail_fraction`) as a purged inner-validation tail for early stopping; input rows are ticker-major, so the split orders by trading date and purges fit rows by the same 5-day label horizon.
- **Tuning is validation-only with a mechanical test lock.** `packages/ml/model/tuning/harness.py` builds every trial's fold data through one locked accessor (`build_locked_fold_views` → `locked_view`), which reuses the training path's purged inner-validation split and raises `TestLockError` on any view index inside a fold's test block or the reserved holdout window; trials train only through those views, and the selection artifact (`selected_on: validation_qlike`) rejects any `test_`-prefixed key. The campaign is opt-in and never runs in CI; tests drive tiny synthetic campaigns under temporary roots.
- **Time alignment is load-bearing.** Most price-derived features are shifted one day; the target uses `shift(-5)` and looks forward on purpose. The exceptions — model inputs in `feature_engineering.py:INDICATORS` whose numerator is the current bar — are `simple_return`, `price_vs_sma_{10,20,50}`, `zscore_price_vs_sma_20`, and `zscore_volume_20`. Do not assume a blanket one-day shift.
- **The training target is log(RV).** `training.py` floors the Yang-Zhang target at `1e-12` before the log and computes R²/MAE/RMSE on the log scale — `test_r2` follows sklearn's fold-own-mean convention, not the non-demeaned R²_oos of the literature, and the pipeline reports the dual OOS-R² conventions beside it per fold and pooled (`oos_r2_prevailing_mean` — fold-own test mean per fold, pooled test mean pooled — and `oos_r2_vs_har` = `1 - SSE_model/SSE_HAR`), each with its scale and floor-retained row set stated. The fold models minimize QLIKE on that log target, the same loss reported in evaluation; R²/MAE/RMSE are log-scale diagnostics. At-or-below-floor target rows (flat windows) are artifacts: one floor policy (`floor_retained_mask`, strict `>`) excludes them from fold fitting, model scoring, and the HAR benchmark, otherwise fitting their extreme labels under-forecasts similar test rows and the QLIKE exponential term explodes. Each fold persists the original test indices, the aligned `test_floor_mask`, the scored indices, the retained/excluded counts, the HAR per-row log forecasts, and the declared headline row set. The evaluation layer also persists one loss-vector row per test-block row (`results/per_row_losses.parquet`; `is_floor` is read from the persisted mask, never re-derived) and writes `results/evaluation_summary.json` from those rows only: dual OOS-R², DM/CW/GW (Newey-West HAC lag 4), Mincer-Zarnowitz regressions, decile/regime QLIKE, QLIKE mean/median/IQR/top-1% tails, the floor with/without decomposition, the date-clustered gap, expanding-origin curves, and the explicit no-calibration status; quarantined folds are excluded from every pooled aggregate. QLIKE (Patton 2011) is the robust volatility loss, and every fold also fits the canonical HAR-RV benchmark (Corsi (2009), 1/5/22 trailing variance components) on training rows only.
- **Phase 1 writes into the model package's tree.** `dataset/artifacts.py` writes `packages/ml/model/artifacts/results/`, so relocating model artifacts requires editing the dataset phase.

## Known dead or broken code

- `.idea/workspace.xml` still carries a `RunManager` entry for the deleted `run_pipeline.py`. It is untracked and stale.

## Conventions

- Use the `logging` module (`logger = logging.getLogger(__name__)`), not `print`. The handler format is identical in `dataset/utils.py` and `model/utils.py`.
- Match each file's local indentation. These files are tab-indented: `model/features/feature_engineering.py`, `model/training/model_trainer.py`, `model/training/training.py`. In `feature_engineering.py` only the `INDICATORS` list uses tabs — its function bodies use 4 spaces. Everything else uses 4 spaces.
- There are no classes; use module-level functions with type hints. `from __future__ import annotations` leads every module; every `model/` package `__init__.py` carries it, the two `dataset/` ones do not.
- Errors fail fast: `raise ValueError` / `FileNotFoundError` with a message. Log-and-re-raise appears in `model/main.py` and in `training/_train_fold_model`; one boundary quarantines instead — `training.py`'s per-fold metric computation records `{"error": ...}` for a failing fold (outside the five-row summary) and continues, filtering invalid metrics with `n_valid`/`n_total` and raising only when the headline QLIKE has none.

## Verification workflow

Run the model pytest suite for any change under `packages/ml/model`:

```bash
uv run --frozen python -m pytest packages/ml/model/tests -q
```

The suite drives the entry points with `--dry-run` and skips checks that need generated artifacts, the dataset parquet, or evaluation outputs when a checkout has none (a fresh clone or CI). The `static` job in `.github/workflows/checks.yml` runs the same command after the byte-compile step. To check behavior, also run the phase you touched — both if the change spans `packages/ml/dataset` and `packages/ml/model`:

1. `python packages/ml/dataset/main.py` — confirm it completes and logs the row and ticker counts it wrote, plus the "Dropped ... NaN rows" line and the parquet-then-JSON save order.
2. `python packages/ml/model/main.py` (or `python -m packages.ml.model.main`) — confirm 5 folds train and `artifacts/results/fold_training_summary.json` (per-fold R²/QLIKE) plus the HAR benchmark report are written. Add `--dry-run` to either form for a fast entry-point check: it logs the planned phases and exits before any phase runs. `--folds 0,2` trains only that selection; `--resume` continues the latest manifest and skips folds whose verified sidecar set is complete.
3. If commands or file layout changed, update `README.md` in the same change.

Cap long-running or chatty commands: `python packages/ml/model/main.py 2>&1 | tail -c 4000`.

## Where to look next

- `README.md` — what the project is, install, and run commands (human-facing).
- `packages/ml/dataset/features/` — read before adding or changing any feature; the target lives in `features/target.py` and the `market_regime` label in `features/regimes.py` (see also `regime_characterization.py`).
- `packages/ml/model/features/feature_engineering.py` — `INDICATORS`, the single source of truth for model inputs.
- `packages/ml/model/cross_validation/cross_validator.py` — fold construction and the leakage guard.
- `packages/ml/model/training/model_trainer.py` — the trainer; its baseline recipe derives from the frozen `training.hyperparameters` config section.
- `packages/ml/model/utils.py` — `get_project_paths`, the central path helper (`project_root`, `model_dir`, `artifacts_dir`, `dataset_path`, `outputs_dir`).
- [`docs/adr/`](docs/adr/) — architecture decision records own the rationale behind these rules; supersede, don't rewrite.
