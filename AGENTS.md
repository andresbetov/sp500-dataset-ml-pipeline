# AGENTS.md

Working guide for AI coding agents in this repository. Read it before editing code.

## Environment

- Python >= 3.12, dependencies managed exclusively with `uv` (`.venv/` is uv-managed, gitignored).
- Kaggle credentials (`~/.kaggle/kaggle.json`) are required for Phase 1 only. Never print, copy, or commit their contents.
- There is no build system, no console entry points, and no test/lint/typecheck configuration. Do not reach for `pytest`, `ruff`, `mypy`, or `make` — none exist, and inventing a fake command wastes the session.

## Commands

```bash
uv sync                             # create or update .venv

python packages/ml/dataset/main.py  # Phase 1 — needs network + Kaggle creds
python packages/ml/model/main.py    # Phases 2-5 — local only
```

Both take no arguments and read no environment variables. They are cwd-independent (paths resolve from `__file__`), but running from the repo root is the documented path. `visualization/visualize.py` is not a documented entry point: run by path it cannot resolve its `visualization.*` and `utils` imports. It only works with `packages/ml/model` on `sys.path` — which is what `model/main.py` provides, or `python -m visualization.visualize` from that directory.

## Hard rules

1. **Run these as scripts.** `dataset/main.py` normalizes its package identity (PEP 366) so package-relative imports work from any cwd, and `import packages.ml.dataset` succeeds when run from the repo root; `model/main.py` keeps flat sibling imports and imports only from inside its directory. Entry calls stay behind `if __name__ == "__main__"`.
2. **Phase 1 before Phases 2-5.** Phase 2 reads `data/processed/dataset.parquet` and fails immediately if it is missing. All generated data is gitignored, so a fresh clone has none.
3. **Never commit generated output.** `packages/ml/model/artifacts/`, `data/model_outputs/`, `data/processed/dataset.parquet`, and `data/processed/ingestion_manifest.json` are gitignored (as are `*.pkl` and `*.joblib` anywhere in the tree, `data/` included); no other `data/` path is ignored, so check before adding anything there. Markdown is no longer ignored — the old blanket `*.md` rule was removed, so `AGENTS.md` and any new `.md` commit normally.
4. **No absolute or machine-specific paths.** Everything resolves from `__file__`; `model/utils.py:get_project_paths` is the model path registry and `dataset/utils.py` holds the Phase 1 path constants. Keep it that way.
5. **No evidence, no "done".** With no test suite, verification means running the phase you touched and quoting the log output you saw.

## Invariants — breaking these corrupts artifacts or silently mislabels results

- **Exactly 5 folds.** `model/training/training.py` raises `ValueError` for any other count, and `visualization/training_diagnostics.py` bakes in five via `FOLD_NAMES = [f"fold_{i}" for i in range(5)]` and bare `range(5)` loops.
- **`INDICATORS` order is positional.** `X = df[INDICATORS].values` fixes column order, and feature-importance arrays are mapped onto `INDICATORS` by index. Reorder the list and every importance chart silently mislabels. Renaming an entry means updating the dataset columns it reads — Phase 2 asserts every name exists in the parquet before training, and the module-level count assertion fails on list drift. The legacy alias columns were removed from the dataset, so no alias-pair bookkeeping remains.
- **`INDICATORS` holds 37 entries** — the pinned model-input list after dedup: the two legacy alias copies, both volume SMAs, `macd_signal`, `log_return` and `market_regime` left the list, and the volatility family joined it. `macd_signal`, `log_return` and `market_regime` stay in the dataset as columns (the first is re-anchored, the second feeds the target, the third is analysis metadata). A module-level count assertion fails fast on drift.
- **`market_regime` is a hardcoded calendar label**, not data-derived: seven calendar buckets cover 2000-2007 through 2026 and a date past 2026 raises. Extend `packages/ml/dataset/features/regimes.py` before ingesting data past 2026.
- **Folds are date-group blocks with a two-sided purge.** `cross_validation/cross_validator.py` splits the sorted unique trading dates into six contiguous blocks; fold k tests on block k+1 and trains on blocks 0..k minus the two-sided purge of the 5-day label overlap. The embargo predicate `first_test_idx - last_train_idx >= 6` (unique-date index space) is asserted per fold, so same-date, different-ticker overlap across the train/test boundary cannot occur.
- **Time alignment is load-bearing.** Most price-derived features are shifted one day; the target uses `shift(-5)` and looks forward on purpose. The exceptions — model inputs in `feature_engineering.py:INDICATORS` whose numerator is the current bar — are `simple_return`, `price_vs_sma_{10,20,50}`, `zscore_price_vs_sma_20`, and `zscore_volume_20`. Do not assume a blanket one-day shift.
- **The training target is log(RV).** `training.py` floors the Yang-Zhang target at `1e-12` before the log and computes R²/MAE/RMSE on the log scale — R² follows sklearn's fold-own-mean convention, not the non-demeaned R²_oos of the literature. Exact-zero target rows (flat windows) are degenerate labels: they stay in the test sets but are excluded from the fold training sets, otherwise a log-MSE model fits their extreme labels and QLIKE's exponential term explodes on similar test rows. QLIKE (Patton 2011) is the robust volatility loss, and every fold also fits the canonical HAR-RV benchmark (Corsi 2008, 1/5/22 trailing variance components) on training rows only.
- **Phase 1 writes into the model package's tree.** `dataset/artifacts.py` writes `packages/ml/model/artifacts/results/`, so relocating model artifacts requires editing the dataset phase.

## Known dead or broken code

- `model/training/inference.py` is never imported; the live path inlines `model.predict()` in `training.py`. Editing it changes nothing.
- The early-stopping branch in `model_trainer.py` is unreachable: `training.py` passes `X_val=None, y_val=None`.
- `cross_validation/cv_utils.py:load_fold_data` is never called.
- `metadata.pkl` carries no `regime` key, so per-fold regime distributions are always empty.
- `.idea/workspace.xml` still carries a `RunManager` entry for the deleted `run_pipeline.py`. It is untracked and stale.

## Conventions

- Use the `logging` module (`logger = logging.getLogger(__name__)`), not `print`. The handler format is identical in `dataset/utils.py` and `model/utils.py`.
- Match each file's local indentation. These files are tab-indented: `model/features/feature_engineering.py`, `model/training/model_trainer.py`, `model/training/training.py`. In `feature_engineering.py` only the `INDICATORS` list uses tabs — its function bodies use 4 spaces. Everything else uses 4 spaces.
- There are no classes; use module-level functions with type hints. `from __future__ import annotations` leads every module; of the seven `__init__.py` files only `model/visualization/__init__.py` carries it.
- Errors fail fast: `raise ValueError` / `FileNotFoundError` with a message. Log-and-re-raise appears in `model/main.py` and in `training/_train_fold_model`; one block swallows instead — `training.py`'s metric computation, which writes NaN metrics.

## Verification workflow

There are no tests. To check a change, run the phase you touched — both if the change spans `packages/ml/dataset` and `packages/ml/model`:

1. `python packages/ml/dataset/main.py` — confirm it completes and logs the row and ticker counts it wrote, plus the "Dropped ... NaN rows" line and the parquet-then-JSON save order.
2. `python packages/ml/model/main.py` — confirm 5 folds train and `artifacts/results/fold_training_summary.json` (per-fold R²/QLIKE) plus the HAR benchmark report are written.
3. If commands or file layout changed, update `README.md` in the same change.

Cap long-running or chatty commands: `python packages/ml/model/main.py 2>&1 | tail -c 4000`.

## Where to look next

- `README.md` — what the project is, install, and run commands (human-facing).
- `packages/ml/dataset/features/` — read before adding or changing any feature; the target lives in `features/target.py` and the `market_regime` label in `features/regimes.py` (see also `regime_characterization.py`).
- `packages/ml/model/features/feature_engineering.py` — `INDICATORS`, the single source of truth for model inputs.
- `packages/ml/model/cross_validation/cross_validator.py` — fold construction and the leakage guard.
- `packages/ml/model/training/model_trainer.py` — XGBoost hyperparameters.
- `packages/ml/model/utils.py` — `get_project_paths`, the central path helper (`project_root`, `model_dir`, `artifacts_dir`, `dataset_path`, `outputs_dir`).
