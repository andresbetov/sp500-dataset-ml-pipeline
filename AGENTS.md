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

1. **Run these as scripts; never import them.** `packages/ml/dataset/__init__.py` imports `main.py`, and `main.py` calls `generate_featured_dataset()` at module top level, so importing the dataset package executes the whole phase. The in-package imports are also flat (`from loader import ...`), so the import only gets that far when the cwd happens to be `packages/ml/dataset`; from anywhere else it dies first with `ModuleNotFoundError: No module named 'loader'`. Either way, run it as a script.
2. **Phase 1 before Phases 2-5.** Phase 2 reads `data/processed/dataset.parquet` and fails immediately if it is missing. All generated data is gitignored, so a fresh clone has none.
3. **Never commit generated output.** `packages/ml/model/artifacts/`, `data/model_outputs/`, and `data/processed/dataset.parquet` are gitignored (as are `*.pkl` and `*.joblib` anywhere in the tree, `data/` included); no other `data/` path is ignored, so check before adding anything there. Markdown is no longer ignored — the old blanket `*.md` rule was removed, so `AGENTS.md` and any new `.md` commit normally.
4. **No absolute or machine-specific paths.** Everything resolves from `__file__`; `model/utils.py:get_project_paths` is the path registry. Keep it that way.
5. **No evidence, no "done".** With no test suite, verification means running the phase you touched and quoting the log output you saw.

## Invariants — breaking these corrupts artifacts or silently mislabels results

- **Exactly 5 folds.** `model/training/training.py` raises `ValueError` for any other count, and `visualization/training_diagnostics.py` bakes in five via `FOLD_NAMES = [f"fold_{i}" for i in range(5)]` and bare `range(5)` loops.
- **`INDICATORS` order is positional.** `X = df[INDICATORS].values` fixes column order, and feature-importance arrays are mapped onto `INDICATORS` by index. Reorder the list and every importance chart silently mislabels. Changing a name means updating both spellings of an alias pair.
- **`INDICATORS` holds 29 entries = 26 distinct indicators + 2 legacy alias columns + `market_regime`.** `volume_zscore` duplicates `zscore_volume_20`, and `zscore_price_20` duplicates `zscore_price_vs_sma_20` (`dataset/features.py:LEGACY_FEATURE_ALIASES`). The model sees both copies.
- **`market_regime` is a hardcoded calendar label**, not data-derived: buckets end at `year >= 2022`. Extend `packages/ml/dataset/features.py` before ingesting data past 2026.
- **`TimeSeriesSplit(gap=CV_GAP_DAYS)` counts rows, not days.** The panel holds hundreds of tickers per date, so the embargo never separates dates, and the leakage guard only compares date extrema. Do not "fix" this without reading `cross_validation/cross_validator.py` first — same-date, different-ticker overlap is deliberate.
- **Time alignment is load-bearing.** Most price-derived features are shifted one day; the target uses `shift(-5)` and looks forward on purpose. The exceptions — model inputs in `feature_engineering.py:INDICATORS` whose numerator is the current bar — are `simple_return`, `log_return`, `price_vs_sma_{10,20,50}`, `zscore_price_vs_sma_20` (aliased as `zscore_price_20`), and `zscore_volume_20`. Do not assume a blanket one-day shift.
- **Phase 1 writes into the model package's tree.** `dataset/main.py` writes `packages/ml/model/artifacts/results/`, so relocating model artifacts requires editing the dataset phase.

## Known dead or broken code

- `model/training/inference.py` is never imported; the live path inlines `model.predict()` in `training.py`. Editing it changes nothing.
- The early-stopping branch in `model_trainer.py` is unreachable: `training.py` passes `X_val=None, y_val=None`.
- `cross_validation/cv_utils.py:load_fold_data` is never called.
- `regime_characterization.json` reports `ticker_correlation_mean`/`_std` as `0.0` in every regime. A `Series.corr()` call with no argument raises and is swallowed by a catch-all `except Exception: pass` in `dataset/features.py`. Treat those fields as always zero; they are not signal.
- `metadata.pkl` carries no `regime` key, so per-fold regime distributions are always empty.
- `.idea/workspace.xml` still carries a `RunManager` entry for the deleted `run_pipeline.py`. It is untracked and stale.

## Conventions

- Use the `logging` module (`logger = logging.getLogger(__name__)`), not `print`. The handler format is identical in `dataset/main.py` and `model/utils.py`.
- Match each file's local indentation. These four files are tab-indented: `dataset/features.py`, `model/features/feature_engineering.py`, `model/training/model_trainer.py`, `model/training/training.py`. In `feature_engineering.py` only the `INDICATORS` list uses tabs — its function bodies use 4 spaces. Everything else uses 4 spaces.
- There are no classes; use module-level functions with type hints. `from __future__ import annotations` leads every module; of the six `__init__.py` files only `model/visualization/__init__.py` carries it.
- Errors fail fast: `raise ValueError` / `FileNotFoundError` with a message. Log-and-re-raise appears in `model/main.py` and in `training/_train_fold_model`; two blocks swallow instead — the catch-all `except Exception: pass` in `dataset/features.py`, and `training.py`'s metric computation, which writes NaN metrics.

## Verification workflow

There are no tests. To check a change, run the phase you touched — both if the change spans `packages/ml/dataset` and `packages/ml/model`:

1. `python packages/ml/dataset/main.py` — confirm it completes and logs the row and ticker counts it wrote, plus the final "Dropped ... NaN rows" line.
2. `python packages/ml/model/main.py` — confirm 5 folds train and `artifacts/results/fold_training_summary.json` is written.
3. If commands or file layout changed, update `README.md` in the same change.

Cap long-running or chatty commands: `python packages/ml/model/main.py 2>&1 | tail -c 4000`.

## Where to look next

- `README.md` — what the project is, install, and run commands (human-facing).
- `packages/ml/dataset/features.py` — read before adding or changing any feature; defines the target, `market_regime`, and the alias map.
- `packages/ml/model/features/feature_engineering.py` — `INDICATORS`, the single source of truth for model inputs.
- `packages/ml/model/cross_validation/cross_validator.py` — fold construction and the leakage guard.
- `packages/ml/model/training/model_trainer.py` — XGBoost hyperparameters.
- `packages/ml/model/utils.py` — `get_project_paths`, the central path helper (`project_root`, `model_dir`, `artifacts_dir`, `dataset_path`, `outputs_dir`).
