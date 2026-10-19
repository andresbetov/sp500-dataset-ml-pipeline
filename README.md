# S&P 500 Volatility Prediction Pipeline

Predicts 5-day realized volatility for S&P 500 constituents from price and volume history alone.

This is a research pipeline: one script builds the dataset, one trains and evaluates the model. Run metrics are written to disk rather than quoted here — they change with every retrain and with every refresh of the source data.

## What it does

Given a multi-ticker daily OHLCV panel, the pipeline:

1. Downloads and cleans the source data from Kaggle.
2. Engineers per-ticker technical features plus a forward-looking volatility target.
3. Trains one gradient-boosted regression model per temporal fold, evaluating each only on strictly later data.
4. Writes an evaluation report and charts for that run.

The prediction target is `realized_volatility_5d` — the Yang–Zhang daily volatility estimate over the next five trading days (per-day σ, not annualized). This is continuous regression, not classification.

Features come from price and volume only. There is deliberately no per-ticker identity encoding: a ticker's behaviour is already carried by its own indicator magnitudes (larger returns, more extreme momentum, different volume patterns), so per-ticker columns would add sparse inputs without adding generalizable signal.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- Kaggle API credentials — needed only for Phase 1

`uv.lock` pins the full dependency tree. `pyarrow` additionally carries an exact pin in `pyproject.toml` (`==24.0.0`) because it writes the dataset artifact; the reference environment is CPython 3.14.7 with pandas 3.0.2 and numpy 2.4.4.

Byte-identical regeneration is a local property: on the machine that wrote `dataset.parquet`, a fresh Phase 1 run reproduces the file byte for byte, but another platform can produce different bytes for identical content. Across machines, compare content instead of bytes — for example the SHA-256 of the `realized_volatility_5d` payload (`a8b097ca3cb9d6b623890aca2dcfa65c3e0d3658761541cc8fb64d51d09aac9f`), which is machine-independent. Phase 1 writes the parquet with pyarrow's default writer options (Snappy compression, `index=False`).

## Setup

```bash
uv sync
```

Phase 1 downloads a Kaggle dataset through `kagglehub`, which reads Kaggle credentials (`~/.kaggle/kaggle.json` by default; create one under Kaggle → Settings → API → Create New Token). `kagglehub` caches the download, but each Phase 1 run still validates the cache, so the machine needs network access and valid credentials. Phases 2-5 are fully local.

## Running the pipeline

There is no single end-to-end entry point. Run the two phases in order, from the repository root:

```bash
# Phase 1 — download, clean, and engineer features
python packages/ml/dataset/main.py
# writes data/processed/dataset.parquet

# Phases 2-5 — feature selection, temporal validation, training, evaluation
python packages/ml/model/main.py
# ... or the module form:
python -m packages.ml.model.main
# writes model artifacts and charts (see Outputs)
```

Both phases work from the repository root or from inside their own package directory, and the model phase runs the same through either form above. The model entry accepts one optional flag, `--dry-run`, which logs the planned phases and exits without running them.

Phase 2 fails immediately if Phase 1 has not been run first. Because generated data is gitignored, a fresh clone always starts with Phase 1.

Phase 1 feature code lives in `packages/ml/dataset/features/` (one family per module, assembled by `assembly.py`):

- `features/__init__.py` — re-exports `build_features_dataframe`
- `features/_common.py` — shared z-score guard and adjusted-OHLC column names
- `features/prices.py` — base returns, adjusted OHLC, return lags, price/SMA normalization, rolling means and extremes
- `features/volume.py` — volume lags, transforms, and z-score
- `features/trend.py` — EMA trend and RSI/MACD momentum
- `features/volatility.py` — range-based and trailing volatility
- `features/target.py` — Yang-Zhang 5-day realized-volatility target
- `features/regimes.py` — hardcoded calendar `market_regime` label
- `features/assembly.py` — validation, history enforcement, and `build_features_dataframe`

### Outputs

| Path | Phase | Contents |
|---|---|---|
| `data/processed/dataset.parquet` | 1 | Cleaned survivors-only panel: raw OHLCV columns plus the retained adjusted OHLC columns (`_adj_open`, `_adj_high`, `_adj_low`), per-ticker features plus `log_price`, `realized_volatility_5d`, `market_regime` |
| `packages/ml/model/artifacts/inputs/` | 2 | `X.npy`, `Y.npy`, `metadata.pkl` — exactly what training consumes |
| `packages/ml/model/artifacts/folds/` | 3 | Fold indices and date ranges for temporal validation |
| `packages/ml/model/artifacts/checkpoints/` | 4 | One trained model per fold plus its `fold_<i>_params.json` hyperparameter sidecar, under a run-scoped `<run_id>/` directory; each file is written atomically (tmp + rename) |
| `packages/ml/model/artifacts/results/` | 1, 4 | `fold_training_summary.json` (per-fold metrics), `har_benchmark.json` (HAR-RV benchmark per fold), `regime_characterization.json`, predictions |
| `packages/ml/model/artifacts/` | 2-5 | `runs.jsonl` (append-only run log: one strict-JSON line per run), `run_manifest.json` (latest run: identity, config hash, environment versions, phase/fold status, durations, artifact hashes), `artifact_registry.json` (byte size + sha256 per recorded artifact, updated atomically at write time; every live `joblib.load` verifies against it before loading) |
| `data/model_outputs/` | 5 | 14 PNG evaluation charts |

Everything in the table is generated output and gitignored.

## How the data is validated

Temporal validation uses date-group folds with expanding training windows: the sorted unique trading dates are split into six contiguous blocks, and each of the five folds trains on every block before its test block and is scored only on later rows.

Each fold applies a train-only purge of the 5-day forward target's label overlap — training rows whose label window reaches into the test block are dropped, while test blocks are kept whole — so exactly five complete trading days separate the last kept training date from the first kept test date. The per-fold guard in `packages/ml/model/cross_validation/cross_validator.py` enforces this in unique-trading-date index space.

Most price-derived features are computed per ticker and shifted one day, so they never see their own bar. A handful of model inputs deliberately carry the close of day t: `simple_return` is a same-bar return, and the `price_vs_sma_*` and `zscore_*` families divide a current-bar value by a lagged statistic. For those inputs the documented contract is the close of t — they are anchored to the bar being predicted, not to the previous close. `log_return` stays in the dataset as the target's input but is not a model input. The target is the single deliberate exception: it looks five days forward.

Six dataset columns are infinite-memory recursions rather than windowed statistics: `ema_12`, `ema_26`, `macd`, `macd_signal`, `macd_hist` and `rsi_14`. With pandas' `adjust=False` the first value seeds each recursion, so the earliest outputs are warm-up approximations of their steady-state definitions rather than exact long-memory estimates; the head cut discards most of that warm-up. `rsi_14` ships the EMA-seeded variant of the smoothed average-gain/average-loss ratio, not Wilder's original 14-period SMA-seeded recursion — on the shipped frame the two agree bit-exactly on ~40% of rows and within 0.5 RSI points on 99.93% of them, with the remaining gap confined to the first ~33 surviving rows of each ticker and decaying with the recursion. Consumers that need steady-state values should recompute the recursion over each ticker's full raw series.

## Model

One `XGBRegressor` per fold, hyperparameters in `packages/ml/model/training/model_trainer.py`. Keeping one model per fold prevents any fold from being influenced by another fold's evaluation data. Edit hazards in the training path — including a dead early-stopping branch — are listed in AGENTS.md.

The target is modeled as `log(RV)` — the Yang-Zhang variance is closer to log-normal, and QLIKE is the robust loss for it. Exact-zero flat windows are floor artifacts: they are floored at `1e-12` only to keep the log finite, and one floor policy excludes them from fold fitting, model scoring, and the HAR benchmark. Each fold persists the original test indices, the aligned retained-row mask, the scored indices, and reconciled counts, and declares the headline floor-retained row set. R²/MAE/RMSE are computed on the log scale, with sklearn's fold-own-mean R² convention. Every fold also fits the canonical HAR-RV benchmark (Corsi (2009), 1/5/22 trailing variance components) on its training rows only and is scored on the same floor-retained test rows; the comparison lands in `har_benchmark.json`.

## Evaluation

Phase 5 writes 14 charts to `data/model_outputs/` in four groups:

- **Metrics** — R² per fold, MAE vs RMSE, and a metrics heatmap (all on the log target)
- **Temporal** — fold windows on a timeline, sample counts per fold, metric trend over time
- **Comparisons** — per-fold R²/MAE/RMSE bars, training vs inference timing, sample distributions, feature counts
- **Diagnostics** — actual vs predicted, residual distributions, feature importance, error by volatility level

For the numbers, read `packages/ml/model/artifacts/results/fold_training_summary.json` and `har_benchmark.json` from the run that produced them. They are not reproduced here because they depend on both the data snapshot and the machine.

## Scope and limitations

- **Survivors-only universe.** The source carries current and recent S&P 500 constituents only — companies that were delisted are absent. Long-horizon return statistics are therefore biased upward by roughly 1.4–3.9 percentage points per year (CAGR), and the ticker cross-section grows monotonically by construction. Phase 1 fails fast if the per-year ticker count ever declines, which is how a frame containing exits would present itself; a membership-aware frame must waive that check explicitly via `prepare_dataframe(..., survivors_only=False)`. The regime characterization JSON records the frame label and the bias note under `_frame`.
- **Price and volume only.** No macro data, fundamentals, sentiment, or cross-ticker features.
- **No sector information.** The source data carries no sector labels, so sector dynamics cannot be learned.
- **`market_regime` is a hardcoded calendar label**, not something derived from the data. Seven calendar buckets span 2000-2007 through 2026; a date past 2026 raises instead of joining the last bucket.
- **QLIKE training targets conditional sigma.** The fold models minimize QLIKE on log daily sigma, the same loss reported in evaluation; the log-scale R²/MAE/RMSE remain diagnostic, and tail-risk use still requires out-of-sample evaluation.
- **Model tests, no lint suite.** `packages/ml/model/tests/` runs under pytest; other changes are verified by running the affected phase and reading the logs.
- **A fresh clone needs Phase 1 first.** Nothing downstream can run without the generated parquet file.

## Contributing

Read [AGENTS.md](AGENTS.md) before editing. It covers the two entry points and their order, the invariants that silently break saved artifacts, and known dead code. Run the model tests with `uv run --frozen python -m pytest packages/ml/model/tests -q` (checks that need generated data are skipped on a fresh clone). Run the phase covering your change — Phase 1 for `packages/ml/dataset`, Phases 2-5 otherwise — and confirm it completes. Repo-wide rationale lives in [docs/adr/](docs/adr/); check whether your change needs a new record before editing an accepted one.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
