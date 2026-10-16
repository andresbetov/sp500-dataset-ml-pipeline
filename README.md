# S&P 500 Volatility Prediction Pipeline

Predicts 5-day realized volatility for S&P 500 constituents from price and volume history alone.

This is a research pipeline: one script builds the dataset, one trains and evaluates the model. Run metrics are written to disk rather than quoted here — they change with every retrain and with every refresh of the source data.

## What it does

Given a multi-ticker daily OHLCV panel, the pipeline:

1. Downloads and cleans the source data from Kaggle.
2. Engineers per-ticker technical features plus a forward-looking volatility target.
3. Trains one gradient-boosted regression model per temporal fold, evaluating each only on strictly later data.
4. Writes an evaluation report and charts for that run.

The prediction target is `realized_volatility_5d` — the standard deviation of log returns over the next five trading days. This is continuous regression, not classification.

Features come from price and volume only. There is deliberately no per-ticker identity encoding: a ticker's behaviour is already carried by its own indicator magnitudes (larger returns, more extreme momentum, different volume patterns), so per-ticker columns would add sparse inputs without adding generalizable signal.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- Kaggle API credentials — needed only for Phase 1

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
# writes model artifacts and charts (see Outputs)
```

Both scripts accept no arguments and read no environment variables, and both work from the repository root or from inside their own package directory.

Phase 2 fails immediately if Phase 1 has not been run first. Because generated data is gitignored, a fresh clone always starts with Phase 1.

### Outputs

| Path | Phase | Contents |
|---|---|---|
| `data/processed/dataset.parquet` | 1 | Cleaned survivors-only panel: raw OHLCV columns, per-ticker features plus `log_price`, `realized_volatility_5d`, `market_regime` |
| `packages/ml/model/artifacts/inputs/` | 2 | `X.npy`, `Y.npy`, `metadata.pkl` — exactly what training consumes |
| `packages/ml/model/artifacts/folds/` | 3 | Fold indices and date ranges for temporal validation |
| `packages/ml/model/artifacts/checkpoints/` | 4 | One trained model per fold |
| `packages/ml/model/artifacts/results/` | 1, 4 | `fold_training_summary.json` (per-fold metrics), `regime_characterization.json`, predictions |
| `data/model_outputs/` | 5 | 14 PNG evaluation charts |

Everything in the table is generated output and gitignored.

## How the data is validated

Temporal validation uses `TimeSeriesSplit` with five folds and expanding training windows — each fold trains on all history preceding its test window and is scored only on later rows.

One caveat worth knowing before trusting the split: the `gap` parameter counts rows, not trading days, and the panel holds hundreds of tickers per date, so the embargo does not separate dates — rows from a test date can appear in the same fold's training set under a different ticker. The leakage guard in `packages/ml/model/cross_validation/cross_validator.py` documents and permits this.

Most price-derived features are computed per ticker and shifted one day, so they never see their own bar. Some model inputs do: `simple_return` and `log_return` are same-bar returns, and the `price_vs_sma_*` and `zscore_*` families divide a current-bar value by a lagged statistic, so they carry information from the close being predicted. The target is the single deliberate exception: it looks five days forward.

## Model

One `XGBRegressor` per fold, hyperparameters in `packages/ml/model/training/model_trainer.py`. Keeping one model per fold prevents any fold from being influenced by another fold's evaluation data. Edit hazards in the training path — including a dead early-stopping branch — are listed in AGENTS.md.

## Evaluation

Phase 5 writes 14 charts to `data/model_outputs/` in four groups:

- **Metrics** — R² per fold, MAE vs RMSE, and a metrics heatmap
- **Temporal** — fold windows on a timeline, sample counts per fold, metric trend over time
- **Comparisons** — per-fold R²/MAE/RMSE bars, training vs inference timing, sample distributions, feature counts
- **Diagnostics** — actual vs predicted, residual distributions, feature importance, error by volatility level

For the numbers, read `packages/ml/model/artifacts/results/fold_training_summary.json` from the run that produced it. They are not reproduced here because they depend on both the data snapshot and the machine.

## Scope and limitations

- **Survivors-only universe.** The source carries current and recent S&P 500 constituents only — companies that were delisted are absent. Long-horizon return statistics are therefore biased upward by roughly 1.4–3.9 percentage points per year (CAGR), and the ticker cross-section grows monotonically by construction. Phase 1 fails fast if the per-year ticker count ever declines, which is how a frame containing exits would present itself; a membership-aware frame must waive that check explicitly via `prepare_dataframe(..., survivors_only=False)`. The regime characterization JSON records the frame label and the bias note under `_frame`.
- **Price and volume only.** No macro data, fundamentals, sentiment, or cross-ticker features.
- **No sector information.** The source data carries no sector labels, so sector dynamics cannot be learned.
- **`market_regime` is a hardcoded calendar label**, not something derived from the data. Its buckets end at "2022 and later", so more recent data needs a new bucket added.
- **Squared-error training predicts conditional means.** Predictions are compressed toward the centre of the historical range by construction; tail-risk use requires post-hoc calibration.
- **No automated test or lint suite.** Changes are verified by running the affected phase and reading the logs.
- **A fresh clone needs Phase 1 first.** Nothing downstream can run without the generated parquet file.

## Contributing

Read [AGENTS.md](AGENTS.md) before editing. It covers the two entry points and their order, the invariants that silently break saved artifacts, and known dead code. Run the phase covering your change — Phase 1 for `packages/ml/dataset`, Phases 2-5 otherwise — and confirm it completes.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
