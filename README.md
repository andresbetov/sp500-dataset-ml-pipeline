# S&P 500 Volatility Prediction Pipeline

Predicts 5-day realized volatility for S&P 500 constituents from price and volume history alone.

This is a research pipeline, not a production risk or trading system. One script builds the dataset; one entry point trains and evaluates the model. Run metrics are written to disk on every run rather than quoted here, because they change with each retrain and each refresh of the source data.

## Research question and scope

The research question is whether the model achieves a lower average QLIKE than the HAR-RV benchmark under a fixed protocol, and where any difference is concentrated across the volatility distribution.

The target is `realized_volatility_5d`: the Yang-Zhang daily volatility estimate over the next five trading days, expressed as per-day sigma and not annualized. The universe is S&P 500 constituents from a survivors-only source, so delisted names are absent.

The model consumes 37 price/volume features and no ticker identity. It trains one XGBoost regressor per temporal fold, each minimizing QLIKE on `log(RV)`, and is scored against the canonical HAR-RV benchmark (Corsi 2009, 1/5/22 trailing variance components) on the same rows.

Intended use is research and benchmarking: comparing specifications under a fixed protocol and quantifying the average difference from HAR. It is not for production risk, trading, or causal use.

## Quickstart

Requirements: Python 3.12 or newer, [uv](https://docs.astral.sh/uv/) for dependency management, and Kaggle API credentials for Phase 1 only (`~/.kaggle/kaggle.json`).

```bash
uv sync
source .venv/bin/activate
python packages/ml/dataset/main.py
python -m packages.ml.model.main
```

`uv sync` creates the virtual environment; activate it before the `python` commands, or prefix each command with `uv run`.

Sizes and runtimes are hardware- and data-dependent; [docs/operations.md](docs/operations.md) records the current disk snapshot and where phase durations are logged.

Phase 1 downloads the pinned Kaggle CSV and needs network access plus Kaggle credentials. Phases 2-5 are local and fail fast if `data/processed/dataset.parquet` is missing, so a fresh clone always starts with Phase 1. The model entry also runs by path:

```bash
python packages/ml/model/main.py
```

To check the model entry point without writing anything, use its dry run:

```bash
python -m packages.ml.model.main --dry-run
```

The model entry also accepts `--resume`, `--folds`, `--evaluate-holdout`, and `--force-holdout`; the run lifecycle is documented in [docs/operations.md](docs/operations.md).

All generated output is gitignored.

## Pipeline at a glance

| Phase | Entry point | Input | Output | Doc |
|---|---|---|---|---|
| 1 — dataset | `python packages/ml/dataset/main.py` | Kaggle CSV | `data/processed/dataset.parquet`, `data/processed/ingestion_manifest.json`, `packages/ml/model/artifacts/results/regime_characterization.json` | [dataset](docs/pipelines/dataset.md) |
| 2 — feature selection | model entry | the parquet | `inputs/X.npy`, `Y.npy`, `metadata.pkl` | [model](docs/pipelines/model.md) |
| 3 — temporal cross-validation | model entry | inputs | `folds/folds_metadata.pkl` | [model](docs/pipelines/model.md) |
| 4 — training and evaluation | model entry | folds | fold summaries, predictions, loss vectors, `evaluation_summary.json` | [model](docs/pipelines/model.md) |
| 5 — visualization | model entry | results | 14 PNGs plus `data/model_outputs/manifest.json` | [model](docs/pipelines/model.md) |

```text
Phase 1 parquet -> Phase 2 inputs -> Phase 3 folds -> Phase 4 models/metrics -> Phase 5 charts
        optional, after Phase 5: locked holdout under --evaluate-holdout
```

The holdout is opt-in and runs only after the normal phases complete, under `--evaluate-holdout`.

## Evaluation design

The protocol is date-group cross-validation with a separate locked holdout. Five design choices matter most:

- **Date-group expanding folds.** Sorted unique trading dates are split into contiguous blocks; each fold trains on earlier blocks and scores only later rows. No shuffle, so a model is never scored on data earlier than its training window.
- **Train-only 5-day purge.** Training rows whose 5-day label window reaches into a test block are dropped, so no test-period outcome leaks into training. Test blocks are kept whole.
- **Strict floor.** The target is floored at `1e-12` before the log; at-or-below-floor rows are excluded from fitting, model scoring, and the benchmark, so flat windows cannot distort the loss.
- **Locked one-shot holdout.** The final 500 unique dates are reserved before fold construction and read only by an opt-in evaluation that writes a lock before scoring anything.
- **QLIKE against HAR.** QLIKE is the headline loss, reported beside dual out-of-sample R², Diebold-Mariano/Clark-West/Giacomini-White (DM/CW/GW) tests, and Mincer-Zarnowitz calibration.

The protocol lives in [docs/pipelines/model.md](docs/pipelines/model.md); how to read the numbers lives in [docs/results.md](docs/results.md).

## Results

Live metrics are written to artifacts and are not maintained in this file. The cross-validation source is `packages/ml/model/artifacts/results/evaluation_summary.json`; the holdout source is `packages/ml/model/artifacts/results/holdout_report.json`. See [docs/results.md](docs/results.md) for how to read them.

The last locked evaluation is dated 2026-10-20 and is documented in [the technical report](docs/reports/model-performance-and-rigor.md). It reports a lower average QLIKE than HAR in both cross-validation and the holdout, with the report's stated caveats: one seed, one 500-date holdout window, calibration rejected, tail safety not established, and a survivors-only universe. The report is historical evidence; later runs on refreshed data may differ. Its tables, intervals, and falsification tests are the canonical narrative source.

## Limitations

- **Survivors-only universe.** The source carries current and recent constituents only; delisted names are absent, so long-horizon returns are biased upward.
- **Price and volume only.** No macro data, fundamentals, sentiment, or cross-ticker features.
- **No sector data.** The source carries no sector labels, so sector dynamics cannot be learned.
- **Calendar `market_regime`.** The regime label is a hardcoded calendar bucket, not a data-derived state, and is not a model input.
- **Calibration rejected, tails not safe.** Mincer-Zarnowitz rejects calibration, and tail safety is not established.
- **Single seed, one holdout window.** The reported results come from one seed and one 500-date holdout; seed and window uncertainty are not estimated.

The full limitations and falsification list is in [docs/reports/model-performance-and-rigor.md](docs/reports/model-performance-and-rigor.md).

## Repository map

- `packages/ml/dataset/` — Phase 1: download, clean, engineer features, build the target.
- `packages/ml/model/` — Phases 2-5: feature selection, folds, training, evaluation, visualization.
- `packages/ml/model/config/base.yaml` — the runtime configuration: fold count, holdout size, retention, floor, and frozen hyperparameters.
- `docs/` — documentation index, operations runbook, results guide, and pipeline references.
- `docs/adr/` — architecture decision records that own the rationale behind the design.
- `data/processed/` — generated dataset (gitignored).
- `packages/ml/model/artifacts/` — generated run state and results (gitignored).
- `data/model_outputs/` — generated charts (gitignored).

## Documentation

| What | Where |
|---|---|
| Quickstart (this file) | [`README.md`](README.md) |
| Dataset pipeline | [`docs/pipelines/dataset.md`](docs/pipelines/dataset.md) |
| Dataset column dictionary | [`docs/reference/dataset-schema.md`](docs/reference/dataset-schema.md) |
| Model pipeline | [`docs/pipelines/model.md`](docs/pipelines/model.md) |
| Command and flag reference | [`docs/reference/cli.md`](docs/reference/cli.md) |
| Running, resuming, retention, holdout | [`docs/operations.md`](docs/operations.md) |
| Reading results | [`docs/results.md`](docs/results.md) |
| Model card | [`docs/model-card.md`](docs/model-card.md) |
| Technical report | [`docs/reports/model-performance-and-rigor.md`](docs/reports/model-performance-and-rigor.md) |
| Architecture decisions | [`docs/adr/`](docs/adr/) |
| Agent and contributor rules | [`AGENTS.md`](AGENTS.md) |

## Contributing

Read [AGENTS.md](AGENTS.md) before editing. Run the hermetic model suite with `uv run --frozen python -m pytest packages/ml/model/tests -q --ignore=packages/ml/model/tests/live`, then run the phase your change touches and confirm it completes. The full suite adds live artifact checks under `tests/live/`; those skip on a fresh clone.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
