# Model card: S&P 500 5-day volatility model

This is a research model card. It summarizes intended use, data, protocol, and standing caveats; the full evidence and numbers are in the [technical report](reports/model-performance-and-rigor.md), not here.

## Model details

- **Name:** S&P 500 volatility model (`sp500-dataset-ml-pipeline`).
- **Provenance:** verified against commit `61eae9e` on 2026-10-25; last locked evaluation 2026-10-20.
- **Type:** one XGBoost regressor per temporal fold, pooled across tickers.
- **Target:** `log(realized_volatility_5d)`, the Yang-Zhang per-day sigma over the next five trading days. Forecasts are exponentiated back to sigma for QLIKE.
- **Objective:** custom QLIKE objective and evaluation metric on the log target.
- **Inputs:** 37 price/volume features, listed in [dataset-schema.md](reference/dataset-schema.md) and [feature_engineering.py](../packages/ml/model/features/feature_engineering.py). No ticker identity, date, sector, or `market_regime` input.
- **Benchmark:** canonical HAR-RV (Corsi 2009), 1/5/22 trailing variance components, OLS in variance levels, trained only on training rows.

## Intended use

Research and benchmarking: comparing specifications under a fixed temporal protocol and quantifying the average difference from HAR. The model is not a production risk system, trading strategy, or causal estimator.

**Out of scope:** VaR/ES, position sizing, intraday forecasting, volatility of a different horizon, a different universe, or any use that assumes calibrated predictive distributions.

## Training data and universe

The source is a pinned Kaggle S&P 500 historical dataset; delisted companies are absent (survivors-only). The frame is current-and-recent constituents, so long-horizon return statistics are biased upward and crisis behavior is not a point-in-time index reconstruction. The current snapshot and source provenance are in [dataset.md](pipelines/dataset.md); the source terms are the Kaggle dataset's terms.

## Evaluation protocol

Date-group, expanding-window cross-validation with a train-only 5-day purge and a separately reserved 500-date holdout. The last 15% of each fold's floor-retained training rows is a purged inner-validation tail. At-or-below-floor targets (`1e-12`) are excluded from fitting, model scoring, and the HAR benchmark. The holdout is opt-in, one-shot, and locked before any metric. Full details: [model.md](pipelines/model.md) and [operations.md](operations.md#locked-holdout).

## Results and caveats

The 2026-10-20 evaluation reports a lower average QLIKE than HAR in cross-validation and in the locked holdout; the report is the canonical source for magnitudes, intervals, and tables. It also reports that:

- calibration is rejected by Mincer-Zarnowitz;
- tail safety is not established (the model is worse in the highest volatility deciles);
- seed robustness, crisis robustness, economic value, and non-HAR superiority are not tested;
- the holdout is one 500-date window, and the universe is survivors-only.

Do not cite a point estimate from this card or the README; cite the report and its caveats.

## Artifacts and provenance

The pipeline records a run manifest, an append-only `runs.jsonl`, an artifact registry with size/SHA-256, fold-manifest hashes, and a holdout lock. Reading and reproduction guidance is in [results.md](results.md); do not run the holdout to audit. The code and documentation are Apache-2.0; the source data keeps its own terms.
