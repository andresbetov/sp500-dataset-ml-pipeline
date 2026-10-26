# Dataset schema

This is the column dictionary for `data/processed/dataset.parquet`, the artifact Phase 2 consumes. It covers all 53 shipped columns: 37 model inputs and 16 extras. The current snapshot is `2,669,959` rows, `469` tickers, and `6,504` dates (`2000-04-04` to `2026-02-12`), but the schema is the contract; the counts are a dated snapshot.

The construction order, cleaning rules, and feature-family behavior are in [dataset.md](../pipelines/dataset.md). The model-input contract and evaluation protocol are in [model.md](../pipelines/model.md). The time-alignment rationale is in [ADR 0007](../adr/0007-time-alignment-contract.md).

## How to read this table

- **Type** is the Arrow type written to the parquet. Every numeric feature is `float64`; `market_regime` is `int8`; `ticker` is a 32-bit Arrow string; `date` is a timezone-naive microsecond timestamp.
- **Input** says whether the column enters `X` through `INDICATORS`. Extras never enter the model matrix.
- **Alignment** says which bar the value is known at. Most features are shifted one day; the documented same-bar exceptions divide a current-bar value by a lagged statistic. The target is the only forward-looking column.
- **Warm-up** is the first 0-based row position within a ticker at which the column is defined after `_enforce_history_based_nan_consistency`. A rolling window of `w` needs `w` prior rows, and entries built on `log_return` start one row later. Rows before the warm-up are NaN and are dropped before the parquet is written.

## Keys and raw columns

| Column | Type | Input | Definition | Alignment | Warm-up |
|---|---|---|---|---|---|
| `ticker` | string | no | S&P 500 constituent symbol; key with `date`. | as-of row | none |
| `date` | timestamp[us] | no | Trading date, timezone-naive, normalized to midnight. | as-of row | none |
| `open` | float64 | no | Raw unadjusted open. | same row | none |
| `high` | float64 | no | Raw unadjusted high. | same row | none |
| `low` | float64 | no | Raw unadjusted low. | same row | none |
| `close` | float64 | no | Raw unadjusted close. | same row | none |
| `adj_close` | float64 | no | Split/dividend-adjusted close; the return and target price base. | same row | none |
| `volume` | float64 | no | Raw share volume; transformed into the volume features below. | same row | none |

## Adjusted and intermediate columns

These are the remaining eight extras. They are shipped for provenance, feature construction, and analysis; none enters `X` directly.

| Column | Type | Input | Definition | Alignment | Warm-up |
|---|---|---|---|---|---|
| `_adj_open` | float64 | no | `open * (adj_close / close)`. | same row | none |
| `_adj_high` | float64 | no | `high * (adj_close / close)`. | same row | none |
| `_adj_low` | float64 | no | `low * (adj_close / close)`. | same row | none |
| `log_price` | float64 | no | Natural log of `adj_close`. | same row | none |
| `log_return` | float64 | no | Per-ticker difference of `log_price`; feeds the target and trailing-volatility features. | same row | 1 |
| `macd_signal` | float64 | no | Re-anchored 9-day EMA of the unshifted MACD difference, divided by the lagged adjusted close. It stays a dataset column but left `INDICATORS`. | shifted one day | 1 |
| `market_regime` | int8 | no | Hardcoded calendar bucket `0..6`, from 2000-2007 through 2026. Analysis metadata, not a model input. | same row | none |
| `realized_volatility_5d` | float64 | no | Yang-Zhang realized volatility over the next five trading days; per-day sigma, not annualized. The model target. | forward five days | last five rows per ticker are NaN |

`market_regime` is not data-derived; a date past 2026 raises until the calendar is extended. The target's first overnight component uses the label row's own close, which is an input at the horizon start rather than leakage past it.

## Model inputs: prices (13)

| Column | Type | Input | Definition | Alignment | Warm-up |
|---|---|---|---|---|---|
| `simple_return` | float64 | yes | Per-ticker percentage change of `adj_close`. | same bar | 1 |
| `return_lag_1` | float64 | yes | `simple_return` shifted one row. | shifted | 1 |
| `return_lag_2` | float64 | yes | `simple_return` shifted two rows. | shifted | 2 |
| `return_lag_3` | float64 | yes | `simple_return` shifted three rows. | shifted | 3 |
| `return_lag_5` | float64 | yes | `simple_return` shifted five rows. | shifted | 5 |
| `return_lag_10` | float64 | yes | `simple_return` shifted ten rows. | shifted | 10 |
| `price_vs_sma_10` | float64 | yes | `adj_close` divided by the one-day-lagged 10-day simple moving average. | same bar numerator | 10 |
| `price_vs_sma_20` | float64 | yes | Same construction with a 20-day SMA. | same bar numerator | 20 |
| `price_vs_sma_50` | float64 | yes | Same construction with a 50-day SMA. | same bar numerator | 50 |
| `zscore_price_vs_sma_20` | float64 | yes | 20-day per-ticker rolling z-score of `price_vs_sma_20`; a constant window is pinned to zero. | same bar | 40 |
| `rolling_mean_5` | float64 | yes | One-day-lagged 5-day mean of `adj_close`, divided by the lagged adjusted close. | shifted | 5 |
| `rolling_max_10` | float64 | yes | One-day-lagged 10-day maximum of `adj_close`, divided by the lagged adjusted close. | shifted | 10 |
| `rolling_min_10` | float64 | yes | One-day-lagged 10-day minimum of `adj_close`, divided by the lagged adjusted close. | shifted | 10 |

## Model inputs: volume (4)

| Column | Type | Input | Definition | Alignment | Warm-up |
|---|---|---|---|---|---|
| `volume_lag_1` | float64 | yes | `log1p` of `volume` shifted one row. | shifted | 1 |
| `volume_lag_3` | float64 | yes | 20-day per-ticker z-score of `log1p(volume)` shifted three rows; constant windows are pinned to zero. | shifted | 23 |
| `volume_lag_5` | float64 | yes | Per-date cross-sectional percentile rank of `volume` shifted five rows, mapped to `[-1, 1]`. | shifted | 5 |
| `zscore_volume_20` | float64 | yes | 20-day per-ticker rolling z-score of raw `volume`; constant windows are pinned to zero. | same bar | 20 |

`volume_lag_5` is one of the two non-reconstructible inputs: it is computed on the pre-drop cross-section and cannot be re-derived from the shipped parquet. `market_abs_ret_5d` is the other.

## Model inputs: trend and momentum (5)

| Column | Type | Input | Definition | Alignment | Warm-up |
|---|---|---|---|---|---|
| `ema_12` | float64 | yes | One-day-lagged 12-day EMA of `adj_close`, divided by the lagged adjusted close. | shifted | 1 |
| `ema_26` | float64 | yes | One-day-lagged 26-day EMA of `adj_close`, divided by the lagged adjusted close. | shifted | 1 |
| `rsi_14` | float64 | yes | EMA-seeded RSI-14 on `adj_close`, shifted one row. This is not Wilder's original SMA seed; see the dataset pipeline note. | shifted | 15 |
| `macd` | float64 | yes | One-day-lagged difference of the 12- and 26-day EMAs, divided by the lagged adjusted close. | shifted | 1 |
| `macd_hist` | float64 | yes | `macd - macd_signal`. | shifted | 1 |

## Model inputs: volatility (15)

The range estimators use the adjusted OHLC bars and are shifted one day. The trailing volatility stack is built on `log_return` and is also shifted one day, so no input sees its own bar.

| Column | Type | Input | Definition | Alignment | Warm-up |
|---|---|---|---|---|---|
| `atr5` | float64 | yes | 5-day mean of `(_adj_high - _adj_low) / adj_close`. | shifted | 5 |
| `atr20` | float64 | yes | 20-day mean of the same average-true-range proxy. | shifted | 20 |
| `park5` | float64 | yes | 5-day mean of the Parkinson range volatility `sqrt(log(H/L)^2 / (4 ln 2))`. | shifted | 5 |
| `park20` | float64 | yes | 20-day mean of the same Parkinson term. | shifted | 20 |
| `gk5` | float64 | yes | 5-day mean of the Garman-Klass volatility; negative bar variances are clipped to zero. | shifted | 5 |
| `rs5` | float64 | yes | 5-day mean of the Rogers-Satchell variance term. | shifted | 5 |
| `gap5` | float64 | yes | 5-day mean absolute overnight gap `abs(log(_adj_open / prior adj_close))`. | shifted | 5 |
| `rv5_past` | float64 | yes | 5-day mean of `abs(log_return)`. | shifted | 6 |
| `rv22` | float64 | yes | 22-day rolling standard deviation of `log_return` (`ddof=1`). | shifted | 23 |
| `rv63` | float64 | yes | 63-day rolling standard deviation of `log_return` (`ddof=1`). | shifted | 64 |
| `har5` | float64 | yes | 5-day sum of squared `log_return`. | shifted | 6 |
| `har22` | float64 | yes | 22-day sum of squared `log_return`. | shifted | 23 |
| `har63` | float64 | yes | 63-day sum of squared `log_return`. | shifted | 64 |
| `volvol22` | float64 | yes | 22-day standard deviation of `abs(log_return)`. | shifted | 23 |
| `market_abs_ret_5d` | float64 | yes | 5-day mean of the per-date cross-sectional mean `abs(log_return)`; the cross-sectional mean skips tickers without a return on a date. | shifted | 6 |

`market_abs_ret_5d` is the second non-reconstructible input: it uses the pre-drop cross-section and cannot be reproduced from the shipped parquet alone. Model code must not re-derive either non-reconstructible column.

## Counts and invariants

- 37 model inputs = 13 prices + 4 volume + 5 trend/momentum + 15 volatility.
- 16 extras = 8 raw/key columns + 3 adjusted OHLC + `log_price` + `log_return` + `macd_signal` + `market_regime` + `realized_volatility_5d`.
- `INDICATORS` order is positional. Reordering the list silently mislabels feature-importance charts; the module asserts the pinned count of 37.
- The target is the only forward-looking column. The same-bar inputs are `simple_return`, `price_vs_sma_{10,20,50}`, `zscore_price_vs_sma_20`, and `zscore_volume_20`; everything else is shifted one day.
- The parquet is written after the feature and target build and after the NaN drop. The floor (`1e-12`) is applied later in model training, not in this artifact.
