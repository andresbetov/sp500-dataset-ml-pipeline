# Dataset pipeline

This is the deep reference for Phase 1: how the raw Kaggle CSV becomes the parquet that Phase 2 consumes. It documents the current-state behavior of `packages/ml/dataset/`. The model side of the pipeline is documented in [model.md](model.md); running and resuming lives in [../operations.md](../operations.md).

## Purpose and contract

Phase 1 turns the pinned Kaggle CSV into `data/processed/dataset.parquet`, the only input Phases 2-5 read. It needs network access and Kaggle credentials (`~/.kaggle/kaggle.json`), and it takes no arguments.

It also writes into the model package's artifact tree at `packages/ml/model/artifacts/results/regime_characterization.json`. Relocating model artifacts therefore means editing the dataset phase too. The entry point is `packages/ml/dataset/main.py`, run from the repository root:

```bash
python packages/ml/dataset/main.py
```

On a fresh clone the parquet does not exist, so Phase 2 fails immediately until Phase 1 has run once. A successful run logs the row and ticker counts it wrote, the "Dropped … NaN rows" line, and the parquet-then-JSON save order.

## Inputs and provenance

The source is a single pinned Kaggle dataset: `jacksaleeby/s-and-p500-historical-data/versions/1`, file `SP500_Historical_Data.csv`. `ingestion.py` downloads it through `kagglehub.dataset_download`, which caches the bundle on disk, then picks the one CSV that matches the expected file name and rejects an empty, ambiguous, or differently named bundle.

Every load writes `data/processed/ingestion_manifest.json`, which records the source handle, the file name, the file's SHA-256, the raw row count, the ticker count, and the date range. The fields are `source`, `file_name`, `source_sha256`, `rows`, `tickers`, `date_min`, and `date_max`.

The current shipped source snapshot is `2,703,531` rows, `472` tickers, dated `2000-01-03` to `2026-02-20`. Treat these as a dated snapshot of the pinned source, not as invariants; a refreshed download can differ. The repository does not redistribute the source CSV; check the Kaggle dataset's terms before redistributing derived data.

## Stage flow

`main.py` runs the stages in this order through one function, `generate_featured_dataset`:

1. **Ingestion** — `load_dataframe()` reads the CSV and writes the ingestion manifest.
2. **Cleaning** — `prepare_dataframe()` validates, repairs, and filters the raw panel.
3. **Feature assembly** — `build_features_dataframe()` derives the engineered columns.
4. **Target and regime** — added inside feature assembly as `realized_volatility_5d` and `market_regime`.
5. **Regime characterization** — `characterize_regimes()` fingerprints each calendar bucket before any row is dropped.
6. **Validation** — `check_contiguous_ticker_frames()` rejects interior ticker gaps; `check_panel_staleness()` warns when the panel's last date lags the run date.
7. **Write** — the parquet is written first, then the regime JSON.

The NaN drop happens after regime characterization: the pipeline characterizes regimes on the full featured frame, drops NaN rows, then writes. This is why the regime JSON's `_frame` record carries both a pre-drop row count and a shipped row count.

## Cleaning rules

`prepare_dataframe` applies these steps in this exact order. Each step exists to catch a specific corruption signature before it can poison a downstream feature.

| Step | What it does | Why it exists |
|---|---|---|
| Normalize column names | Lowercase, strip, and replace spaces with underscores; fail on collisions. | Makes the raw header schema-stable. |
| Validate columns | Require exactly `ticker, date, open, high, low, close, adj_close, volume`; fail on missing or unexpected columns. | Guarantees every later step has its inputs. |
| Cast types | Ticker to string, date to datetime, prices to float64; volume stays int64 while every value is a non-missing integer and becomes float64 otherwise. | Missing or fractional volume is preserved, not coerced to zero. |
| Price-grid guard | Fail when closes collapse onto fewer than `MIN_DISTINCT_CLOSE_PRICES = 50` distinct two-decimal values; count sub-dime closes as a census instead of rejecting them. | A coarse grid means the source came back rounded or corrupt. |
| Survivors-only guard | Fail when the per-year ticker count declines for a frame labelled survivors-only. | A survivors-only universe can only grow its cross-section; a decline means exits are present. |
| OHLC repair/drop | Clip an open back into its own bar when only the open is outside `[low, high]`; drop rows whose range or close is invalid. | Repair a vendor rounding slip; refuse to invent prices for anything worse. |
| Stable dedupe | Stable-sort by `(ticker, date)` and keep the first of each duplicate pair. | Make duplicate resolution deterministic instead of quicksort-order dependent. |
| Zero-volume treatment | Treat a zero-volume row that moved as missing volume; carry forward the ticker's last positive volume, backward-filled at a series start. | A flat, unchanged zero-volume bar is a genuine no-trade day; a moved one is a dropped volume feed. |
| Trading-day minimum | Drop tickers with fewer than `MIN_TRADING_DAYS = 500` trading days. | Short histories cannot carry the longest feature windows. |

The zero-volume census ratio `ZERO_VOLUME_CENSUS_RATIO = 0.05` is not a drop rule: it only controls which per-ticker zero-volume profiles are logged. The survivors-only cross-section guard can be waived by calling `prepare_dataframe(..., survivors_only=False)`, which skips the check and logs a warning; the default is `survivors_only=True`.

## Feature families

Engineered features live under `packages/ml/dataset/features/`. Each module contributes one family, assembled by `assembly.py`.

| Family | Module | What it contributes |
|---|---|---|
| Prices | `prices.py` | Adjusted OHLC columns (`_adj_*`), `log_price`, `simple_return`, `log_return`, return lags, price-vs-SMA normalization, rolling mean and rolling extremes. |
| Volume | `volume.py` | Volume lags transformed to bounded scales, and the per-ticker volume z-score. |
| Trend | `trend.py` | EMA ratios, RSI-14, MACD, and the re-anchored `macd_signal`. |
| Volatility | `volatility.py` | Range-based estimators (ATR, Parkinson, Garman-Klass, Rogers-Satchell, gap) and the trailing volatility stack (`rv5_past`, `rv22`, `rv63`, `har5`, `har22`, `har63`, `volvol22`, `market_abs_ret_5d`). |
| Target | `target.py` | The Yang-Zhang 5-day `realized_volatility_5d`. |
| Regime | `regimes.py` | The hardcoded calendar `market_regime` label. |
| Assembly | `assembly.py` | Orders the build, enforces per-feature history-based NaN consistency, casts numerics, and reorders columns deterministically. |

The build order matters: adjusted OHLC columns come first, then base derived returns, then lags, transforms, trend, momentum, volume, normalization, rolling statistics, range volatility, and trailing volatility. `assembly.py` also enforces a per-feature warm-up with `MIN_HISTORY_BY_FEATURE`, which records the first row position (within a ticker) at which each feature is defined and blanks every column before its warm-up ends. A rolling window of `n` needs `n` prior rows, entries built on `log_return` start one row later because the first return is undefined, and lag entries need their lag count. This warm-up is the main source of the NaN rows the pipeline drops before writing.

The volatility family deserves one clarification. `rv5_past` is the mean absolute log return over five trailing days; `rv22` and `rv63` are trailing close-to-close standard deviations of `log_return`; `har5`/`har22`/`har63` are trailing sums of squared log returns over the same windows; `volvol22` is the volatility of volatility; and `market_abs_ret_5d` is a per-ticker trailing mean of the per-date cross-sectional mean absolute log return. The range estimators (ATR, Parkinson, Garman-Klass, Rogers-Satchell, gap) are window means over their own adjusted-OHLC bars, shifted one day.

## Model-input contract

The model consumes exactly 37 features, listed in `packages/ml/model/features/feature_engineering.py` under `INDICATORS`. The order is positional: `X = df[INDICATORS].values` fixes the column order, and feature-importance arrays are mapped onto `INDICATORS` by index. A module-level assertion fails at import if the list drifts from the pinned count of 37.

| Family | Count | Indicators |
|---|---|---|
| Prices | 13 | `simple_return`, `return_lag_1`, `return_lag_2`, `return_lag_3`, `return_lag_5`, `return_lag_10`, `price_vs_sma_10`, `price_vs_sma_20`, `price_vs_sma_50`, `zscore_price_vs_sma_20`, `rolling_mean_5`, `rolling_max_10`, `rolling_min_10` |
| Volume | 4 | `volume_lag_1`, `volume_lag_3`, `volume_lag_5`, `zscore_volume_20` |
| Trend | 5 | `ema_12`, `ema_26`, `rsi_14`, `macd`, `macd_hist` |
| Volatility | 15 | `atr5`, `atr20`, `park5`, `park20`, `gk5`, `rs5`, `gap5`, `rv5_past`, `rv22`, `rv63`, `har5`, `har22`, `har63`, `volvol22`, `market_abs_ret_5d` |

The remaining 16 dataset columns are extras and never enter `X`: `ticker`, `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `_adj_high`, `_adj_low`, `_adj_open`, `log_price`, `log_return`, `macd_signal`, `market_regime`, and `realized_volatility_5d`.

The first eight are raw or metadata columns. The adjusted OHLC fields feed feature and target construction. `log_price` and `log_return` are intermediate quantities. `macd_signal` stays as a dataset column but left the input list. `market_regime` is analysis metadata, and `realized_volatility_5d` is the target.

Two input columns are computed on per-date cross-sections before the dataset's dropna step and cannot be re-derived from the shipped parquet: `market_abs_ret_5d` and `volume_lag_5`. The model package records this exemption under `NON_RECONSTRUCTIBLE_COLUMNS`, so a fix for either routes to the dataset phase, not to the model.

The full 53-column dictionary, including Arrow types, roles, alignment, and warm-up positions, is in [../reference/dataset-schema.md](../reference/dataset-schema.md).

## Time alignment

Most price-derived features are shifted one day so a feature never sees its own bar. The same-bar exceptions, whose numerator is the current bar, are `simple_return`, `price_vs_sma_10`, `price_vs_sma_20`, `price_vs_sma_50`, `zscore_price_vs_sma_20`, and `zscore_volume_20`.

The target is the only deliberate forward-looking column. `log_return`, `macd_signal`, and `market_regime` remain dataset columns but are not model inputs. The full contract is [../adr/0007-time-alignment-contract.md](../adr/0007-time-alignment-contract.md).

## Target

The target is `realized_volatility_5d`: the Yang-Zhang daily volatility over the next five trading days, expressed as per-day sigma and not annualized. For each row it aggregates the forward window of the next five trading days:

```
variance = var(overnight) + k * var(open_to_close) + (1 - k) * mean(Rogers-Satchell)
k = 0.34 / (1.34 + (5 + 1) / (5 - 1))
realized_volatility_5d = sqrt(max(variance, 0))
```

The variance terms are sample variances (`ddof=1`) over the window, with `overnight = ln(adj_open_i / adj_close_{i-1})`, `open_to_close = ln(adj_close_i / adj_open_i)`, and the Rogers-Satchell range term. The stored value is a per-day sigma, not annualized; multiply by `sqrt(252)` to annualize.

The last five rows of every ticker are NaN because they lack a complete forward window. The volatility floor is applied later, in model training, not here; see [model.md](model.md). The rationale is [../adr/0006-yang-zhang-5day-target.md](../adr/0006-yang-zhang-5day-target.md).

## Regime label

`market_regime` is a hardcoded calendar label, not a data-derived state. Seven buckets cover the panel's years: `0` = 2000-2007, `1` = 2008-2010, `2` = 2011-2019, `3` = 2020-2021, `4` = 2022, `5` = 2023-2025, `6` = 2026. A year past 2026 raises rather than silently joining the last bucket, so later source data cannot be mislabelled; extend `regimes.py` before ingesting data past 2026. The label is analysis metadata, not a model input. See [../adr/0008-market-regime-calendar-label.md](../adr/0008-market-regime-calendar-label.md).

The label feeds `characterize_regimes`, which fingerprints each bucket before the NaN drop: volatility, return, and volume summary statistics, plus a pooled cross-ticker correlation estimated over trailing 60-day windows with a Geyer integrated-autocorrelation-time correction for the standard error. The characterization is descriptive metadata written to the regime JSON, not a model input.

## Outputs

| Path | Contents | Notes |
|---|---|---|
| `data/processed/dataset.parquet` | The cleaned, featured, NaN-dropped panel. | Current snapshot is `2,669,959` rows and `53` columns (37 inputs plus 16 extras); the `ticker` column is pinned to 32-bit Arrow `string`. |
| `data/processed/ingestion_manifest.json` | Source handle, file name, source SHA-256, raw rows, tickers, date range. | Written during ingestion, before any cleaning. |
| `packages/ml/model/artifacts/results/regime_characterization.json` | Per-regime volatility, return, volume, and cross-ticker correlation statistics, plus a `_frame` record. | `_frame` declares the survivors-only universe and records `pre_dropna_rows` and `shipped_rows`. |

The parquet is written before the regime JSON, so a failure in the JSON step cannot leave a characterization describing a dataset that was never written. All generated output under `data/` and `packages/ml/model/artifacts/` is gitignored.

The shipped frame is survivors-only: the source carries current and recent constituents only, so delisted names are absent and long-horizon return statistics are biased upward by roughly 1.4-3.9 percentage points per year (CAGR). The regime JSON's `_frame` record states this explicitly through its `label`, `survivors_only`, and `bias_note` fields, so any reader of the characterization sees the universe caveat next to the numbers.

## Failure modes

| Failure | Raised or logged |
|---|---|
| No CSV, multiple CSVs, or a CSV not named `SP500_Historical_Data.csv` | `FileNotFoundError` or `ValueError` from `_pick_csv_file`. |
| Close prices collapse onto fewer than 50 distinct two-decimal values | `ValueError` from `_check_price_grid`. |
| Missing or unexpected columns | `ValueError` from `_validate_required_columns`. |
| Empty ticker, unparseable date, or non-numeric price/volume | `ValueError` from `_cast_columns`. |
| Declining per-year ticker count on a survivors-only frame | `ValueError` from `_check_survivor_cross_section`. |
| Zero-volume treatment leaves missing volume | `ValueError` from `_treat_zero_volume_price_moves`. |
| Non-finite Yang-Zhang variance or non-finite feature values | `ValueError` from the feature build. |
| A year with no regime bucket (past 2026) | `ValueError` from `regimes.py`. |
| An interior ticker date gap (silent row drops) | `ValueError` from `check_contiguous_ticker_frames`. |
| Panel last date more than 30 days behind the run date | Warning from `check_panel_staleness`. |

The row reductions between the raw source and the shipped frame come from cleaning — the OHLC drop, deduplication, and the 500-trading-day filter. The exact per-step split is not persisted, and the 1,211-row difference between the ingestion census and the regime JSON's `_frame.pre_dropna_rows` is cleaning attrition, not an unresolved source discrepancy.

## Verification

Run Phase 1 and confirm it completes with these log lines: `Loading raw dataset...`, `Prepared: ...`, `Featured: ...`, `Dropped ... NaN rows`, `Dataset saved to data/processed/dataset.parquet`, and `Regime stats saved to ...`. The parquet save precedes the regime-JSON save in the log.

```bash
python packages/ml/dataset/main.py
```

This needs network and Kaggle credentials and is not part of the model test suite. The CI `dataset-smoke` job exercises the dataset path, but the model static job does not run Phase 1. A fresh clone has no parquet until Phase 1 runs once. Do not run the holdout to verify this phase; the holdout is a model-side gate documented in [../operations.md](../operations.md).
