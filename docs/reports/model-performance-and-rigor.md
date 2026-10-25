# Model performance and rigor: a five-day S&P 500 constituent volatility model

**Technical report for machine-learning engineers, quantitative researchers, and technical auditors**

> **Key result.**
>
> - **Headline.** Lower QLIKE is better. In the locked 500-date holdout, model QLIKE is 0.068804 vs 0.094570 for HAR; the edge sign is `HAR − model`, the holdout edge is 0.025766, and QLIKE is 27.25% lower than HAR’s QLIKE. The equal-weight CV edge is 0.049189 (45.68%).
> - **Volatility pattern.** Edges are positive in within-window realized-volatility deciles 0–5 (0 = lowest, 9 = highest) and negative in deciles 6–9. Top-1% model vs HAR mean losses are 1.3376 vs 0.9714 in CV and 1.5518 vs 1.2425 in holdout; the holdout top-1% loss share is 22.6% vs 13.1%, though the share is confounded by lower total loss.
> - **Limits.** Calibration is rejected by MZ and a calibration layer is untested; tail safety is not established. Seed robustness, crisis robustness, economic value, and non-HAR superiority are not tested. See §11 (Limitations and falsification).

**Purpose.** This report documents what was built, on what data, under which protocol, and what the evidence does and does not support. It is a read-only synthesis of the evaluated artifacts; the locked holdout was not re-evaluated.

**Status.** Issued 2026-10-25. The evaluated model and its holdout report are unchanged from the 2026-10-20 evaluation. This branch adds only this documentation file; the test-only CI fix is already in `develop`, and the production subtree is identical to the evaluated tree.

**Contents.** [0 Provenance](#0-provenance) · [1 Executive summary](#1-executive-summary-and-claim-card) · [2 Question and target](#2-question-target-and-intended-use) · [3 Data](#3-data-and-universe) · [4 Protocol](#4-protocol-and-design) · [5 Model and tuning](#5-model-and-tuning) · [6 Cross-validation](#6-cross-validation-performance) · [7 Holdout](#7-holdout-performance) · [8 Robustness](#8-robustness-and-diagnostic-limits) · [9 Construction](#9-construction-robustness) · [10 Engineering](#10-engineering-discipline) · [11 Limitations](#11-limitations-and-falsification) · [12 Conclusion](#12-conclusion) · [Appendices A–E](#appendix-a-formulas-and-definitions).

## 0. Provenance

- **Evaluated model commit:** `df5aa5a6ee45cdef8361a8acd7c2a6c6876c092b`.
- **Publication merge:** `466d76dd283c9abdbd8a3790fb6e467e6efa8c8b` (2026-10-22); published tree `718b04fdb500de3145ace1f4a7a7a3a09400c3a8`, identical to the evaluated tree.
- **Base develop commit:** `178e74c69e33eacb2fa299fa0e085146e92f5b7a` (test-only CI fix merge; production subtree identical to the evaluated tree).
- **Report commit:** the tip of branch `docs/model-performance-report` (the commit that adds this file; SHA in `git log`).
- **Holdout report SHA-256:** `45118752fd6dcb5de1f8a816270931d2bb453ba3d434107d842dde9fb81b82d5`.
- **Seeds:** base/campaign `42`; selected model seed `43` (`random_state=43`).
- **Dates:** holdout evaluated `2026-10-20`; report generated `2026-10-20T06:58:05.097434+00:00`; lock recorded `2026-10-20T06:52:01.673063+00:00`; this report `2026-10-25`.

Full lock, dataset, fold, config, run-id, and environment provenance is in Appendix C.

> **Conventions**
>
> - **Signs:** headline edge = `HAR − model` (positive = model better); DM differential = `model − HAR` (negative = model better); floor artifact and expanding-origin edge = `model − HAR`.
> - **Units:** dimensionless QLIKE computed on per-day sigma forecasts; inputs are per-day sigma, not annualized.
> - **Precision:** Tables 1, 3–8 round continuous values to six decimals; counts are exact; derived values are rounded to six decimals; exact values remain in the cited artifacts.
> - **Weighting:** equal-weight = each fold one vote; pooled = each row one vote.
> - **Abbreviations:** DM Diebold–Mariano; GW Giacomini–White; MZ Mincer–Zarnowitz; HAR heterogeneous autoregressive; OOS out-of-sample; MSPE mean squared prediction error; HAC heteroskedasticity/autocorrelation-consistent.
> - **Tags:** `[V]` verified this session by file, hash, or command; `[R]` recorded evidence not re-executed; `[D]` derived from fact-sheet values.

## 1. Executive summary and claim card

The model is a per-row XGBoost regressor that forecasts the next five trading days' average daily volatility (per-day sigma) for S&P 500 constituents. It uses 37 price/volume features and no ticker identity, and it is compared with a canonical HAR-RV benchmark on the same floor-retained rows.

Two definitions come first. **QLIKE** is `h/f − ln(h/f) − 1`, where `h` is realized per-day sigma and `f` is the forecast. It is zero at a perfect forecast, positive otherwise, and penalizes underprediction more heavily. The **headline edge** is `HAR QLIKE − model QLIKE`; positive means the model has lower loss. **Relative reduction** is edge divided by HAR QLIKE.

**Table 1. Results at a glance**

| Evaluation | n | Model QLIKE | HAR QLIKE | Edge (HAR − model) | Relative reduction | Source |
|---|---:|---:|---:|---:|---:|---|
| CV, equal-weight mean of five folds | 5 fold means (2,094,049 rows) | 0.058485 | 0.107674 | **0.049189** | **45.683348%** | `evaluation_summary.json#headline.aggregate` |
| CV, pooled and row-weighted | 2,094,049 rows | 0.058520 | 0.107151 | 0.048631 | 45.385840% | `evaluation_summary.json#headline.pooled` |
| Locked holdout, one 500-date block | 234,473 rows | 0.068804 | 0.094570 | **0.025766** | **27.245124%** | `holdout_report.json#headline.aggregate` |

*Table 1 notes — n:* as shown; *row set:* floor-retained rows (raw RV strictly above `1e-12`); *scale:* dimensionless QLIKE computed on per-day sigma forecasts; inputs are per-day sigma, not annualized; *weighting:* fold-equal, row-equal, row-equal; *source:* `packages/ml/model/artifacts/results/` keys as listed; continuous values rounded to six decimals, counts exact.

> **Limits.** **Single seed** (selected 43; base 42; no sweep). **One-shot holdout** (500 dates once; no second window). **Calibration rejected by MZ** (no calibration layer applied). **Floor policy** (at-or-below-`1e-12` targets excluded; including them slightly reduces the edge). **HAR only** (no GARCH, alternative ML, ensemble, or implied vol). **Survivor-only universe** (delisted names absent; long-horizon returns biased upward by roughly 1.4–3.9 pp/year CAGR `[R]`). **Fold 2 at the cap** (499 of 500; convergence not demonstrated). **No economic test** (no costs, P&L, VaR, or ES).

- **CV.** All five fold edges are positive; equal-weight `0.049189`, pooled `0.048631`, fold range `0.020528` to `0.077302`. Pooled DM `-28.754226`, p `1.6e-181`.
- **Holdout.** Edge `0.025766` (`27.25%`), DM `-7.301276`, HAC SE `0.003529`, p `5.7e-13`; derived 95% CI `[0.018849, 0.032683]`.
- **Tails.** Better in within-window deciles 0–5, worse in 6–9. Top-1% mean loss is `1.3376` vs `0.9714` (CV) and `1.5518` vs `1.2425` (holdout); the top-1% share (`0.225561` vs `0.131400` holdout) can be confounded by lower total loss. Average gain does not establish tail safety.

## 2. Question, target, and intended use

The question is whether the model achieves lower average QLIKE than HAR under this protocol, and where the difference is concentrated across the volatility distribution.

The unit is a ticker-date row. Let `t` be the row/date index, `i` the ticker, `w = {t+1, …, t+5}` the forward five-trading-day window, and `d ∈ w`. At the close of day `t`, the model observes 37 price/volume features for ticker `i` and forecasts the Yang–Zhang 5-day per-day sigma for that ticker over `w`, not annualized. The target is

```
variance_{i,t} = Var_w({overnight_{i,d}}) + k * Var_w({open_to_close_{i,d}})
                 + (1-k) * Mean_w(RS_{i,d})
realized_volatility_5d_{i,t} = sqrt(max(variance_{i,t}, 0))
k = 0.34 / (1.34 + (5+1)/(5-1))
```

`Var_w` is the sample variance (`ddof=1`) over `w`; `Mean_w` is the mean over `w`; `overnight_{i,d} = ln(open_{i,d}/close_{i,d−1})`, `open_to_close_{i,d} = ln(close_{i,d}/open_{i,d})`, and `RS_{i,d}` is the Rogers–Satchell range term. The model fits `log(realized_volatility_5d_{i,t})` and exponentiates forecasts back to sigma before QLIKE. The estimand is the conditional per-day sigma under QLIKE; the report does not claim that the exponentiated forecast is the conditional mean or a full distribution. Consecutive rows have overlapping five-day labels. That is not leakage—each forecast uses only information at its own date—but it induces serial dependence, handled by a train-only purge and HAC lag 4.

The 37 features cover return lags, adjusted OHLC ranges and gaps, trailing volatility and HAR components, price location and z-scores, rolling means/extremes, EMA/MACD/RSI momentum, volume transforms, and a market absolute-return term. There is no ticker, date, sector, or `market_regime` input; the model is pooled across names. Intended use is research and benchmarking: comparing specifications under a fixed protocol and quantifying the average difference from HAR. It is not for production risk, trading, or causal use.

## 3. Data and universe

The source is the Kaggle dataset `jacksaleeby/s-and-p500-historical-data/versions/1`, file `SP500_Historical_Data.csv`, SHA-256 `82f3ab3ab6821c54b67a555c1f52c60dc4c162923499f2a1403c51d08944722a`. Ingestion records `2,703,531` rows, `472` tickers, and `2000-01-03` to `2026-02-20`. The processed dataset has `2,669,959` rows, `469` tickers, `6,504` dates (`2000-04-04` to `2026-02-12`), and `53` columns: 37 model indicators plus 16 extras, including adjusted-price fields (`adj_close`, `_adj_*`) and note/metadata columns. The extras never enter `X` and are not enumerated completely here; input checks show zero X/Y NaNs, infinities, and duplicate pairs.

**Table 2. Sample accounting**

| Stage | Rows | Tickers / dates | Note | Source |
|---|---:|---|---|---|
| Raw source, ingestion census | 2,703,531 | 472 tickers; 2000-01-03 to 2026-02-20 | source CSV | `ingestion_manifest.json#rows`, `#tickers` |
| Pre-drop frame census | 2,702,320 | — | differs from ingestion by 1,211; unresolved | `regime_characterization.json#_frame.pre_dropna_rows` |
| Shipped dataset | 2,669,959 | 469 tickers; 6,504 dates; 2000-04-04 to 2026-02-12 | 53 columns (37 + 16) | `metadata.pkl#dataset` |
| At-floor / zero-RV | 2,620 | — | rate `0.000981`; floor `1e-12` | `metadata.pkl#input_checks.degenerate_count`, `#degenerate_rate` |
| CV test blocks | 2,095,971 | 5 folds | all test rows | `evaluation_summary.json#floor_decomposition.counts` |
| CV scored | 2,094,049 | 5 folds | floor-retained | `evaluation_summary.json#headline.pooled.n_rows` |
| Holdout block | 234,488 | 500 dates; 2024-02-15 to 2026-02-12 | all reserved rows | `holdout_report.json#holdout.n_rows` |
| Holdout scored | 234,473 | 500 dates | 15 floor-excluded | `holdout_report.json#holdout.n_scored_rows` |
| Fold fit rows (sum) | 4,777,450 | 5 folds | post-inner-val/purge/floor | `fold_training_summary.json#fold_i.n_train_samples` [D] |
| Inner validation (sum) | 844,997 | 5 folds | last 15% of each tail | `fold_training_summary.json#fold_i.n_inner_val_samples` [D] |
| Inner purge (sum) | 10,868 | 5 folds | five-day purge | `fold_training_summary.json#fold_i.n_inner_purge_rows` [D] |

*Table 2 notes — n:* rows as shown; *row set:* data-accounting stages; *scale:* rows, except the dimensionless floor rate; *weighting:* not applicable; *source:* `data/processed/` and `packages/ml/model/artifacts/` keys as listed

**Survivor bias.** The source is current-and-recent constituents, not point-in-time membership. The frame records `survivors_only: true`, `pre_dropna_rows=2,702,320`, `shipped_rows=2,669,959`, and an upward long-horizon return bias of roughly `1.4–3.9` pp/year (CAGR) `[R]`. Delisted failures are absent and the cross-section grows mechanically. Crisis and average-universe claims are conditional on this frame.

**Raw census discrepancy.** Ingestion says `2,703,531`; the pre-drop frame says `2,702,320`; difference `1,211`; both cite the same source SHA. Shipped rows are `2,669,959` in both accounts, so implied drops are `33,572` versus `32,361`. The difference is unresolved; model rows are pinned by the dataset and input hashes.

**Floor census.** There are `2,620` total floor rows: `1,922` in CV test blocks and `698` never evaluated. Per-fold test-block rows are `7 / 581 / 733 / 557 / 44`; holdout `15` of `234,488`. A pre-remediation split is listed in Appendix D; the total is unchanged.

## 4. Protocol and design

The design is date-group, expanding-window cross-validation with a train-only purge and a separately reserved holdout. Six contiguous date blocks yield five folds; each trains on every earlier block and scores only later rows. The final 500 dates are removed before fold construction.

**Folds and purge.** Fold test starts are 0-based indices `1000`, `2001`, `3002`, `4002`, `5003`; fold dates total `6,004` (`6,504` processed minus `500` holdout). Test ranges are in Table 4. The purge is `5` training dates; the last kept train date to first test date is `6` index units, i.e. five complete excluded dates. Observed last/first pairs are `994/1000`, `1995/2001`, `2996/3002`, `3996/4002`, `4997/5003`. Test blocks are kept whole; there is no shuffle.

The purge follows from the five-day label: a training row at `t` reaching `t+5` can overlap the test block and carry test-period outcome information. Dropping the last five training dates removes it. Test rows are not purged: each forecast is made at its own date, and overlap among test labels creates serial correlation, not leakage; HAC lag 4 addresses it.

**Inner validation.** The last `15%` of each training tail is an inner validation view, separated by a `5`-day inner purge. Inner-validation rows are `50,560 / 105,705 / 165,060 / 228,548 / 295,124`; inner-purge rows are `2,035 / 1,990 / 2,281 / 2,297 / 2,265`. The final holdout refit has no validation tail and no early stopping.

**Holdout and refit.** The holdout is `2024-02-15` to `2026-02-12` (500 dates), reserved before folds and never used by tuning or folds. The last fit date is `2024-02-07`; purge `5`; separation `6`. The final refit uses `2,430,521` rows and `2,345` purged rows.

**HAR.** Canonical HAR-RV: `1/5/22` trailing variance components, OLS in variance levels, training-only fit, `21`-date per-ticker warm-up, forecasts mapped to log and floored, no clip. The fold fit-row difference from XGBoost is protocol, not a different test set: fold 0 `329,731` versus `284,472`, difference `45,259 = 50,560 + 2,035 − 7,336`.

**Floor, HAC, lock.** The floor is a single strict `1e-12`; at-or-below-floor rows are excluded from fitting, model scoring, and HAR scoring. The scored row set is floor-retained; no forecast clip fired (`clip_count=0`, `clip_rate=0.0`, `folds_with_clips=[]`). HAC lag is `4`. The lock is written before any holdout metric (`lock_before_metrics=true`), with `overrides=[]`; a second evaluation is refused, and a post-lock resume records a skip with `0` trainer starts. No force was used; no fold was quarantined (`quarantined_folds={}`).

```
2000-04-04                                                         2024-02-14
    |---- fold 0 train ----|== test 0 ==|                                  |
    |------ fold 1 train ------|== test 1 ==|                              |
    |-------- fold 2 train --------|== test 2 ==|                          |
    |---------- fold 3 train ----------|== test 3 ==|                      |
    |------------ fold 4 train ------------|== test 4 ==|                  |
                                                      last fit 2024-02-07  |
                                                                  purge  |
                                                      holdout start 2024-02-15
                                                      |===== locked holdout =====|
                                                                  2026-02-12
```

## 5. Model and tuning

The model is one XGBoost regressor per fold, trained on `log(Y)` with a custom QLIKE objective and QLIKE evaluation metric. The selected configuration is `learning_rate=0.03`, `max_depth=8`, `min_child_weight=25`, `gamma=0.1`, `n_estimators=500`, `subsample=0.8`, `colsample_bytree=0.8`, `colsample_bylevel=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `random_state=43`, `tree_method=hist`, artifact `n_jobs=-1` (effective `1` under the determinism policy).

**Table 3. Tuning stages**

| Stage | Trials | Best validation QLIKE | Best trial | Source |
|---|---:|---:|---:|---|
| Coarse grid | 27 | 0.059303 | 8 | `selection.json#stages.coarse.*` |
| Finalist refinement | 2 | 0.059255 | 28 | `selection.json#stages.refinement.*` |
| Seed finalists | 3 | 0.059173 | 30 | `selection.json#stages.seed_finalists.*` |
| Total | 32 | 0.059173 | 30 | `selection.json#n_trials`, `#selected_trial.validation_qlike` |

*Table 3 notes — n:* 32 trials (27+2+3); *row set:* inner-validation tails of the five folds, floor-retained; *scale:* dimensionless QLIKE computed on per-day sigma forecasts; *weighting:* mean of five per-fold validation QLIKEs, fold-equal as recorded; *source:* `tuning/20261019T100655Z-5e822810/selection.json`; values rounded to six decimals.

The selected trial is 30, stage `seed_finalists`, seed `43`, mean validation QLIKE `0.059173`. The gain from coarse best to selected is `0.000130` absolute (`0.219%` relative); worst trial `0.062012` → selected is `0.002839` (`4.58%`). The refinement→seed gain is `0.000082` (`0.138%`). The three seed finalists recorded `0.059255 / 0.059173 / 0.059320`; the max−min spread is `0.000147` (about `0.25%` of the selected value). The tuning search is ruled out as the sole source of the gap to HAR, but the decomposition among model class, features, and tuning is not identified.

The base/campaign seed is `42`; coarse and refinement ran at it. Seed finalists ran `42`, `43`, `44`; the selected model seed is `43`, and `hyperparameters.random_state=43`. The lock payload's `selection.seed=42` is the campaign seed, not the model seed. The finalist stage is selection on validation QLIKE, not a seed sweep; `results/seed_sweep.json` is absent.

Fold 2 selected best iteration `499` of `500`; other folds selected `117`, `81`, `134`, `447`. The validation minimum is at the cap, so convergence is not demonstrated. The final refit also uses 500 rounds without early stopping. Fold 2 has the largest CV edge (`0.077302`), so the cap could affect its level and the cross-fold comparison. Selection config `5e822810` is recorded but not independently recomputable; frozen config `407a2ad5` is verified.

## 6. Cross-validation performance

**Metrics.** QLIKE and edge are defined in §1. **Dual OOS R²:** `1 − SSE_model/SSE_mean` and `1 − SSE_model/SSE_HAR` on the log target; `SSE_mean` uses fold-own means; `test_r2`/`har_r2` keep sklearn's convention. **DM** tests the `model − HAR` differential (negative = lower model loss, two-sided); **HAC** is Newey–West with `L=4`. **CW** is the Clark–West adjusted MSPE diagnostic; nesting is not claimed. **GW** is a Wald χ²(2) on the date-mean differential with a constant and one lag, null `E[d_t|I_{t−1}]=0`. **MZ** is `y_true = α + β y_pred + e` on the log target with HAC SEs; null `α=0, β=1`.

**Table 4. Per-fold performance and dual R²**

| Fold | Test dates | n scored | Model QLIKE | HAR QLIKE | Edge (HAR − model) | Model R² | HAR R² | OOS R² vs HAR | Best iter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2004-03-30 to 2008-03-20 | 367,810 | 0.057833 | 0.125721 | 0.067888 | 0.555690 | -0.314936 | 0.662105 | 117 |
| 1 | 2008-03-24 to 2012-03-09 | 395,811 | 0.054991 | 0.075519 | 0.020528 | 0.681642 | 0.509239 | 0.351297 | 81 |
| 2 | 2012-03-12 to 2016-03-02 | 423,389 | 0.060505 | 0.137807 | 0.077302 | 0.472410 | -0.642068 | 0.678704 | 499 |
| 3 | 2016-03-03 to 2020-02-24 | 443,943 | 0.066046 | 0.125609 | 0.059564 | 0.351841 | -0.462956 | 0.556953 | 134 |
| 4 | 2020-02-25 to 2024-02-14 | 463,096 | 0.053051 | 0.073716 | 0.020665 | 0.593750 | 0.350199 | 0.374809 | 447 |
| Equal-weight | five folds | 2,094,049 | 0.058485 | 0.107674 | 0.049189 | 0.531067 | -0.112104 | — | — |
| Pooled | five folds | 2,094,049 | 0.058520 | 0.107151 | 0.048631 | 0.593511 | — | 0.562568 | — |

*Table 4 notes — n:* scored rows per row; *row set:* floor-retained CV test rows; *scale:* dimensionless QLIKE computed on per-day sigma forecasts; R² log target; *weighting:* row within fold; fold-equal for the equal-weight row; row-weighted for pooled; *source:* `evaluation_summary.json#headline.folds.*`, `#dual_oos_r2.*`, `#headline.pooled`; `har_benchmark.json#folds.fold_i.har_r2`; `fold_training_summary.json#fold_i.best_iteration`; continuous values rounded to six decimals, counts exact.

All five fold edges are positive, with substantial dispersion (`0.0205–0.0773`); no leave-one-out contribution analysis is reported. Model R² is positive in every fold (`0.351841` to `0.681642`; mean `0.531067`); HAR's is positive only in folds 1 and 4 (mean `-0.112104`). A negative R² means HAR's squared log error exceeds the fold mean's; it does not make HAR a weak QLIKE benchmark, because the losses differ. Model `oos_r2_vs_har` is positive in every fold (`0.351297` to `0.678704`; pooled `0.562568`).

Equal-weight gives each fold one vote; pooled gives each row one vote. Later folds are larger; HAR QLIKE varies by fold and does not decline monotonically, so pooling and fold-equal weighting differ. The pooled edge is smaller by `0.000558` [D].

**Inference.** Pooled DM is `-28.754226` (p `1.6e-181`); fold DM values are all negative, from `-20.957449` to `-8.454065`. CW pooled is `44.799426` (p `0.0`); nesting is not claimed. GW pooled is `21913.099581` on χ²(2) (p `0.0`). Pooled MZ gives α `-0.261560` (t `-59.813106`) and β `0.950368` (t(β=0) `898.911`; t(β−1) `-46.944`), `n_obs=2,094,049`; calibration is rejected. DM is computed on the date-mean differential (`model − HAR`), so it is date-equal; HAC covers serial and heteroskedastic dependence. The pooled edge is row-weighted; the difference is weighting, not a missing cross-sectional correction.

**Date-clustered gap and signs.** The date-equal DM mean differential is `−0.049287` (`model − HAR`; descriptive date-clustered SE `0.000845`, not the DM HAC SE; `5,004` dates) and the row-weighted pooled edge is `+0.048631` (`HAR − model`); they differ by `0.000656` in magnitude and `0.097918` when signed because the conventions are opposite and the aggregations differ (date-equal versus row-weighted).

**Table 5. Floor decomposition**

| Row set | n | Model QLIKE | HAR QLIKE | Artifact edge (model − HAR) | HAR − model edge | Source |
|---|---:|---:|---:|---:|---:|---|
| Pooled floor-retained (row-weighted) | 2,094,049 | 0.058520 | 0.107151 | -0.048631 | +0.048631 | `evaluation_summary.json#floor_decomposition.pooled` |
| All test rows including floor | 2,095,971 | 0.079386 | 0.127928 | -0.048542 | +0.048542 | same |
| Change | +1,922 | +0.020867 [D] | +0.020777 [D] | +0.000089 | -0.000089 | `#floor_decomposition.pooled.edge_change` |

*Table 5 notes — n:* scored rows; all-test row adds `1,922` at-or-below-floor test rows; *row set:* pooled CV test blocks, floor-retained vs all; *scale:* dimensionless QLIKE computed on per-day sigma forecasts; *weighting:* one observation per row; *source:* `evaluation_summary.json#floor_decomposition.counts`, `#floor_decomposition.pooled`; continuous values rounded to six decimals, counts exact.

Including floor rows raises both QLIKE levels by about `0.0209` and `0.0208`; the advantage shrinks by `0.000089`, from `0.048631` to `0.048542`. Per-fold floor rows are `7 / 581 / 733 / 557 / 44`; the largest per-fold `edge_change` is fold 1 (`0.000589` artifact convention), still far below its edge. Floor rows are flat windows, not random; the ranking is stable, but levels are not policy-neutral.

**Loss distribution.** CV medians are `0.020065` (model) and `0.066148` (HAR), with IQRs `0.052083` and `0.134624`; holdout medians are `0.020728` and `0.055232`, with IQRs `0.055774` and `0.106576`. The model improves the center more than the tail. In the holdout, the top-1% loss share is higher (22.6% vs 13.1%), though the share is confounded by lower total loss; the top-1% mean loss is also higher (1.55 vs 1.24). Tail safety is not established.

**Table 6. QLIKE by realized-volatility decile**

| Decile | CV n | CV model | CV HAR | CV edge (HAR − model) | HO n | HO model | HO HAR | HO edge (HAR − model) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 209,405 | 0.072146 | 0.311775 | +0.239629 | 23,448 | 0.079986 | 0.225095 | +0.145109 |
| 1 | 209,405 | 0.037614 | 0.187332 | +0.149718 | 23,447 | 0.040072 | 0.131695 | +0.091623 |
| 2 | 209,405 | 0.030073 | 0.133216 | +0.103143 | 23,447 | 0.029784 | 0.096216 | +0.066433 |
| 3 | 209,405 | 0.027018 | 0.095162 | +0.068144 | 23,447 | 0.025295 | 0.071887 | +0.046592 |
| 4 | 209,404 | 0.026602 | 0.066184 | +0.039582 | 23,447 | 0.022602 | 0.052997 | +0.030394 |
| 5 | 209,405 | 0.028599 | 0.043938 | +0.015340 | 23,448 | 0.023961 | 0.038346 | +0.014385 |
| 6 | 209,405 | 0.033614 | 0.027469 | -0.006145 | 23,447 | 0.028102 | 0.027301 | -0.000801 |
| 7 | 209,405 | 0.044097 | 0.019183 | -0.024915 | 23,447 | 0.040172 | 0.022410 | -0.017763 |
| 8 | 209,405 | 0.070492 | 0.027337 | -0.043155 | 23,447 | 0.072883 | 0.034149 | -0.038734 |
| 9 | 209,405 | 0.214940 | 0.159912 | -0.055028 | 23,448 | 0.325176 | 0.245596 | -0.079580 |

*Table 6 notes — n:* rows per decile; CV total `2,094,049`, holdout `234,473`; *row set:* floor-retained pooled CV and holdout; *scale:* dimensionless QLIKE computed on per-day sigma forecasts; *weighting:* row within decile; deciles are equal-count bins of realized log sigma, formed on pooled CV rows for CV and within holdout for HO; *source:* `evaluation_summary.json#decile_qlike.pooled.bins`; `holdout_report.json#holdout_metrics.decile_qlike.bins`; edges [D]; continuous values rounded to six decimals, counts exact.

Deciles are computed within each evaluation window. The pattern is the same in both windows: better in 0–5, worse in 6–9, crossover between 5 and 6. Holdout decile 6 is near zero but still negative (`-0.000801`); decile 9 is worst (`-0.079580` holdout, `-0.055028` CV). Decile 0 has the largest gain; deciles 0–5 drive the positive average. No multiplicity-adjusted inference is attached to individual bins.

**Table 7. QLIKE by calendar regime**

| Regime | CV n | CV model | CV HAR | CV edge (HAR − model) | HO n | HO model | HO HAR | HO edge (HAR − model) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 346,686 | 0.057388 | 0.128559 | +0.071171 | 0 | — | — | — |
| 1 | 295,724 | 0.057289 | 0.073200 | +0.015911 | 0 | — | — | — |
| 2 | 972,164 | 0.060799 | 0.125586 | +0.064787 | 0 | — | — | — |
| 3 | 231,761 | 0.064237 | 0.082568 | +0.018331 | 0 | — | — | — |
| 4 | 116,635 | 0.042239 | 0.050838 | +0.008599 | 0 | — | — | — |
| 5 | 131,079 | 0.051756 | 0.083967 | +0.032212 | 220,884 | 0.068197 | 0.095674 | +0.027477 |
| 6 | 0 | — | — | — | 13,589 | 0.078678 | 0.076632 | -0.002046 |

*Table 7 notes — n:* rows per regime; CV total `2,094,049`, holdout `234,473`; *row set:* floor-retained pooled CV and holdout; *scale:* dimensionless QLIKE computed on per-day sigma forecasts; *weighting:* row within regime; labels are hardcoded calendar buckets `0..6`, not estimated volatility states; calendar mapping: 0=2000–2007, 1=2008–2010, 2=2011–2019, 3=2020–2021, 4=2022, 5=2023–2025, 6=2026; CV has no bucket 6 and holdout spans 5–6; *source:* `evaluation_summary.json#regime_qlike.pooled.bins`, `#regime_qlike.labels`; `holdout_report.json#holdout_metrics.regime_qlike`; edges [D]; continuous values rounded to six decimals, counts exact.

The regime table is descriptive: hardcoded calendar buckets 0–6, not volatility states; the holdout spans 5–6. It is not used as high-volatility or tail evidence.

**Expanding origin.** Eleven checkpoints per fold; edge is `model − HAR`, so negative is better. Folds 0, 2, 3, and 4 are negative at every checkpoint; fold 1 starts positive (`+0.022035`) and ends negative (`-0.019939`). This is a path-dependence diagnostic, not a separate test.

## 7. Holdout performance

The holdout is `2024-02-15` to `2026-02-12`, `500` dates, `234,488` rows, `234,473` scored, `15` floor-excluded. Model QLIKE is `0.068804`, HAR `0.094570`, edge `0.025766` (`27.25%` relative). Model R² `0.426642`; HAR R² `0.037529`; `oos_r2_prevailing_mean` `0.426642`; `oos_r2_vs_har` `0.404286`; MAE `0.260997`; RMSE `0.349028` (all log target except QLIKE).

**Table 8. Holdout inference**

| Quantity | Estimate | Standard error / note | Test / interval | Convention | Source |
|---|---:|---|---|---|---|
| Edge | 0.025766 | 0.003529 (HAC SE of mean differential, lag 4) | 95% CI `[0.018849, 0.032683]` [D] | HAR − model | `holdout_report.json#holdout_metrics.edge`, `#dm.hac_se` |
| DM | -7.301276 | 0.003529 (HAC SE of mean differential) | two-sided p `5.7e-13` | model − HAR, date-mean | `#holdout_metrics.dm` |
| DM mean | -0.025766 | — | — | model − HAR | `#dm.mean_differential` |
| MZ α | -0.306146 | 0.017005 | t `-18.003804`; p `1.818865767245556e-72` | log target, OLS/HAC | `#mz` |
| MZ β | 0.933458 | 0.004130 | t(β=0) `226.042218` (p `0.0`); t(β−1) `-16.114` | log target, OLS/HAC | `#mz` |
| CW | 17.014853 | — | one-sided p `3.1866457467402745e-65` | adjusted squared error (MSPE); nesting not claimed | `#cw` |
| GW | 1037.668167 | — | χ²(2); p `4.7121647770872576e-226` | date-mean differential, constant + one lag | `#gw` |

*Table 8 notes — n:* `234,473` rows and `500` dates; *row set:* floor-retained holdout; *scale:* dimensionless QLIKE on per-day sigma forecasts for model/HAR, edge, and DM; adjusted squared error (MSPE) for CW; loss differential for GW; log target for MZ; *weighting:* row for edge and MZ, date-equal for DM/CW/GW; *source:* `holdout_report.json#holdout_metrics.*`; CI derived as `edge ± 1.959963984540054 × DM HAC SE` (no CI field exists); continuous values rounded to six decimals, counts exact; DM p-value rounded, other p-values as emitted, exact values in the artifact; MZ t for β is under H0 β=0, and t(β−1) tests β=1. MZ `t(β−1)` values are shown to three decimal places; exact values are in the artifact.

DM is negative and far below conventional thresholds; the interval is entirely above zero, conditional on the fitted model and HAC specification. It excludes seed, model-selection, and window-to-window uncertainty; the holdout was evaluated once, so no empirical window distribution exists. MZ rejects calibration (α t `-18.004`; β−1 t `-16.114`). CW is an adjusted squared-error (MSPE) diagnostic, not a QLIKE test; GW tests the date-mean loss differential. They point in the same direction but use different losses and are not independent confirmations of one QLIKE differential. The holdout DM mean differential is a near-identity to the row-weighted holdout edge at displayed precision (the exact gap is `6.5e-7`); it is not an additional test.

**Refit and benchmark.** Final refit: `2,430,521` rows, `2,345` purged, no early stopping, no validation tail. The HAR fit uses `2,420,697` rows and excludes `9,824` warm-up rows; it scores `234,473` test rows, `0` warm-up test rows, `0` clips. Timing: `352.5` s refit, `3.9` s inference, `0.7` s HAR; these sum to about `357.1` s, while the reported total is `363.5` s, including overhead. The lock was recorded before the report, with `overrides=[]`, `post_hoc_adjustment.applied=false`, `SHA256SUMS` `4/4`, and a post-lock resume recording a skip, `0` trainer starts, and an unchanged report.

**Comparison with CV.** The holdout edge is smaller: relative reduction `27.25%` versus `45.68%` equal-weight and `45.39%` pooled. The model's absolute QLIKE is worse (`0.068804` vs `0.058485`/`0.058520`); HAR's is better (`0.094570` vs `0.107674`/`0.107151`). The holdout refit uses a fixed 500 rounds while CV best iterations range 81–499. CV test blocks include 2008 and March 2020, but the design is not a clean crisis stress test: training windows expand and overlap; test blocks are disjoint and multi-year; the holdout covers 2024–2026. Crisis behavior is therefore not separately identified; the smaller holdout edge is consistent with a different window and the fixed 500-round refit.

## 8. Robustness and diagnostic limits

**Floor.** Table 5 gives the floor sensitivity: ranking is unchanged, levels are policy-dependent, and no alternative floor or imputation was tested.

**Loss distribution.** The cross-validation loss-distribution discussion shows the center improves and the tail worsens. Model/HAR median ratios are `0.30` (CV) and `0.38` (holdout); IQR ratios `0.39` and `0.52`. Lower mean QLIKE does not imply lower probability of a large loss, lower expected shortfall, or establish tail safety; no VaR/ES test was run.

**Deciles.** Table 6 shows positive edges in 0–5, negative edges in 6–9, and the worst result in 9; deciles 0–5 drive the average. No multiplicity correction or tail-weighted loss was run.

**Calibration.** MZ rejects calibration; no calibration layer or predictive distribution ships.

**Expanding origin.** The cross-validation expanding-origin discussion shows fold 1 starting positive and ending negative; this is path dependence in one period, not a separate test.

## 9. Construction robustness

**Table 9. Threat–control–evidence matrix**

| Threat | Control | Evidence |
|---|---|---|
| Future information in features | Default one-day shift; four documented same-bar families (six columns); target is the only forward-looking column; no ticker identity | `metadata.pkl#n_indicators=37`, `#n_ticker_features=0`; zero X/Y NaN/inf/duplicate pairs; ADR 0007 |
| Label overlap across folds | Train-only 5-date purge; separation 6; test blocks whole; fold guard in unique-date index space | `folds_metadata.pkl#fold_i.purge_train_count`, `#fold_i.separation_units`; last/first `994/1000` … `4997/5003`; inner purge `2,035…2,265` |
| Test-set tuning | Tuning uses inner-validation tails only; test and holdout never enter selection | `selection.json#trials[*].folds.fold_i.validation_qlike`; `one_shot`; `lock_before_metrics=true` |
| HAR not comparable | Training-only fit; 21-date warm-up; same scored rows; no forecast clip | `har_benchmark.json#benchmark`, `#warmup_days`, `#folds.*`; holdout coefficients; `clip_count=0` |
| Floor policy changes results | Single strict floor; consistent exclusion from fit/model/HAR; with/without sensitivity | `floor_decomposition.counts` `2,620/1,922/698`; `#pooled`; per-fold rows |
| Holdout reuse | Lock before metrics; second evaluation refused; post-lock resume skips; force requires reason and refuses changed identity | Payload `859a…`; file `1445…`; `overrides=[]`; `SHA256SUMS` `4/4`; `HoldoutLockError` |
| Artifact corruption / stale pairing | Registry size/SHA; report hashes; `SHA256SUMS`; atomic writes | Dataset `d6b7…`; X `81cdfd…`; Y `77845a…`; metadata `b84269…`; folds file `fcf728…`; report `4511…` |
| Silent run drift / resume retraining | Manifest identity; sidecar/predictions hash checks; strict-JSON append-only log | Resume `5/5`, `0` trained, summary hash `b10a0ac…`; determinism `20/24`; `runs.jsonl` `6` lines |

*Table 9 notes — n:* eight threat rows; *row set:* construction controls; *scale:* qualitative, with counts/hashes; *weighting:* not applicable; *source:* artifacts in the Evidence column plus the audit archive for resume/determinism

**Leakage and alignment.** The target is the sole forward-looking column. Most features shift one day; four feature families (six columns) are anchored to close `t`: `simple_return`, `price_vs_sma_{10,20,50}`, `zscore_price_vs_sma_20`, `zscore_volume_20`. They divide a current value by a lagged statistic and are available at decision time; the target's first overnight component uses close `t` as previous close. ADR 0007 enumerates the contract. Zero X/Y NaNs, infinities, and duplicate pairs make the matrix finite and uniquely keyed; they do not prove economic alignment. Fold and holdout separation are described in §4.

**HAR, integrity, resume.** HAR fits only training rows and scores exactly the same floor-retained rows; fold 0's `45,259 = 50,560 + 2,035 − 7,336` difference is inner validation plus warm-up, not a different test set. The payload digest `859a2a8e…` binds the recipe, dataset, inputs, fold digest, and bounds; file SHA `14457233…` verifies lock bytes; report hash `4511…` verifies the report; `SHA256SUMS` covers `4/4`. This is local integrity, not external attestation. Resume requires manifest identity plus matching sidecar/predictions hashes; the recorded run resumed `5/5` folds, trained `0`, and left the summary hash unchanged (`b10a0ac…`). `runs.jsonl` has `6` lines; strict JSON means no NaN/Infinity extensions, not that numbers are true.

## 10. Engineering discipline

**Table 10. Engineering evidence**

| Item | Result | What it establishes | Source |
|---|---|---|---|
| Test collection | 421 tests | 421 tests collect at the base develop commit | pytest collection [V] |
| Local gate suite | 421 passed, 0 skipped, 583.98 s | Local suite passed; CI is separate | audit archive [R] |
| Initial CI | Run `39868305183`: static `380` passed, `39` skipped, `2` failed; dataset-smoke success | Static job does not run Phase 1; `39` live-artifact checks skipped; `2` synthetic tests failed | CI record [R] |
| CI fix | Run `39869905251` success on PR #63; run `39870292080` success at `178e74c` | Test-only fix; both CI jobs passed | CI record [R] |
| Determinism | 20/24 tracked artifacts byte-identical; 4 run-dependent | Core evaluation stable; four run-scoped artifacts differ | audit archive [R] |
| Resume | 5/5 folds resumed; 0 trained; summary hash unchanged (`b10a0ac…`) | Resume reuses verified sidecars | audit archive [R] |
| VmRSS | Phase-2 peak 3379.4 MiB; phase-3 start 1934.4 MiB (drop 1445.0 MiB); phase-4 peak 3438.4 MiB; process end 1592.6 MiB | Peak memory by phase | audit archive [R] |
| Atomic writes | 14 tests pass; `*.tmp` + `os.replace`; no residue | Torn writes do not leave half-artifacts | audit archive [R] |
| Retention | 6 + 4 tests pass; 4 run dirs kept (≤5); per-dir README; `runs.jsonl` 6 lines | Retention keeps four run directories; log is append-only | audit archive [R] |

*Table 10 notes — n:* nine evidence items; *row set:* engineering process; *scale:* counts, seconds, MiB, truncated hash; *weighting:* not applicable; *source:* pytest collection, CI record, audit archive

The local/gate suite passed `421` tests. CI ran the same suite, but the static job does not run Phase 1: on the publication merge, `39` live-artifact checks skipped and `2` synthetic tests that required the dataset failed (`380` passed). Dataset-smoke passed. A test-only fix (PR #63) merged at `178e74c`; after it, `static` and `dataset-smoke` passed. The local 421-pass count is not a CI count. Published and evaluated trees are identical and the report hash is unchanged.

Determinism is 20/24, not “fully deterministic.” The four run-dependent artifacts are `manifest.json`, `timing_breakdown.png` (the presentation timing artifact), `fold_training_summary.json`, and `validation_predictions.pkl`. The latter two differ only in timing, run-id, or container serialization; their metric values and arrays match. The core evaluation artifacts—`per_row_losses.parquet`, `har_benchmark.json`, `evaluation_summary.json`, fold manifest—are byte-identical across two full runs, and per-fold metrics match exactly. Same-machine result; cross-platform checks should compare content.

Operational controls reduce silent corruption and accidental holdout reuse but cannot detect source leakage or seed-specific results.

## 11. Limitations and falsification

The Limits callout in §1 lists the standing caveats; this section expands the ones that most affect interpretation.

**Single seed.** One run at seed `43`; the seed-finalist stage ran `42`, `43`, `44` to select on validation QLIKE, not to estimate a distribution. `results/seed_sweep.json` is absent. The three finalist values span `0.000147` (about `0.25%` of the selected value), but that is selection on validation, not a test or holdout uncertainty estimate. The `0.000082` (`0.138%`) figure is the refinement→seed gain, not a seed spread. Seed uncertainty is not in the holdout interval.

**One-shot holdout.** One 500-date block, evaluated once. This is strong against repeated-holdout tuning and weak for uncertainty: no second window, so window-specific results cannot be separated from a stable edge. The CI is within-window, not a new-window CI.

**No crisis window.** CV test blocks include 2008 and March 2020, but the design is not a clean crisis stress test: training windows expand and overlap; test blocks are disjoint and multi-year; the holdout covers 2024–2026. Crisis behavior is therefore not separately identified. The regime table is descriptive calendar-bucket analysis; decile 9 is the weakest decile.

**HAR only.** No GARCH/EGARCH, HAR-X, factor model, alternative boosting, random forest, neural network, or ensemble. “Better” means lower QLIKE than this HAR under this protocol, not best-in-class.

**No economic test.** QLIKE is statistical; no trading strategy, costs, position sizing, VaR, or ES. Lower QLIKE need not translate into economic value; the mapping depends on an unevaluated decision rule.

**Calibration rejected.** MZ rejects the calibration null (`α=0, β=1`); no calibration layer is applied, and no predictive distribution, quantiles, VaR, or ES are produced. A lower average QLIKE can coexist with miscalibration; tail safety is not established, and density claims are unsupported.

**Fold 2 at the cap.** Best iteration `499` of `500`; validation minimum at the cap; final refit also 500 rounds, no early stopping. Fold 2 has the largest CV edge, so the cap is not trivial. More rounds could raise or lower it; a convergence study is needed.

**Survivor-only universe.** Current/recent constituents only; delisted names absent; long-horizon returns biased upward by roughly `1.4–3.9` pp/year (CAGR) `[R]`. A membership-aware panel is required to test whether the edge survives delisting; the report does not claim it does.

**Multiplicity.** Ten deciles and seven regimes across two windows; no multiple-testing correction. The 5/6 crossover is consistent, but individual bins are descriptive.

**Other.** Raw census differs by `1,211` unresolved rows; selection config hash `5e822810` is unverified.

**(a) Tests that would weaken the supported average-advantage claim.**

1. A pre-registered seed sweep (at least ten seeds) with a seed-level interval including zero.
2. A second non-overlapping holdout or rolling-origin evaluation with edge ≤ 0.
3. A dedicated high-volatility/crisis window where the model underperforms HAR.
4. A membership-aware universe including delisted names that removes the edge.

**(b) Different estimands and open questions.**

5. A stronger benchmark (GARCH/EGARCH, HAR-X, factor, ensemble) that tests non-HAR superiority; this is a different estimand and does not weaken the HAR average edge.
6. A calibration layer that fixes MZ but degrades QLIKE, or reveals tail miscalibration.
7. A tail-weighted loss or quantile evaluation; this changes the estimand and does not by itself weaken the average QLIKE claim.
8. An economic test with realistic costs showing utility or no utility.
9. A census or selection-hash reconciliation that changes the comparison or its provenance.

## 12. Conclusion

**Table 11. Claims ladder**

| Claim | Status | Evidence | Test that would weaken / resolve the claim |
|---|---|---|---|
| Average QLIKE advantage over HAR in CV | Supported | All five fold edges positive; equal-weight and pooled agree; DM p-values tiny; floor sensitivity preserves ranking | Seed sweep interval including zero; different HAR; weighting reversal (none observed) |
| Average advantage in locked holdout | Supported, conditional on this window | Edge `0.025766`; DM `-7.301276`; CI `[0.018849, 0.032683]` | Second window with edge ≤ 0; seed uncertainty; refit capacity (fixed 500 rounds vs CV 81–499) and window composition |
| Advantage strongest in low/mid volatility | Supported, descriptive | Within-window deciles: 0–5 positive, 6–9 negative, same crossover | Multiplicity-adjusted tests; different deciles; tail-weighted loss |
| Calibration | Rejected by MZ; layer untested | MZ α `-0.306146`, β `0.933458`; joint null rejected; no calibration layer | Calibrated model with new evaluation and improved/unchanged QLIKE |
| Tail safety | Not established | Top-1% mean/share higher than HAR in CV and holdout; decile 9 worse | Tail-focused model or VaR/ES backtest showing acceptable tails |
| Seed robustness | Not tested | One selected seed; no sweep; three-seed stage is selection | Multi-seed sweep with stable distribution |
| Crisis robustness | Not tested | No crisis window; decile 9 negative; CV crisis periods not a clean crisis test | Dedicated crisis-window evaluation |
| Economic value | Not tested | No P&L, costs, sizing, VaR, or ES | Economic backtest with pre-specified rule |
| Superiority to non-HAR models | Not tested | HAR only | GARCH/HAR-X/factor/ensemble comparison |

*Table 11 notes — n:* nine claims; *row set:* claims synthesized from Tables 1–10; *scale:* ordinal support status; *weighting:* not applicable; *source:* artifact keys cited in those tables. Sign note: evidence uses `HAR − model` for the headline edge and `model − HAR` for the DM differential; positive headline edge and negative DM mean model better.

The holdout edge is `0.025766` QLIKE (`27.25%` of HAR); CV is larger (`0.049189` equal-weight, `0.048631` pooled). The holdout is the cleaner estimate for this window because it was untouched by selection, though refit capacity and composition are confounded. The derived 95% interval `[0.018849, 0.032683]` is conditional on the model and HAC and is narrower than full production uncertainty.

The ranking is stable in the floor sensitivity reported, while absolute QLIKE levels change; deciles 6–9 have negative edges (worst in 9).

**Verdict.** The average QLIKE advantage over HAR is measured but conditional: concentrated in low/mid volatility and smaller in the holdout. Calibration is rejected; tail safety is not established. Seed robustness, crisis robustness, economic value, and non-HAR superiority are not tested.

**Next evidence.** Multi-seed sweep; a second holdout or rolling-origin evaluation; a dedicated crisis window; stronger benchmarks; a membership-aware universe; a tail-weighted loss. Until then, cite a conditional average advantage over HAR.

## Appendix A. Formulas and definitions

**Target.** Using the notation of §2 (`t` the row/date index, `i` the ticker, `w = {t+1, …, t+5}` the forward window, `d ∈ w`):

```
variance_{i,t} = Var_w({overnight_{i,d}}) + k * Var_w({open_to_close_{i,d}})
                 + (1-k) * Mean_w(RS_{i,d})
realized_volatility_5d_{i,t} = sqrt(max(variance_{i,t}, 0))
k = 0.34 / (1.34 + (5+1)/(5-1))
```

Terms are defined in §2; the stored value is per-day sigma, not annualized. Rows without a complete forward window are dropped.

**QLIKE.** `QLIKE(h,f) = h/f − ln(h/f) − 1 = exp(d) − d − 1`, with `d = ln(h) − ln(f)`. It is zero at `h=f`, positive otherwise, and penalizes underprediction more.

**Edge and relative reduction.** `edge = QLIKE_HAR − QLIKE_model`; `relative = edge / QLIKE_HAR`. Positive edge means model better. The DM differential is the opposite sign.

**DM/HAC.** With `d_t` the loss differential (or date-mean differential), `DM = mean(d_t)/sqrt(HAC_var(d_t)/n)`. Newey–West with `L=4`: `Omega = Gamma_0 + Σ_{l=1..4}(1−l/5)(Gamma_l + Gamma_l')`. Negative DM means lower model loss under `model − HAR`.

**CW.** Clark–West adjusted MSPE diagnostic; nesting not claimed. **GW.** Wald χ²(2) on `d_t = c + φ d_{t−1} + u_t`, null `c=φ=0`. **MZ.** `y_true = α + β y_pred + e`, HAC SEs; null `α=0, β=1`.

**Dual R².** `1 − SSE_model/SSE_mean` and `1 − SSE_model/SSE_HAR` on the log target. `SSE_mean` uses the fold-own mean per fold and pooled mean pooled; `test_r2`/`har_r2` use sklearn's fold-own mean. Negative means worse than the corresponding mean under squared error.

**Deciles/regimes/origins.** Deciles are equal-count bins of realized log sigma (pooled CV bins for CV, holdout bins for holdout). Regimes are hardcoded calendar buckets `0..6`. Expanding origin is the cumulative mean of per-row QLIKE in date order, eleven checkpoints per fold; its edge is `model − HAR`.

**Floor/HAR.** Single strict `1e-12`; at-or-below-floor rows excluded from fit, model score, and HAR score. HAR is canonical `1/5/22` trailing variance, OLS in variance levels, 21-date warm-up, forecasts mapped to log and floored, no clip in these runs.

## Appendix B. Claim–evidence map

- **CV advantage:** `evaluation_summary.json#headline.aggregate`, `#headline.pooled`, `#headline.folds.*`, `#dm_cw_gw.pooled.dm`.
- **Holdout advantage:** `holdout_report.json#headline.aggregate`, `#holdout_metrics.model_qlike`, `#har_qlike`, `#edge`, `#dm`, `#cw`, `#gw`.
- **Calibration:** `evaluation_summary.json#mz.pooled`; `holdout_report.json#holdout_metrics.mz`. MZ rejects `α=0, β=1`; calibration is not applied and a calibration layer is untested.
- **Low/mid vs high vol:** `evaluation_summary.json#decile_qlike.pooled.bins`; holdout equivalents. Deciles 0–5 are positive, 6–9 negative.
- **Descriptive regimes/calendar buckets:** `evaluation_summary.json#regime_qlike.pooled.bins`, `#regime_qlike.labels`; holdout `#holdout_metrics.regime_qlike`. Not volatility evidence.
- **Tail concentration:** `evaluation_summary.json#loss_distributions.pooled.model`, `#loss_distributions.pooled.har`; holdout loss distribution.
- **Floor sensitivity:** `evaluation_summary.json#floor_decomposition.counts`, `#floor_decomposition.pooled`, `#floor_decomposition.folds.*`.
- **Folds/purge:** `folds_metadata.pkl#fold_i.*`; `config/base.yaml#folds`; `fold_training_summary.json#fold_i.n_inner_val_samples`, `#n_inner_purge_rows`.
- **Holdout/refit:** `holdout_report.json#protocol.holdout`, `#protocol.purge`, `#protocol.refit`, `#protocol.benchmark`.
- **Lock:** `holdout_lock.json#sha256`, `#payload`; `run_manifest.json#holdout`; `holdout_report.json#protocol.one_shot`, `#overrides`, `#post_hoc_adjustment`; `results/SHA256SUMS`.
- **Data/universe:** `ingestion_manifest.json#rows`, `#tickers`, `#source_sha256`; `metadata.pkl#dataset`, `#input_checks`; `regime_characterization.json#_frame`.
- **Tuning:** `selection.json#stages`, `#selected_trial`, `#n_trials`; `config/base.yaml#training.hyperparameters`, `#tuning.selected`.
- **Engineering:** audit archive gate/determinism/resume/operations/CI records; current pytest collection.
- **Stale sources:** Appendix D; the pre-remediation audit, provisional baseline, legacy snapshot, and presentation root are not current evidence.

## Appendix C. Reproduction and checksums

Run from the repository root; these are read-only except for pytest's normal cache. Do not run `--evaluate-holdout` or `--force-holdout` to audit: the first is refused by the lock, the second appends an override.

1. **Commits/trees.**

```bash
git rev-parse HEAD
git rev-parse df5aa5a^{tree} 466d76d^{tree}
```

`HEAD` is the report commit, not the base develop commit `178e74c`; it adds only this file. The evaluated and publication trees are both `718b04fdb500de3145ace1f4a7a7a3a09400c3a8`.

2. **Lock payload.**

```bash
uv run --frozen python - <<'PY'
import json
from packages.ml.model.evaluation.holdout import lock_sha256
lock = json.load(open("packages/ml/model/artifacts/results/holdout_lock.json"))
print(lock_sha256(lock["payload"]))
PY
```

Expected `859a2a8e…`.

3. **Config hash.**

```bash
uv run --frozen python -c "from packages.ml.model.run_log import config_hash; print(config_hash())"
```

Expected `407a2ad5`.

4. **Test collection.**

```bash
uv run --frozen python -m pytest packages/ml/model/tests --collect-only -q | tail -1
```

Expected `421 tests collected`.

5. **Read the locked headline.**

```bash
uv run --frozen python - <<'PY'
import json
r = json.load(open("packages/ml/model/artifacts/results/holdout_report.json"))
print(r["headline"]["aggregate"])
print(r["holdout_metrics"]["dm"])
PY
```

6. **Full suite.** `uv run --frozen python -m pytest packages/ml/model/tests -q`; recorded `421 passed`, `0` skipped, `583.98 s`.

**Table C.1. Canonical provenance and artifact hashes**

| Identifier | SHA-256 / run id | Role |
|---|---|---|
| Run ids | pipeline `20261020T070010Z-407a2ad5`; holdout `20261020T063951Z-407a2ad5`; tuning `20261019T100655Z-5e822810` | run provenance |
| `holdout_report.json` | `45118752fd6dcb5de1f8a816270931d2bb453ba3d434107d842dde9fb81b82d5` | report bytes |
| `holdout_lock.json` | `14457233cacf1a8c04e74b73324e893c6403e860a1cee8a896b688ebbe113c14` | lock file bytes |
| Lock payload | `859a2a8ef3fa63d95b87bb26650a8a0532bc1431328a16c7dfa215c0739aba07` | lock identity |
| `dataset.parquet` | `d6b77cfb50504e14e048c8134857bf98bd32a1aa8cefd481fd046196fe89c3b2` | dataset bytes |
| `X.npy` | `81cdfd77a9579addac6a75bc7bb7d44e1f3460cd94f52e1942a484f0a1f19332` | model input |
| `Y.npy` | `77845ae3670677ed0d602d22b2c30824bc27db4159ea92f4e0b50814fb166325` | target input |
| `metadata.pkl` | `b842696a7bb514fd21ba4f917f4b7958c2fa0172692a82e500aa0e220223db29` | input metadata |
| `folds_metadata.pkl` | `fcf7280045509a68788dc6cc9de26f25c7a8e80ef7be7aef2901562863184514` | fold file bytes |
| Fold payload | `c4b9f8c7943e9bf162ede21ea1497b3e0163c83957a282f071f0c54df32b8a9d` | fold identity |
| `holdout_model.pkl` | `3c068f485c2798bc548dbc99f594a86673af576e7dad9a0f85b98db728e33291` | final model |
| `holdout_predictions.pkl` | `042ca7618e8763c179baedc0cec7dd887ce5268962ba20edd1919dfd91b371eb` | holdout predictions |

*Table C.1 notes — n:* twelve entries; *row set:* provenance and artifact identity; *scale:* SHA-256 or run id; *weighting:* not applicable; *source:* `holdout_report.json#artifact_hashes`, `run_manifest.json#input_identity`, `run_manifest.json#run_id`, direct `sha256sum`.

**Environment.** Python `3.14.8`, numpy `2.4.4`, pandas `3.0.2`, scikit-learn `1.8.0`, xgboost `3.2.0`, pyarrow `24.0.0`, joblib `1.5.3`, PyYAML `6.0.3`; lock hash `c69de45a…`; `n_jobs=1`, `OMP_NUM_THREADS=1`. Environments can change model bytes; cross-platform checks should compare content.

## Appendix D. Current versus stale source inventory

**Table D.1. Source inventory**

| Source | What it is | Status | Used for |
|---|---|---|---|
| `run_manifest.json` | identity, config, environment, input identity, holdout block | current | provenance, lock |
| `results/holdout_report.json` | locked holdout metrics, protocol, hashes | current (`4511…`) | holdout |
| `results/holdout_lock.json` | lock payload and file identity | current (`859a…` / `1445…`) | lock |
| `results/evaluation_summary.json` | CV headline, folds, inference, deciles, regimes, floor | current | CV |
| `results/fold_training_summary.json` | per-fold metrics, counts, best iteration | current | folds |
| `results/har_benchmark.json` | HAR fit, coefficients, per-fold QLIKE/R² | current | benchmark |
| `results/per_row_losses.parquet` | per-row loss vectors | current | tails/floor/deciles |
| `results/regime_characterization.json` | frame label, survivor flag, pre-drop census | current; `2,702,320` | data |
| `inputs/metadata.pkl` | dataset metadata, 37 features, checks | current | data/features |
| `folds/folds_metadata.pkl` | fold geometry, purge, holdout bounds | current (`c4b9…` / `fcf728…`) | protocol |
| `config/base.yaml` | frozen config, tuning provenance | current (`407a2ad5`) | protocol/tuning |
| `tuning/20261019T100655Z-5e822810/selection.json` | selected trial and stages | current; `5e822810` unverified | tuning |
| `data/processed/ingestion_manifest.json` | raw source census | current; `2,703,531` conflicts | data |
| `data/processed/dataset.parquet` | shipped dataset | current (`d6b77…`) | data |
| Pre-remediation audit | fold-0 `test_qlike=0.063238`, `har_qlike=0.141701`; `n_train=367,202`, `n_train_samples=366,518`; floor `1,928/692` | forbidden for current numbers | none |
| Provisional post-remediation baseline | `0.058621 / 0.107674 / 0.049053` | superseded by `0.058485 / 0.107674 / 0.049189` | none |
| `baseline_v1/` (`2,668,192 × 39`) | legacy snapshot | provenance only; never an input | none |
| Presentation output root | generated charts and manifest | presentation only | none |
| `results/seed_sweep.json` | seed sweep | absent | none |

*Table D.1 notes — n:* nineteen entries; *row set:* source inventory; *scale:* paths/statuses/hashes; *weighting:* not applicable; *source:* direct inspection and artifact keys

## Appendix E. Glossary

- **Calibration:** mapping raw forecasts to unbiased/linear calibration; rejected by MZ; no layer applied.
- **CW:** Clark–West adjusted MSPE diagnostic; nesting not claimed.
- **DM:** Diebold–Mariano equal-average-loss test with HAC covariance.
- **Dual R²:** OOS R² against the prevailing mean and against HAR, log target.
- **Floor:** strict `1e-12` threshold; at-or-below-floor rows excluded from fit and scoring.
- **Fold-manifest digest:** canonical SHA-256 of the fold payload (`c4b9f8…`), not the file hash (`fcf728…`).
- **HAC:** heteroskedasticity/autocorrelation-consistent covariance; Newey–West lag 4.
- **HAR:** canonical HAR-RV with `1/5/22` trailing variance components, OLS variance levels.
- **Headline edge:** `HAR − model`; positive = model better. Floor block uses the opposite sign.
- **Holdout:** final 500 dates, `2024-02-15` to `2026-02-12`, reserved before folds and evaluated once.
- **Inner validation:** last 15% of each training tail, used for tuning/early stopping, five-day inner purge.
- **Lock payload:** digest `859a2a8e…` binding selection, dataset, inputs, fold digest, bounds; file hash `14457233…`.
- **MZ:** Mincer–Zarnowitz calibration regression; null `α=0, β=1`.
- **Purge:** train-only removal of five dates whose labels overlap the test block.
- **QLIKE:** `h/f − ln(h/f) − 1` in per-day sigma.
- **Survivor bias:** upward bias from a current/recent constituent list; roughly `1.4–3.9` pp/year (CAGR).
