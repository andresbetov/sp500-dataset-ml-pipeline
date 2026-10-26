# Model pipeline

This is the deep reference for Phases 2-5: how the parquet becomes inputs, folds, trained models, metrics, and charts. It owns the protocol, fold geometry, training, and evaluation definitions.

Running, resuming, retention, and the holdout gate live in [../operations.md](../operations.md); how to read the numbers lives in [../results.md](../results.md); formulas and the full analysis live in [../reports/model-performance-and-rigor.md](../reports/model-performance-and-rigor.md). The dataset phase is documented in [dataset.md](dataset.md).

## Purpose and phase map

Phases 2-5 consume `data/processed/dataset.parquet` and produce the model inputs, the temporal folds, the trained models and metrics, and the charts. Phase 1 must run first: Phase 2 reads the parquet and fails immediately if it is missing.

- **Phase 2 — feature selection** reads the parquet, selects the 37 positional `INDICATORS`, validates finite `X`/`Y`, and writes `inputs/X.npy`, `Y.npy`, and `metadata.pkl`.
- **Phase 3 — temporal cross-validation** reserves the final 500 unique dates as holdout, splits the sorted remaining dates into contiguous blocks, builds folds with a train-only purge, and writes `folds/folds_metadata.pkl`.
- **Phase 4 — training and evaluation** floors the target, trains one QLIKE XGBoost model per fold with a purged inner-validation tail, fits the canonical HAR-RV benchmark on training rows, and writes fold summaries, predictions, loss vectors, and the evaluation summary.
- **Phase 5 — visualization** writes 14 PNGs plus `data/model_outputs/manifest.json`.

The holdout is optional and lives outside this map: only `--evaluate-holdout` reads the reserved window, and it is governed by the lock documented in [../operations.md](../operations.md).

## Entry points and CLI

The model entry runs Phases 2-5. Both command forms are equivalent and run from the repository root:

```bash
python -m packages.ml.model.main
python packages/ml/model/main.py
```

With no arguments it runs the full pipeline unchanged and never reads the holdout. The five flags are `--dry-run`, `--resume`, `--folds`, `--evaluate-holdout`, and `--force-holdout`; the canonical contract, constraints, and examples are in [../reference/cli.md](../reference/cli.md). The run lifecycle behind `--resume` and the holdout flags is in [../operations.md](../operations.md).

## Phase 2 — feature selection

`phase_2_feature_selection` loads the parquet, extracts the 37 pinned indicators in contract order as `X = df[INDICATORS].values`, and extracts the target as `Y = df["realized_volatility_5d"].values`. There is no ticker encoding: `n_ticker_features` is 0 and the model is pooled across names.

Before anything is written, `validate_extracted_inputs` rejects non-finite `X` and non-finite or negative `Y` by name, and records per-column NaN/infinity rates, duplicate `(ticker, date)` pairs, zero-variance columns, and the degenerate target mass at or below the floor. These checks are stored under `input_checks` with keys `n_rows`, `n_features`, `x_nan_count`, `x_inf_count`, `duplicate_pairs`, `zero_variance_columns`, `y_nan_count`, `y_inf_count`, `y_negative_count`, `degenerate_count`, and `degenerate_rate`, plus the per-column `x_nan_rates` and `x_inf_rates`. Non-finite or negative targets are contract violations that raise before any fit; constant columns and duplicate pairs are recorded, not raised.

The metadata records the date and ticker columns, the row-aligned calendar regime labels, and a dataset pin: the parquet path, its SHA-256 and mtime, the row and unique-date counts, the extra-column list, and the `INDICATORS` order hash.

Two input columns are computed upstream on per-date cross-sections before the dataset's dropna step and cannot be re-derived from the shipped parquet: `market_abs_ret_5d` and `volume_lag_5`. They are listed in `NON_RECONSTRUCTIBLE_COLUMNS` and stay in the pinned input list, but model code never re-derives them; a fix for them routes to the dataset phase.

The outputs are `inputs/X.npy`, `inputs/Y.npy`, and `inputs/metadata.pkl`, each written through a sibling temporary file and an atomic replace. The metadata pickle is written last, so its presence marks a complete Phase 2. The metadata is registered in `artifact_registry.json` so later phases verify it before loading.

## Phase 3 — temporal cross-validation

Phase 3 builds date-group, expanding-window folds from the sorted unique trading dates. The final `holdout.n_dates = 500` unique dates are reserved as the holdout before any fold construction, so no fold train or test row can fall inside it. The remaining dates are split into `n_folds + 1` contiguous blocks; with the default `folds.n_folds = 5` that is six blocks. Fold `k` tests on block `k+1` and trains on blocks `0..k`, so the training window expands with every fold and the first block is train-only.

Each fold drops only the training rows whose 5-day label window reaches into the test block. The last kept training date therefore sits exactly five complete trading dates before the first kept test date, enforced as `first_test_idx - last_train_idx >= separation_min = 6` in unique-date index space. Test blocks are kept whole, so there is no shuffle and no AFML-style embargo. The persisted payload carries, per fold, the train/test indices, date bounds, row counts, the purge count, the typed gap keys, and the per-regime train/test distribution over the seven calendar buckets.

```
sorted unique trading dates (holdout reserved first)
|<--------------- folds (n_dates) ---------------->|<-- holdout -->|
| block 0 | block 1 | block 2 | block 3 | block 4 | block 5 | reserved |

fold 0: train = block 0 (minus purge)            test = block 1
fold 1: train = blocks 0-1 (minus purge)         test = block 2
fold 2: train = blocks 0-2 (minus purge)         test = block 3
fold 3: train = blocks 0-3 (minus purge)         test = block 4
fold 4: train = blocks 0-4 (minus purge)         test = block 5
```

Each fold also holds out the temporally last `training.validation_tail_fraction = 0.15` of its floor-retained training rows as a purged inner-validation tail for early stopping, with the same 5-day label-horizon purge. This tail is strictly inside fold-train, so it can never intersect the fold's test block or the reserved holdout. Input rows are stored ticker-major, so the inner split orders by trading date rather than stored row order; that date ordering is what makes the tail temporal.

The purge follows from the five-day label: a training row at `t` aggregates `t+1..t+5`, so a row within five days of a test block can carry test-period outcome information. Dropping the last five training dates removes it. Test rows are not purged because each forecast uses only information at its own date; the overlap among test labels creates serial correlation, not leakage, and the HAC lag 4 in evaluation absorbs it.

Phase 3 also enforces a minimum calendar size before building anything: the frame must hold at least the reserved holdout plus `(n_folds + 1) * (purge_horizon_days + 1)` fold dates, so the earliest fold still keeps a training date after its train-side purge. A frame that fails this check raises rather than producing an empty block.

The whole payload — schema version, pinned feature names, holdout bounds, remainder date count, and every fold record — is hashed by `fold_manifest_hash`, a canonical SHA-256 over sorted keys and compact JSON. The digest changes with any membership, date, regime, or gap change, and it is stable across runs and processes. The payload is written to `folds/folds_metadata.pkl` atomically and registered in the artifact registry.

## Phase 4 — training

Training fits one `XGBRegressor` per fold on the log target. The raw Yang-Zhang target is floored at `rv_floor = 1e-12` before the log, so `log(RV)` is always finite. At-or-below-floor rows are flat-window artifacts and are excluded from fitting, from model scoring, and from the HAR benchmark under the single `floor_retained_mask` predicate. Without the floor, a flat window's log target is undefined, and fitting those extreme labels under-forecasts similar test rows so the QLIKE exponential term explodes; the floor policy exists to keep those rows out of the fit, the model score, and the benchmark alike.

Each fold model minimizes QLIKE through a custom objective and evaluation metric. The hyperparameters come from the frozen `training.hyperparameters` section in `config/base.yaml`, read at call time by `model_trainer.default_params()`. The determinism policy pins `training.n_jobs = 1` and `OMP_NUM_THREADS = 1` before every fit, with `training.seed = 42` as the baseline seed and `hyperparameters.random_state = 43` as the selected model seed. The effective values, seeds, `PYTHONHASHSEED`, library versions, git sha, and `uv.lock` hash are recorded in the run manifest.

Early stopping is live in fold training: the model stops after `EARLY_STOPPING_ROUNDS = 30` rounds without a validation-QLIKE improvement on the purged inner-validation tail, and the best iteration and per-round QLIKE history are persisted. The final holdout refit has no validation tail and no early stopping.

The canonical HAR-RV benchmark (Corsi 2009) is fitted on training rows only, using 1/5/22 trailing variance components built from the same Yang-Zhang estimator family as the target, with OLS in variance levels and a 21-date per-ticker warm-up. Its forecasts are mapped to log and floored, and it scores exactly the same floor-retained test rows as the model. The fit scale is declared as variance and the report scale as log-variance for R² and sigma for QLIKE, so the log-scale R² is never mistaken for the fit objective; QLIKE is the headline model-vs-benchmark comparison.

The benchmark applies one declared forecast constraint: a variance-space forecast below `rv_floor**2` is clipped to the floor, which is equivalently a lower bound of `ln(rv_floor)` in log space. Each clip is counted per fold as `clip_count` and `clip_rate`, so the counter cannot drift from the applied clip. The unconstrained OLS fit itself stays canonical; only the reported forecast is clipped.

Each fold writes a run-scoped checkpoint `packages/ml/model/artifacts/checkpoints/<run_id>/fold_<i>.pkl` plus three sidecars: `fold_<i>_params.json` (the full `get_params()` set, tool versions, and git sha), `fold_<i>_summary.json` (the metric block plus identity), and `fold_<i>_predictions.pkl` (predictions, HAR result, and the repeated summary). The summary records the predictions file's size and SHA-256, so a stale summary beside a newer predictions file is detected and the fold recomputed. Every fold record carries an identity block of `run_id`, `config_hash`, `fold_manifest_hash`, and the coefficient-interpretation setting, which resume must match before trusting a sidecar.

## Evaluation

QLIKE is the headline loss. For per-day sigma forecasts it is `QLIKE(h, f) = h/f - ln(h/f) - 1`, which in log space is `exp(delta) - delta - 1` for `delta = actual - forecast`. It is zero at a perfect forecast and positive otherwise. The loss is computed on per-day sigma, not variance, so its absolute values are not directly comparable to literature numbers on a variance scale.

The headline is the unweighted mean of the fold metrics: each fold contributes one equal-weight QLIKE regardless of its row count, and the edge is `har_qlike - model_qlike` (positive means the model is better). A separate pooled, row-weighted value is computed from the persisted loss vectors over every floor-retained row of the valid folds, so it answers a different question than an average of fold ratios.

The other metrics are log-scale diagnostics. R²/MAE/RMSE are computed on the log target; `test_r2` keeps sklearn's fold-own-mean convention. The pipeline reports dual out-of-sample R² beside it, per fold and pooled: `oos_r2_prevailing_mean = 1 - SSE_model/SSE_mean` and `oos_r2_vs_har = 1 - SSE_model/SSE_HAR`, both on the log target over the floor-retained scored rows.

Statistical tests use Newey-West HAC covariance with lag 4 on the date-mean differential. The Diebold-Mariano test compares model and HAR loss differentials; Clark-West is an adjusted MSPE diagnostic with nesting not claimed; Giacomini-White is a Wald test on a constant plus one lag. Mincer-Zarnowitz regresses `y_true` on `y_pred` to test the calibration null `alpha = 0, beta = 1`. Decile and regime tables are descriptive; no multiplicity correction is attached to individual bins.

The evaluation layer persists one loss-vector row per test-block row in `packages/ml/model/artifacts/results/per_row_losses.parquet` (with `is_floor` read from the persisted mask, never re-derived) and writes `packages/ml/model/artifacts/results/evaluation_summary.json` from those rows only. The summary carries the promoted QLIKE headline (equal-weight fold mean plus row-weighted pooled values), the dual OOS-R², DM/CW/GW, MZ, decile and regime QLIKE, the loss distribution, the floor with/without decomposition, the date-clustered gap, expanding-origin curves, and an explicit no-calibration status. Quarantined folds are excluded from every pooled aggregate. The floor decomposition reports QLIKE with and without the at-or-below-floor rows, so the floor policy's effect on both levels and the edge is visible; the expanding-origin curves are a path-dependence diagnostic, not a separate test. Known pandas/scikit-learn edge cases, including the constant-target R² convention, are triaged in [../pandas-trap-triage.md](../pandas-trap-triage.md).

## Phase 5 — visualization

Phase 5 renders 14 PNGs plus `data/model_outputs/manifest.json`, in four categories:

| Category | Plots |
|---|---|
| Metrics | `r2_scores_by_fold`, `mae_rmse_comparison`, `metrics_heatmap` |
| Temporal | `temporal_fold_ranges`, `fold_sample_distribution`, `temporal_metric_trend` |
| Performance / comparison | `metrics_comparison`, `timing_breakdown`, `sample_distribution`, `feature_count_summary` |
| Diagnostics | `actual_vs_predicted`, `residuals_distribution`, `feature_importance`, `error_by_volatility_bin` |

Each plot is isolated: a per-plot failure is recorded in the manifest and the other plots still render. The manifest records per-plot statuses and a stats block. Charts are presentation-only renderings of the results and are not authoritative; when a chart disagrees with an artifact, trust the artifact. Feature importance is XGBoost's default `gain` importances mapped onto `INDICATORS` by position, and the plot refuses to render if the model's importance width or the persisted feature names drift from the pinned 37. The manifest's stats block also records the number of checkpoint models the feature-importance plot loaded, so a partial checkout that contributes zeros to the aggregate is visible rather than hidden. Known pandas/matplotlib edge cases at these sites are triaged in [../pandas-trap-triage.md](../pandas-trap-triage.md).

## Run state and resume

Every run owns one run record written to `run_manifest.json` plus one appended line in `runs.jsonl`. The identity that resume depends on is the manifest's `run_id`, `config_hash`, `fold_manifest_hash`, and `input_identity` (the dataset SHA-256 plus the `X.npy`/`Y.npy` hashes).

A fold is reused only when its checkpoint, params, summary, and predictions sidecars all exist and verify against the artifact registry, and the summary matches the predictions hash. An incomplete fold is retrained; an identity mismatch refuses by name. `--folds` writes a partial selection recorded in `run_selection.json`. Retention prunes directories, never the append-only `runs.jsonl`. The full runbook is in [../operations.md](../operations.md).

## Configuration and fixed constants

`packages/ml/model/config/base.yaml` owns the runtime constants, and the `MODEL_CONFIG_PATH` environment variable redirects the pipeline to a different config file. The sections are:

| Section | Owns |
|---|---|
| `rv_floor` | The `1e-12` target floor. |
| `folds` | `n_folds`, `purge_horizon_days`, `label_horizon_days`, `separation_min`. |
| `training` | Determinism policy, `seed`, `n_jobs`, `omp_num_threads`, `validation_tail_fraction`, and the frozen `hyperparameters`. |
| `paths` | The dataset, artifacts, and outputs paths. |
| `run` | The log path and `retention`. |
| `tuning` | Selection provenance and the search space. |
| `holdout` | The reserved `n_dates`. |

The determinism policy is `bit-reproducible` for the fitted model: identical inputs, seed, and environment produce identical checkpoint model bytes and identical metric values. Run-manifest timing/run-id fields and container serialization of two summary payloads may differ; the report records the 20/24 byte-identical audit. The `tuning.selected` keys name the campaign run and winning trial whose values were frozen into `training.hyperparameters`; the selection artifact under `packages/ml/model/artifacts/tuning/<run_id>/selection.json` remains the source of truth, and Phase 4 runs record that provenance in their run manifest.

The tuning campaign is library-level (`tuning/harness.py:run_tuning_campaign`), opt-in, and never runs in CI or the pipeline; the frozen `training.hyperparameters` are the only values Phase 4 reads. Re-running the campaign is a deliberate manual research action, not a CLI path; the selection artifact remains the source of truth for provenance.

Structural constants remain literals in code, not config keys: early stopping rounds (30), HAC lag (4), HAR warm-up days (21), the target window (5), and the minimum trading days for a ticker (500).

## Failure modes and quarantine

| Failure | Behavior |
|---|---|
| Missing parquet | Phase 2 fails fast with a missing-file error. |
| Non-finite `X`, non-finite or negative `Y` | `ValueError` before any fit path sees the arrays. |
| Identity mismatch on `--resume` | Refuses by name; never falls back to a fresh run silently. |
| Fold metric computation fails | The fold is quarantined — recorded with an error and excluded from pooled aggregates — and the run continues. |
| No fold contributes a finite headline QLIKE | `AllFoldsInvalidError` is raised. |
| Holdout already locked | A second evaluation raises `HoldoutLockError`; a post-lock resume records a skip. |
| Partial run | `run_selection.json` records `partial: true` and the fold selection. |

## Verification

Check the entry point without writing anything:

```bash
python -m packages.ml.model.main --dry-run
```

A full run trains one model per configured fold (five with the default config) and writes `packages/ml/model/artifacts/results/fold_training_summary.json` plus the HAR benchmark report. Run the test suite for any change under `packages/ml/model`:

```bash
uv run --frozen python -m pytest packages/ml/model/tests -q
```

Do not run `--evaluate-holdout` or `--force-holdout` as verification: the holdout is one-shot and its lock is recorded before any metric. The contributor workflow is in [../../AGENTS.md](../../AGENTS.md).
