# AGENTS.md

Working guide for AI coding agents in this repository. Read it before editing code. Instructions from the user in the session override this file.

## Repository map

| Path | What it is |
|---|---|
| `packages/ml/dataset/main.py` | Phase 1 entry: download, clean, engineer features, build the target |
| `packages/ml/model/main.py` | Phases 2-5 entry (also `python -m packages.ml.model.main`) |
| `packages/ml/dataset/` | Phase 1 code, including `features/` for the feature families |
| `packages/ml/model/` | Phases 2-5 code: features, folds, training, evaluation, visualization |
| `packages/ml/model/config/base.yaml` | Runtime constants: fold count, holdout size, retention, floor, frozen hyperparameters |
| `data/processed/` | Generated dataset parquet and ingestion manifest (gitignored) |
| `packages/ml/model/artifacts/` | Generated pipeline state and results (gitignored) |
| `data/model_outputs/` | Generated presentation charts (gitignored) |
| `README.md` | Front door and links |
| `docs/` | Current-state reference: `operations.md`, `results.md`, `pipelines/` |
| `docs/adr/` | Frozen architecture decision records |
| `docs/reports/` | Dated evaluation reports |

## Commands

Run everything from the repository root. Both entries resolve their paths from `__file__`, but the root is the documented working directory. After `uv sync`, activate `.venv` (`source .venv/bin/activate`) or prefix commands with `uv run`.

```bash
uv sync                                             # create or update .venv
python packages/ml/dataset/main.py                  # Phase 1 — needs network + Kaggle creds
python packages/ml/model/main.py                    # Phases 2-5 — by-path form
python -m packages.ml.model.main                    # Phases 2-5 — module form (equivalent)
python -m packages.ml.model.main --dry-run          # log the planned phases, exit without writing
python -m packages.ml.model.main --folds 0,2        # train only the named folds (explicitly partial)
python -m packages.ml.model.main --resume           # continue the latest manifest, reuse verified folds
uv run --frozen python -m pytest packages/ml/model/tests -q --ignore=packages/ml/model/tests/live  # hermetic unit suite (CI)
uv run --frozen python -m pytest packages/ml/model/tests -q                                      # full suite, includes live checks
python -m packages.ml.model.determinism             # opt-in seed sweep -> packages/ml/model/artifacts/results/seed_sweep.json
python -m packages.ml.model.visualization.visualize # render the figures into data/model_outputs/
```

The holdout flags exist but are a deliberate gate action, never a verification step:

```bash
python -m packages.ml.model.main --evaluate-holdout       # one-shot; writes the lock before any metric
python -m packages.ml.model.main --force-holdout "reason" # override a recorded lock; implies --evaluate-holdout
```

On a fresh checkout with no lock, the first `--evaluate-holdout` creates the lock and scores the holdout, and it cannot be repeated. Exercise the holdout only through the synthetic/temporary-root tests. See `docs/operations.md`. The canonical flag contract is `docs/reference/cli.md`.

## Hard rules

1. **Run these as scripts.** Both entries normalize their package identity on direct execution (PEP 366) so they work from any cwd; the model entry uses package-absolute imports and also runs as `python -m packages.ml.model.main`, while the dataset entry keeps package-relative imports. Entry calls stay behind `if __name__ == "__main__"`.
2. **Phase 1 before Phases 2-5.** Phase 2 reads `data/processed/dataset.parquet` and fails immediately if it is missing. All generated data is gitignored, so a fresh clone has none.
3. **Never commit generated output.** `packages/ml/model/artifacts/`, `data/model_outputs/`, `data/processed/dataset.parquet`, and `data/processed/ingestion_manifest.json` are gitignored (as are `*.pkl` and `*.joblib` anywhere in the tree, `data/` included). No other `data/` path is ignored, so check before adding anything there. Markdown commits normally; the old blanket `*.md` ignore is gone.
4. **No absolute or machine-specific paths.** Everything resolves from `__file__`. `packages/ml/model/utils.py:get_project_paths` is the model path registry; `packages/ml/dataset/utils.py` holds the Phase 1 path constants. Keep it that way.
5. **No evidence, no "done".** Verification means running the model pytest suite for model-package changes and running the phase you touched; quote the output you saw.
6. **Never run `--evaluate-holdout` or `--force-holdout` as verification.** On a fresh checkout the first evaluation creates the lock and scores the holdout; there is no second chance.
7. **Config is the source for numeric training/CV constants.** Change them in `packages/ml/model/config/base.yaml`, not as literals in code. The documented structural literals are the exceptions, listed under "Invariants".
8. **Avoid duplicating explanatory prose.** README, AGENTS, and the docs under `docs/` each own their subject; critical invariants may be summarized here with a link to the canonical page. See "Documentation ownership".

## Invariants — breaking these corrupts artifacts or silently mislabels results

- **Fold count is config-sourced.** `config/base.yaml` owns `folds.n_folds` (default `5`); `cross_validation/cross_validator.py` re-exports it as `N_FOLDS` and builds that many date-group folds. `training/training.py` compares the persisted `fold_*` count against it and fails by name on a mismatch; `visualization/training_diagnostics.py` derives its loops and panel grid from it. Change the count in config, never as a literal.
- **Numeric constants come from config; structural literals do not.** The trainer's baseline recipe is the frozen `training.hyperparameters` section, read at call time by `model_trainer.default_params()`. Structural constants that stay literals in code are early stopping rounds (`30` in `model_trainer.py`), HAC lag (`4`), HAR warm-up days (`21`), the target window (`5`), and the minimum trading days (`500` in the dataset phase). Do not promote or demote these silently.
- **Determinism is pinned and recorded.** `training.n_jobs: 1` and `training.omp_num_threads: 1` pin the trainer and `OMP_NUM_THREADS` to one thread before every fit; `run_log.new_run` records the effective values, the seeds (`training.seed: 42`, selected `random_state: 43`), `PYTHONHASHSEED`, library versions, git sha, and `uv.lock` hash in the run manifest. `determinism.run_seed_sweep` writes `packages/ml/model/artifacts/results/seed_sweep.json` from validation-only fold views. Do not reintroduce multi-thread defaults.
- **Resume trusts only complete, verified fold records.** A fold is reused only when its checkpoint, params, summary, and predictions sidecars all exist and match their artifact-registry size/SHA-256 entries, and the summary matches the predictions hash. Resume verifies the manifest `run_id`, `config_hash`, `fold_manifest_hash`, and `input_identity` before Phase 2/3 can rewrite anything, and refuses by name on any mismatch — never a silent fallback to a fresh run. The runbook is `docs/operations.md`.
- **Retention prunes directories, never history.** `run.retention` (default `5`) keeps the newest `packages/ml/model/artifacts/checkpoints/<run_id>/` directories; `packages/ml/model/artifacts/runs.jsonl` is append-only and never pruned. Each kept directory carries a `README.md`. See `docs/operations.md`.
- **Subset runs are explicitly partial.** `--folds` emits only the selected folds and sets `partial: true`. `packages/ml/model/artifacts/results/run_selection.json` is written on every run with `partial: true/false`; a full run keeps one summary row per configured fold. Phase 5 renders only the emitted folds.
- **`INDICATORS` order is positional.** `feature_engineering.py` fixes column order via `X = df[INDICATORS].values`, and feature-importance arrays are mapped onto `INDICATORS` by index. Reorder the list and every importance chart silently mislabels. The module asserts the pinned count of `37`; renaming an entry means updating the dataset columns it reads. The full list and the two non-reconstructible columns are in `docs/pipelines/dataset.md`.
- **`market_regime` is a hardcoded calendar label**, not data-derived and not a model input. Seven buckets cover 2000-2007 through 2026, and a date past 2026 raises. Extend `packages/ml/dataset/features/regimes.py` before ingesting data past 2026.
- **Folds are date-group blocks with a train-only purge.** `cross_validator.py` splits the sorted unique dates into `n_folds + 1` contiguous blocks (six with the default five); fold k tests on block k+1 and trains on blocks 0..k minus the train-only purge. The last kept training date sits five complete trading dates before the first test date, enforced as `separation_min: 6` in unique-date index space. Each fold holds out the last `training.validation_tail_fraction: 0.15` of floor-retained training rows as a purged inner-validation tail. The geometry is `docs/pipelines/model.md`.
- **Tuning is validation-only with a mechanical test lock.** `tuning/harness.py` builds every trial's fold data through one locked accessor that reuses the purged inner-validation split and raises `TestLockError` on any view index inside a test block or the reserved holdout. The selection artifact keys on `validation_qlike` and rejects any `test_`-prefixed key. The campaign is opt-in and never runs in CI.
- **The holdout is locked before it is scored.** `evaluation/holdout.py` writes `packages/ml/model/artifacts/results/holdout_lock.json` and records the lock in the run manifest before the final refit or any holdout metric. A second evaluation raises `HoldoutLockError`; a post-lock resume records a skip; `--force-holdout` requires a recorded lock and still refuses a changed identity. The default no-argument run never reads the holdout. The full gate is `docs/operations.md`.
- **Time alignment is load-bearing.** Most price-derived features shift one day; the target uses `shift(-5)` and looks forward on purpose. The same-bar exceptions are `simple_return`, `price_vs_sma_{10,20,50}`, `zscore_price_vs_sma_20`, and `zscore_volume_20`. Do not assume a blanket one-day shift.
- **The training target is log(RV).** `training.py` floors the Yang-Zhang target at `1e-12` before the log and reports QLIKE as the headline loss, with R²/MAE/RMSE as log-scale diagnostics and dual OOS-R² (`1 - SSE_model/SSE_mean` and `1 - SSE_model/SSE_HAR`). At-or-below-floor rows are excluded from fitting, model scoring, and the HAR benchmark. The full metric definitions are `docs/pipelines/model.md` and `docs/results.md`.
- **Phase 1 writes into the model package's tree.** `packages/ml/dataset/artifacts.py` writes `packages/ml/model/artifacts/results/`, so relocating model artifacts requires editing the dataset phase too.
- **Early stopping is live in fold training and absent from the final holdout refit.** `model_trainer.py` sets `EARLY_STOPPING_ROUNDS = 30`; the holdout refit has no validation tail and no early stopping.
- **Per-run READMEs do not record input identity.** `retention.py` writes run id, config hash, fold-manifest hash, status, fold selection, resumed flag, contents, and the retention rule. Resume identity is the manifest's job; do not claim the README records it.

## Conventions

- Use the `logging` module (`logger = logging.getLogger(__name__)`), not `print`. The handler format is identical in `dataset/utils.py` and `model/utils.py`.
- Match each file's local indentation. Tab-indented files: `model/features/feature_engineering.py` (only the `INDICATORS` list uses tabs; its function bodies use 4 spaces), `model/training/model_trainer.py`, `model/training/training.py`. Everything else uses 4 spaces.
- There are no classes; use module-level functions with type hints. `from __future__ import annotations` leads every module; every `model/` package `__init__.py` carries it, the two `dataset/` ones do not.
- Errors fail fast: `raise ValueError` / `FileNotFoundError` with a message. Log-and-re-raise appears in `model/main.py` and in `training/_train_fold_model`; one boundary quarantines instead — `training.py` records `{"error": ...}` for a failing fold and continues, filtering invalid metrics with `n_valid`/`n_total` and raising only when the headline QLIKE has none.

## Known dead or broken code

- `.idea/workspace.xml` still carries a `RunManager` entry for the deleted `run_pipeline.py`. It is untracked and stale.

## Verification workflow

Run the model pytest suite for any change under `packages/ml/model`. The hermetic unit suite is the CI gate; the full suite adds the live artifact checks:

```bash
uv run --frozen python -m pytest packages/ml/model/tests -q --ignore=packages/ml/model/tests/live
uv run --frozen python -m pytest packages/ml/model/tests -q
```

The hermetic suite collects 270 tests. The full suite collects 275: the five extra checks live in `packages/ml/model/tests/live/` and skip when generated artifacts, the dataset parquet, or evaluation outputs are absent (a fresh clone or CI). The `static` job in `.github/workflows/checks.yml` runs the hermetic command after the byte-compile step. To check behavior, also run the phase you touched — both if the change spans `packages/ml/dataset` and `packages/ml/model`:

1. `python packages/ml/dataset/main.py` — confirm it completes and logs the row and ticker counts it wrote, the "Dropped ... NaN rows" line, and the parquet-then-JSON save order.
2. `python packages/ml/model/main.py` (or `python -m packages.ml.model.main`) — confirm one fold per configured fold trains and `packages/ml/model/artifacts/results/fold_training_summary.json` (per-fold R²/QLIKE) plus the HAR benchmark report are written. `--dry-run` logs the planned phases and exits before any phase runs; `--folds 0,2` trains only that selection; `--resume` continues the latest manifest and skips folds whose verified sidecar set is complete.
3. If commands or file layout changed, update the owning document (see below) in the same change.

Do not run `--evaluate-holdout` or `--force-holdout` as a verification step. Cap long-running or chatty commands: `python packages/ml/model/main.py 2>&1 | tail -c 4000`.

## Documentation ownership

- `README.md` — front door: identity, quickstart, pipeline summary, dated snapshot.
- `docs/pipelines/dataset.md` — Phase 1 deep reference.
- `docs/pipelines/model.md` — Phases 2-5 protocol, folds, training, evaluation.
- `docs/operations.md` — run lifecycle, resume, retention, artifacts, holdout gate.
- `docs/results.md` — reading the generated result artifacts.
- `docs/reports/` — dated evaluation reports (historical evidence, never rewritten).
- `docs/adr/` — frozen decision history; supersede, don't rewrite.
- `AGENTS.md` — this file, agent-critical rules and commands only.

Update the owning document when behavior changes; do not restate a fact in a second file when its canonical home already covers it.

## Where to look next

- `README.md` — what the project is, install, and run commands (human-facing).
- `docs/pipelines/dataset.md` and `docs/pipelines/model.md` — read before changing the phases.
- `docs/reference/cli.md` — canonical commands, flags, and examples.
- `docs/reference/dataset-schema.md` — 53-column dictionary with types, alignment, and warm-up.
- `docs/model-card.md` — intended use, protocol, and standing caveats.
- `packages/ml/model/features/feature_engineering.py` — `INDICATORS`, the single source of truth for model inputs.
- `packages/ml/model/cross_validation/cross_validator.py` — fold construction and the leakage guard.
- `packages/ml/model/training/model_trainer.py` — the trainer; its baseline recipe derives from the frozen `training.hyperparameters` config section.
- `packages/ml/model/utils.py` — `get_project_paths`, the central path helper.
- `docs/adr/` — architecture decision records own the rationale behind these rules.
