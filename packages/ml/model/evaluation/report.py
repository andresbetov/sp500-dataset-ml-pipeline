"""Evaluation report: per-row loss vectors and the audit statistics.

The module owns two artifacts, both written by ``phase_4_training``:

- ``results/per_row_losses.parquet``: one row per fold-test-block row with
  ``fold, row_idx, date, ticker, y_true, y_pred_model, y_pred_har, loss_model,
  loss_har, is_floor``. ``row_idx`` is the original input index; ``is_floor``
  is the persisted ``test_floor_mask`` inverted (True marks the at-or-below-
  floor rows the headline excludes), read from the mask rather than re-derived
  from the target; ``loss_model``/``loss_har`` are the elementwise
  ``exp(delta) - delta - 1`` terms of the single QLIKE functional.
- ``results/evaluation_summary.json``: the report computed from those vectors
  only (no report-time refit), projected through the shared strict-JSON
  sanitizer and written with ``allow_nan=False``.

The report carries the dual OOS-R² conventions the training phase emitted
(consumed, not recomputed), the **promoted QLIKE headline** (the unweighted
mean of the fold metrics with per-fold edges, per-fold `n_test`, the
n_valid/n_total counts, the separate row-weighted pooled values from the
persisted vectors, and the single-seed caveat), the DM/CW/GW tests on the
model-vs-HAR loss differentials (Newey-West HAC lag = horizon - 1 = 4),
Mincer-Zarnowitz regressions pooled and per fold with ``n_valid``, decile
QLIKE on global edges, regime QLIKE on the persisted metadata labels, QLIKE
mean/median/IQR/top-1% tail tables, the floor with/without decomposition plus
its top-1% shares, the date-clustered gap (descriptive mean ± SE with
``n_dates`` and ``n_valid``), expanding-origin loss curves per fold, and the
calibration stub.

Inference roles are labelled, not interchangeable: the cluster-by-date
``mean ± SE`` is descriptive (equal-weight dates; it understates uncertainty
under date-level serial dependence), while the DM/CW/GW statistics are the
inferential tests and use a Newey-West HAC covariance with lag 4 on the
date-mean differential series. Quarantined folds are deterministically
excluded from every pooled aggregate and from the gap/SE/DM/CW/GW/MZ inputs;
their per-fold descriptive blocks stay in the report flagged as quarantined.

The reporting layer never re-derives floor membership from the target (the
persisted ``is_floor`` column is the only source) and never reloads a model or
refits a benchmark.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from packages.ml.model.evaluation.stats import sanitize_json, spread
from packages.ml.model.training.har_benchmark import HAC_LAG
from packages.ml.model.training.qlike_objective import qlike_losses
from packages.ml.model.utils import normalize_dates

# The persisted vector schema, in column order. The per-row losses are derived
# from (y_true, y_pred_*) with the one QLIKE functional, never passed in.
VECTOR_COLUMNS = (
    "fold",
    "row_idx",
    "date",
    "ticker",
    "y_true",
    "y_pred_model",
    "y_pred_har",
    "loss_model",
    "loss_har",
    "is_floor",
)

# The per-fold record arrays a caller supplies before the loss columns are
# derived; ``loss_model``/``loss_har`` are never passed in.
VECTOR_INPUT_COLUMNS = (
    "fold",
    "row_idx",
    "date",
    "ticker",
    "y_true",
    "y_pred_model",
    "y_pred_har",
    "is_floor",
)

# The distribution tail: the top 1% of rows by loss, and its share of the
# total loss (k = max(1, ceil(0.01 * n)) rows).
TOP_TAIL_FRACTION = 0.01

# Decile tables use global (pooled) target edges applied to every fold, so the
# bins are comparable across folds. Regime tables cover the seven persisted
# integer calendar buckets.
N_DECILES = 10
N_REGIMES = 7

SCHEMA_VERSION = 1
SCALE_NOTE = "log target (log daily sigma); QLIKE in per-day sigma units"

# The promoted headline convention: one equal-weight metric per fold. The
# number of folds is config-sourced, so the note is formatted with the fold
# count actually reported (five in the pipeline configuration).
QLIKE_HEADLINE_CONVENTION = "unweighted mean of the {n_folds} fold metrics"
QLIKE_HEADLINE_WEIGHTING = (
    "each fold contributes one equal-weight metric regardless of its n_test; "
    "the aggregate is not row-weighted"
)
QLIKE_EDGE_DEFINITION = (
    "edge = har_qlike - model_qlike per fold and har_qlike - model_qlike "
    "between the aggregate means; positive means the model has the lower QLIKE"
)
QLIKE_POOLED_WEIGHTING = (
    "row-weighted mean over every floor-retained row of the valid folds, "
    "computed from the persisted per-row loss vectors; not an average of the "
    "per-fold ratios"
)
QLIKE_SINGLE_SEED_CAVEAT = (
    "single seed: the fold metrics come from one training run (one seed); "
    "seed-to-seed uncertainty is not estimated, so the aggregate is a "
    "run-level result, not a seed-level confidence statement"
)

# The deferred post-hoc adjustment route was not selected (the QLIKE-consistent
# objective is the estimand): the report declares the status explicitly instead
# of shipping a fitted factor, so no fake numbers can enter an artifact.
CALIBRATION_STATUS = "not applied"
CALIBRATION_NOTE = (
    "Calibration is not applied: no fitted per-fold factor and no "
    "distributional correction of the log forecasts ships with this report. "
    "The deferred interface (per-row predictions + purged validation tails) "
    "remains documented for a future ruling."
)
CALIBRATION_INTERFACE = "per-row log forecasts + purged inner-validation tails"


def _fold_index(fold_key: str) -> int:
    """The integer index behind a ``fold_<i>`` key, for deterministic ordering."""
    try:
        return int(str(fold_key).split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"fold keys must be fold_<index>; got {fold_key!r}") from exc


def _finite_array(values, name: str) -> np.ndarray:
    """A flat float64 array that refuses non-finite values by name."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} received no values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} received non-finite values")
    return array


def loss_vector_frame(records: list[dict]) -> pd.DataFrame:
    """Assemble the per-row loss-vector frame from per-fold record arrays.

    Each record carries the fold key, the original row indices, the metadata
    date/ticker for those rows, the floored log target, the model and HAR log
    forecasts for every test-block row, and the persisted floor flag
    (``True`` = at-or-below-floor, headline-excluded). The per-row losses are
    computed here with ``qlike_losses`` — the elementwise core of the same
    functional as ``qlike_loss`` — so the vector and the headline cannot
    diverge.
    """
    if not records:
        raise ValueError("loss_vector_frame received no fold records")
    frames = []
    for record in records:
        missing = [key for key in VECTOR_INPUT_COLUMNS if key not in record]
        if missing:
            raise ValueError(f"loss vector record is missing keys: {missing}")
        row_idx = np.asarray(record["row_idx"], dtype=np.int64).reshape(-1)
        dates = normalize_dates(record["date"]).reshape(-1)
        tickers = np.asarray(record["ticker"]).reshape(-1)
        y_true = _finite_array(record["y_true"], "loss vector y_true")
        y_pred_model = _finite_array(
            record["y_pred_model"], "loss vector y_pred_model"
        )
        y_pred_har = _finite_array(record["y_pred_har"], "loss vector y_pred_har")
        is_floor = np.asarray(record["is_floor"], dtype=bool).reshape(-1)
        lengths = {
            len(row_idx),
            len(dates),
            len(tickers),
            len(y_true),
            len(y_pred_model),
            len(y_pred_har),
            len(is_floor),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"{record['fold']}: loss vector arrays are misaligned: "
                f"{sorted(lengths)}"
            )
        frames.append(
            pd.DataFrame(
                {
                    "fold": str(record["fold"]),
                    "row_idx": row_idx,
                    "date": dates,
                    "ticker": tickers,
                    "y_true": y_true,
                    "y_pred_model": y_pred_model,
                    "y_pred_har": y_pred_har,
                    "loss_model": qlike_losses(y_true - y_pred_model),
                    "loss_har": qlike_losses(y_true - y_pred_har),
                    "is_floor": is_floor,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _validate_frame(vectors: pd.DataFrame) -> pd.DataFrame:
    """Validate the persisted vector schema before any statistic consumes it."""
    if not isinstance(vectors, pd.DataFrame):
        raise TypeError(
            f"evaluation vectors must be a DataFrame, got {type(vectors).__name__}"
        )
    missing = [column for column in VECTOR_COLUMNS if column not in vectors.columns]
    if missing:
        raise ValueError(f"evaluation vectors are missing columns: {missing}")
    if vectors.empty:
        raise ValueError("evaluation vectors are empty")
    if vectors["fold"].isna().any():
        raise ValueError("evaluation vectors carry a missing fold key")
    for fold_key in vectors["fold"].unique():
        _fold_index(fold_key)
    for column in (
        "y_true",
        "y_pred_model",
        "y_pred_har",
        "loss_model",
        "loss_har",
    ):
        values = vectors[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"evaluation vectors carry non-finite {column}")
    return vectors


def loss_distribution(losses) -> dict:
    """The QLIKE distribution block: mean, median, IQR, and the top-1% tail.

    ``top1_mean`` is the mean of the largest ``max(1, ceil(0.01 * n))`` losses
    and ``top1_share`` their share of the total loss, so a heavy tail is
    described by both its level and its contribution. An empty input returns
    the block with ``n = 0`` and null statistics.
    """
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    n = int(values.size)
    empty = {
        "n": 0,
        "mean": None,
        "median": None,
        "q25": None,
        "q75": None,
        "iqr": None,
        "top1_mean": None,
        "top1_share": None,
    }
    if n == 0:
        return empty
    if not np.isfinite(values).all():
        raise ValueError("loss_distribution received non-finite losses")

    q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    tail_count = max(1, int(math.ceil(TOP_TAIL_FRACTION * n)))
    tail = np.sort(values)[-tail_count:]
    total = float(values.sum())
    return {
        "n": n,
        "mean": float(values.mean()),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "top1_mean": float(tail.mean()),
        "top1_share": float(tail.sum() / total) if total != 0.0 else None,
    }


def decile_edges(values, n_deciles: int = N_DECILES) -> list[float]:
    """The global decile edges (n_deciles + 1 quantiles) of a target set."""
    array = _finite_array(values, "decile_edges")
    if n_deciles < 2:
        raise ValueError(f"decile_edges requires at least 2 bins, got {n_deciles}")
    return [
        float(edge)
        for edge in np.quantile(array, np.linspace(0.0, 1.0, n_deciles + 1))
    ]


def _assign_deciles(values: np.ndarray, edges: list[float], n_deciles: int) -> np.ndarray:
    """Bin values by the inner global edges, matching (edge_low, edge_high]."""
    inner = np.asarray(edges[1:-1], dtype=np.float64)
    if len(edges) != n_deciles + 1:
        raise ValueError(
            f"decile table needs {n_deciles + 1} edges, got {len(edges)}"
        )
    return np.clip(
        np.searchsorted(inner, values, side="right"), 0, n_deciles - 1
    )


def decile_table(
    y_true,
    loss_model,
    loss_har,
    edges: list[float],
    n_deciles: int = N_DECILES,
) -> list[dict]:
    """QLIKE by target decile, with the global edges restated on each bin.

    The target edges come from the pooled headline rows and are applied to
    every frame, so bin k covers the same log-volatility range everywhere.
    ``mean_y_true_log`` is the bin's mean log target (the sigma-scale centre is
    its exponential); empty bins keep their index and null statistics.
    """
    targets = _finite_array(y_true, "decile_table y_true")
    model = _finite_array(loss_model, "decile_table loss_model")
    har = _finite_array(loss_har, "decile_table loss_har")
    if not (targets.size == model.size == har.size):
        raise ValueError("decile_table received misaligned arrays")
    if len(edges) != n_deciles + 1:
        raise ValueError(
            f"decile_table needs {n_deciles + 1} edges, got {len(edges)}"
        )
    bins = _assign_deciles(targets, edges, n_deciles)
    rows = []
    for decile in range(n_deciles):
        mask = bins == decile
        count = int(mask.sum())
        if count:
            mean_target = float(targets[mask].mean())
            sigma_center = (
                float(math.exp(mean_target)) if math.isfinite(mean_target) else None
            )
        else:
            mean_target = sigma_center = None
        rows.append(
            {
                "decile": decile,
                "edge_low": float(edges[decile]),
                "edge_high": float(edges[decile + 1]),
                "n_rows": count,
                "mean_y_true_log": mean_target,
                "sigma_center": sigma_center,
                "qlike_model": float(model[mask].mean()) if count else None,
                "qlike_har": float(har[mask].mean()) if count else None,
            }
        )
    return rows


def regime_table(
    regime,
    loss_model,
    loss_har,
    n_regimes: int = N_REGIMES,
) -> list[dict]:
    """QLIKE by the persisted integer calendar bucket (0..n_regimes - 1)."""
    labels = np.asarray(regime)
    model = _finite_array(loss_model, "regime_table loss_model")
    har = _finite_array(loss_har, "regime_table loss_har")
    if not (labels.size == model.size == har.size):
        raise ValueError("regime_table received misaligned arrays")
    numeric = labels.astype(np.float64)
    in_buckets = (
        np.isfinite(numeric)
        & (numeric >= 0)
        & (numeric <= n_regimes - 1)
        & (numeric == np.floor(numeric))
    )
    if not in_buckets.all():
        raise ValueError(
            f"regime_table labels must be integers in 0..{n_regimes - 1}; "
            f"got {np.unique(labels[~in_buckets]).tolist()}"
        )
    rows = []
    for bucket in range(n_regimes):
        mask = labels.astype(np.int64) == bucket
        count = int(mask.sum())
        rows.append(
            {
                "regime": bucket,
                "n_rows": count,
                "qlike_model": float(model[mask].mean()) if count else None,
                "qlike_har": float(har[mask].mean()) if count else None,
            }
        )
    return rows


def _date_means(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Per-date means in ascending date order, one equal-weight value per date."""
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    if unique_dates.size == 0:
        raise ValueError("date aggregation received no dates")
    sums = np.bincount(inverse, weights=values, minlength=unique_dates.size)
    counts = np.bincount(inverse, minlength=unique_dates.size)
    return sums / counts


def _hac_ols(
    design: np.ndarray, target: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett) HAC covariance.

    The sandwich is ``(X'X)^-1 X' Omega X (X'X)^-1`` with the residual second
    moments at lag 0 plus the Bartlett-weighted autocovariances up to ``lag``.
    A regression on a constant gives the HAC variance of the mean, so the same
    implementation serves the DM/CW/GW denominators and the MZ coefficients.
    """
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if design.ndim != 2 or target.ndim != 1:
        raise ValueError("hac_ols expects a 2-D design and a 1-D target")
    n_rows, n_columns = design.shape
    if target.size != n_rows:
        raise ValueError(
            f"hac_ols received {n_rows} design rows for {target.size} targets"
        )
    if n_rows <= n_columns:
        raise ValueError(
            f"hac_ols requires more rows than parameters: {n_rows} rows for "
            f"{n_columns} parameters"
        )
    if lag < 0:
        raise ValueError(f"hac_ols lag must be nonnegative, got {lag}")
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    bread = np.linalg.pinv(design.T @ design)
    meat = design.T @ (design * (residual**2)[:, None])
    for offset in range(1, lag + 1):
        if offset >= n_rows:
            break
        weight = 1.0 - offset / (lag + 1)
        cross = (
            design[offset:] * (residual[offset:] * residual[:-offset])[:, None]
        ).T @ design[:-offset]
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    covariance = 0.5 * (covariance + covariance.T)
    return coefficients, covariance


def _coefficient_blocks(coefficients: np.ndarray, covariance: np.ndarray) -> tuple:
    """Standard errors, t statistics, and two-sided p-values per coefficient."""
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    t_stats: list[float | None] = []
    p_values: list[float | None] = []
    for value, error in zip(coefficients, standard_errors):
        if error > 0.0 and math.isfinite(error):
            statistic = float(value / error)
            t_stats.append(statistic)
            p_values.append(math.erfc(abs(statistic) / math.sqrt(2.0)))
        else:
            t_stats.append(None)
            p_values.append(None)
    return standard_errors, t_stats, p_values


def _mean_hac_se(series: np.ndarray, lag: int) -> float:
    """The Newey-West HAC standard error of a series' mean."""
    design = np.ones((series.size, 1), dtype=np.float64)
    _coefficients, covariance = _hac_ols(design, series, lag)
    return float(math.sqrt(max(float(covariance[0, 0]), 0.0)))


def dm_test(loss_model, loss_har, dates, lag: int = HAC_LAG) -> dict:
    """Diebold-Mariano test on the date-mean QLIKE differential.

    The loss differential is ``loss_model - loss_har``; rows are first
    aggregated to equal-weight per-date means, then the DM statistic is the
    mean of that series over its Newey-West HAC standard error (lag 4, the
    five-day label overlap). A zero differential gives exactly ``0``; a
    degenerate zero-variance non-zero mean leaves the statistic undefined
    (``None``) rather than fabricating a value.
    """
    model = _finite_array(loss_model, "dm_test loss_model")
    har = _finite_array(loss_har, "dm_test loss_har")
    dates = np.asarray(dates).reshape(-1)
    if not (model.size == har.size == dates.size):
        raise ValueError("dm_test received misaligned arrays")
    differential = model - har
    date_means = _date_means(differential, dates)
    mean = float(date_means.mean())
    se = 0.0 if date_means.size < 2 else _mean_hac_se(date_means, lag)
    if se == 0.0:
        statistic = 0.0 if mean == 0.0 else None
    else:
        statistic = mean / se
    p_value = (
        min(1.0, 2.0 * math.erfc(abs(statistic) / math.sqrt(2.0)))
        if statistic is not None
        else None
    )
    return {
        "loss": "QLIKE",
        "differential": "model - HAR",
        "hac_lag": int(lag),
        "mean_differential": mean,
        "hac_se": se,
        "statistic": statistic,
        "p_value": p_value,
        "n_rows": int(differential.size),
        "n_dates": int(date_means.size),
        "null": "equal expected loss (two-sided)",
    }


def cw_test(
    y_true,
    y_pred_model,
    y_pred_har,
    dates,
    lag: int = HAC_LAG,
) -> dict:
    """Clark-West adjusted MSPE diagnostic on the date-mean differential.

    ``f_t = e_har^2 - (e_model^2 - (y_pred_model - y_pred_har)^2)`` is the
    nested-model adjusted squared-error differential (positive favours the
    larger model). HAR is not nested in the XGB model, so ``nesting_claimed``
    stays ``False`` and the block is reported as a diagnostic with that
    precondition stated; the aggregation and HAC covariance match the DM
    test (equal-weight dates, lag 4).
    """
    target = _finite_array(y_true, "cw_test y_true")
    model = _finite_array(y_pred_model, "cw_test y_pred_model")
    har = _finite_array(y_pred_har, "cw_test y_pred_har")
    dates = np.asarray(dates).reshape(-1)
    if not (target.size == model.size == har.size == dates.size):
        raise ValueError("cw_test received misaligned arrays")
    adjusted = (target - har) ** 2 - (
        (target - model) ** 2 - (model - har) ** 2
    )
    date_means = _date_means(adjusted, dates)
    mean = float(date_means.mean())
    se = 0.0 if date_means.size < 2 else _mean_hac_se(date_means, lag)
    if se == 0.0:
        statistic = 0.0 if mean == 0.0 else None
    else:
        statistic = mean / se
    p_value = (
        0.5 * math.erfc(statistic / math.sqrt(2.0))
        if statistic is not None
        else None
    )
    return {
        "loss": "adjusted squared error (Clark-West MSPE adjustment)",
        "hac_lag": int(lag),
        "mean_adjusted_differential": mean,
        "hac_se": se,
        "statistic": statistic,
        "p_value": p_value,
        "n_rows": int(adjusted.size),
        "n_dates": int(date_means.size),
        "nesting_claimed": False,
        "note": (
            "HAR is not nested in the XGB model; the adjusted MSPE statistic "
            "is reported as a diagnostic with the nesting precondition not "
            "claimed (one-sided upper p-value)."
        ),
    }


def gw_test(loss_model, loss_har, dates, lag: int = HAC_LAG) -> dict:
    """Giacomini-White conditional-ability test on the date-mean differential.

    The date-mean differential is regressed on a constant and its first lag
    (the instrument set); the reported statistic is the joint Wald test that
    both coefficients are zero, with the same Newey-West HAC covariance as the
    DM test. A zero differential gives a zero statistic (p = 1).
    """
    model = _finite_array(loss_model, "gw_test loss_model")
    har = _finite_array(loss_har, "gw_test loss_har")
    dates = np.asarray(dates).reshape(-1)
    if not (model.size == har.size == dates.size):
        raise ValueError("gw_test received misaligned arrays")
    date_means = _date_means(model - har, dates)
    if date_means.size < 3:
        return {
            "instruments": "constant + one lag of the date-mean differential",
            "hac_lag": int(lag),
            "statistic": None,
            "p_value": None,
            "coefficients": [],
            "t_stats": [],
            "p_values": [],
            "n_dates": int(date_means.size),
            "null": "E[d_t | I_{t-1}] = 0 (conditional equal predictive ability)",
            "note": "fewer than three dates: the conditional test is undefined",
        }
    target = date_means[1:]
    design = np.column_stack([np.ones(target.size), date_means[:-1]])
    coefficients, covariance = _hac_ols(design, target, lag)
    standard_errors, t_stats, p_values = _coefficient_blocks(
        coefficients, covariance
    )
    wald = float(coefficients @ np.linalg.pinv(covariance) @ coefficients)
    wald = max(wald, 0.0)
    return {
        "instruments": "constant + one lag of the date-mean differential",
        "hac_lag": int(lag),
        "statistic": wald,
        "p_value": math.exp(-0.5 * wald),
        "coefficients": [float(value) for value in coefficients],
        "t_stats": t_stats,
        "p_values": p_values,
        "n_dates": int(date_means.size),
        "null": "E[d_t | I_{t-1}] = 0 (conditional equal predictive ability)",
        "note": "Wald chi-square with 2 degrees of freedom",
    }


def mz_regression(y_true, y_pred, lag: int = HAC_LAG) -> dict:
    """Mincer-Zarnowitz regression ``y_true = alpha + beta * y_pred + e``.

    OLS with Newey-West HAC standard errors (lag 4). Unbiased forecasts give
    ``alpha = 0`` and ``beta = 1``; a constant forecast bias moves ``alpha``
    and a scale bias moves ``beta``. A degenerate zero-residual fit reports
    null t statistics instead of dividing by zero.
    """
    target = _finite_array(y_true, "mz_regression y_true")
    forecast = _finite_array(y_pred, "mz_regression y_pred")
    if target.size != forecast.size:
        raise ValueError("mz_regression received misaligned arrays")
    design = np.column_stack([np.ones(target.size), forecast])
    coefficients, covariance = _hac_ols(design, target, lag)
    standard_errors, t_stats, p_values = _coefficient_blocks(
        coefficients, covariance
    )
    return {
        "spec": "y_true = alpha + beta * y_pred + e (OLS, Newey-West HAC SEs)",
        "hac_lag": int(lag),
        "alpha": float(coefficients[0]),
        "beta": float(coefficients[1]),
        "se_alpha": float(standard_errors[0]),
        "se_beta": float(standard_errors[1]),
        "t_alpha": t_stats[0],
        "p_alpha": p_values[0],
        "t_beta": t_stats[1],
        "p_beta": p_values[1],
        "n_obs": int(target.size),
    }


def date_clustered_gap(loss_model, loss_har, dates) -> dict:
    """Descriptive cluster-by-date mean ± SE of the QLIKE gap.

    The differential is averaged per date (equal weight per date), and the
    standard error is the sample standard deviation of those date means over
    ``sqrt(n_dates)``. This is the descriptive uncertainty; the DM/CW/GW
    blocks are the HAC-robust inferential tests. A constant gap has SE 0.
    """
    model = _finite_array(loss_model, "date_clustered_gap loss_model")
    har = _finite_array(loss_har, "date_clustered_gap loss_har")
    dates = np.asarray(dates).reshape(-1)
    if not (model.size == har.size == dates.size):
        raise ValueError("date_clustered_gap received misaligned arrays")
    date_means = _date_means(model - har, dates)
    if date_means.size < 2:
        se = 0.0
    else:
        se = float(spread(date_means, ddof=1) / math.sqrt(date_means.size))
    return {
        "mean": float(date_means.mean()),
        "se": se,
        "n_dates": int(date_means.size),
        "n_rows": int(model.size),
    }


def expanding_origin_curve(dates, loss_model, loss_har) -> dict:
    """Expanding-mean loss curves in date order (the forward-origin diagnostic).

    Every row of the fold's test block is included and ordered by (date, row
    order); the curve at position ``i`` is the mean loss over the first
    ``i + 1`` rows. ``n`` is the vector row count (``n_test`` of the fold).
    """
    model = _finite_array(loss_model, "expanding_origin_curve loss_model")
    har = _finite_array(loss_har, "expanding_origin_curve loss_har")
    dates = np.asarray(dates).reshape(-1)
    if not (model.size == har.size == dates.size):
        raise ValueError("expanding_origin_curve received misaligned arrays")
    order = np.lexsort((np.arange(model.size), dates))
    positions = np.arange(1, model.size + 1, dtype=np.float64)
    sorted_model = model[order]
    sorted_har = har[order]
    cumulative_model = np.cumsum(sorted_model) / positions
    cumulative_har = np.cumsum(sorted_har) / positions
    return {
        "n": int(model.size),
        "order": order,
        "dates": dates[order],
        "loss_model": cumulative_model,
        "loss_har": cumulative_har,
        "edge": cumulative_model - cumulative_har,
    }


def _iso_date(value) -> str:
    """A JSON-safe day stamp for a datetime64 value."""
    return np.datetime_as_string(np.asarray(value).astype("datetime64[ns]"), unit="D")


def expanding_origin_block(
    dates, loss_model, loss_har, n_checkpoints: int = 11
) -> dict:
    """The JSON projection of one fold's expanding-origin curve.

    The full curve is one value per vector row (``n_points``); the summary
    stores evenly spaced checkpoints plus the final value instead of the whole
    array, and the curve itself is reproducible from the persisted vectors.
    """
    curve = expanding_origin_curve(dates, loss_model, loss_har)
    n = curve["n"]
    positions = sorted(
        {
            int(round(position))
            for position in np.linspace(0.0, n - 1.0, min(n_checkpoints, n))
        }
    )
    checkpoints = [
        {
            "index": position,
            "date": _iso_date(curve["dates"][position]),
            "loss_model": float(curve["loss_model"][position]),
            "loss_har": float(curve["loss_har"][position]),
            "edge": float(curve["edge"][position]),
        }
        for position in positions
    ]
    return {
        "n_points": n,
        "checkpoints": checkpoints,
        "final": {
            "loss_model": float(curve["loss_model"][-1]),
            "loss_har": float(curve["loss_har"][-1]),
            "edge": float(curve["edge"][-1]),
        },
    }


def _side_block(frame: pd.DataFrame) -> dict:
    """One with/without-floor side: model/HAR distributions plus their edge."""
    model = loss_distribution(frame["loss_model"].to_numpy())
    har = loss_distribution(frame["loss_har"].to_numpy())
    edge = (
        None
        if model["mean"] is None or har["mean"] is None
        else float(model["mean"] - har["mean"])
    )
    return {
        "n_rows": int(len(frame)),
        "model": model,
        "har": har,
        "edge": edge,
    }


def floor_decomposition(
    vectors: pd.DataFrame, *, total_floor_rows: int | None = None
) -> dict:
    """The with/without-floor QLIKE decomposition for one frame.

    ``with_floor`` covers every vector row (the fold's whole test block);
    ``without_floor`` covers only the persisted ``is_floor == False`` headline
    rows. Both sides carry the full distribution block (mean, median, IQR,
    top-1% mean and loss share) and the model-minus-HAR edge, so the floor
    policy's effect on levels and on the edge is visible. Floor membership is
    read from the ``is_floor`` column, never re-derived from the target.
    """
    is_floor = vectors["is_floor"].to_numpy(dtype=bool)
    with_floor = _side_block(vectors)
    without_floor = _side_block(vectors[~is_floor])
    edge_change = (
        None
        if with_floor["edge"] is None or without_floor["edge"] is None
        else float(with_floor["edge"] - without_floor["edge"])
    )
    block = {
        "n_rows": int(len(vectors)),
        "n_floor_rows": int(is_floor.sum()),
        "with_floor": with_floor,
        "without_floor": without_floor,
        "edge_change": edge_change,
    }
    if total_floor_rows is not None:
        in_test_blocks = int(is_floor.sum())
        if int(total_floor_rows) < in_test_blocks:
            raise ValueError(
                f"total_floor_rows={total_floor_rows} is smaller than the "
                f"{in_test_blocks} floor rows in the vectors"
            )
        block["counts"] = {
            "n_floor_rows_total": int(total_floor_rows),
            "n_floor_rows_in_test_blocks": in_test_blocks,
            "n_floor_rows_never_evaluated": int(total_floor_rows) - in_test_blocks,
            "n_rows": int(len(vectors)),
        }
    return block


def calibration_stub() -> dict:
    """The calibration status block: explicit "not applied", no fake numbers."""
    return {
        "status": CALIBRATION_STATUS,
        "applied": False,
        "interface": CALIBRATION_INTERFACE,
        "note": CALIBRATION_NOTE,
    }


def _inference_roles() -> dict:
    """The role labels that keep the two variance estimates distinguishable."""
    return {
        "cluster_se": (
            "descriptive: cluster-by-date standard error of the equal-weight "
            "per-date mean QLIKE gap (understates uncertainty under date-level "
            "serial dependence)"
        ),
        "dm_hac": (
            "inferential: DM/CW/GW on the date-mean differential with a "
            "Newey-West HAC covariance (lag 4, the five-day label overlap)"
        ),
    }


def build_headline_block(
    vectors: pd.DataFrame,
    fold_keys: list[str],
    valid_folds: list[str],
    quarantined_folds: dict,
    headline_row_set: str,
) -> dict:
    """The promoted QLIKE headline: equal-weight fold mean plus pooled context.

    ``aggregate`` is the **unweighted mean of the fold metrics**: each fold's
    model and HAR QLIKE is first averaged over its own floor-retained rows
    (exactly the fold metric the training summary reports), then the fold
    values are averaged with equal weight. ``pooled`` is the separate
    row-weighted mean over every retained vector row of the valid folds, so it
    answers a different question (the average row's loss) and is never an
    average of fold ratios. Per-fold edges use ``har_qlike - model_qlike``
    (positive = model lower loss), and the aggregate edge is the difference of
    the two aggregate means. Quarantined folds stay listed with null metrics
    and are excluded from both aggregates.
    """
    retained = vectors[~vectors["is_floor"].to_numpy(dtype=bool)]
    folds = {}
    model_values = []
    har_values = []
    for index, fold_key in enumerate(fold_keys, start=1):
        fold_retained = retained[retained["fold"] == fold_key]
        fold_all = vectors[vectors["fold"] == fold_key]
        quarantined = fold_key in quarantined_folds
        if quarantined or fold_retained.empty:
            model_qlike = har_qlike = edge = None
        else:
            model_qlike = float(fold_retained["loss_model"].mean())
            har_qlike = float(fold_retained["loss_har"].mean())
            edge = har_qlike - model_qlike
            model_values.append(model_qlike)
            har_values.append(har_qlike)
        folds[fold_key] = {
            "label": f"{fold_key} ({index} of {len(fold_keys)})",
            "n_test": int(len(fold_retained)),
            "n_test_total": int(len(fold_all)),
            "model_qlike": model_qlike,
            "har_qlike": har_qlike,
            "edge": edge,
            "quarantined": bool(quarantined),
        }

    aggregate_model = float(np.mean(model_values)) if model_values else None
    aggregate_har = float(np.mean(har_values)) if har_values else None
    aggregate_edge = (
        None
        if aggregate_model is None or aggregate_har is None
        else aggregate_har - aggregate_model
    )

    pooled_frame = retained[retained["fold"].isin(valid_folds)]
    pooled_model = (
        float(pooled_frame["loss_model"].mean()) if not pooled_frame.empty else None
    )
    pooled_har = (
        float(pooled_frame["loss_har"].mean()) if not pooled_frame.empty else None
    )
    pooled_edge = (
        None
        if pooled_model is None or pooled_har is None
        else pooled_har - pooled_model
    )

    return {
        "metric": "QLIKE",
        "convention": QLIKE_HEADLINE_CONVENTION.format(n_folds=len(fold_keys)),
        "weighting": QLIKE_HEADLINE_WEIGHTING,
        "edge_definition": QLIKE_EDGE_DEFINITION,
        "row_set": headline_row_set,
        "scale": SCALE_NOTE,
        "n_valid": len(model_values),
        "n_total": len(fold_keys),
        "excluded_folds": sorted(quarantined_folds),
        "aggregate": {
            "model_qlike": aggregate_model,
            "har_qlike": aggregate_har,
            "edge": aggregate_edge,
        },
        "pooled": {
            "model_qlike": pooled_model,
            "har_qlike": pooled_har,
            "edge": pooled_edge,
            "n_rows": int(len(pooled_frame)),
            "n_valid_folds": len(valid_folds),
            "n_total_folds": len(fold_keys),
            "excluded_folds": sorted(quarantined_folds),
            "weighting": QLIKE_POOLED_WEIGHTING,
            "source": (
                "persisted per-row loss vectors "
                "(results/per_row_losses.parquet)"
            ),
        },
        "folds": folds,
        "caveat": QLIKE_SINGLE_SEED_CAVEAT,
    }


def build_evaluation_summary(
    vectors: pd.DataFrame,
    *,
    dual_oos_r2: dict,
    regime,
    headline_row_set: str,
    total_floor_rows: int,
    quarantined_folds: dict | None = None,
    hac_lag: int = HAC_LAG,
) -> dict:
    """Build the strict-JSON evaluation summary from the persisted vectors.

    Every statistic is computed from ``vectors`` (and the persisted regime
    labels indexed by ``row_idx``); the dual OOS-R² values are consumed from
    ``dual_oos_r2`` as the training phase emitted them. The ``headline`` block
    is the promoted QLIKE aggregate: the unweighted mean of the fold metrics
    (each fold's loss over its own floor-retained rows), with per-fold edges
    and counts, plus the separate row-weighted pooled values from the same
    vectors and the single-seed caveat. Quarantined folds are excluded from
    every pooled aggregate and from the gap/SE/DM/CW/GW/MZ inputs; their
    per-fold descriptive blocks remain in the report flagged with
    ``quarantined: true``.
    """
    vectors = _validate_frame(vectors)
    quarantine = dict(quarantined_folds or {})
    fold_keys = sorted(vectors["fold"].unique(), key=_fold_index)
    valid_folds = [fold_key for fold_key in fold_keys if fold_key not in quarantine]
    if not valid_folds:
        raise ValueError(
            "build_evaluation_summary: every fold is quarantined; no valid "
            "fold contributes a pooled statistic"
        )
    retained = vectors[~vectors["is_floor"].to_numpy(dtype=bool)]
    valid_retained = retained[retained["fold"].isin(valid_folds)]
    valid_all = vectors[vectors["fold"].isin(valid_folds)]
    if valid_retained.empty:
        raise ValueError(
            "build_evaluation_summary: the non-quarantined folds hold no "
            "headline (floor-retained) rows"
        )

    regime_values = np.asarray(regime)
    if regime_values.ndim != 1 or regime_values.size <= int(
        vectors["row_idx"].max()
    ):
        raise ValueError(
            "build_evaluation_summary: the regime labels are not aligned with "
            f"the original row indices ({regime_values.size} labels for a max "
            f"row_idx of {int(vectors['row_idx'].max())})"
        )

    def regime_for(frame: pd.DataFrame) -> np.ndarray:
        return regime_values[frame["row_idx"].to_numpy(dtype=np.int64)]

    def distribution_block(frame: pd.DataFrame) -> dict:
        return {
            "n_rows": int(len(frame)),
            "model": loss_distribution(frame["loss_model"].to_numpy()),
            "har": loss_distribution(frame["loss_har"].to_numpy()),
        }

    loss_distributions = {
        "folds": {
            fold_key: {
                **distribution_block(retained[retained["fold"] == fold_key]),
                "quarantined": fold_key in quarantine,
            }
            for fold_key in fold_keys
        },
        "pooled": {
            **distribution_block(valid_retained),
            "n_valid_folds": len(valid_folds),
            "n_total_folds": len(fold_keys),
            "excluded_folds": sorted(quarantine),
        },
    }

    in_test_blocks = int(vectors["is_floor"].sum())
    if int(total_floor_rows) < in_test_blocks:
        raise ValueError(
            f"total_floor_rows={total_floor_rows} is smaller than the "
            f"{in_test_blocks} floor rows in the vectors"
        )
    floor = {
        "counts": {
            "n_floor_rows_total": int(total_floor_rows),
            "n_floor_rows_in_test_blocks": in_test_blocks,
            "n_floor_rows_never_evaluated": int(total_floor_rows) - in_test_blocks,
            "n_rows": int(len(vectors)),
        },
        "row_set": (
            "headline = without_floor (floor-retained rows); with_floor adds "
            "the persisted at-or-below-floor test-block rows"
        ),
        "folds": {
            fold_key: floor_decomposition(vectors[vectors["fold"] == fold_key])
            for fold_key in fold_keys
        },
        "pooled": {
            **floor_decomposition(valid_all),
            "n_valid_folds": len(valid_folds),
            "n_total_folds": len(fold_keys),
            "excluded_folds": sorted(quarantine),
        },
    }

    edges = decile_edges(valid_retained["y_true"].to_numpy())
    deciles = {
        "n_deciles": N_DECILES,
        "edges": edges,
        "edges_source": (
            "global pooled edges over the headline (floor-retained) rows of "
            "the non-quarantined folds, applied to every fold"
        ),
        "folds": {
            fold_key: {
                "n_rows": int(len(retained[retained["fold"] == fold_key])),
                "quarantined": fold_key in quarantine,
                "bins": decile_table(
                    retained.loc[retained["fold"] == fold_key, "y_true"].to_numpy(),
                    retained.loc[
                        retained["fold"] == fold_key, "loss_model"
                    ].to_numpy(),
                    retained.loc[
                        retained["fold"] == fold_key, "loss_har"
                    ].to_numpy(),
                    edges,
                ),
            }
            for fold_key in fold_keys
        },
        "pooled": {
            "n_rows": int(len(valid_retained)),
            "n_valid_folds": len(valid_folds),
            "n_total_folds": len(fold_keys),
            "excluded_folds": sorted(quarantine),
            "bins": decile_table(
                valid_retained["y_true"].to_numpy(),
                valid_retained["loss_model"].to_numpy(),
                valid_retained["loss_har"].to_numpy(),
                edges,
            ),
        },
    }

    regimes = {
        "labels": "metadata['regime'] integer calendar buckets 0..6",
        "folds": {
            fold_key: {
                "n_rows": int(len(retained[retained["fold"] == fold_key])),
                "quarantined": fold_key in quarantine,
                "bins": regime_table(
                    regime_for(retained[retained["fold"] == fold_key]),
                    retained.loc[
                        retained["fold"] == fold_key, "loss_model"
                    ].to_numpy(),
                    retained.loc[
                        retained["fold"] == fold_key, "loss_har"
                    ].to_numpy(),
                ),
            }
            for fold_key in fold_keys
        },
        "pooled": {
            "n_rows": int(len(valid_retained)),
            "n_valid_folds": len(valid_folds),
            "n_total_folds": len(fold_keys),
            "excluded_folds": sorted(quarantine),
            "bins": regime_table(
                regime_for(valid_retained),
                valid_retained["loss_model"].to_numpy(),
                valid_retained["loss_har"].to_numpy(),
            ),
        },
    }

    gap_folds = {}
    dm_folds = {}
    cw_folds = {}
    gw_folds = {}
    mz_folds = {}
    for fold_key in valid_folds:
        frame = retained[retained["fold"] == fold_key]
        dates = frame["date"].to_numpy()
        model = frame["loss_model"].to_numpy()
        har = frame["loss_har"].to_numpy()
        gap_folds[fold_key] = {
            **date_clustered_gap(model, har, dates),
            "n_valid": 1,
            "n_total": 1,
        }
        dm_folds[fold_key] = dm_test(model, har, dates, lag=hac_lag)
        cw_folds[fold_key] = cw_test(
            frame["y_true"].to_numpy(),
            frame["y_pred_model"].to_numpy(),
            frame["y_pred_har"].to_numpy(),
            dates,
            lag=hac_lag,
        )
        gw_folds[fold_key] = gw_test(model, har, dates, lag=hac_lag)
        mz_folds[fold_key] = mz_regression(
            frame["y_true"].to_numpy(),
            frame["y_pred_model"].to_numpy(),
            lag=hac_lag,
        )

    pooled_dates = valid_retained["date"].to_numpy()
    pooled_model = valid_retained["loss_model"].to_numpy()
    pooled_har = valid_retained["loss_har"].to_numpy()
    pooled_gap = {
        **date_clustered_gap(pooled_model, pooled_har, pooled_dates),
        "n_valid": len(valid_folds),
        "n_total": len(fold_keys),
        "excluded_folds": sorted(quarantine),
    }
    pooled_tests = {
        "dm": dm_test(pooled_model, pooled_har, pooled_dates, lag=hac_lag),
        "cw": cw_test(
            valid_retained["y_true"].to_numpy(),
            valid_retained["y_pred_model"].to_numpy(),
            valid_retained["y_pred_har"].to_numpy(),
            pooled_dates,
            lag=hac_lag,
        ),
        "gw": gw_test(pooled_model, pooled_har, pooled_dates, lag=hac_lag),
    }
    pooled_mz = {
        **mz_regression(
            valid_retained["y_true"].to_numpy(),
            valid_retained["y_pred_model"].to_numpy(),
            lag=hac_lag,
        ),
        "n_valid": len(valid_folds),
        "n_total": len(fold_keys),
        "excluded_folds": sorted(quarantine),
    }

    expanding = {
        "definition": (
            "expanding mean of the per-row QLIKE in date order over every "
            "persisted vector row of the fold's test block (the full curve is "
            "reproducible from the vectors; checkpoints sample it evenly)"
        ),
        "folds": {
            fold_key: {
                **expanding_origin_block(
                    vectors.loc[vectors["fold"] == fold_key, "date"].to_numpy(),
                    vectors.loc[
                        vectors["fold"] == fold_key, "loss_model"
                    ].to_numpy(),
                    vectors.loc[
                        vectors["fold"] == fold_key, "loss_har"
                    ].to_numpy(),
                ),
                "quarantined": fold_key in quarantine,
            }
            for fold_key in fold_keys
        },
    }

    dual = {
        "note": (
            "consumed from the per-fold and pooled conventions the training "
            "phase already emitted; not recomputed at report time"
        ),
        "folds": {
            fold_key: {
                "oos_r2_prevailing_mean": (
                    (dual_oos_r2.get("folds") or {}).get(fold_key) or {}
                ).get("oos_r2_prevailing_mean"),
                "oos_r2_vs_har": (
                    (dual_oos_r2.get("folds") or {}).get(fold_key) or {}
                ).get("oos_r2_vs_har"),
                "n_scored_rows": int(
                    len(retained[retained["fold"] == fold_key])
                ),
            }
            for fold_key in fold_keys
        },
        "pooled": dict(dual_oos_r2.get("pooled") or {}),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "persisted per-row loss vectors (results/per_row_losses.parquet)",
        "row_set": headline_row_set,
        "scale": SCALE_NOTE,
        "headline": build_headline_block(
            vectors,
            fold_keys,
            valid_folds,
            quarantine,
            headline_row_set,
        ),
        "n_folds": len(fold_keys),
        "n_valid_folds": len(valid_folds),
        "n_total_folds": len(fold_keys),
        "quarantined_folds": quarantine,
        "quarantine_policy": (
            "quarantined folds are excluded from every pooled aggregate and "
            "from the gap/SE/DM/CW/GW/MZ inputs; their per-fold descriptive "
            "blocks stay flagged quarantined"
        ),
        "dual_oos_r2": dual,
        "loss_distributions": loss_distributions,
        "floor_decomposition": floor,
        "decile_qlike": deciles,
        "regime_qlike": regimes,
        "date_clustered_gap": {
            "roles": _inference_roles(),
            "excluded_folds": sorted(quarantine),
            "folds": gap_folds,
            "pooled": pooled_gap,
        },
        "dm_cw_gw": {
            "hac_lag": int(hac_lag),
            "differential": "model - HAR on the headline (floor-retained) rows",
            "folds": {
                fold_key: {
                    "dm": dm_folds[fold_key],
                    "cw": cw_folds[fold_key],
                    "gw": gw_folds[fold_key],
                }
                for fold_key in valid_folds
            },
            "pooled": {
                **pooled_tests,
                "n_valid": len(valid_folds),
                "n_total": len(fold_keys),
                "excluded_folds": sorted(quarantine),
            },
        },
        "mz": {
            "spec": (
                "Mincer-Zarnowitz regression y_true = alpha + beta * y_pred "
                f"+ e (OLS, Newey-West HAC SEs, lag {hac_lag})"
            ),
            "hac_lag": int(hac_lag),
            "folds": mz_folds,
            "pooled": pooled_mz,
        },
        "expanding_origin": expanding,
        "calibration": calibration_stub(),
    }


def _write_atomically(path: Path, write_payload) -> None:
    """Write ``path`` via a sibling ``*.tmp`` + ``os.replace``.

    The parent directory is created first; the final path appears only through
    the atomic rename, and any failure removes the temporary file, so no
    partial artifact and no ``*.tmp`` residue is left behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        write_payload(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_loss_vectors(vectors: pd.DataFrame, path: Path | str) -> None:
    """Write the per-row loss-vector frame as a parquet artifact, atomically."""
    target = Path(path)
    frame = _validate_frame(vectors)

    def _dump(tmp_path: Path) -> None:
        frame.to_parquet(tmp_path, index=False)

    _write_atomically(target, _dump)


def write_evaluation_summary(payload: dict, path: Path | str) -> None:
    """Write the evaluation summary through the shared strict-JSON sanitizer."""
    target = Path(path)
    document = sanitize_json(payload)

    def _dump(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, allow_nan=False)

    _write_atomically(target, _dump)
