"""
FastAPI backend for volatility prediction model.

Endpoints:
  GET /api/health          → Model metadata and status
  GET /api/model/info      → Complete model report (fold 4)
  GET /api/predict/{ticker} → Latest prediction + recent history for a ticker (params: period, history_limit)
  GET /api/screener         → All S&P 500 tickers ranked by predicted volatility (params: limit, offset, sort, sort_by, search, min_vol, max_vol)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import load_model, predict

logger = logging.getLogger(__name__)

app = FastAPI(
    title="S&P 500 Volatility Predictor",
    description="Predicción de volatilidad realizada a 5 días para el S&P 500",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cached resources ──────────────────────────────────────────────
_model = None
_cached_screener = None
_cached_screener_ts = 0
_cached_predictions: dict[str, tuple[float, dict]] = {}
SCREENER_CACHE_TTL = 300  # 5 minutes
PREDICT_CACHE_TTL = 300  # 5 minutes

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent.parent / "data" / "processed" / "dataset.parquet"
ARTIFACTS_DIR = SCRIPT_DIR.parent / "ml" / "model" / "artifacts"

INDICATORS = [
    "ema_12", "ema_26", "macd", "macd_hist", "macd_signal",
    "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_50",
    "return_lag_1", "return_lag_10", "return_lag_2", "return_lag_3",
    "return_lag_5", "rolling_max_10", "rolling_mean_5", "rolling_min_10",
    "rsi_14", "simple_return",
    "volume_lag_1", "volume_lag_3", "volume_lag_5",
    "volume_sma_10", "volume_sma_20", "volume_zscore",
    "zscore_price_20", "zscore_price_vs_sma_20", "zscore_volume_20",
    "log_return", "market_regime",
]

FEATURE_GROUPS = {
    "Trend": ["price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_50"],
    "Volatility": ["rolling_max_10", "rolling_min_10", "rolling_mean_5"],
    "Momentum": ["ema_12", "ema_26", "macd", "macd_signal", "macd_hist", "rsi_14"],
    "Returns": ["simple_return", "log_return", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10"],
    "Volume": ["volume_lag_1", "volume_lag_3", "volume_lag_5", "volume_sma_10", "volume_sma_20", "volume_zscore", "zscore_volume_20"],
    "Regime": ["market_regime"],
    "Z-Score": ["zscore_price_20", "zscore_price_vs_sma_20"],
}

INDICATOR_INDEX = {name: i for i, name in enumerate(INDICATORS)}


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _ticker_exists_in_dataset(ticker: str) -> bool:
    """Quick check if ticker exists in the historical dataset."""
    if not DATASET_PATH.exists():
        return False
    df = pd.read_parquet(DATASET_PATH, columns=["ticker"])
    return ticker in df["ticker"].values


def _get_ticker_prediction(
    ticker: str,
    period: str = "6mo",
    history_limit: int = 252,
) -> dict:
    """Compute prediction for a single ticker using yfinance data."""
    now = time.time()
    cache_key = f"{ticker}_{period}"
    if cache_key in _cached_predictions:
        ts, data = _cached_predictions[cache_key]
        if (now - ts) < PREDICT_CACHE_TTL:
            return data

    from data_preparation import (
        compute_features,
        download_ticker_data,
        prepare_ticker_data,
        select_features,
    )

    download_start = time.time()
    raw = download_ticker_data(ticker, period=period)
    download_elapsed = time.time() - download_start

    prepared = prepare_ticker_data(raw)
    total_prepared = int(len(prepared))
    total_ohlc_dropped = int(len(raw) - len(prepared))

    featured = compute_features(prepared, include_target=False)
    total_nan_dropped = int(total_prepared - len(featured))

    dates = featured["date"]
    X = select_features(featured)

    model = _get_model()
    inference_start = time.time()
    result = predict(X, model=model, dates=dates, ticker=ticker)
    inference_elapsed = time.time() - inference_start

    preds = result["realized_volatility_5d_pred"].values.astype(float)
    annualized = result["realized_volatility_5d_pred_annualized"].values.astype(float)

    history = [
        {
            "date": str(row["date"].date()),
            "volatility_daily": round(float(preds[i]), 6),
            "volatility_annualized": round(float(annualized[i]), 6),
        }
        for i, (_, row) in enumerate(result.iterrows())
    ]

    latest_pred = preds[-1]
    hist_mean = float(np.mean(preds))
    hist_std = float(np.std(preds))
    hist_min = float(np.min(preds))
    hist_max = float(np.max(preds))

    volatile_pct_change = ((latest_pred - hist_mean) / hist_mean * 100) if hist_mean > 0 else 0.0

    market_regime_map = {0: "pre-crisis", 1: "financial-crisis", 2: "post-crisis", 3: "covid", 4: "post-covid"}
    last_date = pd.Timestamp(dates.iloc[-1])
    last_year = last_date.year
    if last_year <= 2007:
        current_regime = 0
    elif last_year <= 2010:
        current_regime = 1
    elif last_year <= 2019:
        current_regime = 2
    elif last_year <= 2021:
        current_regime = 3
    else:
        current_regime = 4

    total_elapsed = time.time() - download_start

    data = {
        "ticker": ticker,
        "latest": {
            "date": str(dates.iloc[-1].date()),
            "volatility_daily": round(float(latest_pred), 6),
            "volatility_annualized": round(float(annualized[-1]), 6),
            "vs_historical_mean_pct": round(float(volatile_pct_change), 2),
        },
        "history": history[-history_limit:] if history_limit else history,
        "history_stats": {
            "count": len(history),
            "mean": round(hist_mean, 6),
            "std": round(hist_std, 6),
            "min": round(hist_min, 6),
            "max": round(hist_max, 6),
            "median": round(float(np.median(preds)), 6),
            "percentiles": {
                "p10": round(float(np.percentile(preds, 10)), 6),
                "p25": round(float(np.percentile(preds, 25)), 6),
                "p75": round(float(np.percentile(preds, 75)), 6),
                "p90": round(float(np.percentile(preds, 90)), 6),
            },
        },
        "metadata": {
            "period": period,
            "data_quality": {
                "downloaded_rows": len(raw),
                "ohlc_invalid_dropped": total_ohlc_dropped,
                "rows_after_preparation": total_prepared,
                "nan_features_dropped": total_nan_dropped,
                "valid_rows": len(history),
                "date_range": {
                    "min": str(dates.iloc[0].date()),
                    "max": str(dates.iloc[-1].date()),
                },
            },
            "market_regime": {
                "id": current_regime,
                "label": market_regime_map[current_regime],
            },
            "timing_seconds": {
                "download": round(download_elapsed, 2),
                "feature_engineering": round(inference_start - download_start - download_elapsed, 2),
                "inference": round(inference_elapsed, 3),
                "total": round(total_elapsed, 2),
            },
        },
    }

    _cached_predictions[cache_key] = (now, data)
    return data


def _build_screener() -> tuple[list[dict], dict]:
    """Compute predictions for all tickers from the cached dataset.
    
    Returns:
        (rows, metadata) where rows is sorted list of ticker predictions
        and metadata contains dataset info.
    """
    global _cached_screener, _cached_screener_ts

    now = time.time()
    if _cached_screener is not None and (now - _cached_screener_ts) < SCREENER_CACHE_TTL:
        return _cached_screener

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_parquet(DATASET_PATH)
    dataset_date_min = str(df["date"].min().date())
    dataset_date_max = str(df["date"].max().date())
    dataset_built = datetime.fromtimestamp(DATASET_PATH.stat().st_mtime).isoformat()

    # For each ticker, take the last row (most recent date) with valid indicators
    latest_rows = df.groupby("ticker", sort=False).last().reset_index()

    # Drop tickers with any NaN in indicators
    indicator_cols = [c for c in INDICATORS if c in latest_rows.columns]
    n_total_tickers = len(latest_rows)
    before_drop = len(latest_rows)
    latest_rows = latest_rows.dropna(subset=indicator_cols)
    n_excluded_nan = before_drop - len(latest_rows)

    # Keep only ticker + indicators
    X = latest_rows[INDICATORS].values.astype(np.float64)

    model = _get_model()
    preds = model.predict(X)

    rows = []
    for i, ticker in enumerate(latest_rows["ticker"]):
        vol = round(float(preds[i]), 6)
        rows.append({
            "ticker": str(ticker),
            "volatility_daily": vol,
            "volatility_annualized": round(vol * np.sqrt(252), 6),
            "last_date": str(latest_rows["date"].iloc[i].date()),
            "last_price": round(float(latest_rows["adj_close"].iloc[i]), 2),
            "regime": int(latest_rows["market_regime"].iloc[i]),
        })

    rows.sort(key=lambda r: r["volatility_daily"], reverse=True)

    metadata = {
        "dataset": {
            "date_range": {"min": dataset_date_min, "max": dataset_date_max},
            "built_at": dataset_built,
            "n_tickers_total": n_total_tickers,
            "n_tickers_excluded_nan": n_excluded_nan,
            "n_tickers_valid": len(rows),
        },
        "generated_at": datetime.now().isoformat(),
    }

    _cached_screener = (rows, metadata)
    _cached_screener_ts = now

    return rows, metadata


def _build_model_info() -> dict:
    """Aggregate comprehensive information about the last fold (fold 4)."""
    import joblib

    model_path = ARTIFACTS_DIR / "checkpoints" / "fold_4_model.pkl"
    summary_path = ARTIFACTS_DIR / "results" / "fold_training_summary.json"
    metadata_path = ARTIFACTS_DIR / "inputs" / "metadata.pkl"
    folds_path = ARTIFACTS_DIR / "folds" / "folds_metadata.pkl"
    predictions_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"
    X_path = ARTIFACTS_DIR / "inputs" / "X.npy"
    Y_path = ARTIFACTS_DIR / "inputs" / "Y.npy"

    # ── Model ──
    model = joblib.load(model_path)
    params = model.get_params()
    fi = model.feature_importances_

    # ── Training summary ──
    with open(summary_path) as f:
        all_summaries = json.load(f)
    fold4 = all_summaries["fold_4"]

    # ── Fold metadata ──
    folds = joblib.load(folds_path)
    fold4_cv = folds["fold_4"]

    # ── Validation predictions (fold 4 statistics) ──
    preds = joblib.load(predictions_path)["fold_4"]
    y_pred = preds["test_pred"]
    y_true = preds["test_true"]
    residuals = y_true - y_pred

    pred_stats = {
        "test_samples": len(y_true),
        "predictions": {
            "min": round(float(y_pred.min()), 6),
            "max": round(float(y_pred.max()), 6),
            "mean": round(float(y_pred.mean()), 6),
        },
        "actual": {
            "min": round(float(y_true.min()), 6),
            "max": round(float(y_true.max()), 6),
            "mean": round(float(y_true.mean()), 6),
        },
        "residuals": {
            "mean": round(float(residuals.mean()), 6),
            "std": round(float(residuals.std()), 6),
            "min": round(float(residuals.min()), 6),
            "max": round(float(residuals.max()), 6),
        },
    }

    # ── Dataset ──
    X = np.load(X_path, mmap_mode="r")
    y = np.load(Y_path, mmap_mode="r")
    metadata = joblib.load(metadata_path)

    dataset_info = {
        "total_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "date_range": {
            "min": str(pd.Timestamp(metadata["date"].min()).date()),
            "max": str(pd.Timestamp(metadata["date"].max()).date()),
        },
        "unique_tickers": int(pd.Series(metadata["ticker"]).nunique()),
        "target": {
            "min": round(float(y.min()), 6),
            "max": round(float(y.max()), 6),
            "mean": round(float(y.mean()), 6),
            "std": round(float(y.std()), 6),
        },
    }
    del X, y

    # ── Feature importance ranked ──
    ranked = sorted(
        [(name, float(fi[INDICATOR_INDEX[name]])) for name in INDICATORS],
        key=lambda x: -x[1],
    )
    feature_importance = []
    for rank, (name, imp) in enumerate(ranked, 1):
        group = next((g for g, feats in FEATURE_GROUPS.items() if name in feats), "Other")
        feature_importance.append({
            "rank": rank,
            "name": name,
            "importance": imp,
            "importance_pct": round(imp * 100, 4),
            "group": group,
        })

    group_importance = {}
    for entry in feature_importance:
        g = entry["group"]
        group_importance[g] = round(group_importance.get(g, 0) + entry["importance_pct"], 2)

    return {
        "fold": 4,
        "model": {
            "type": type(model).__name__,
            "hyperparameters": {
                "objective": params.get("objective"),
                "eval_metric": params.get("eval_metric"),
                "max_depth": params.get("max_depth"),
                "min_child_weight": params.get("min_child_weight"),
                "gamma": params.get("gamma"),
                "learning_rate": params.get("learning_rate"),
                "n_estimators": params.get("n_estimators"),
                "reg_alpha": params.get("reg_alpha"),
                "reg_lambda": params.get("reg_lambda"),
                "subsample": params.get("subsample"),
                "colsample_bytree": params.get("colsample_bytree"),
                "colsample_bylevel": params.get("colsample_bylevel"),
                "tree_method": params.get("tree_method"),
                "random_state": params.get("random_state"),
            },
            "n_features": int(model.n_features_in_),
            "file_size_bytes": model_path.stat().st_size,
        },
        "training": {
            "train_samples": int(fold4["n_train_samples"]),
            "test_samples": int(fold4["n_test_samples"]),
            "n_features": int(fold4["n_features"]),
            "n_features_indicators": int(fold4["n_features_indicators"]),
            "n_features_ticker": int(fold4["n_features_ticker"]),
            "date_ranges": {
                "train": {"min": fold4["train_date_min"], "max": fold4["train_date_max"]},
                "test": {"min": fold4["test_date_min"], "max": fold4["test_date_max"]},
            },
            "gap_days": int(fold4_cv["gap_days"]),
            "timing_seconds": {
                "train": round(fold4["train_time_seconds"], 2),
                "inference": round(fold4["inference_time_seconds"], 2),
                "total": round(fold4["fold_time_seconds"], 2),
            },
        },
        "performance": {
            "test_r2": round(fold4["test_r2"], 6),
            "test_mae": round(fold4["test_mae"], 6),
            "test_rmse": round(fold4["test_rmse"], 6),
        },
        "predictions": pred_stats,
        "dataset": dataset_info,
        "feature_importance": {
            "ranked": feature_importance,
            "by_group": group_importance,
        },
        "cross_validation": {
            "method": "TimeSeriesSplit",
            "n_splits": 5,
            "regime_distribution_train": {str(k): v for k, v in fold4_cv.get("regime_distribution_train", {}).items()},
            "regime_distribution_test": {str(k): v for k, v in fold4_cv.get("regime_distribution_test", {}).items()},
        },
    }


# ── Endpoints ──────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    model = _get_model()
    return {
        "status": "ok",
        "model": {
            "type": type(model).__name__,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "n_features": model.n_features_in_,
        },
        "dataset": {
            "path": str(DATASET_PATH),
            "exists": DATASET_PATH.exists(),
        },
    }


@app.get("/api/model/info")
def model_info():
    try:
        info = _build_model_info()
        return info
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/{ticker}")
def predict_ticker_endpoint(
    ticker: str,
    period: str = "6mo",
    history_limit: int = 252,
):
    ticker = ticker.upper().strip()
    if not _ticker_exists_in_dataset(ticker):
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found in S&P 500 historical dataset",
        )
    try:
        data = _get_ticker_prediction(ticker, period=period, history_limit=history_limit)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/screener")
def screener(
    limit: int = 50,
    offset: int = 0,
    sort: str = "desc",
    sort_by: str = "volatility",
    search: str = "",
    min_vol: float = 0.0,
    max_vol: float = 1.0,
):
    try:
        rows, metadata = _build_screener()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    filtered = rows
    if search:
        search_upper = search.upper().strip()
        filtered = [r for r in filtered if search_upper in r["ticker"]]

    filtered = [r for r in filtered if min_vol <= r["volatility_daily"] <= max_vol]

    if sort_by == "ticker":
        filtered = sorted(filtered, key=lambda r: r["ticker"], reverse=(sort == "desc"))
    else:
        filtered = sorted(filtered, key=lambda r: r["volatility_daily"], reverse=(sort == "desc"))

    paginated = filtered[offset:offset + limit] if limit else filtered

    return {
        "total_tickers": len(filtered),
        "total_all": metadata["dataset"]["n_tickers_valid"],
        "limit": limit,
        "offset": offset,
        "sorted_by": sort_by,
        "sort_order": sort,
        "filters": {
            "search": search or None,
            "min_volatility": min_vol if min_vol > 0 else None,
            "max_volatility": max_vol if max_vol < 1.0 else None,
        },
        "dataset": metadata["dataset"],
        "generated_at": metadata["generated_at"],
        "tickers": paginated,
    }


@app.get("/api/screener/summary")
def screener_summary():
    try:
        rows, metadata = _build_screener()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    vols = [r["volatility_daily"] for r in rows]
    annualized = [r["volatility_annualized"] for r in rows]
    mean_vol = float(np.mean(vols))
    std_vol = float(np.std(vols))

    regimes = {}
    for r in rows:
        regimes[r["regime"]] = regimes.get(r["regime"], 0) + 1

    regime_labels = {0: "pre-crisis", 1: "financial-crisis", 2: "post-crisis", 3: "covid", 4: "post-covid"}
    regime_distribution = {
        str(k): {"label": regime_labels[k], "count": v, "pct": round(v / len(rows) * 100, 1)}
        for k, v in sorted(regimes.items())
    }

    return {
        "total_tickers": len(rows),
        "mean_daily_volatility": round(mean_vol, 6),
        "mean_annualized_volatility": round(mean_vol * np.sqrt(252), 6),
        "std_daily_volatility": round(std_vol, 6),
        "min_daily_volatility": round(float(np.min(vols)), 6),
        "max_daily_volatility": round(float(np.max(vols)), 6),
        "median_daily_volatility": round(float(np.median(vols)), 6),
        "percentiles": {
            "p10": round(float(np.percentile(vols, 10)), 6),
            "p25": round(float(np.percentile(vols, 25)), 6),
            "p50": round(float(np.percentile(vols, 50)), 6),
            "p75": round(float(np.percentile(vols, 75)), 6),
            "p90": round(float(np.percentile(vols, 90)), 6),
        },
        "regime_distribution": regime_distribution,
        "dataset": metadata["dataset"],
        "generated_at": metadata["generated_at"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
