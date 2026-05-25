"""
Model inference module.

Loads a trained XGBoost model, takes prepared features, and returns
predictions with associated metadata (dates, ticker, volatility).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "ml" / "model" / "artifacts" / "checkpoints"
RESULTS_DIR = SCRIPT_DIR.parent / "ml" / "model" / "artifacts" / "results"

MODEL_PATH = MODEL_DIR / "fold_4_model.pkl"
SUMMARY_PATH = RESULTS_DIR / "fold_training_summary.json"

PREDICTION_COLUMN = "realized_volatility_5d_pred"


def load_model(model_path: str | Path = MODEL_PATH):
    """
    Load trained XGBoost model from disk.

    Args:
        model_path: Path to fold_4_model.pkl (or any fold model)

    Returns:
        Loaded XGBRegressor model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run the training pipeline first (python main.py)"
        )
    model = joblib.load(model_path)
    logger.info("Loaded model from %s (%s trees, %d features)", model_path, model.n_estimators, model.n_features_in_)
    return model


def predict(
    features: pd.DataFrame | np.ndarray,
    model=None,
    model_path: str | Path = MODEL_PATH,
    dates: pd.Series | np.ndarray | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Predict realized_volatility_5d from feature matrix.

    Args:
        features: DataFrame (29 indicator columns) or numpy array (n, 29)
        model: Pre-loaded model. If None, loads from model_path.
        model_path: Path to model file (used only if model is None)
        dates: Optional array of dates to include in output
        ticker: Optional ticker symbol to include in output

    Returns:
        DataFrame with columns:
          - date (if provided)
          - ticker (if provided)
          - realized_volatility_5d_pred: predicted volatility
          - realized_volatility_5d_pred_annualized: annualized volatility (×√252)
    """
    if model is None:
        model = load_model(model_path)

    if isinstance(features, pd.DataFrame):
        X = features.values.astype(np.float64)
    else:
        X = np.asarray(features, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    logger.info("Predicting for %d samples, %d features", X.shape[0], X.shape[1])

    pred = model.predict(X)

    result = pd.DataFrame({PREDICTION_COLUMN: pred})
    result[PREDICTION_COLUMN + "_annualized"] = pred * np.sqrt(252)

    if dates is not None:
        dates_arr = np.asarray(dates)
        if len(dates_arr) != len(pred):
            raise ValueError(
                f"dates length ({len(dates_arr)}) != predictions ({len(pred)})"
            )
        result.insert(0, "date", dates_arr)

    if ticker is not None:
        result.insert(0 if dates is None else 1, "ticker", ticker)

    logger.info(
        "Predictions: range [%.6f, %.6f], mean %.6f",
        pred.min(), pred.max(), pred.mean(),
    )

    return result


def predict_ticker(
    ticker: str,
    period: str = "6mo",
    start: str | None = None,
    end: str | None = None,
    model=None,
    return_features: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline for a single ticker: download → features → predict.

    Args:
        ticker: Stock symbol (e.g. 'AAPL')
        period: yfinance period (default '6mo')
        start: Override start date
        end: Override end date
        model: Pre-loaded model (loads fold_4 if None)
        return_features: If True, returns (predictions, features) tuple

    Returns:
        DataFrame with date, ticker, predicted volatility, annualized volatility.
        If return_features=True, also returns the features DataFrame.
    """
    from data_preparation import (
        compute_features,
        download_ticker_data,
        prepare_ticker_data,
        select_features,
    )

    if model is None:
        model = load_model()

    raw = download_ticker_data(ticker, start=start, end=end, period=period)
    prepared = prepare_ticker_data(raw)
    featured = compute_features(prepared, include_target=False)

    dates = featured["date"]
    features = select_features(featured)

    result = predict(features, model=model, dates=dates, ticker=ticker)

    if return_features:
        return result, features
    return result


def get_model_metadata() -> dict:
    """
    Return metadata about the trained model for reference.

    Returns:
        Dict with training date ranges, metrics, and model params
    """
    import json

    metadata = {}

    model = load_model()
    params = model.get_params()
    metadata["model"] = {
        "type": type(model).__name__,
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "n_features": model.n_features_in_,
    }

    summary_path = Path(SUMMARY_PATH)
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        metadata["training"] = summary

    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = predict_ticker("AAPL", period="6mo")
    print(result.to_string(index=False))
