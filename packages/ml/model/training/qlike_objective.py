"""QLIKE objective and metric for log-sigma volatility forecasts.

The model target is log daily sigma, so for a log gap
``delta = actual_log - forecast_log`` the loss is::

    QLIKE(delta) = exp(delta) - delta - 1

Training and evaluation both use that single functional. The analytic
derivatives with respect to the forecast are::

    grad = 1 - exp(delta)
    hess = exp(delta)

The exponential is evaluated only after the log gap passes a finite bound, so
an extreme residual raises a named error instead of silently producing inf.
"""

from __future__ import annotations

import numpy as np

MAX_LOG_RATIO = 700.0


class QlikeOverflowError(ValueError):
    """Raised when a log-forecast gap would overflow the exponential term."""


def _log_delta(actual_log, forecast_log) -> np.ndarray:
    actual = np.asarray(actual_log, dtype=np.float64)
    forecast = np.asarray(forecast_log, dtype=np.float64)
    if actual.shape != forecast.shape:
        raise ValueError(
            f"QLIKE received mismatched shapes: {actual.shape} vs {forecast.shape}"
        )
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("QLIKE received non-finite log targets or forecasts")
    return actual - forecast


def _raise_if_overflow(delta: np.ndarray) -> None:
    if delta.size == 0:
        raise ValueError("QLIKE received an empty delta array")
    largest = float(np.max(delta))
    if largest > MAX_LOG_RATIO:
        raise QlikeOverflowError(
            f"QLIKE log ratio {largest:.6g} exceeds the safe maximum "
            f"of {MAX_LOG_RATIO:g}"
        )


def qlike_mean(delta) -> float:
    """Mean QLIKE over a finite log-ratio array.

    This is the one formula behind both the training metric and the reported
    fold loss. The exponential bound is checked before ``exp`` is evaluated.
    """
    values = np.asarray(delta, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("QLIKE received non-finite log targets or forecasts")
    _raise_if_overflow(values)

    loss = np.exp(values) - values - 1.0
    if not np.isfinite(loss).all():
        raise ValueError("QLIKE produced non-finite values")
    return float(loss.mean())


def qlike_grad_hess(actual_log, forecast_log) -> tuple[np.ndarray, np.ndarray]:
    """Analytic gradient and Hessian of the sigma-scale QLIKE loss.

    ``actual_log`` and ``forecast_log`` are log daily sigma values. Shapes must
    match and every value must be finite. A log gap above ``MAX_LOG_RATIO`` is
    rejected before the exponential term is evaluated.
    """
    delta = _log_delta(actual_log, forecast_log)
    _raise_if_overflow(delta)

    exp_delta = np.exp(delta)
    grad = 1.0 - exp_delta
    hess = exp_delta
    return grad, hess


def qlike_objective(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """XGBoost custom objective adapter returning ``(grad, hess)``."""
    return qlike_grad_hess(y_true, y_pred)


def qlike_eval_metric(y_true, y_pred) -> tuple[str, float]:
    """XGBoost custom evaluation metric adapter returning ``("qlike", value)``."""
    return "qlike", qlike_mean(_log_delta(y_true, y_pred))
