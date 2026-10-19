"""Tests for the QLIKE-consistent training objective and its wiring.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_qlike.py
"""

from __future__ import annotations

import re
import traceback

import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model.training import model_trainer, training
from packages.ml.model.training.qlike_objective import (
    MAX_LOG_RATIO,
    QlikeOverflowError,
    qlike_eval_metric,
    qlike_grad_hess,
    qlike_mean,
    qlike_objective,
)

MODEL_ROOT = REPO_ROOT / "packages" / "ml" / "model"
TESTS_DIR = MODEL_ROOT / "tests"

GRAD_STEP = 1e-6
HESS_STEP = 1e-5
GRAD_RTOL = 1e-6
HESS_RTOL = 1e-5
FIXTURE_POINTS = 21

FORBIDDEN_ROUTE = re.compile(r"smear|calibrat|Duan|Jensen", re.IGNORECASE)
SCALER_TOKENS = ("scaler", "standardscaler", "minmaxscaler", "robustscaler")


def _qlike(delta: np.ndarray) -> np.ndarray:
    return np.exp(delta) - delta - 1.0


def _analytic_grad(actual_log: np.ndarray, forecast_log: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(actual_log - forecast_log)


def _relative_error(
    numeric: np.ndarray, analytic: np.ndarray, floor: float = 1e-12
) -> np.ndarray:
    """Relative error with a denominator floor.

    The floor keeps a vanishing analytic value well defined: with ``floor=1.0``
    the check becomes an absolute tolerance where the gradient crosses zero.
    """
    denominator = np.maximum(
        np.maximum(np.abs(numeric), np.abs(analytic)), floor
    )
    return np.abs(numeric - analytic) / denominator


def _fixture_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """21 pairs whose log gap spans [-2, 2]."""
    deltas = np.linspace(-2.0, 2.0, FIXTURE_POINTS)
    forecast_log = np.full_like(deltas, 0.3)
    actual_log = forecast_log + deltas
    return actual_log, forecast_log, deltas


def test_gradient_and_hessian_fixture() -> None:
    actual_log, forecast_log, deltas = _fixture_pairs()
    assert len(deltas) == FIXTURE_POINTS
    assert deltas.min() <= -2.0 and deltas.max() >= 2.0

    grad, hess = qlike_grad_hess(actual_log, forecast_log)
    assert grad.shape == deltas.shape
    assert hess.shape == deltas.shape

    numeric_grad = (
        _qlike(actual_log - (forecast_log + GRAD_STEP))
        - _qlike(actual_log - (forecast_log - GRAD_STEP))
    ) / (2.0 * GRAD_STEP)
    grad_error = _relative_error(numeric_grad, grad, floor=1.0)
    assert np.all(grad_error <= GRAD_RTOL), (
        f"gradient fixture failed at {int(np.count_nonzero(grad_error > GRAD_RTOL))} "
        f"of {FIXTURE_POINTS} points; max relative error {grad_error.max():.3e}"
    )

    numeric_hess = (
        _analytic_grad(actual_log, forecast_log + HESS_STEP)
        - _analytic_grad(actual_log, forecast_log - HESS_STEP)
    ) / (2.0 * HESS_STEP)
    hess_error = _relative_error(numeric_hess, hess)
    assert np.all(hess_error <= HESS_RTOL), (
        f"hessian fixture failed at {int(np.count_nonzero(hess_error > HESS_RTOL))} "
        f"of {FIXTURE_POINTS} points; max relative error {hess_error.max():.3e}"
    )


def test_gradient_negative_control() -> None:
    actual_log, forecast_log, _ = _fixture_pairs()
    grad, _ = qlike_grad_hess(actual_log, forecast_log)
    numeric_grad = (
        _qlike(actual_log - (forecast_log + GRAD_STEP))
        - _qlike(actual_log - (forecast_log - GRAD_STEP))
    ) / (2.0 * GRAD_STEP)

    scaled_error = _relative_error(numeric_grad, 1.5 * grad, floor=1.0)
    failures = int(np.count_nonzero(scaled_error > GRAD_RTOL))
    assert failures >= 20, (
        f"scaled-gradient control only failed {failures} of {FIXTURE_POINTS} points"
    )


def test_overflow_guard() -> None:
    assert MAX_LOG_RATIO == 700.0

    with pytest.raises(QlikeOverflowError) as excinfo:
        qlike_grad_hess(np.array([1000.0]), np.array([0.0]))
    assert isinstance(excinfo.value, ValueError)

    with pytest.raises(QlikeOverflowError):
        qlike_mean(np.array([MAX_LOG_RATIO + 1.0]))
    assert np.isfinite(qlike_mean(np.array([MAX_LOG_RATIO])))

    for actual, forecast in (
        (np.nan, 0.0),
        (0.0, np.inf),
        (np.inf, 0.0),
        (-np.inf, 0.0),
    ):
        with pytest.raises(ValueError) as excinfo:
            qlike_grad_hess(
                np.array([actual], dtype=np.float64),
                np.array([forecast], dtype=np.float64),
            )
        assert not isinstance(excinfo.value, QlikeOverflowError)

    with pytest.raises(ValueError):
        qlike_grad_hess(np.zeros(3), np.zeros(2))


def test_metric_matches_loss() -> None:
    actual_log = np.array([-9.0, -8.5, -8.0, -7.5, -9.5])
    forecast_log = np.array([-8.8, -8.7, -8.2, -7.1, -9.9])

    # The xgboost adapter returns the scalar metric; the history key comes from
    # the function name, which the sklearn wrapper reads.
    value = qlike_eval_metric(actual_log, forecast_log)

    assert qlike_eval_metric.__name__ == "qlike_eval_metric"
    assert value == pytest.approx(
        training.qlike_loss(actual_log, forecast_log), rel=1e-12, abs=1e-12
    )
    assert value == pytest.approx(
        qlike_mean(actual_log - forecast_log), rel=1e-12, abs=1e-12
    )


def test_trainer_wiring_and_tiny_fit() -> None:
    trainer_source = (MODEL_ROOT / "training" / "model_trainer.py").read_text(
        encoding="utf-8"
    )
    assert "reg:squarederror" not in trainer_source
    assert "rmse" not in trainer_source.lower()

    captured: dict = {}
    real_constructor = model_trainer.xgb.XGBRegressor

    def spy_constructor(*args, **kwargs):
        captured.update(kwargs)
        tiny_kwargs = dict(kwargs)
        tiny_kwargs["n_estimators"] = 5
        return real_constructor(*args, **tiny_kwargs)

    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 2))
    y = rng.normal(loc=-8.0, scale=0.4, size=40)

    model_trainer.xgb.XGBRegressor = spy_constructor
    try:
        model, label_mapping = model_trainer.train_xgboost_model(X, y)
    finally:
        model_trainer.xgb.XGBRegressor = real_constructor

    assert label_mapping == {}
    assert captured["objective"] is qlike_objective
    assert captured["eval_metric"] is qlike_eval_metric
    defaults = model_trainer.default_params()
    assert captured["random_state"] == defaults["random_state"]
    assert captured["n_jobs"] == defaults["n_jobs"]

    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()


def test_no_deferred_route_tokens() -> None:
    hits = []
    for path in sorted(MODEL_ROOT.rglob("*.py")):
        if TESTS_DIR in path.parents:
            continue
        if FORBIDDEN_ROUTE.search(path.read_text(encoding="utf-8")):
            hits.append(str(path.relative_to(REPO_ROOT)))

    assert not hits, "recalibration vocabulary remains: " + ", ".join(hits)


def test_reported_loss_floor_magnitudes() -> None:
    low = training.qlike_loss(np.array([-23.025851]), np.array([0.0]))
    assert low == pytest.approx(22.025851, rel=1e-9)

    high = training.qlike_loss(np.array([23.025851]), np.array([0.0]))
    # 23.025851 rounds ln(1e10); the stated 1e-6 relative tolerance absorbs
    # the ~700 absolute shift that rounding imposes on exp(delta).
    assert high == pytest.approx(9999999976.0, rel=1e-6)

    for actual, forecast in ((np.nan, 0.0), (0.0, np.inf)):
        with pytest.raises(ValueError):
            training.qlike_loss(
                np.array([actual], dtype=np.float64),
                np.array([forecast], dtype=np.float64),
            )

    for relative_path in (
        "training/qlike_objective.py",
        "training/model_trainer.py",
        "training/training.py",
    ):
        source = (MODEL_ROOT / relative_path).read_text(encoding="utf-8").lower()
        introduced = [token for token in SCALER_TOKENS if token in source]
        assert not introduced, f"{relative_path} introduces scaler wording: {introduced}"


def test_gradient_sign() -> None:
    actual_log = np.array([1.0, -1.0, 0.0])
    forecast_log = np.zeros(3)

    grad, hess = qlike_grad_hess(actual_log, forecast_log)

    assert grad[0] < 0.0
    assert grad[1] > 0.0
    assert grad[2] == 0.0
    assert np.all(hess > 0.0)


_DASH_VARIANTS = "\u2212\u2010\u2011\u2012\u2013\u2014\u2015\uFE58\uFE63\uFF0D"


def _normalized_docstring(function) -> str:
    """Lowercase, collapse whitespace, and unify unicode dashes to ASCII hyphens."""
    text = (function.__doc__ or "").lower()
    for dash in _DASH_VARIANTS:
        text = text.replace(dash, "-")
    return " ".join(text.split())


def _has_signed_identity(text: str) -> bool:
    """Accept either orientation of the signed QLIKE/Hansen-Lunde identity."""
    sigma = r"(?:h|sigma|σ)"
    ln_sigma = rf"l(?:n|og)\s*\(?\s*{sigma}\s*\)?"
    qlike_first = rf"qlike\s*=\s*hl\s*-\s*{ln_sigma}\s*-\s*1"
    hl_first = rf"hl\s*=\s*qlike\s*\+\s*{ln_sigma}\s*\+\s*1"
    return bool(re.search(qlike_first, text) or re.search(hl_first, text))


def _has_scale_note(text: str) -> bool:
    """Require sigma-scale, per-day, variance-comparison, and 4-6x ratio tokens."""
    sigma_scale = "sigma" in text
    per_day = bool(re.search(r"per[- ]?day|daily", text))
    variance_scale = "variance" in text
    ratio = bool(
        re.search(r"4\s*(?:-|to)\s*6\s*(?:x|×|times)", text)
        or re.search(r"4\s*(?:x|×|times)\s*(?:-|to)\s*6", text)
    )
    return sigma_scale and per_day and variance_scale and ratio


def test_docsign_identity_and_scale_note() -> None:
    """The qlike_loss docstring states the signed identity and the sigma scale.

    Signal: this test fails if the documented sign is flipped (for example
    ``QLIKE = HL + ln(h) - 1``) or if the scale note is missing. At the branch
    point the docstring carries neither the signed identity nor the scale tokens,
    so both content checks below are red; the trailing block pins the numeric
    anchors, proving the change is docstring-only.
    """
    text = _normalized_docstring(training.qlike_loss)

    # Negative controls: the matchers reject a flipped sign either way and an
    # empty document, so merely mentioning "QLIKE" cannot satisfy them.
    for flipped in ("qlike = hl + ln(h) - 1", "hl = qlike - ln(h) + 1"):
        assert not _has_signed_identity(flipped)
    assert not _has_scale_note("")

    assert _has_signed_identity(text), (
        "qlike_loss docstring lacks the signed identity QLIKE = HL - ln(h) - 1 "
        "(or its HL = QLIKE + ln(h) + 1 equivalent)"
    )
    assert _has_scale_note(text), (
        "qlike_loss docstring lacks the sigma scale declaration "
        "(sigma, per-day, variance, and 4-6x tokens)"
    )

    low = training.qlike_loss(np.array([-23.025851]), np.array([0.0]))
    assert low == pytest.approx(22.025851, rel=1e-9)
    high = training.qlike_loss(np.array([23.025851]), np.array([0.0]))
    assert high == pytest.approx(1e10, rel=1e-6)


if __name__ == "__main__":
    failures = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All tests passed")
