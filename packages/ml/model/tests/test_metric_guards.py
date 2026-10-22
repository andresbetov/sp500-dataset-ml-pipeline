"""Tests for the pandas-trap guards and their recorded triage.

The diagnostics decile plot bins the stored log target with
``pd.qcut(..., duplicates="drop")``. A constant target realizes no bins and a
duplicate-heavy target can realize fewer than ten; the plot must warn with the
fold and the realized count and still render. Live folds realize all ten bins
and must not warn. The remaining known traps are recorded as intended/no-fix
(``YearLocator`` tick placement, ``min_periods=1`` warm-up windows) or deferred
(the heatmap's constant-metric normalization), and the constant-target
``r2_score`` convention is documented rather than changed.

All writes go under a temporary directory; the live check reads the stored
predictions only.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_metric_guards.py
"""

from __future__ import annotations

import logging
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import conftest  # noqa: F401  (puts the repository root on sys.path)
from conftest import ARTIFACTS_DIR, REPO_ROOT, requires_live_artifacts
from packages.ml.model.training import har_benchmark
from packages.ml.model.visualization import temporal_analysis, training_diagnostics

LOGGER_NAME = training_diagnostics.__name__


def _single_fold_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "fold_0": {
            "test_true": np.asarray(y_true, dtype=np.float64),
            "test_pred": np.asarray(y_pred, dtype=np.float64),
        }
    }


def _render_single_fold(predictions: dict, output_path: Path) -> Path:
    with (
        mock.patch.object(training_diagnostics, "FOLD_NAMES", ["fold_0"]),
        mock.patch.object(training_diagnostics, "FOLD_COLORS", ["#1f77b4"]),
    ):
        return training_diagnostics.plot_error_by_volatility_bin(
            predictions, output_path=output_path
        )


class _WarningRecorder(logging.Handler):
    """Collect this module's warning records without pytest's logging fixture."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == LOGGER_NAME:
            self.records.append(record)


@contextmanager
def _captured_warnings():
    recorder = _WarningRecorder()
    logger = logging.getLogger(LOGGER_NAME)
    previous_level = logger.level
    logger.addHandler(recorder)
    logger.setLevel(logging.WARNING)
    try:
        yield recorder
    finally:
        logger.removeHandler(recorder)
        logger.setLevel(previous_level)


def _warnings(recorder: _WarningRecorder) -> list[str]:
    return [record.getMessage() for record in recorder.records]


def test_constant_target_warns_and_renders_without_crashing() -> None:
    """A constant target realizes zero bins: warn with fold/count, still render."""
    y_true = np.full(50, -2.0)
    y_pred = y_true + 0.10

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "error_by_volatility_bin.png"
        with _captured_warnings() as recorder:
            returned = _render_single_fold(
                _single_fold_predictions(y_true, y_pred), output_path
            )

        warnings = _warnings(recorder)
        assert len(warnings) == 1, warnings
        assert "fold_0" in warnings[0]
        assert "0 of 10" in warnings[0]
        assert returned == output_path
        assert output_path.is_file()
        assert output_path.stat().st_size > 0


def test_live_predictions_realize_ten_bins_and_do_not_warn() -> None:
    """The live target realizes 10/10 bins per fold and the guard stays silent."""
    requires_live_artifacts(ARTIFACTS_DIR)
    predictions_path = ARTIFACTS_DIR / "results" / "validation_predictions.pkl"
    if not predictions_path.is_file():
        pytest.skip("requires the stored live validation predictions")

    predictions = training_diagnostics.load_validation_predictions(predictions_path)

    for fold_name in training_diagnostics.FOLD_NAMES:
        deciles = np.asarray(
            pd.qcut(
                np.asarray(predictions[fold_name]["test_true"]),
                q=10,
                labels=False,
                duplicates="drop",
            )
        )
        finite = deciles[np.isfinite(deciles.astype(float))]
        n_bins = int(finite.max()) + 1 if finite.size else 0
        assert n_bins == 10, f"{fold_name} realized {n_bins} bins"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "live_error_by_volatility_bin.png"
        with _captured_warnings() as recorder:
            returned = training_diagnostics.plot_error_by_volatility_bin(
                predictions, output_path=output_path
            )

        assert _warnings(recorder) == []
        assert returned == output_path
        assert output_path.is_file()
        assert output_path.stat().st_size > 0


def test_trap_triage_records_intended_and_deferred_cases() -> None:
    """The triage records the no-fix cases and defers the heatmap guard."""
    triage_path = REPO_ROOT / "docs" / "pandas-trap-triage.md"
    assert triage_path.is_file(), f"missing triage record: {triage_path}"
    text = triage_path.read_text(encoding="utf-8")
    lower = text.lower()

    assert "qcut" in lower
    assert "warn-only" in lower
    assert "constant" in lower and "r2_score" in lower
    assert "min_periods=1" in text and "intended, no-fix" in lower
    assert "yearlocator" in lower and "intended, no-fix" in lower
    assert "heatmap" in lower and "deferred" in lower
    assert "not fixed here" in lower

    diagnostics_source = Path(training_diagnostics.__file__).read_text(encoding="utf-8")
    assert "fewer than ten" in diagnostics_source.lower()
    assert "r2_score" in diagnostics_source

    assert "min_periods=1" in Path(har_benchmark.__file__).read_text(encoding="utf-8")
    assert "YearLocator()" in Path(temporal_analysis.__file__).read_text(
        encoding="utf-8"
    )


def test_shaped_target_rendering_is_unchanged() -> None:
    """A ten-bin target keeps the exact errorbar centers/means/stds and no warning."""
    y_true = np.linspace(-5.0, 0.0, 100)
    y_pred = y_true + np.linspace(0.05, -0.05, len(y_true))
    abs_errors = np.abs(y_true - y_pred)
    deciles = np.asarray(
        pd.qcut(y_true, q=10, labels=False, duplicates="drop")
    )

    expected_centers = np.array(
        [y_true[deciles == b].mean() for b in range(10)]
    )
    expected_means = np.array(
        [abs_errors[deciles == b].mean() for b in range(10)]
    )
    expected_stds = np.array(
        [abs_errors[deciles == b].std() for b in range(10)]
    )

    captured: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    real_errorbar = training_diagnostics.plt.Axes.errorbar

    def spy_errorbar(self, x, y, *args, **kwargs):
        captured.append(
            (
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                np.asarray(kwargs["yerr"], dtype=np.float64),
            )
        )
        return real_errorbar(self, x, y, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "shaped_error_by_volatility_bin.png"
        with (
            mock.patch.object(
                training_diagnostics.plt.Axes, "errorbar", spy_errorbar
            ),
            _captured_warnings() as recorder,
        ):
            returned = _render_single_fold(
                _single_fold_predictions(y_true, y_pred), output_path
            )

        assert _warnings(recorder) == []
        assert len(captured) == 1, captured
        centers, means, stds = captured[0]
        np.testing.assert_allclose(centers, expected_centers, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(means, expected_means, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(stds, expected_stds, rtol=0.0, atol=1e-12)
        assert returned == output_path
        assert output_path.is_file()


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except pytest.skip.Exception as exc:
                skipped += 1
                print(f"SKIP {_name}: {exc}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    suffix = f" ({skipped} skipped)" if skipped else ""
    print(f"All tests passed{suffix}")
