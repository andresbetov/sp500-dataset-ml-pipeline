"""Tests for metric-panel units, benchmark gaps, and constant-metric ties.

The metrics comparison figure must not put R² (dimensionless) beside MAE/RMSE
(log points) on one shared "Actual Value" axis, and it must not draw a
normalized twin-axis overlay over absolute bars. Each metric gets its own
panel with a unit-bearing label. A per-fold table carries the absolute metrics
and the model-vs-HAR gaps under one sign convention: positive means the model
is better than the benchmark.

Only absolute values are rendered. A constant metric therefore has no
normalized score to turn into a 1.0/0.0 tie artifact; the tests pin the
constant-metric bars and the absence of any normalized series or twin axis.

Every rendering writes under a temporary directory; no live artifact is
written.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_plot_units.py
"""

from __future__ import annotations

import ast
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conftest  # noqa: F401  (puts the repository root on sys.path)
from matplotlib.text import Text

from packages.ml.model.visualization import performance_comparison
from packages.ml.model.visualization.data_loader import (
    QLIKE_GAP_SIGN_CONVENTION,
    extract_qlike_stats,
)

MODULE_PATH = Path(performance_comparison.__file__)
PERFORMANCE_PLOTS = {
    "metrics_comparison.png",
    "timing_breakdown.png",
    "sample_distribution.png",
    "feature_count_summary.png",
}


def _summary_frame() -> pd.DataFrame:
    """Five folds with distinct metric and QLIKE/HAR values."""
    qlike = [
        (0.060, 0.090, 0.510),
        (0.050, 0.080, -0.640),
        (0.065, 0.095, -0.460),
        (0.055, 0.085, 0.350),
        (0.085, 0.075, 0.120),
    ]
    rows = []
    for index, (test_qlike, har_qlike, har_r2) in enumerate(qlike):
        rows.append(
            {
                "fold": f"fold_{index}",
                "train_date_min": f"200{index}-01-03",
                "train_date_max": f"200{index}-06-29",
                "test_date_min": f"200{index}-07-01",
                "test_date_max": f"200{index}-12-31",
                "n_train_samples": 1_000 + index,
                "n_test_samples": 200 + index,
                "test_r2": 0.10 + index * 0.01,
                "test_mae": 0.20 + index * 0.01,
                "test_rmse": 0.30 + index * 0.01,
                "train_time_seconds": 10.0 + index,
                "inference_time_seconds": 1.0 + index,
                "har_time_seconds": 2.0 + index,
                "fold_time_seconds": 13.0 + index,
                "n_features_indicators": 37,
                "n_features_ticker": 0,
                "test_qlike": test_qlike,
                "har_qlike": har_qlike,
                "har_r2": har_r2,
            }
        )
    return pd.DataFrame(rows)


def _constant_frame() -> pd.DataFrame:
    frame = _summary_frame()
    for column in ("test_r2", "test_mae", "test_rmse"):
        frame[column] = 0.25
    return frame


def _render_and_capture(frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Render once and keep the figure open for inspection before it closes."""
    captured: dict[str, object] = {}
    real_savefig = performance_comparison.plt.savefig

    def capture(*args, **kwargs):
        captured["figure"] = performance_comparison.plt.gcf()
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(performance_comparison.plt, "savefig", capture)
    output_path = tmp_path / "metrics_comparison.png"
    returned = performance_comparison.plot_metrics_comparison(frame, output_path)
    assert returned == output_path
    assert "figure" in captured, "plot did not reach savefig"
    return output_path, captured["figure"]


def _figure_text(figure) -> str:
    """All text drawn in the figure, including axes labels and table cells."""
    return "\n".join(text.get_text() for text in figure.findobj(Text))


def _plot_function_node() -> ast.FunctionDef:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "plot_metrics_comparison":
            return node
    raise AssertionError("plot_metrics_comparison is missing")


def test_each_metric_panel_declares_its_own_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R² is dimensionless; MAE/RMSE are log points on the log target."""
    _, figure = _render_and_capture(_summary_frame(), tmp_path, monkeypatch)

    bar_axes = [axis for axis in figure.axes if axis.patches]
    labels = [axis.get_ylabel() for axis in bar_axes]
    assert len(bar_axes) == 3, labels

    r2_labels = [label for label in labels if "R²" in label]
    mae_labels = [label for label in labels if "MAE" in label]
    rmse_labels = [label for label in labels if "RMSE" in label]
    assert len(r2_labels) == 1, labels
    assert len(mae_labels) == 1, labels
    assert len(rmse_labels) == 1, labels
    assert "dimensionless" in r2_labels[0].lower()
    assert "log points" in mae_labels[0].lower()
    assert "log points" in rmse_labels[0].lower()

    text = _figure_text(figure)
    assert "Actual Value" not in text
    assert "Normalized Score" not in text


def test_constant_metrics_render_absolute_values_without_normalized_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constant metrics render finite bars, never a normalized 1.0/0.0 series."""
    frame = _constant_frame()
    _, figure = _render_and_capture(frame, tmp_path, monkeypatch)

    bar_axes = [axis for axis in figure.axes if axis.patches]
    assert len(bar_axes) == 3
    for axis in bar_axes:
        heights = np.array([patch.get_height() for patch in axis.patches])
        np.testing.assert_allclose(
            heights, np.full(len(frame), 0.25), rtol=0.0, atol=1e-12
        )
        assert np.isfinite(heights).all()

    # Three metric panels plus the table axis: no twin axis was reintroduced.
    assert len(figure.axes) == 4
    assert "norm" not in _figure_text(figure).lower()
    for axis in figure.axes:
        for line in axis.lines:
            assert "norm" not in line.get_label().lower()


def test_benchmark_records_match_the_summary_and_shared_stats() -> None:
    """One row per fold; QLIKE gaps reuse the shared gap definition."""
    frame = _summary_frame()
    stats = extract_qlike_stats(frame)
    records = performance_comparison._benchmark_records(frame)

    assert list(records["fold"]) == list(frame["fold"])
    for position, record in enumerate(records.itertuples(index=False)):
        row = frame.iloc[position]
        fold = str(row["fold"])
        assert record.test_r2 == pytest.approx(float(row["test_r2"]), abs=1e-12)
        assert record.test_mae == pytest.approx(float(row["test_mae"]), abs=1e-12)
        assert record.test_rmse == pytest.approx(float(row["test_rmse"]), abs=1e-12)
        shared = stats["per_fold"][fold]
        assert record.qlike_gap == pytest.approx(
            shared["qlike_gap"], rel=0.0, abs=1e-12
        )
        assert record.qlike_gap == pytest.approx(
            float(row["har_qlike"]) - float(row["test_qlike"]),
            rel=0.0,
            abs=1e-12,
        )
        assert record.r2_gap == pytest.approx(
            float(row["test_r2"]) - float(row["har_r2"]),
            rel=0.0,
            abs=1e-12,
        )


def test_gap_table_is_emitted_with_one_row_per_fold_and_one_sign_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cell values and signs match the summary and shared stats per fold."""
    frame = _summary_frame()
    stats = extract_qlike_stats(frame)
    records = performance_comparison._benchmark_records(frame)
    _, figure = _render_and_capture(frame, tmp_path, monkeypatch)

    table_axes = [axis for axis in figure.axes if axis.tables]
    assert len(table_axes) == 1, [axis.tables for axis in figure.axes]
    table = table_axes[0].tables[0]
    n_rows = max(row for row, _ in table.get_celld()) + 1
    assert n_rows == len(frame) + 1  # header + one row per fold

    assert len(records) == len(frame)
    for position, record in enumerate(records.itertuples(index=False)):
        table_row = position + 1
        cells = [
            table[(table_row, column)].get_text().get_text()
            for column in range(9)
        ]
        row = frame.iloc[position]
        fold = str(row["fold"])
        assert cells[0] == fold == record.fold

        # Absolute values (columns 1-4 and 6-7).
        assert float(cells[1]) == pytest.approx(record.test_r2, abs=5e-5)
        assert float(cells[2]) == pytest.approx(record.test_mae, abs=5e-5)
        assert float(cells[3]) == pytest.approx(record.test_rmse, abs=5e-5)
        assert float(cells[4]) == pytest.approx(record.har_r2, abs=5e-5)
        assert float(cells[6]) == pytest.approx(record.test_qlike, abs=5e-5)
        assert float(cells[7]) == pytest.approx(record.har_qlike, abs=5e-5)

        # Gap signs and values, checked against the summary/shared stats.
        assert float(cells[5]) == pytest.approx(record.r2_gap, abs=5e-5)
        assert float(cells[5]) == pytest.approx(
            float(row["test_r2"]) - float(row["har_r2"]), abs=5e-5
        )
        shared = stats["per_fold"][fold]
        assert float(cells[8]) == pytest.approx(shared["qlike_gap"], abs=5e-5)
        assert float(cells[8]) == pytest.approx(
            float(row["har_qlike"]) - float(row["test_qlike"]), abs=5e-5
        )

    text = _figure_text(figure)
    assert QLIKE_GAP_SIGN_CONVENTION in text
    assert "R² gap = test_r2 - har_r2" in text
    assert text.count("Gap convention") == 1
    assert text.count("positive = model better than HAR") == 1

    qlike_gaps = [
        table[(row_index, 8)].get_text().get_text()
        for row_index in range(1, len(frame) + 1)
    ]
    r2_gaps = [
        table[(row_index, 5)].get_text().get_text()
        for row_index in range(1, len(frame) + 1)
    ]
    assert any(gap.startswith("+") for gap in qlike_gaps)
    assert any(gap.startswith("-") for gap in qlike_gaps)
    assert any(gap.startswith("+") for gap in r2_gaps)
    assert any(gap.startswith("-") for gap in r2_gaps)


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                if "tmp_path" in _fn.__code__.co_varnames or "monkeypatch" in _fn.__code__.co_varnames:
                    continue
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
