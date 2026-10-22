"""Feature-importance plotting is source-only and documents its limits.

The plot resolves every checkpoint from the run's recorded artifact paths, so
the old ``predictions_dict``/``summary_df`` head parameters were dead weight:
callers could mutate them without changing a pixel. The public signature takes
only the output path, the single caller passes only that path, and the
docstring names the gain/correlation interpretation limits and the pinned
37-feature contract.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import conftest  # noqa: F401  (puts the repository root on sys.path)

from packages.ml.model.features.feature_engineering import INDICATORS
from packages.ml.model.visualization import training_diagnostics

MODULE_PATH = Path(training_diagnostics.__file__)


def test_plot_takes_only_the_output_path() -> None:
    """The two unused data parameters are gone from the public signature."""
    parameters = list(
        inspect.signature(training_diagnostics.plot_feature_importance).parameters.values()
    )
    assert [parameter.name for parameter in parameters] == ["output_path"]
    assert parameters[0].default is None
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_dropped_data_arguments_raise_type_error(tmp_path: Path) -> None:
    """Passing the old head arguments is a TypeError, not a silent render."""
    with pytest.raises(TypeError):
        training_diagnostics.plot_feature_importance(
            {}, None, tmp_path / "feature_importance.png"
        )


def test_the_single_caller_passes_only_the_output_path() -> None:
    """``generate_training_diagnostics`` no longer forwards the dead data."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    caller = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate_training_diagnostics"
    )
    calls = [
        node
        for node in ast.walk(caller)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "plot_feature_importance"
    ]
    assert len(calls) == 1, calls
    assert len(calls[0].args) == 1, ast.dump(calls[0])
    assert calls[0].keywords == [], ast.dump(calls[0])


def test_docstring_documents_gain_and_correlation_limits() -> None:
    """The docstring discloses the interpretation limits of the metric."""
    docstring = inspect.getdoc(training_diagnostics.plot_feature_importance) or ""
    lowered = docstring.lower()
    assert "gain" in lowered
    assert "correlat" in lowered
    assert f"{len(INDICATORS)} features" in lowered
