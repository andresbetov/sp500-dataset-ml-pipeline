"""Tests pinning the HAR benchmark's corrected bibliographic labels.

The canonical HAR-RV benchmark is Corsi (2009); the older ``Corsi 2008``
working-paper label and the ``Corsi's objective`` fit-scale shorthand are both
stale. The fit is a variance-level variant of the equation Corsi specifies in
realized-volatility levels, and the written artifact has to say so. These
checks pin:

- ``Corsi (2009)`` in the module header, the fit-scale disclosure, and the
  written ``har_benchmark.json`` benchmark label, with no ``Corsi 2008`` left
  in product sources or the top-level docs;
- the variance-level variant note replaces ``Corsi's objective`` while the
  whitespace-normalized fit/report-scale disclosure survives;
- a fixture-written artifact carries the corrected label and the same fold
  coefficients the fit returned (stored artifacts are not touched; the check
  is fixture-scoped);
- the label-only change leaves the fit math byte-identical on the shared
  fixture.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_har_labels.py
"""

from __future__ import annotations

import inspect
import json
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import numpy as np

from conftest import REPO_ROOT
from packages.ml.model.training import har_benchmark, training

FLOOR = float(training.RV_FLOOR)

FIXTURE_ROWS = 80
FIXTURE_TRAIN_ROWS = 60

CORRECT_LABEL = "Corsi (2009)"
STALE_LABEL = "Corsi 2008"
STALE_SHORTHAND = "Corsi's objective"

# User-facing docs that carried the same benchmark label as the product code.
LABEL_DOC_PATHS = (REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _label_sources() -> dict[str, str]:
    """Whitespace-normalized product/doc sources that carry the Corsi label."""
    sources = {
        str(Path(module.__file__).relative_to(REPO_ROOT)): Path(
            module.__file__
        ).read_text(encoding="utf-8")
        for module in (har_benchmark, training)
    }
    for path in LABEL_DOC_PATHS:
        sources[str(path.relative_to(REPO_ROOT))] = path.read_text(encoding="utf-8")
    return {name: _normalize(text) for name, text in sources.items()}


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic live-shaped regressors and log targets for the fit checks."""
    index = np.arange(FIXTURE_ROWS, dtype=np.float64)
    regressors = np.column_stack(
        [
            6e-4 * (1.0 + 0.1 * np.sin(0.13 * index)),
            6e-4 * (1.0 + 0.1 * np.sin(0.11 * index + 1.0)),
            6e-4 * (1.0 + 0.1 * np.sin(0.07 * index + 2.0)),
        ]
    )
    target_log = 0.5 * np.log(0.04 * (1.0 + 0.1 * np.cos(0.2 * index)))
    return regressors, target_log


def _fit_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit the real benchmark on the fixture's first 60 rows."""
    regressors, target_log = _fixture()
    indices = np.arange(len(regressors))
    train_indices = indices[:FIXTURE_TRAIN_ROWS]
    test_indices = indices[FIXTURE_TRAIN_ROWS:]
    result = har_benchmark.fit_har_benchmark(
        train_indices=train_indices,
        test_indices=test_indices,
        target_log_train=target_log[train_indices],
        target_log_test=target_log[test_indices],
        regressors=regressors,
        forecast_floor=FLOOR,
        warmup_mask=np.zeros(len(regressors), dtype=bool),
        fold_key="fold_0",
    )
    return regressors, target_log, train_indices, test_indices, result


def _write_fixture_artifact() -> tuple[dict, dict]:
    """Write and read back ``har_benchmark.json`` under a temporary root."""
    _, _, _, _, result = _fit_fixture()
    validation = {
        "fold_0": {"test_pred": np.array([0.0]), "test_true": np.array([0.0])}
    }
    summary = {"fold_0": {"test_r2": 0.5, "test_qlike": 0.25}}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "results").mkdir()
        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            training._save_training_results(
                validation, summary, {"fold_0": result}
            )
        report = json.loads(
            (root / "results" / "har_benchmark.json").read_text(encoding="utf-8")
        )
    return report, result


def test_corsi_label_updated_everywhere() -> None:
    """``Corsi (2009)`` is present and no ``Corsi 2008`` label remains."""
    sources = _label_sources()
    har_source = sources["packages/ml/model/training/har_benchmark.py"]
    training_source = sources["packages/ml/model/training/training.py"]

    assert CORRECT_LABEL in har_source
    assert CORRECT_LABEL in training_source

    stale = sorted(name for name, text in sources.items() if STALE_LABEL in text)
    assert not stale, f"stale {STALE_LABEL!r} label remains in: {', '.join(stale)}"

    assert "Canonical HAR-RV benchmark (Corsi (2009))" in har_source
    assert (
        "HAR-RV (Corsi (2009)): canonical 1/5/22 trailing variance components"
        in training_source
    )


def test_variance_level_variant_note_replaces_fit_objective_claim() -> None:
    """The fit is disclosed as a variance-level variant of Corsi (2009) Eq. (8)."""
    fit_doc = _normalize(inspect.getdoc(har_benchmark.fit_har_benchmark) or "")
    module_doc = _normalize(inspect.getdoc(har_benchmark) or "")
    combined = f"{module_doc} {fit_doc}"

    assert STALE_SHORTHAND not in combined, (
        "the fit-scale disclosure still uses the stale 'Corsi's objective' "
        "shorthand instead of the variance-level variant wording"
    )
    assert "variance-level variant" in combined
    assert "Corsi (2009) Eq. (8)" in combined
    assert "realized-volatility levels" in combined
    assert "variance and log are the variants" in combined

    # The earlier fit-scale disclosure content survives around the replaced note.
    assert "The fit optimizes variance-level squared error" in combined
    assert (
        "so the log-scale R² below is a comparability metric, not the fit target; "
        "QLIKE is the headline model-vs-benchmark comparison."
    ) in combined


def test_fixture_artifact_carries_the_corrected_benchmark_label() -> None:
    """A fixture-written artifact carries Corsi (2009) and the fitted values."""
    report, result = _write_fixture_artifact()

    assert report["benchmark"].startswith("HAR-RV (Corsi (2009)):")
    assert STALE_LABEL not in json.dumps(report)
    assert report["folds"]["fold_0"]["coefficients"] == result["coefficients"]
    assert report["folds"]["fold_0"]["n_test"] == result["n_test"]
    assert report["folds"]["fold_0"]["fit_scale"] == "variance"


def test_labels_only_change_leaves_fit_math_identical() -> None:
    """Coefficients and forecasts still equal the canonical variance-level OLS."""
    regressors, target_log, train_indices, test_indices, result = _fit_fixture()

    train_design = np.column_stack(
        [np.ones(train_indices.size), regressors[train_indices]]
    )
    test_design = np.column_stack(
        [np.ones(test_indices.size), regressors[test_indices]]
    )
    expected_coefficients = np.linalg.lstsq(
        train_design, np.exp(2.0 * target_log[train_indices]), rcond=None
    )[0]
    expected_forecast_log = 0.5 * np.log(
        np.maximum(test_design @ expected_coefficients, FLOOR**2)
    )

    assert np.array_equal(np.asarray(result["coefficients"]), expected_coefficients)
    assert np.array_equal(result["forecast_log"], expected_forecast_log)
    assert result["fit_scale"] == "variance"
    assert result["clip_count"] == 0


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
