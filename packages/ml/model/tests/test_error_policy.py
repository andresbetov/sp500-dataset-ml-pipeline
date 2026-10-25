"""Tests for the one metric error policy: quarantine, finite filtering, strict JSON.

The policy under test:

- a fold whose metric computation fails is quarantined — the fold record
  ``{"error": "<type>: <message>"}`` is recorded outside the five-row
  ``fold_training_summary.json`` (in the run record), its metric values are
  invalid, and the fold loop continues instead of aborting;
- global metrics are filtered to the folds that contributed a finite value,
  each with its own ``n_valid``/``n_total`` count and a warning, and the run
  raises only when the headline QLIKE has ``n_valid == 0``;
- the QLIKE summary is decoupled from R² validity: a fold with NaN R² but a
  valid QLIKE still contributes its QLIKE;
- results writers serialize strictly through the evaluation sanitizer
  (``NaN``/``inf`` → ``null``, unencodable objects raise) with
  ``allow_nan=False``;
- the run completion does not recompute or re-log the fold-mean QLIKE: the
  report layer owns the single aggregate, and a quarantined fold cannot stop
  the orchestration from completing;
- ``qlike_loss`` itself stays fail-loud on non-finite input.

Every phase check drives a real ``phase_4_training`` against synthetic
artifacts under a temporary root (stub trainer, constant HAR regressors), so
no live ``artifacts/**`` or ``data/model_outputs/**`` file is written; the
live check reads the stored artifacts only. The evaluation-package checks
import the package inside the test bodies, so this file still collects when
run against the branch point and the behavioral red failures (a fold abort,
silently dropped NaN metrics, a ``NaN`` token in JSON) stay observable there.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_error_policy.py
"""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import math
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pytest

from conftest import ARTIFACTS_DIR, REPO_ROOT, requires_live_artifacts
from packages.ml.model.artifact_integrity import REGISTRY_FILENAME, record_artifact
from packages.ml.model.cross_validation import cv_utils
from packages.ml.model.training import training

# One row per trading date over 100 dates; five expanding folds whose test
# blocks have pairwise-distinct sizes (6/8/10/12/14 scored rows), so an
# injected metric failure can be aimed at exactly one fold by its row count.
N_ROWS = 100
N_INDICATORS = 3
N_TICKER_FEATURES = 1
TRAIN_STOPS = (20, 26, 34, 44, 56)
TEST_STOPS = (26, 34, 44, 56, 70)
FOLD_TEST_SIZES = tuple(
    test_stop - train_stop for train_stop, test_stop in zip(TRAIN_STOPS, TEST_STOPS)
)
QUARANTINED_FOLD = 2
QUARANTINED_FOLD_KEY = f"fold_{QUARANTINED_FOLD}"
QUARANTINED_TEST_SIZE = FOLD_TEST_SIZES[QUARANTINED_FOLD]

# The pipeline entry point whose completion path must not aggregate QLIKE.
MAIN_PATH = REPO_ROOT / "packages" / "ml" / "model" / "main.py"

# The per-fold metric keys the policy filters and a quarantine invalidates.
METRIC_KEYS = (
    "test_r2",
    "test_mae",
    "test_rmse",
    "test_qlike",
    "har_qlike",
    "har_r2",
)


def _reject_constant(token: str):
    raise AssertionError(f"non-strict JSON constant emitted: {token}")


def _parse_strict(text: str):
    """Parse strict JSON, rejecting the NaN/Infinity extension tokens."""
    return json.loads(text, parse_constant=_reject_constant)


class _WarningRecorder(logging.Handler):
    """Collect formatted log messages without pytest's caplog fixture."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _InfoRecorder(logging.Handler):
    """Collect formatted INFO messages without pytest's caplog fixture."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _RecordingModel:
    """Picklable fold-model double with deterministic row-indexed forecasts."""

    def get_params(self) -> dict:
        return {"n_estimators": 1, "missing": float("nan")}

    def predict(self, X: np.ndarray) -> np.ndarray:
        rows = np.asarray(X)[:, 0].astype(np.float64)
        return -1.4 + 0.05 * np.sin(rows)


def _stub_trainer(X_train, y_train, X_val=None, y_val=None):
    return _RecordingModel(), {}


def _stub_har_regressors(dates: np.ndarray, tickers: np.ndarray) -> tuple:
    """Constant components with no warm-up rows; the HAR fit stays real."""
    return (
        np.full((len(dates), 3), 0.04, dtype=np.float64),
        np.zeros(len(tickers), dtype=bool),
    )


def _write_five_fold_artifacts(root: Path) -> None:
    """Write the synthetic five-fold Phase 2-3 input set under ``root``."""
    rng = np.random.default_rng(23)
    y = rng.uniform(0.05, 0.6, size=N_ROWS)
    X = rng.normal(size=(N_ROWS, N_INDICATORS + N_TICKER_FEATURES))
    X[:, 0] = np.arange(N_ROWS, dtype=np.float64)
    dates = np.datetime64("2000-01-03") + np.arange(N_ROWS)

    metadata = {
        "ticker": np.array([f"T{row % 3}" for row in range(N_ROWS)]),
        "date": dates,
        "regime": (np.arange(N_ROWS) % 7).astype(np.int8),
        "n_indicators": N_INDICATORS,
        "n_ticker_features": N_TICKER_FEATURES,
    }
    folds_metadata = {}
    for fold_idx, (train_stop, test_stop) in enumerate(
        zip(TRAIN_STOPS, TEST_STOPS)
    ):
        folds_metadata[f"fold_{fold_idx}"] = {
            "train_indices": list(range(train_stop)),
            "test_indices": list(range(train_stop, test_stop)),
            "n_train": train_stop,
            "n_test": test_stop - train_stop,
            "train_date_min": str(dates[0]),
            "train_date_max": str(dates[train_stop - 1]),
            "test_date_min": str(dates[train_stop]),
            "test_date_max": str(dates[test_stop - 1]),
        }
    folds_metadata["holdout"] = {
        "bounds": {"start": str(dates[TEST_STOPS[-1]]), "end": str(dates[-1])},
        "n_dates": N_ROWS - TEST_STOPS[-1],
    }

    inputs = root / "inputs"
    folds = root / "folds"
    inputs.mkdir(parents=True)
    folds.mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "results").mkdir()
    np.save(inputs / "X.npy", X)
    np.save(inputs / "Y.npy", y)
    joblib.dump(metadata, inputs / "metadata.pkl")
    joblib.dump(folds_metadata, folds / "folds_metadata.pkl")
    # The live readers verify recorded hashes before loading; the synthetic
    # inputs must carry their registry entries like the real writers'.
    record_artifact(inputs / "metadata.pkl", registry_path=root / REGISTRY_FILENAME)
    record_artifact(
        folds / "folds_metadata.pkl", registry_path=root / REGISTRY_FILENAME
    )


def _prepared_root(tmpdir: str) -> Path:
    """A temporary artifacts root with the synthetic Phase 2-3 inputs."""
    root = Path(tmpdir) / "artifacts"
    assert not root.resolve().is_relative_to(REPO_ROOT), (
        "the phase run must target a temporary root, never live artifacts"
    )
    root.mkdir()
    _write_five_fold_artifacts(root)
    return root


@contextlib.contextmanager
def _phase_environment(
    root: Path,
    *,
    failing_qlike_sizes: tuple = (),
    nan_r2_size: int | None = None,
):
    """Redirect one phase run: temp artifact root, stub trainer/HAR, injections.

    ``failing_qlike_sizes`` makes ``training.qlike_loss`` raise for calls whose
    actual-log array has one of the given sizes (the fold test blocks are
    size-distinct), and ``nan_r2_size`` makes ``training.r2_score`` return NaN
    for that size — a metric value that is invalid without raising.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(training, "ARTIFACTS_DIR", root))
        stack.enter_context(mock.patch.object(cv_utils, "ARTIFACTS_DIR", root))
        stack.enter_context(
            mock.patch.object(training, "load_har_regressors", _stub_har_regressors)
        )
        stack.enter_context(
            mock.patch.object(training, "train_xgboost_model", _stub_trainer)
        )
        if failing_qlike_sizes:
            real_qlike_loss = training.qlike_loss

            def failing_qlike_loss(actual_log, forecast_log):
                size = int(np.asarray(actual_log).size)
                if size in failing_qlike_sizes:
                    raise ValueError(
                        f"injected metric failure on the {size}-row fold test block"
                    )
                return real_qlike_loss(actual_log, forecast_log)

            stack.enter_context(
                mock.patch.object(training, "qlike_loss", failing_qlike_loss)
            )
        if nan_r2_size is not None:
            real_r2_score = training.r2_score

            def nan_r2_score(y_true, y_pred):
                if int(np.asarray(y_true).size) == nan_r2_size:
                    return float("nan")
                return real_r2_score(y_true, y_pred)

            stack.enter_context(
                mock.patch.object(training, "r2_score", nan_r2_score)
            )
        yield


def _run_phase(root: Path, **injections) -> tuple:
    """Drive one real ``phase_4_training`` run under the temporary root."""
    with _phase_environment(root, **injections):
        validation, summary, model_paths = training.phase_4_training()
    summary_json = _parse_strict(
        (root / "results" / "fold_training_summary.json").read_text(encoding="utf-8")
    )
    manifest = _parse_strict(
        (root / "run_manifest.json").read_text(encoding="utf-8")
    )
    return validation, summary, model_paths, summary_json, manifest


@contextlib.contextmanager
def _captured_package_logs():
    """Capture the package's INFO lines; the standalone runner has no caplog."""
    logger = logging.getLogger("packages.ml.model")
    handler = _InfoRecorder()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _main_function() -> ast.FunctionDef:
    """The parsed ``main`` function of the pipeline entry point."""
    module = ast.parse(
        MAIN_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PATH)
    )
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _references_name(node: ast.AST, name: str) -> bool:
    """Whether any descendant of ``node`` loads the plain name ``name``."""
    return any(
        isinstance(child, ast.Name) and child.id == name for child in ast.walk(node)
    )


def _run_main_with_stubbed_phases(root: Path, summary: dict) -> None:
    """Drive ``main`` with lightweight phase doubles under ``root``."""
    from packages.ml.model import main as main_module

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(main_module, "ARTIFACTS_DIR", root))
        stack.enter_context(
            mock.patch.object(
                main_module,
                "phase_2_feature_selection",
                return_value=(np.zeros((2, 2)), np.zeros(2), {}, None),
            )
        )
        stack.enter_context(
            mock.patch.object(
                main_module, "phase_3_cross_validation", return_value=[{}] * 5
            )
        )
        stack.enter_context(
            mock.patch.object(
                main_module,
                "phase_4_training",
                return_value=(
                    {},
                    summary,
                    [f"model_{fold_idx}" for fold_idx in range(5)],
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                main_module, "generate_all_visualizations", return_value={}
            )
        )
        main_module.main()


def test_fold_metric_failure_is_quarantined_not_fatal() -> None:
    """An injected per-fold metric failure quarantines that fold; the rest run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        handler = _WarningRecorder()
        logger = training.logger
        logger.addHandler(handler)
        try:
            validation, summary, model_paths, summary_json, manifest = _run_phase(
                root, failing_qlike_sizes=(QUARANTINED_TEST_SIZE,)
            )
        finally:
            logger.removeHandler(handler)

        # Every fold still completes its row: the loop never aborted.
        fold_keys = {f"fold_{fold_idx}" for fold_idx in range(5)}
        assert set(validation) == set(summary) == fold_keys
        assert len(model_paths) == 5

        # The failing fold is recorded outside the summary, as the one
        # quarantine record shape, and only that fold is quarantined.
        quarantined = manifest["quarantined_folds"]
        assert set(quarantined) == {QUARANTINED_FOLD_KEY}
        error = quarantined[QUARANTINED_FOLD_KEY]["error"]
        assert error.startswith("ValueError: ")
        assert "injected metric failure" in error
        assert manifest["folds"][QUARANTINED_FOLD]["status"] == "quarantined"
        assert [
            fold["status"] for fold in manifest["folds"]
        ] == ["completed", "completed", "quarantined", "completed", "completed"]

        # The quarantined fold contributes no metric value; the others do.
        for key in METRIC_KEYS:
            assert math.isnan(summary[QUARANTINED_FOLD_KEY][key]), key
            assert summary_json[QUARANTINED_FOLD_KEY][key] is None, key
        for fold_idx in range(5):
            if fold_idx == QUARANTINED_FOLD:
                continue
            fold_key = f"fold_{fold_idx}"
            for key in METRIC_KEYS:
                assert math.isfinite(summary[fold_key][key]), (fold_key, key)
                assert math.isfinite(summary_json[fold_key][key]), (fold_key, key)

        # The finite filtering is recorded: 4 of 5 folds valid, per metric.
        global_metrics = manifest["global_metrics"]
        assert global_metrics["n_valid_folds"] == 4
        assert global_metrics["n_total_folds"] == 5
        qlike_block = global_metrics["metrics"]["test_qlike"]
        assert qlike_block["n_valid"] == 4 and qlike_block["n_total"] == 5
        assert math.isfinite(qlike_block["mean"])

        # The alert fires: a fold-level warning names the fold and the reason,
        # and a global warning states the valid/total count.
        assert any(
            "Fold 2 metric computation failed" in message
            and "quarantining the fold" in message
            for message in handler.messages
        ), handler.messages
        assert any(
            "4/5" in message and "headline QLIKE" in message
            for message in handler.messages
        ), handler.messages


def test_qlike_summary_is_decoupled_from_r2_validity() -> None:
    """A fold with NaN R² but a valid QLIKE still contributes its QLIKE."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        handler = _WarningRecorder()
        logger = training.logger
        logger.addHandler(handler)
        try:
            _validation, summary, _model_paths, _summary_json, manifest = _run_phase(
                root, nan_r2_size=QUARANTINED_TEST_SIZE
            )
        finally:
            logger.removeHandler(handler)

        # The NaN R² is an invalid value, not a quarantine: the fold's QLIKE
        # is valid and the run records no quarantined fold.
        assert math.isnan(summary[QUARANTINED_FOLD_KEY]["test_r2"])
        assert math.isfinite(summary[QUARANTINED_FOLD_KEY]["test_qlike"])
        assert manifest["quarantined_folds"] == {}

        metrics = manifest["global_metrics"]["metrics"]
        assert metrics["test_r2"]["n_valid"] == 4
        assert metrics["test_r2"]["n_total"] == 5
        assert metrics["test_qlike"]["n_valid"] == 5
        assert metrics["test_qlike"]["n_total"] == 5

        # The QLIKE summary includes the NaN-R² fold's QLIKE: the recorded mean
        # is the mean over all five fold values, not over the four valid-R² ones.
        all_qlike = [summary[f"fold_{idx}"]["test_qlike"] for idx in range(5)]
        assert metrics["test_qlike"]["mean"] == pytest.approx(
            float(np.mean(all_qlike)), rel=1e-12, abs=1e-12
        )
        valid_r2_qlike = [
            summary[f"fold_{idx}"]["test_qlike"]
            for idx in range(5)
            if math.isfinite(summary[f"fold_{idx}"]["test_r2"])
        ]
        assert abs(
            metrics["test_qlike"]["mean"] - float(np.mean(valid_r2_qlike))
        ) > 1e-12

        assert any(
            "test_r2 4/5" in message for message in handler.messages
        ), handler.messages


def test_all_folds_invalid_raises_the_named_error() -> None:
    """``n_valid == 0`` raises the named error instead of reporting nothing."""
    from packages.ml.model.evaluation.stats import AllFoldsInvalidError

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        with _phase_environment(root, failing_qlike_sizes=FOLD_TEST_SIZES):
            with pytest.raises(AllFoldsInvalidError) as excinfo:
                training.phase_4_training()

    message = str(excinfo.value)
    assert "0/5" in message
    assert "test_qlike" in message
    assert issubclass(AllFoldsInvalidError, RuntimeError)


def test_five_row_summary_keeps_quarantine_outside() -> None:
    """The fold summary keeps exactly five rows; the quarantine lives outside."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        _validation, _summary, _model_paths, summary_json, manifest = _run_phase(
            root, failing_qlike_sizes=(QUARANTINED_TEST_SIZE,)
        )
        text = (root / "results" / "fold_training_summary.json").read_text(
            encoding="utf-8"
        )

    fold_rows = [key for key in summary_json if key.startswith("fold_")]
    assert len(fold_rows) == 5, sorted(summary_json)
    assert set(summary_json) == set(fold_rows)
    assert "quarantined_folds" not in summary_json
    assert set(manifest["quarantined_folds"]) == {QUARANTINED_FOLD_KEY}

    # The written summary is strict JSON: the quarantined row's invalid
    # metrics are null, never a NaN token.
    assert "NaN" not in text and "Infinity" not in text
    assert _parse_strict(text) == summary_json


def test_main_has_no_duplicate_fold_qlike_mean() -> None:
    """The report layer is the only QLIKE aggregator; main must not repeat it."""
    main_function = _main_function()
    duplicate_means = []
    for node in ast.walk(main_function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue
        if name not in {"mean", "nanmean", "average"}:
            continue
        if any(
            _references_name(argument, "fold_training_summary")
            for argument in node.args
        ):
            duplicate_means.append(node.lineno)

    assert not duplicate_means, (
        "main.py recomputes the fold-mean QLIKE, duplicating the report-layer "
        f"aggregate at line(s) {duplicate_means}"
    )


def test_main_completion_is_safe_with_a_quarantined_fold() -> None:
    """A NaN/quarantined fold still reaches the orchestration's completion."""
    summary = {
        f"fold_{fold_idx}": {
            "test_qlike": (
                float("nan")
                if fold_idx == QUARANTINED_FOLD
                else 0.1 + fold_idx
            ),
            "har_qlike": (
                float("nan")
                if fold_idx == QUARANTINED_FOLD
                else 0.2 + fold_idx
            ),
        }
        for fold_idx in range(5)
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        _run_main_with_stubbed_phases(root, summary)
        manifest = _parse_strict(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )

    assert manifest["status"] == "completed"
    assert [phase["phase"] for phase in manifest["phases"]] == [
        "phase_2_feature_selection",
        "phase_3_cross_validation",
        "phase_4_training",
        "phase_5_visualization",
    ]


def test_pipeline_reports_the_aggregate_qlike_exactly_once() -> None:
    """The report-layer aggregate is logged once, never by ``main`` as well."""
    from packages.ml.model import main as main_module

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _prepared_root(tmpdir)
        with _captured_package_logs() as messages:
            with _phase_environment(
                root, failing_qlike_sizes=(QUARANTINED_TEST_SIZE,)
            ):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(main_module, "ARTIFACTS_DIR", root)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module,
                            "phase_2_feature_selection",
                            return_value=(
                                np.zeros((2, 2)),
                                np.zeros(2),
                                {},
                                None,
                            ),
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module,
                            "phase_3_cross_validation",
                            return_value=[{}] * 5,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            main_module,
                            "generate_all_visualizations",
                            return_value={},
                        )
                    )
                    main_module.main()

        manifest = _parse_strict(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )

    assert manifest["status"] == "completed"
    assert set(manifest["quarantined_folds"]) == {QUARANTINED_FOLD_KEY}

    aggregate_lines = [
        message
        for message in messages
        if "mean QLIKE" in message or "aggregate QLIKE" in message
    ]
    assert len(aggregate_lines) == 1, aggregate_lines
    assert aggregate_lines[0].startswith("Headline aggregate QLIKE"), (
        aggregate_lines
    )


def test_results_writers_serialize_strictly() -> None:
    """NaN/inf metrics serialize as null; the files parse with no NaN token."""
    validation = {
        "fold_0": {
            "test_pred": np.array([0.0, 0.1]),
            "test_true": np.array([0.05, 0.15]),
        }
    }
    summary = {
        "fold_0": {
            "test_r2": float("nan"),
            "test_mae": float("inf"),
            "test_rmse": np.float64(0.5),
            "test_qlike": -float("inf"),
            "test_note": ("tuple", 1),
        }
    }
    har_results = {
        "fold_0": {
            "n_test": 2,
            "clip_count": 0,
            "clip_rate": 0.0,
            "har_r2": float("nan"),
            "har_qlike": 0.2,
            "coefficient_interpretation": False,
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            training._save_training_results(validation, summary, har_results)
        summary_text = (root / "results" / "fold_training_summary.json").read_text(
            encoding="utf-8"
        )
        report_text = (root / "results" / "har_benchmark.json").read_text(
            encoding="utf-8"
        )

    for text in (summary_text, report_text):
        assert "NaN" not in text, text
        assert "Infinity" not in text, text
        _parse_strict(text)

    summary_json = _parse_strict(summary_text)
    assert summary_json["fold_0"]["test_r2"] is None
    assert summary_json["fold_0"]["test_mae"] is None
    assert summary_json["fold_0"]["test_qlike"] is None
    assert summary_json["fold_0"]["test_rmse"] == 0.5
    assert summary_json["fold_0"]["test_note"] == ["tuple", 1]

    report_json = _parse_strict(report_text)
    assert report_json["folds"]["fold_0"]["har_r2"] is None
    assert report_json["folds"]["fold_0"]["model_r2"] is None
    assert report_json["folds"]["fold_0"]["model_qlike"] is None


def test_unserializable_objects_fail_instead_of_being_masked() -> None:
    """The sanitizer raises on values JSON cannot encode; writers use it."""
    from packages.ml.model.evaluation.stats import (
        UnserializableError,
        sanitize_json,
    )

    with pytest.raises(UnserializableError):
        sanitize_json({"value": object()})
    with pytest.raises(UnserializableError):
        sanitize_json({1: "non-string key"})
    assert sanitize_json(
        {"nested": [np.float64("nan"), np.int64(3)], "ok": None}
    ) == {"nested": [None, 3], "ok": None}

    validation = {
        "fold_0": {
            "test_pred": np.array([0.0]),
            "test_true": np.array([0.0]),
        }
    }
    summary = {"fold_0": {"test_r2": 0.5, "test_qlike": 0.25, "bad": object()}}
    har_results = {
        "fold_0": {
            "n_test": 1,
            "clip_count": 0,
            "clip_rate": 0.0,
            "har_r2": 0.1,
            "har_qlike": 0.2,
            "coefficient_interpretation": False,
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "artifacts"
        with mock.patch.object(training, "ARTIFACTS_DIR", root):
            with pytest.raises(UnserializableError):
                training._save_training_results(validation, summary, har_results)


def test_sanitizer_spread_and_quarantine_contract() -> None:
    """The shared evaluation pieces keep their declared contracts."""
    from packages.ml.model.evaluation import stats as evaluation_stats

    # The sanitizer: non-finite floats become null at every nesting depth.
    assert evaluation_stats.sanitize_json(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "neg_inf": -float("inf"),
            "finite": 2.5,
        }
    ) == {"nan": None, "inf": None, "neg_inf": None, "finite": 2.5}
    assert evaluation_stats.sanitize_json(np.array([1.0, np.nan])) == [1.0, None]

    # The dispersion helper: ddof=1 by default, finite values only.
    values = [1.0, 2.0, 3.0, 4.0]
    assert evaluation_stats.spread(values) == pytest.approx(
        float(np.std(values, ddof=1)), rel=1e-12
    )
    assert evaluation_stats.spread(values, ddof=0) == pytest.approx(
        float(np.std(values, ddof=0)), rel=1e-12
    )
    with pytest.raises(ValueError):
        evaluation_stats.spread([])
    with pytest.raises(ValueError):
        evaluation_stats.spread([1.0, float("nan")])
    with pytest.raises(ValueError):
        evaluation_stats.spread([1.0])

    # The quarantine record shape.
    assert evaluation_stats.quarantine_record(ValueError("boom")) == {
        "error": "ValueError: boom"
    }


def test_qlike_loss_guards_stay_fail_loud() -> None:
    """The computation-layer QLIKE guards are untouched: non-finite input raises."""
    from packages.ml.model.training.qlike_objective import (
        MAX_LOG_RATIO,
        QlikeOverflowError,
        qlike_mean,
    )

    for actual, forecast in (
        (np.nan, 0.0),
        (0.0, np.inf),
        (np.inf, 0.0),
        (-np.inf, 0.0),
    ):
        with pytest.raises(ValueError):
            training.qlike_loss(
                np.array([actual], dtype=np.float64),
                np.array([forecast], dtype=np.float64),
            )

    assert MAX_LOG_RATIO == 700.0
    with pytest.raises(QlikeOverflowError):
        qlike_mean(np.array([MAX_LOG_RATIO + 1.0]))

    # A clean fold's loss value is unchanged.
    assert training.qlike_loss(np.array([-9.0]), np.array([-8.8])) == pytest.approx(
        math.exp(-0.2) + 0.2 - 1.0, rel=1e-12
    )


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
