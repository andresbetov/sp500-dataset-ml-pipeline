"""Per-plot isolation and the on-disk visualization manifest.

Each plot is rendered through :func:`render_plot`: a failure is logged and
recorded but never aborts its siblings. :func:`collect_plot_statuses` activates
a context-local collector so the orchestrator can write one ``manifest.json``
next to the PNGs. Entries carry exactly ``category``, ``plot``, ``status``, and
``error``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import matplotlib.pyplot as plt

from packages.ml.model.evaluation.stats import sanitize_json

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"

_ACTIVE_STATUSES: ContextVar[list[dict] | None] = ContextVar(
    "visualization_plot_statuses", default=None
)


@contextmanager
def collect_plot_statuses() -> Iterator[list[dict]]:
    """Collect the statuses recorded by every plot in the context."""
    statuses: list[dict] = []
    token = _ACTIVE_STATUSES.set(statuses)
    try:
        yield statuses
    finally:
        _ACTIVE_STATUSES.reset(token)


def _record(category: str, plot: str, status: str, error: str | None) -> None:
    statuses = _ACTIVE_STATUSES.get()
    if statuses is None:
        return
    statuses.append(
        {
            "category": category,
            "plot": plot,
            "status": status,
            "error": error,
        }
    )


def render_plot(
    category: str,
    plot: str,
    render: Callable[[Path], Path],
    output_dir: Path,
) -> Path | None:
    """Render one plot, isolating a failure from its siblings.

    Returns the saved path on success and ``None`` on failure. A failure is
    logged with its traceback and recorded in the active status collector (when
    one is open) as ``status: "error"`` with the exception type and message.
    """
    output_path = output_dir / f"{plot}.png"
    try:
        path = render(output_path)
    except Exception as exc:
        logger.exception("Visualization %s/%s failed: %s", category, plot, exc)
        # A raise can leave a half-built figure open; close the current figure
        # without touching any other figure the caller may own.
        plt.close()
        _record(category, plot, "error", f"{type(exc).__name__}: {exc}")
        return None

    _record(category, plot, "ok", None)
    return path


def write_plot_manifest(
    output_dir: Path,
    statuses: list[dict],
    stats: dict | None = None,
) -> Path:
    """Write the per-plot manifest (and optional run stats) atomically.

    ``stats`` is sanitized with the shared strict-JSON policy before the
    write, so non-finite values become ``null`` instead of an invalid
    ``NaN``/``Infinity`` token.
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    payload: dict[str, object] = {"plots": statuses}
    if stats is not None:
        payload["stats"] = sanitize_json(stats)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(tmp_path, manifest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return manifest_path
