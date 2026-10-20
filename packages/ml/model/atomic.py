"""Atomic file writes: a sibling ``*.tmp`` file plus ``os.replace``.

Every writer that must never expose a partially written file routes through
:func:`write_via_tmp`: the parent directory is created first, the payload is
written to ``<name>.tmp``, and the final path appears only through the atomic
rename. Any failure — including a failure in the rename itself — removes the
temporary file, so the previous file at the target path stays intact and no
``*.tmp`` residue is left behind.

The typed helpers below cover the model writers (bytes, text, strict JSON,
NumPy arrays, joblib pickles). Callers that need their own serializer (e.g. a
module whose tests inject a failure into its local ``json``/``joblib``) can
pass a writer to :func:`write_via_tmp` directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np


def write_via_tmp(path: Path | str, writer: Callable[[Path], None]) -> Path:
    """Write ``path`` through a sibling temporary file and an atomic rename.

    ``writer`` receives the temporary path and must leave a complete payload
    there. The parent directory is created before the temporary file is
    opened, so a dir-less target cannot fail with ``FileNotFoundError``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    try:
        writer(tmp_path)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target


def write_bytes_atomically(path: Path | str, payload: bytes) -> Path:
    """Write raw bytes atomically."""

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "wb") as handle:
            handle.write(payload)

    return write_via_tmp(path, _write)


def write_text_atomically(
    path: Path | str, text: str, *, encoding: str = "utf-8"
) -> Path:
    """Write text atomically."""

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding=encoding) as handle:
            handle.write(text)

    return write_via_tmp(path, _write)


def write_json_atomically(
    path: Path | str,
    payload: Any,
    *,
    trailing_newline: bool = False,
    **dump_kwargs: Any,
) -> Path:
    """Write one JSON document atomically.

    ``dump_kwargs`` pass through to ``json.dump`` (``indent``,
    ``sort_keys``, ``allow_nan``, ``default``, ...). A failure while dumping
    removes the temporary file and leaves any previous document intact.
    """

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, **dump_kwargs)
            if trailing_newline:
                handle.write("\n")

    return write_via_tmp(path, _write)


def write_npy_atomically(path: Path | str, array: np.ndarray) -> Path:
    """Write a NumPy array atomically, preserving the target suffix.

    The array is written through an open file handle so ``np.save`` never
    appends its own ``.npy`` suffix to the temporary name.
    """

    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "wb") as handle:
            np.save(handle, array)

    return write_via_tmp(path, _write)


def write_pickle_atomically(path: Path | str, payload: Any) -> Path:
    """Write a joblib pickle atomically."""

    def _write(tmp_path: Path) -> None:
        joblib.dump(payload, tmp_path)

    return write_via_tmp(path, _write)
