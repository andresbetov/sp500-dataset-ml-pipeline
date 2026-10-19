"""Shared evaluation statistics and the one strict-JSON metric policy.

Three pieces, shared by the training and reporting layers:

- ``sanitize_json`` is the single results-writer sanitizer. It recursively
  projects a payload onto strict-JSON data: non-finite floats (``NaN``,
  ``inf``, ``-inf``) become ``None``, numpy scalars and arrays become their
  Python/list equivalents, and any value JSON cannot encode raises
  ``UnserializableError`` naming the offending type. Writers pair it with
  ``json.dump(..., allow_nan=False)``, so a non-standard ``NaN``/``Infinity``
  token can never reach a results file and a stray object fails the write
  instead of being masked by ``str()``.
- ``spread`` is the sample standard deviation of a finite value set
  (``ddof=1`` by default), the shared dispersion helper for evaluation
  summaries.
- ``quarantine_record`` is the failed-fold record shape,
  ``{"error": "<type>: <message>"}``; ``AllFoldsInvalidError`` is the named
  error raised when no fold contributes a valid headline metric
  (``n_valid == 0``).
"""

from __future__ import annotations

import math

import numpy as np


class UnserializableError(TypeError):
    """Raised when a payload holds a value strict JSON cannot encode."""


class AllFoldsInvalidError(RuntimeError):
    """Raised when no fold contributes a valid headline metric (``n_valid == 0``)."""


def quarantine_record(exc: BaseException) -> dict:
    """The failed-fold quarantine record: ``{"error": "<type>: <message>"}``.

    The one shape a quarantined fold is recorded with, wherever the fold's
    failure is kept: a fold whose metric computation raises contributes this
    record instead of a metric set, and the run records it outside the
    five-row fold summary (``quarantined_folds`` in the run manifest/record).
    """
    return {"error": f"{type(exc).__name__}: {exc}"}


def sanitize_json(value: object) -> object:
    """Recursively project ``value`` onto strict-JSON data.

    Non-finite floats (``NaN``, ``inf``, ``-inf``) become ``None`` — the only
    strict-JSON spelling of an invalid number — numpy scalars become Python
    scalars, numpy arrays and tuples become lists, and dict/list containers
    are sanitized item by item. Anything else that JSON cannot encode raises
    ``UnserializableError`` naming the type, so an unencodable object fails
    the write instead of being masked by ``str()``. Dict keys must be strings:
    JSON's integer-key coercion would silently relabel a key.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UnserializableError(
                    "strict JSON sanitizer requires string dict keys, got "
                    f"{type(key).__name__}: {key!r}"
                )
            sanitized[key] = sanitize_json(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        # tolist() carries 0-d arrays to scalars and n-d arrays to nested
        # lists, so one recursive pass covers every shape.
        return sanitize_json(value.tolist())
    raise UnserializableError(
        f"strict JSON sanitizer cannot encode {type(value).__name__}: {value!r}"
    )


def spread(values, ddof: int = 1) -> float:
    """Sample standard deviation of a finite value set, ``ddof=1`` by default.

    The shared dispersion helper for evaluation summaries: the caller filters
    invalid (NaN/None) values first — a non-finite value here is a contract
    violation and raises — and the returned spread uses the requested degrees
    of freedom. An empty set or a ``ddof`` the set cannot support raises
    rather than returning a silent NaN.
    """
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("spread received no values")
    if not np.isfinite(array).all():
        raise ValueError("spread received non-finite values; filter them first")
    if not 0 <= ddof < array.size:
        raise ValueError(f"spread ddof={ddof} is outside [0, {array.size})")
    return float(np.std(array, ddof=ddof))
