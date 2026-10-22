"""Tests pinning the Hansen-Lunde (2005) citation in the QLIKE docstring.

The QLIKE / Hansen-Lunde identity in ``qlike_loss`` previously carried only a
bare ``Hansen-Lunde 2005`` year mention. The 2005 paper is the QLIKE-origin
comparison and must be pinned in full — journal, volume(issue), pages, and
DOI — and distinguished from the Hansen-Lunde (2006) papers. These checks pin:

- the full journal pin in the ``qlike_loss`` docstring, whitespace-normalized
  because the source wraps it across lines;
- the clause marking the 2005 JAE paper as the QLIKE-origin comparison and
  distinct from the 2006 papers;
- no bare ``Hansen-Lunde 2005`` mention without the full pin, and every
  author-year 2005 reference using the ampersand form the pin establishes;
- the QLIKE numbers are untouched (docstring-only change).

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_har_citations.py
"""

from __future__ import annotations

import inspect
import traceback
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from packages.ml.model.training import training

# The exact full pin required here. Matched whitespace-normalized because the
# docstring wraps it across source lines.
FULL_PIN = (
    "Hansen & Lunde (2005), Journal of Applied Econometrics 20(7), 873-889, "
    "doi:10.1002/jae.800"
)

_DASH_VARIANTS = "\u2212\u2010\u2011\u2012\u2013\u2014\u2015\uFE58\uFE63\uFF0D"


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and unify unicode dashes to ASCII."""
    for dash in _DASH_VARIANTS:
        text = text.replace(dash, "-")
    return " ".join(text.lower().split())


def _qlike_doc() -> str:
    module_path = Path(training.__file__).resolve()
    assert module_path.is_relative_to(REPO_ROOT), (
        f"training resolves outside the repository: {module_path}"
    )
    return _normalize(inspect.getdoc(training.qlike_loss) or "")


def test_full_journal_pin_present_in_qlike_docstring() -> None:
    """The full 2005 JAE citation is pinned, not a bare year mention."""
    doc = _qlike_doc()

    assert _normalize(FULL_PIN) in doc, (
        "qlike_loss docstring lacks the full Hansen-Lunde (2005) pin "
        "(journal, volume(issue), pages, DOI)"
    )
    assert "journal of applied econometrics" in doc
    assert "20(7)" in doc
    assert "873-889" in doc
    assert "doi:10.1002/jae.800" in doc


def test_2006_distinction_clause_present() -> None:
    """The docstring separates the 2005 QLIKE-origin paper from the 2006 ones."""
    doc = _qlike_doc()

    assert "the 2005 jae paper is the qlike-origin comparison" in doc, (
        "qlike_loss docstring must state the 2005 JAE paper is the "
        "QLIKE-origin comparison"
    )
    assert "hansen-lunde (2006) papers" in doc, (
        "qlike_loss docstring must distinguish the 2005 pin from the "
        "Hansen-Lunde (2006) papers"
    )


def test_no_bare_hansen_lunde_2005_mention() -> None:
    """A hyphenated bare-year mention cannot replace the full pin."""
    doc = _qlike_doc()

    assert "hansen-lunde 2005" not in doc, (
        "the bare 'Hansen-Lunde 2005' mention must carry the full pin "
        "instead"
    )

    # Author-year references outside the pin must use the ampersand form the
    # pin establishes, not the bare hyphenated one.
    assert "hansen-lunde (2005)" not in doc.replace(_normalize(FULL_PIN), "")


def test_qlike_numbers_unchanged() -> None:
    """The citation change is docstring-only; the loss values still match."""
    assert training.qlike_loss(
        np.array([-23.025851]), np.array([0.0])
    ) == pytest.approx(22.025851, rel=1e-9)
    assert training.qlike_loss(
        np.array([23.025851]), np.array([0.0])
    ) == pytest.approx(9999999976.0, rel=1e-6)


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
