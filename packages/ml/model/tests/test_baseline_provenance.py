"""Tests for the legacy baseline artifact's provenance README and hash manifest.

``packages/ml/model/artifacts/baseline_v1/`` holds the pre-remediation baseline
snapshot (a legacy ``dataset.parquet`` plus ``regime_characterization.json``)
and is gitignored, so it is absent from a fresh clone. The directory is kept in
place for provenance: ``README.md`` states the disposition and retention rule,
``manifest.json`` records every file's repo-relative path, byte size, SHA-256,
and created timestamp, and the tracked one-liner ``docs/model-artifacts.md``
names the provenance and the explicit-override requirement. No deletion code
path exists; if the override is ever exercised, the directory goes away only
with a strict-JSON ``artifact_removal`` record appended to ``runs.jsonl`` and
the tracked reference updated. That removal contract is exercised against a
temporary fixture root here -- the live directory is never deleted or rewritten.

The live provenance checks skip when the gitignored tree is absent (a fresh
clone or CI); the tracked-reference check runs everywhere.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_baseline_provenance.py
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pytest

from conftest import REPO_ROOT

BASELINE_REL = "packages/ml/model/artifacts/baseline_v1"
BASELINE_DIR = REPO_ROOT / BASELINE_REL
DOCS_REL = "docs/model-artifacts.md"
DOCS_PATH = REPO_ROOT / DOCS_REL
RUNS_REL = "packages/ml/model/artifacts/runs.jsonl"
README_NAME = "README.md"
MANIFEST_NAME = "manifest.json"
REMOVAL_PHASE = "artifact_removal"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Whitespace-normalized phrases the provenance files must carry: the schema
# divergence numbers and the retention / explicit-override rule.
REQUIRED_NOTES = (
    "2,668,192 rows",
    "39 columns",
    "2,669,959 rows",
    "53 columns",
    "retention",
    "explicit",
    "override",
    "runs.jsonl",
)


def _reject_constant(token: str) -> None:
    raise AssertionError(f"non-strict JSON constant emitted: {token}")


def _load_strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _require_baseline_dir() -> None:
    """Skip the live provenance checks when the gitignored tree is absent.

    A fresh clone legitimately lacks the artifact tree while the tracked
    reference still names it, so absence is a skip, not a failure; the removal
    contract itself is pinned by the temporary-fixture checks below.
    """
    if not BASELINE_DIR.is_dir():
        pytest.skip(
            "baseline artifact tree absent from this checkout (gitignored): "
            + BASELINE_REL
        )


def _directory_violations(
    directory: Path, manifest: Mapping[str, Any], project_root: Path
) -> list[str]:
    """Coverage, size, and sha256 violations of a manifest for a directory.

    Every file under ``directory`` except the manifest itself must be listed
    with a repo-relative path, byte size, and SHA-256; every listed file must
    exist and match its recorded size and hash.
    """
    violations: list[str] = []
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return ["manifest has no files list"]

    recorded: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            violations.append(f"malformed manifest entry: {entry!r}")
            continue
        stored = Path(entry["path"])
        if stored.is_absolute():
            violations.append(f"entry path is not repo-relative: {entry['path']!r}")
            continue
        recorded[stored.as_posix()] = entry

    on_disk = {
        path.relative_to(project_root).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    }
    for rel in sorted(on_disk - set(recorded)):
        violations.append(f"{rel}: on disk but not in the manifest")
    for rel in sorted(set(recorded) - on_disk):
        violations.append(f"{rel}: in the manifest but not on disk")
    for rel in sorted(on_disk & set(recorded)):
        entry = recorded[rel]
        path = project_root / rel
        size = path.stat().st_size
        if entry.get("size") != size:
            violations.append(f"{rel}: size {entry.get('size')!r} != {size}")
        digest = _sha256_file(path)
        if entry.get("sha256") != digest:
            violations.append(f"{rel}: sha256 {entry.get('sha256')!r} != {digest}")
    return violations


def _removal_record_present(runs_path: Path) -> bool:
    if not runs_path.is_file():
        return False
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except (AssertionError, json.JSONDecodeError):
            continue
        if (
            isinstance(record, dict)
            and record.get("phase") == REMOVAL_PHASE
            and record.get("path") == BASELINE_REL
        ):
            return True
    return False


def _absence_violations(root: Path) -> list[str]:
    """The post-removal contract for a fixture root with no baseline directory."""
    violations: list[str] = []
    if (root / BASELINE_REL).exists():
        return [f"{BASELINE_REL}: still present"]
    docs = root / DOCS_REL
    if docs.is_file() and "baseline_v1" in docs.read_text(encoding="utf-8"):
        violations.append(f"{DOCS_REL}: still references the removed baseline")
    if not _removal_record_present(root / RUNS_REL):
        violations.append(f"{RUNS_REL}: no {REMOVAL_PHASE} record for {BASELINE_REL}")
    return violations


def _write_absence_fixture(
    root: Path, *, keep_reference: bool, write_removal: bool
) -> None:
    docs = root / DOCS_REL
    docs.parent.mkdir(parents=True, exist_ok=True)
    if keep_reference:
        line = f"`{BASELINE_REL}/` is retained with a provenance README."
    else:
        line = "No baseline artifact snapshot is retained."
    docs.write_text(line + "\n", encoding="utf-8")

    if write_removal:
        runs_path = root / RUNS_REL
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "phase": REMOVAL_PHASE,
            "path": BASELINE_REL,
            "override": True,
            "reason": "explicit owner override",
            "removed_utc": "2026-10-18T00:00:00Z",
        }
        with open(runs_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def test_provenance_files_exist_and_manifest_covers_every_file() -> None:
    _require_baseline_dir()
    readme_path = BASELINE_DIR / README_NAME
    manifest_path = BASELINE_DIR / MANIFEST_NAME
    assert readme_path.is_file(), f"missing provenance README: {readme_path}"
    assert manifest_path.is_file(), f"missing provenance manifest: {manifest_path}"

    manifest = _load_strict_json(manifest_path)
    assert isinstance(manifest, dict), "manifest is not a JSON object"
    assert manifest.get("directory") == BASELINE_REL

    violations = _directory_violations(BASELINE_DIR, manifest, REPO_ROOT)
    assert not violations, "; ".join(violations)

    for entry in manifest["files"]:
        assert isinstance(entry["size"], int) and entry["size"] > 0, entry
        assert SHA256_RE.fullmatch(entry["sha256"]), entry
        created = entry.get("created_utc")
        assert created, entry
        datetime.fromisoformat(created.replace("Z", "+00:00"))


def test_recomputed_hashes_match_and_a_flipped_byte_is_detected() -> None:
    _require_baseline_dir()
    manifest = _load_strict_json(BASELINE_DIR / MANIFEST_NAME)
    entries = manifest["files"]

    for entry in entries:
        path = REPO_ROOT / entry["path"]
        assert _sha256_file(path) == entry["sha256"], entry["path"]

    # The verifier has teeth: the smallest recorded file copied intact passes,
    # and one flipped byte in the copy is reported (tamper signal).
    smallest = min(entries, key=lambda entry: entry["size"])
    source = REPO_ROOT / smallest["path"]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        copy = root / smallest["path"]
        copy.parent.mkdir(parents=True)
        raw = bytearray(source.read_bytes())
        copy.write_bytes(bytes(raw))
        fixture_manifest = {"directory": BASELINE_REL, "files": [dict(smallest)]}

        assert _directory_violations(copy.parent, fixture_manifest, root) == []

        raw[len(raw) // 2] ^= 0x01
        copy.write_bytes(bytes(raw))
        violations = _directory_violations(copy.parent, fixture_manifest, root)
        assert any("sha256" in violation for violation in violations), violations
        assert any(
            smallest["path"] in violation for violation in violations
        ), violations


def test_retention_and_divergence_notes_are_present() -> None:
    _require_baseline_dir()
    readme_text = _normalized(
        (BASELINE_DIR / README_NAME).read_text(encoding="utf-8")
    ).lower()
    manifest_text = _normalized(
        (BASELINE_DIR / MANIFEST_NAME).read_text(encoding="utf-8")
    ).lower()

    for note in REQUIRED_NOTES:
        assert note.lower() in readme_text, f"README is missing the note: {note!r}"
        assert note.lower() in manifest_text, f"manifest is missing the note: {note!r}"

    manifest = _load_strict_json(BASELINE_DIR / MANIFEST_NAME)
    divergence = manifest.get("divergence")
    assert isinstance(divergence, dict), divergence
    assert divergence.get("legacy") == {"rows": 2668192, "columns": 39}
    assert divergence.get("current") == {"rows": 2669959, "columns": 53}


def test_tracked_reference_names_the_provenance_and_the_override() -> None:
    assert DOCS_PATH.is_file(), f"missing tracked reference: {DOCS_PATH}"
    text = _normalized(DOCS_PATH.read_text(encoding="utf-8")).lower()
    assert "baseline_v1" in text, text
    assert "readme" in text and "manifest" in text, text
    assert "override" in text, text


def test_absence_contract_holds_for_a_removed_directory_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_absence_fixture(root, keep_reference=False, write_removal=True)
        assert _absence_violations(root) == []


def test_absence_contract_flags_a_stale_tracked_reference() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_absence_fixture(root, keep_reference=True, write_removal=True)
        violations = _absence_violations(root)
        assert any("references" in violation for violation in violations), violations


def test_absence_contract_flags_a_missing_removal_record() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_absence_fixture(root, keep_reference=False, write_removal=False)
        violations = _absence_violations(root)
        assert any(REMOVAL_PHASE in violation for violation in violations), violations


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
