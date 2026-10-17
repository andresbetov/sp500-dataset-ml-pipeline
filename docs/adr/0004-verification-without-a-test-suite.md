# ADR 0004: Verification without a test suite

- **Status:** accepted
- **Date:** 2026-10-17
- **Scope:** how changes are verified repo-wide

## Context

There is no build system and no test, lint, or typecheck configuration — no
pytest, ruff, mypy, or make. The pipeline's correctness lives in run logs and
in artifacts (parquet bytes, per-fold metrics), not in assertions. Any
verification story has to work with that, and has to catch the failure mode
that actually bites here: import breakage from moving modules between
packages, which no static check in the current toolchain would flag.

## Decision

Verification is running the phase you touched and quoting the log output you
saw (`AGENTS.md` verification workflow). For structural refactors that move
modules without changing behavior, the acceptance test is stronger: a
golden-master byte comparison — baseline SHA-256 hashes of the artifacts plus
a normalized-log diff (timestamps and logger names stripped) that must show
at most the single intentional log-line change, with all hashes `OK`.
Anything else is a failure: stop and investigate.

## Alternatives considered

- **A smoke test now:** add a minimal test file for the refactor. Rejected
  (deferred, not dismissed) — a one-off test written to bless its own change
  proves little, and inventing a suite mid-refactor without a runner anyone
  agreed on is how dead test files are born.
- **Golden-file parquet test as the permanent story:** check in expected
  hashes and diff against them in CI. Rejected for now — it is the natural
  next investment (see revisit trigger), but it needs a home for baselines
  and a runner first; until then, the one-off golden-master comparison above
  covers behavior-preserving moves without pretending that infrastructure
  exists.
- **CI-only static checks (`compileall` and friends):** cheap and
  automatable. Rejected as sufficient — byte-compiling every module cannot
  catch the import breakage this layout risks; only executing the phase does.

## Consequences

"No evidence, no done" stays enforceable with zero infrastructure: every
change ships with the log lines that prove it works. The cost is that
verification is manual and only as disciplined as the contributor — silent
regressions have no tripwire.

- **Revisit trigger:** the first silent regression, or a second contributor —
  either means the honor system stopped scaling, and the golden-file parquet
  test is where to start.

## Evidence

- `AGENTS.md:9` (Environment: no build system, no test/lint/typecheck
  config), `AGENTS.md:28` (hard rule 5: no evidence, no "done"),
  `AGENTS.md:56-64` (verification workflow: per-phase run commands, expected
  log lines, README update rule).
- `README.md:115` (no automated test or lint suite),
  `README.md:116` (a fresh clone needs Phase 1 first).
- The golden-master byte comparison for behavior-preserving refactors —
  baseline SHA-256 hashes of the dataset parquet, the ingestion manifest,
  and the regime characterization, plus a normalized-log diff expected to
  show exactly one changed line (the intentional rename).
