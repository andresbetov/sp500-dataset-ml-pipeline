# ADR 0003: Phases couple exclusively through on-disk artifacts

- **Status:** accepted
- **Date:** 2026-10-17
- **Scope:** the Phase 1 → Phases 2-5 boundary

## Context

One script builds the dataset (needs network + Kaggle credentials); another
trains and evaluates (fully local). They run at different times, on possibly
different machines, with different requirements — so the boundary between
them has to survive all of that.

## Decision

The two phases are coupled exclusively through on-disk artifacts. Phase 1
writes `data/processed/dataset.parquet` (plus the ingestion manifest and the
regime characterization); Phase 2 reads the parquet and fails immediately if
it is missing. All generated output is gitignored and never committed, so a
fresh clone always starts with Phase 1. One known odd corner is documented,
not fixed: Phase 1 writes `regime_characterization.json` into the model
package's tree (`packages/ml/model/artifacts/results/`), so relocating model
artifacts requires editing the dataset phase.

## Alternatives considered

- **Single end-to-end script:** one command does everything. Rejected — it
  forces credentials and network on every local retrain and collapses two
  different failure modes (download vs. training) into one log.
- **In-process import:** the model phase imports the dataset phase and calls
  it. Rejected — same coupling as a single script with extra steps, and it
  drags the Kaggle/network dependency into the local-only phase.
- **Committing artifacts (or LFS):** ship a canned parquet so Phase 1 is
  optional. Rejected — the frame grows with every refresh, byte-identical
  regeneration is a local-only property, and checked-in data goes stale
  silently while looking authoritative.

## Consequences

Each phase runs, fails, and is reasoned about independently; rerunning
training never re-downloads anything. The cost is the fail-fast cliff: no
parquet, no training — and the odd corner above means the "model artifacts"
directory is not owned by exactly one phase.

- **Revisit trigger:** a second consumer of the artifacts appears, or an
  orchestration need (scheduling, lineage, partial reruns) shows up — either
  turns a file-path contract into something that wants a manifest or a
  runner.

## Evidence

- `README.md:40` (no single end-to-end entry point), `README.md:54` (Phase 2
  fails fast without Phase 1), `README.md:68-79` (Outputs table + gitignored
  note).
- `AGENTS.md:25` (hard rule 2: Phase 1 before Phases 2-5),
  `AGENTS.md:26` (hard rule 3: never commit generated output, artifact paths
  listed), `AGENTS.md:39` (the odd-corner invariant: Phase 1 writes into
  `packages/ml/model/artifacts/results/`).
