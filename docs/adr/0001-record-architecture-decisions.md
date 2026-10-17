# ADR 0001: Record architecture decisions

- **Status:** accepted
- **Date:** 2026-10-17
- **Scope:** repo-wide rationale — where cross-cutting decisions are written down

## Context

The repo's load-bearing decisions — entry-point layout, phase coupling,
verification without a suite — were scattered across code comments,
`AGENTS.md` rules, and commit history. `AGENTS.md` states what to do, not why
the alternatives were rejected, so each new change risked relitigating a
settled trade-off.

## Decision

Keep architecture decision records in `docs/adr/` as Nygard-style one-pagers
(Status / Date / Scope / Context / Decision / Alternatives considered /
Consequences with a revisit trigger / Evidence). Records are numbered
sequentially starting at 0001; numbers are never reused; once accepted, a
record's content is frozen — a changed mind gets a new superseding record,
not an edit.

## Alternatives considered

- **No ADRs (status quo):** rationale stays implicit. Rejected — the rules in
  `AGENTS.md` already needed "why" footnotes, and commit messages bury them.
- **Wiki or issue tracker:** rationale lives off-repo. Rejected — nothing in
  this repo depends on external services (Kaggle credentials aside), and an
  off-repo record rots the moment access lapses.
- **Decision docstrings in code:** rationale next to the mechanism. Rejected
  for repo-wide decisions — entry points and phase coupling span packages,
  and no single module owns them.

## Consequences

Cross-cutting rationale has one home, and `AGENTS.md` can stay a rulebook
instead of growing justifications. The cost is upkeep: every repo-wide change
should check whether a record needs superseding.

- **Revisit trigger:** more than ~15 records, or a second contributor —
  either means the flat index and manual numbering need tooling or a
  lighter-weight convention.

## Evidence

The scaffold itself: `docs/adr/` did not exist before this record, and no
numbering collides with anything in the tree.
