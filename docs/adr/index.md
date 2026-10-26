# Architecture Decision Records

Decisions that shape the whole repo live here as Nygard-style one-pagers.
Status is `accepted` for decisions already lived; numbers are never reused;
accepted content is frozen — supersede, don't rewrite. Copy
[template.md](template.md) for a new record.

| Number | Title | Status | Date |
|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | accepted | 2026-10-17 |
| [0002](0002-entry-points-scripts-with-normalized-imports.md) | Entry points stay plain scripts with normalized imports | accepted; partially superseded 2026-10-18 | 2026-10-17 |
| [0003](0003-phase-coupling-via-artifacts.md) | Phases couple exclusively through on-disk artifacts | accepted | 2026-10-17 |
| [0004](0004-verification-without-a-test-suite.md) | Verification without a test suite | accepted; partially superseded 2026-10-18 | 2026-10-17 |
| [0005](0005-survivors-only-universe.md) | Survivors-only universe | accepted | 2026-10-17 |
| [0006](0006-yang-zhang-5day-target.md) | Yang–Zhang 5-day volatility target | accepted | 2026-10-17 |
| [0007](0007-time-alignment-contract.md) | Time-alignment contract | accepted | 2026-10-17 |
| [0008](0008-market-regime-calendar-label.md) | Market regime as a hardcoded calendar label | accepted | 2026-10-17 |

Evidence line numbers in accepted ADRs refer to the tree at the ADR date. Current-state references live in the docs listed in [`../README.md`](../README.md); accepted decision text is not rewritten.
