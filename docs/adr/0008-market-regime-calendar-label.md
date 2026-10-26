# ADR 0008: Market regime as a hardcoded calendar label

- **Status:** accepted
- **Date:** 2026-10-17
- **Scope:** definition, limits, and role of the `market_regime` column

## Context

Volatility behavior differs across market eras — crisis years do not look
like calm ones — so per-regime analysis (how metrics vary between calm and
stressed periods) needs an era label on every row. Deriving regimes from
the data itself (volatility quantiles, hidden-state models) would make the
label a function of the series being studied, entangling the grouping with
the phenomenon. A fixed calendar mapping keeps the label independent of the
data, at the price of hardcoding history: the mapping knows only the eras
its author enumerated, and data from beyond its last bucket has no label.

## Decision

`market_regime` is a hardcoded calendar label, not something derived from
the data. Seven calendar buckets span 2000–2026: 0 for 2000–2007
(pre-crisis), 1 for 2008–2010 (financial crisis), 2 for 2011–2019
(post-crisis recovery), 3 for 2020–2021 (COVID), 4 for 2022 (post-COVID
inflation), 5 for 2023–2025 (post-COVID recovery), and 6 for 2026 (terminal
year). A date past 2026 raises instead of joining the last bucket, so data
from a later source period cannot be silently mislabelled — ingesting data
past 2026 requires extending the calendar buckets first. The column ships
in the dataset as analysis metadata; it is not a model input.

## Alternatives considered

- **Data-derived regimes:** assign eras from trailing volatility quantiles
  or a fitted regime-switching model. Rejected — the label would then be a
  function of the same price series the model studies, so per-regime
  breakdowns would group rows by the phenomenon being measured rather than
  by an independent era. The calendar label stays exogenous by
  construction.
- **Rolling quantiles:** label each row by its trailing-window volatility
  rank. Rejected for the same exogeneity reason, with an added failure
  mode: the label would need its own window-length decision and warm-up
  handling, importing a second estimation problem into what should be a
  fixed grouping key.

## Consequences

Per-regime analysis gets a stable, data-independent grouping key whose
meaning is readable from the bucket definitions — no fitting, no warm-up,
no leakage from the label into the features. The costs are twofold: the
buckets are coarse historical judgments (a single label for 2011–2019
asserts those nine years belong together), and the mapping is frozen at
2026 — any ingestion past that year fails fast until the calendar is
extended, which forces a conscious relabelling decision rather than silent
drift. Excluding the column from model inputs keeps identity-by-era from
leaking a coarse time trend into training.

- **Revisit trigger:** ingesting data past 2026 — extending the calendar
  buckets is mandatory before that ingestion, and the extension should
  re-examine whether the existing bucket boundaries still carve history
  at its joints.

## Evidence

- `packages/ml/dataset/features/regimes.py:7-49` (seven buckets 0–6:
  2000–2007 through 2026 at `:27-37`; post-2026 raises at `:40-46` with
  "extend the calendar buckets before ingesting later data").
- `AGENTS.md:35` (invariant: hardcoded calendar label, seven buckets
  2000–2007 through 2026, past-2026 raises; points at
  `features/regimes.py`), `AGENTS.md:34` (`market_regime` left the model
  input list and stays as a dataset column: analysis metadata).
- `README.md:113` (scope bullet: hardcoded calendar label, seven buckets
  2000-2007 through 2026, post-2026 raises), `README.md:72` (shipped
  parquet contains `market_regime`), `README.md:65` (module listing:
  `features/regimes.py` — hardcoded calendar label).
