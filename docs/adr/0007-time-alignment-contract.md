# ADR 0007: Time-alignment contract

- **Status:** accepted
- **Date:** 2026-10-17
- **Scope:** which bar each feature and the target are anchored to

## Context

A daily-bar model decides at the close of day t, so every input must be
computable from information available at that close — anything that peeks at
day t+1 or later is leakage, and it is the quiet kind: metrics improve and
nothing raises. The default discipline is a one-day shift, but a blanket
shift would mis-anchor inputs whose definition is inherently about the
current bar (a same-bar return divided by nothing from the future is not
leakage). The pipeline needs the default stated, the exceptions enumerated
exactly, and the target's forward look called out as the single deliberate
deviation — otherwise each new feature family re-litigates what "aligned"
means.

## Decision

Most price-derived features are shifted one day and never see their own
bar. The exceptions — model inputs whose numerator is the current bar,
anchored to the close of day t rather than the previous close — are exactly
four: `simple_return`, `price_vs_sma_{10,20,50}`, `zscore_price_vs_sma_20`,
and `zscore_volume_20`. For those inputs the documented contract is the
close of t: they divide a current-bar value by a lagged statistic, so they
are anchored to the bar being predicted, not to the previous close. The
target is the sole forward-looker: it aggregates the next five trading days
via `shift(-5)` on purpose. Any new feature family must declare its anchor
bar against this contract.

## Alternatives considered

- **Blanket one-day shift for everything:** shift all inputs with no
  exceptions. Rejected — it would mis-anchor the four same-bar inputs,
  turning a current-bar normalization (a value divided by its own lagged
  statistic) into a stale-bar one and degrading features whose definition
  is meaningful only at the close of t.
- **No shift anywhere:** compute every feature from its own bar. Rejected —
  most price-derived features would then see the bar being predicted,
  which is textbook leakage: in-sample metrics would look better while the
  deployed decision, made before that bar's information exists, could not
  reproduce them.
- **Ad-hoc per-feature choice:** let each feature module decide its own
  alignment without a stated contract. Rejected — alignment scattered
  across modules with no canonical list is unauditable; the enumerated
  exception list exists precisely so a reviewer can check four names
  instead of re-deriving every feature's timing.

## Consequences

Alignment is auditable from one record: default shift, four named
exceptions, one forward-looking target. The cost is maintenance discipline —
every added input must be classified as shifted or same-bar at authoring
time, and the exception list must stay exact rather than drifting into
family shorthand.

- **Revisit trigger:** any new feature family must declare its anchor bar
  when added; revisit this record if that discipline erodes — if
  undeclared alignments appear or the exception list stops matching the
  model-input list, the contract has decayed and needs re-cutting.

## Evidence

- `AGENTS.md:37` (time-alignment invariant: one-day shift default; sole
  enumerated same-bar exceptions `simple_return`,
  `price_vs_sma_{10,20,50}`, `zscore_price_vs_sma_20`, `zscore_volume_20`;
  target uses `shift(-5)`).
- `README.md:87` (validation-section prose: most price-derived features
  shifted one day; same-bar inputs carry the close of t; target is the
  single deliberate forward-looking exception).
- `packages/ml/model/features/feature_engineering.py:31-33`
  (`price_vs_sma_{10,20,50}` in the model-input list), `:47`
  (`simple_return`), `:52-53` (`zscore_price_vs_sma_20`,
  `zscore_volume_20`) — the four exceptions are all pinned model inputs.
- `packages/ml/dataset/features/target.py:38`
  (`shift(-TARGET_WINDOW_DAYS)`, the target's forward look).
