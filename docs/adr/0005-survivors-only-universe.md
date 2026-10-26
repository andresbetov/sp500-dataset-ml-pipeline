# ADR 0005: Survivors-only universe

- **Status:** accepted
- **Date:** 2026-10-17
- **Scope:** what the dataset panel covers and how that coverage is guarded and disclosed

## Context

The source carries current and recent S&P 500 constituents only — companies
that were delisted are absent. A panel drawn from it therefore describes the
winners that stayed listed, not the index as it was constituted at each point
in time. Long-horizon return statistics computed on such a panel are biased
upward, and any consumer that mistakes the frame for a point-in-time index
reconstruction will overstate what a tradable strategy could have earned.
The pipeline needs the frame it actually ships to be labelled, guarded, and
disclosed, not silently assumed.

## Decision

The shipped panel is explicitly a survivors-only frame: current and recent
constituents, delisted names absent. Three mechanisms make that explicit:

1. A monotonic cross-section guard. Every name in a survivors-only universe
   is still listed at the end of the panel, so distinct tickers per calendar
   year can only grow. `prepare_dataframe` runs that check and fails fast
   with a per-year summary if the count ever declines — which is how a frame
   containing exits would present itself.
2. An explicit waiver. A membership-aware frame must opt out by name via
   `prepare_dataframe(..., survivors_only=False)`; the waiver logs a warning
   and skips the guard, so no frame is ever treated as survivors-only by
   accident.
3. Disclosure in the artifact. The regime characterization JSON reserves a
   `_frame` key recording the source-universe label, the `survivors_only`
   flag, the bias note, and both row counts, so the shipped bytes describe
   their own coverage.

The disclosed bias magnitude is roughly 1.4–3.9 percentage points per year
(CAGR) on long-horizon return statistics. That figure is the estimate stated
in the pipeline's own disclosure, not a value recomputed from the shipped
data.

## Alternatives considered

- **Point-in-time membership reconstruction:** rebuild the historical
  constituent list at each date so entries and exits are represented
  faithfully. Rejected — the source carries no historical membership
  information to reconstruct from; the panel cannot represent what its
  source never contained.
- **Delisting-inclusive source:** switch to a vendor feed that carries
  delisted names with their exit prices. Rejected for now — no such source
  is wired into the pipeline, and adopting one would change the download,
  cleaning, and guard logic end to end. This is the natural upgrade path
  (see revisit trigger), not a change to make speculatively.
- **Silent acceptance:** ship the panel without the guard, waiver, or
  `_frame` disclosure. Rejected — the bias is material at multi-year
  horizons, and an undisclosed survivors-only frame invites exactly the
  backtest-overstatement error the guard exists to prevent.

## Consequences

Consumers can trust the frame label: a shipped dataset either passed the
monotonicity guard or was explicitly waived, and the JSON says which. The
cost is that the guard is a blunt instrument — it detects exits only through
a declining yearly cross-section, so a frame with subtly wrong membership
but a non-declining count would pass. Long-horizon return levels remain
biased upward by the disclosed amount regardless of the guard.

- **Revisit trigger:** a membership-aware or delisting-inclusive source
  becomes available — at that point the guard, the waiver default, and the
  `_frame` disclosure all need re-examination.

## Evidence

- `packages/ml/dataset/cleaning.py:307` (`_check_survivor_cross_section`:
  monotonic per-year ticker-count guard, raises with per-year summary),
  `:314-319` (waiver path: `survivors_only=False` logs a warning and
  returns), `:341` (`prepare_dataframe(df, survivors_only: bool = True)`
  signature), `:351` (guard invocation).
- `packages/ml/dataset/artifacts.py:14-18`
  (`SURVIVORS_ONLY_FRAME_LABEL`, `SURVIVORSHIP_BIAS_NOTE` stating the
  1.4–3.9 percentage-points-per-year CAGR figure), `:42-48` (reserved
  `_frame` key in `regime_characterization.json`: label, `survivors_only`
  flag, bias note, row counts).
- `README.md:110` (scope bullet: survivors-only coverage, bias magnitude,
  monotonic cross-section, fail-fast behavior, explicit waiver, `_frame`
  disclosure).
