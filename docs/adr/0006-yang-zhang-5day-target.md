# ADR 0006: Yang–Zhang 5-day volatility target

- **Status:** accepted
- **Date:** 2026-10-17
- **Scope:** definition of the prediction target the dataset ships

## Context

Every downstream phase — feature selection, temporal validation, training,
evaluation — consumes a single label column, so the target's estimator,
horizon, and scale fix the meaning of all reported metrics. Volatility can
be estimated several defensible ways (close-to-close, range-based, intraday),
over any horizon, at daily or annualized scale, for regression or for
classification. The dataset needs one pinned definition with the horizon and
scale stated, otherwise per-fold R², QLIKE, and the HAR-RV benchmark
comparison have no shared referent.

## Decision

The target is `realized_volatility_5d`: the Yang–Zhang daily volatility
estimate over the next five trading days, stored as a per-day σ (not
annualized; multiply by sqrt(252) to annualize). Concretely, for every row
the estimator aggregates the forward window i = j+1..j+5 from three
components — overnight log returns, open-to-close log returns, and the
Rogers-Satchell drift-independent term — combining sample variances
(ddof=1) of the first two with the mean of the third under the standard
Yang–Zhang weight for n=5, and storing sqrt(max(variance, 0)). Rows without
a complete forward window — the last five rows of every ticker — are NaN and
dropped before shipping. The forward look is implemented by shifting the
aggregates by the window length, and the modeling problem is continuous
regression on this value (log-transformed downstream), not classification.

## Alternatives considered

- **Close-to-close realized volatility:** the simplest estimator, variance
  of daily close log returns over the window. Rejected — it discards the
  information in opens, highs, and lows that the overnight and
  Rogers-Satchell components capture, making it noisier at a fixed five-day
  horizon.
- **Intraday realized volatility:** sum of squared high-frequency returns.
  Rejected — the source is daily OHLCV bars, so no intraday series exists
  to build it from. This becomes the option of choice only if an intraday
  feed arrives (see revisit trigger).
- **Annualized scale:** multiply by sqrt(252) at write time so the column
  reads directly as annualized volatility. Rejected — a constant rescaling
  carries no information and fixes a convention (252 trading days) inside
  the artifact; storing per-day σ keeps the raw estimate and leaves
  annualization to the consumer.
- **Classification bins:** discretize the target into low/medium/high
  volatility regimes. Rejected — bin edges are arbitrary, ordering
  information is lost, and the downstream training path is regression
  throughout (log-scale R²/MAE/RMSE plus QLIKE), so bins would orphan every
  reported metric.

## Consequences

All downstream metrics speak the same language: five-trading-day-ahead
daily σ, continuous. The Yang–Zhang construction makes the label more
efficient than close-to-close at the same horizon, at the cost of a more
involved definition whose components (especially the overnight term's use
of the label row's own close at the horizon start) must be understood as
horizon-start inputs rather than leakage. The fixed five-day horizon also
propagates: the cross-validation purge width and the NaN tail per ticker
are both consequences of this window.

- **Revisit trigger:** a horizon or estimator revision — for example,
  intraday data arrives and justifies an intraday estimator, or a different
  forecast horizon is wanted. Changing either means re-cutting the label
  column, the purge width, and the benchmark comparison together.

## Evidence

- `packages/ml/dataset/features/target.py:17` (`TARGET_WINDOW_DAYS = 5`),
  `:19-21` (Yang–Zhang open-to-close weight for n=5), `:38`
  (`shift(-TARGET_WINDOW_DAYS)`, the forward-look mechanism), `:43-58`
  (docstring: forward window i=j+1..j+5, overnight / open-to-close /
  Rogers-Satchell components, sample variances with ddof=1,
  sqrt(max(variance, 0)), per-day σ not annualized, last five rows per
  ticker NaN), `:95` (writes `realized_volatility_5d`).
- `README.md:16` ("What it does": Yang–Zhang daily volatility over the next
  five trading days, per-day σ not annualized, continuous regression not
  classification), `README.md:64` (module listing: `features/target.py` —
  Yang-Zhang 5-day realized-volatility target).
