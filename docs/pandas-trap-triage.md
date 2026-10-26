# Pandas-trap triage

The diagnostics and reporting code paths carry a small set of known pandas and
scikit-learn edge cases. This file records the decision for each; a deferred
case is not a fixed case.

| Site | Trap | Decision |
| --- | --- | --- |
| `visualization/training_diagnostics.py` — `plot_error_by_volatility_bin` | `pd.qcut(..., duplicates="drop")` collapses duplicate quantile edges. A constant target realizes zero bins, and a duplicate-heavy target can realize fewer than ten. | **Warn-only.** When fewer than ten bins are realized, the fold and the realized count are logged and the available bins still render. Live folds realize 10 of 10 and do not warn. |
| `training/training.py` — dual OOS-R² | A constant `test_true` makes `sklearn.metrics.r2_score` return `1.0` when the forecasts match and `0.0` otherwise; it never raises and never returns `NaN`. | **Documented convention**, no code change. The prevailing-mean SSE form states the same rule; the vs-HAR ratio still raises when the HAR squared error is zero. |
| `visualization/temporal_analysis.py` — fold range plot | `mdates.YearLocator()` only positions year ticks; it changes no date, fold boundary, or plotted value. | **Intended, no-fix.** |
| `training/har_benchmark.py` — trailing components | `rolling(window, min_periods=1)` uses the observations available at a ticker's series start. The warm-up mask excludes those rows from the fit. | **Intended, no-fix.** |
| `visualization/metrics_visualizer.py` — metrics heatmap | Per-column min-max normalization has no constant-column guard: a constant metric yields `0/0` (`NaN`), while the comparison plots use a `ones_like` tie value. The two tie conventions are inconsistent. | **Deferred** to the visualization normalization work; not fixed here. |
