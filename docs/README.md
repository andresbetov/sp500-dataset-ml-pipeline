# Documentation

This set covers how the pipeline is built, run, and read. It separates current-state reference (how the code and artifacts behave today) from the dated evaluation report (what one 2026-10-20 run measured).

## Where to start

| You are | Start here |
|---|---|
| New user | [`README.md`](../README.md) — what it does and how to run it |
| Operator | [`operations.md`](operations.md) — run lifecycle, resume, retention, holdout gate |
| Contributor or agent | [`AGENTS.md`](../AGENTS.md) — rules and commands |
| Researcher or auditor | [`model-card.md`](model-card.md), [`results.md`](results.md), and the [report](reports/model-performance-and-rigor.md) |
| Architecture history | [`adr/`](adr/) — decision records |

## Current-state docs

| Document | Purpose |
|---|---|
| [`README.md`](../README.md) | Front door: identity, quickstart, pipeline summary, results pointer |
| [`model-card.md`](model-card.md) | Intended use, data, protocol, and standing caveats |
| [`operations.md`](operations.md) | Canonical runbook: run state, resume, retention, artifacts, holdout |
| [`results.md`](results.md) | Reading guide for the generated result artifacts |
| [`reference/cli.md`](reference/cli.md) | Canonical command and flag reference |
| [`reference/dataset-schema.md`](reference/dataset-schema.md) | 53-column dataset dictionary |
| [`pipelines/dataset.md`](pipelines/dataset.md) | Phase 1 deep reference |
| [`pipelines/model.md`](pipelines/model.md) | Phases 2-5 protocol, folds, training, evaluation |
| [`pandas-trap-triage.md`](pandas-trap-triage.md) | Known pandas/scikit-learn edge cases and their decisions |

The dated evaluation report is [`reports/model-performance-and-rigor.md`](reports/model-performance-and-rigor.md).

The legacy path [`model-artifacts.md`](model-artifacts.md) now redirects to the operations runbook; it is kept only so old links do not break.

## Provenance

The current-state documents in this directory were verified against commit `61eae9e` on 2026-10-25. The report is a dated record of the 2026-10-20 evaluation and does not change with later runs.

ADR evidence that cites line numbers is historical: those line numbers refer to the tree at each ADR's date, not the current tree.
