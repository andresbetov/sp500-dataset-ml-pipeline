# ADR 0002: Entry points stay plain scripts with normalized imports

- **Status:** accepted
- **Date:** 2026-10-17
- **Scope:** both phase entry points (`packages/ml/dataset/main.py`, `packages/ml/model/main.py`)

## Context

The pipeline has two entry points run by path from the repo root, taking no
arguments and reading no environment variables. There is no build system and
no console scripts. The dataset phase uses package-relative imports, which
break when its `main.py` is run by path unless the module normalizes its own
package identity first (PEP 366); the model phase uses flat sibling imports
and needs no normalization.

## Decision

Entry points stay plain scripts run by path — no build system, no console
scripts. The dataset phase normalizes its package identity on direct
execution (`packages/ml/dataset/main.py:7-12`) so its package-relative
imports resolve from any cwd; the model phase is unchanged. Entry calls stay
behind `if __name__ == "__main__"`.

## Alternatives considered

- **Console scripts:** needs a `[project.scripts]` section and a build
  system. Rejected — `pyproject.toml` has neither (18 lines, `[project]`
  dependencies only), and adding a build backend to serve two scripts is
  machinery without a consumer.
- **src-layout + editable install:** makes imports work by installing the
  packages. Rejected — it replaces the documented `python path/to/main.py`
  command with an install step and breaks the cwd-independent scripts users
  already run.
- **Flat imports everywhere:** rewrite the dataset phase to import siblings
  like the model phase. Rejected — it keeps the failure this layout fixes:
  run by path, package-relative imports in a multi-module phase cannot
  resolve without either normalization or an install.
- **`python -m` entry:** run as `python -m packages.ml.dataset` from the
  root. Rejected — it changes the documented command and only works from one
  cwd, while the scripts work from the root or their own directory.

## Consequences

Two commands, zero install, and each phase keeps the import style that fits
its size. The cost is asymmetry: contributors must know the dataset phase
normalizes PEP 366 identity while the model phase does not — which is why
`AGENTS.md` hard rule 1 exists.

- **Revisit trigger:** either phase becomes an installed or imported library,
  or a third phase appears — at that point a shared entry-point convention
  (or a real package) earns its keep.

## Evidence

- `README.md:38-52` (`## Running the pipeline`): no single end-to-end entry
  point, the two by-path commands, no-args/no-env, cwd-independence.
- `AGENTS.md:24` (hard rule 1): PEP 366 normalization in dataset, flat
  sibling imports in model, `__main__` guard.
- `pyproject.toml:1-18`: `[project]` only — no `[project.scripts]`, no
  `[build-system]`, no console entry points.
- `packages/ml/dataset/main.py:7-12`: the PEP 366 `__package__`
  normalization block.

## Supersession note — 2026-10-18

The model phase now also normalizes its package identity on direct execution
(`packages/ml/model/main.py`, PEP 366) and uses package-absolute imports
throughout; the dataset entry keeps package-relative imports. The model phase
also documents a `python -m packages.ml.model.main` entry form. The rest of the
decision above — plain script entry points, no build system, no console
scripts, `__main__` guard — stands; the context and alternatives that describe
the model phase as flat-imported are historical.
