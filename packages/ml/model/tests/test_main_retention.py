"""Retention contract for the Phase-2 arrays in the pipeline entry point.

Phase 4 reloads its inputs from disk, so ``main`` must release the Phase-2
frame copies once they have been logged instead of holding them across
cross-validation, training, and visualization.

Run with pytest from the repository root:

    uv run --frozen python -m pytest packages/ml/model/tests -q

This file also runs standalone:

    .venv/bin/python -B packages/ml/model/tests/test_main_retention.py
"""

from __future__ import annotations

import ast
import re
import traceback
from pathlib import Path

from conftest import REPO_ROOT

MAIN_PATH = REPO_ROOT / "packages" / "ml" / "model" / "main.py"
RELEASED_NAMES = ("X", "y", "metadata", "encoder")
SOURCE_CHECKED_NAMES = ("X", "y", "metadata")


def _parse_main() -> tuple[str, ast.Module, ast.FunctionDef]:
    source = MAIN_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(MAIN_PATH))
    main_function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    return source, module, main_function


def _release_statement(main_function: ast.FunctionDef) -> ast.Delete:
    statements = [
        node for node in ast.walk(main_function) if isinstance(node, ast.Delete)
    ]
    assert len(statements) == 1, (
        f"expected one del statement in main, found {len(statements)}"
    )
    return statements[0]


def _only_call_line(main_function: ast.FunctionDef, predicate) -> ast.Call:
    calls = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call) and predicate(node)
    ]
    assert len(calls) == 1, f"expected one matching call, found {len(calls)}"
    return calls[0]


def _phase_2_unpack(main_function: ast.FunctionDef) -> ast.Assign:
    unpack = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "phase_2_feature_selection"
    ]
    assert len(unpack) == 1, "expected one phase_2_feature_selection unpack"
    assignment = unpack[0]
    assert len(assignment.targets) == 1
    target = assignment.targets[0]
    assert isinstance(target, ast.Tuple)
    names = tuple(
        element.id for element in target.elts if isinstance(element, ast.Name)
    )
    assert names == RELEASED_NAMES, (
        f"phase_2_feature_selection must bind {RELEASED_NAMES}, got {names}"
    )
    return assignment


def test_release_statement_targets_phase_2_arrays() -> None:
    _, _, main_function = _parse_main()
    assignment = _phase_2_unpack(main_function)
    statement = _release_statement(main_function)

    targets = [
        target.id for target in statement.targets if isinstance(target, ast.Name)
    ]
    assert len(targets) == len(statement.targets), "del targets must be plain names"
    assert tuple(targets) == RELEASED_NAMES, (
        f"del must release {RELEASED_NAMES}, got {tuple(targets)}"
    )
    assert statement.lineno > assignment.lineno, (
        "del must follow the phase_2_feature_selection unpack"
    )


def test_release_is_between_phase_2_log_and_training() -> None:
    _, _, main_function = _parse_main()
    statement = _release_statement(main_function)

    phase_2_log = _only_call_line(
        main_function,
        lambda call: isinstance(call.func, ast.Attribute)
        and call.func.attr == "info"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "logger"
        and bool(call.args)
        and any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.lstrip().startswith("Phase 2 complete")
            for node in ast.walk(call.args[0])
        ),
    )
    phase_4_call = _only_call_line(
        main_function,
        lambda call: isinstance(call.func, ast.Name)
        and call.func.id == "phase_4_training",
    )

    assert statement.lineno > phase_2_log.end_lineno, (
        "del must follow the Phase-2 completion log"
    )
    assert statement.end_lineno < phase_4_call.lineno, (
        "del must precede the phase_4_training call"
    )


def test_no_released_name_is_loaded_after_the_del() -> None:
    _, _, main_function = _parse_main()
    statement = _release_statement(main_function)

    retained = [
        (node.lineno, node.id)
        for node in ast.walk(main_function)
        if isinstance(node, ast.Name)
        and node.id in RELEASED_NAMES
        and isinstance(node.ctx, ast.Load)
        and node.lineno > statement.end_lineno
    ]
    assert not retained, (
        "released names are still loaded after the del: "
        + ", ".join(f"{name} at line {line}" for line, name in retained)
    )


def test_no_shadow_copy_outside_the_del() -> None:
    source, module, main_function = _parse_main()
    statement = _release_statement(main_function)

    outside_loads = [
        (node.lineno, node.id)
        for node in module.body
        if node is not main_function
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and child.id in SOURCE_CHECKED_NAMES
        and isinstance(child.ctx, ast.Load)
        and child.lineno > statement.end_lineno
    ]
    assert not outside_loads, (
        "released names are loaded outside main after the del: "
        + ", ".join(f"{name} at line {line}" for line, name in outside_loads)
    )

    after_release = "\n".join(source.splitlines()[statement.end_lineno :])
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(name) for name in SOURCE_CHECKED_NAMES) + r")\b"
    )
    source_hits = pattern.findall(after_release)
    assert not source_hits, (
        "released names still appear in source after the del: "
        + ", ".join(sorted(set(source_hits)))
    )


if __name__ == "__main__":
    failures = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All tests passed")
