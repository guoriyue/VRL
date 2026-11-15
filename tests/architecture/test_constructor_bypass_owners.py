"""Constructor bypasses live in named package helpers, never inline in a test module."""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]


def _bypass_lines(source: str) -> list[int]:
    """Lines calling ``X.__new__(...)`` outside a ``__new__`` override."""

    lines: list[int] = []

    def visit(node: ast.AST, *, in_new_override: bool) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            in_new_override = node.name == "__new__"
        if (
            not in_new_override
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__new__"
        ):
            lines.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child, in_new_override=in_new_override)

    visit(ast.parse(source), in_new_override=False)
    return lines


def test_test_modules_build_bare_objects_only_through_package_helpers() -> None:
    """``object.__new__(Cls)`` plus attribute pokes belongs in one helper per package
    (``bare_trainer`` in ``tests/trainers/online/_helpers.py``), so the attribute names
    a method reads appear in one place.
    """
    violations = [
        f"{path.relative_to(TESTS_ROOT.parent)}:{line}"
        for path in sorted(TESTS_ROOT.rglob("test_*.py"))
        for line in _bypass_lines(path.read_text(encoding="utf-8"))
    ]
    assert not violations, violations


def test_bypass_scan_skips_new_overrides_and_sees_both_spellings() -> None:
    source = """
class Probe:
    def __new__(cls):
        return super().__new__(cls)

def build():
    a = object.__new__(Probe)
    b = Probe.__new__(Probe)
"""
    assert _bypass_lines(source) == [7, 8]
