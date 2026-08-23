"""Dead-flag lint: run `python -m vrl.scripts.lint.dead_flags` from the repo root.

A CLI flag is DEAD when nothing consumes it: its ``dest`` is never read as
``args.<dest>`` anywhere in ``vrl/`` or ``tests/``, and its option string
appears in no doc, Makefile target, or other module. Such a flag is worse than
unused code — a user can pass it, the run accepts it, and nothing happens.

This is the institutional half of the 2026-08 bloat audit: that sweep removed
the dead flags by hand, and this keeps the count at zero without another audit.
The same "the consumer is the source of truth" rule the config lint applies to
schema keys, applied to argparse.

Exit code 0 = clean; 1 = findings (suitable for CI / pre-commit), matching
``vrl.config.lint``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# A flag whose dest is read through **vars(args) / getattr(args, name) rather
# than a literal args.<dest> cannot be proven live by grep. Declare those here
# with the reason, so an unprovable case is an explicit decision, not a silent
# pass. Empty today: the audit left no such flag behind.
DYNAMIC_CONSUMERS: dict[str, str] = {}


def declared_flags() -> list[tuple[str, int, str, str]]:
    """Every ``add_argument`` in ``vrl/`` as ``(file, line, option, dest)``."""

    found: list[tuple[str, int, str, str]] = []
    for path in sorted((REPO_ROOT / "vrl").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", "") != "add_argument":
                continue
            option = next(
                (
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ),
                None,
            )
            if option is None:
                continue
            dest = next(
                (
                    kw.value.value
                    for kw in node.keywords
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant)
                ),
                None,
            )
            if dest is None:
                dest = option.lstrip("-").replace("-", "_")
            found.append((str(path.relative_to(REPO_ROOT)), node.lineno, option, dest))
    return found


def _corpus() -> dict[str, str]:
    sources: dict[str, str] = {}
    for root in ("vrl", "tests"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            sources[str(path.relative_to(REPO_ROOT))] = path.read_text()
    return sources


def dead_flags() -> list[tuple[str, int, str, str]]:
    """Declared flags with neither a ``args.<dest>`` reader nor any mention."""

    sources = _corpus()
    prose = "\n".join(
        path.read_text() for path in (REPO_ROOT / "docs").rglob("*.md") if path.is_file()
    )
    makefile = REPO_ROOT / "Makefile"
    prose += makefile.read_text() if makefile.exists() else ""

    dead: list[tuple[str, int, str, str]] = []
    for file, line, option, dest in declared_flags():
        if dest in DYNAMIC_CONSUMERS:
            continue
        reader = re.compile(rf"\bargs\.{re.escape(dest)}\b")
        if any(reader.search(text) for text in sources.values()):
            continue
        # An option string named by a doc, a Makefile target, or another module
        # is a documented entry point even without a literal attribute read.
        if option in prose or any(
            option in text for name, text in sources.items() if name != file
        ):
            continue
        dead.append((file, line, option, dest))
    return dead


def main() -> int:
    findings = dead_flags()
    total = len(declared_flags())
    if findings:
        print(f"✗ {len(findings)} CLI flag(s) declared but never consumed:")
        for file, line, option, dest in findings:
            print(f"    {file}:{line}  {option}  (dest={dest})")
        print("  fix: delete the flag, wire its consumer, or — if it is read")
        print("       dynamically — register it in DYNAMIC_CONSUMERS with a reason")
        return 1
    print(f"✓ dead-flag sweep: all {total} declared CLI flags have a consumer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
