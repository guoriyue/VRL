"""Parser for the ``real_cover`` labels that record honest coverage gaps.

``@pytest.mark.real_cover(...)`` marks a test whose double cannot be made real
in-process and names where the real counterpart lives. A label is only worth
writing if something reads it — a prose note with no consumer rots (this repo
already shipped one: a docstring promised a memory-effect twin "when a
vLLM-equipped GPU lane exists" while that twin sat 1090 lines below it). So this
module is the single parser behind both consumers:

- ``tests/architecture/test_real_cover_labels.py`` fails the suite when a label
  names a target that does not exist, omits its reason, or leaves a tracked gap
  without an on-disk owner;
- ``--real-cover-report`` (``tests/conftest.py``) prints the register with each
  target's lane metadata.

Lane metadata neither proves that a counterpart is real nor that it ran: a
double can carry a ``gpu`` marker, a genuine counterpart can run in the default
lane, and an opt-in target can still skip on a matching host. The mandatory
``why=`` records the argument a reviewer must assess; the register reports lane
metadata without turning it into validation.

The scan skips exactly one file — this one, whose docstring names the marker.
Everything else that spells ``pytest.mark.real_cover`` must yield a parsed label
or ``test_the_scan_understood_every_file_that_mentions_the_marker`` fails, so a
spelling the AST walk does not understand (a bare decorator with no arguments,
say) reddens instead of vanishing.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

_SELF = Path(__file__).resolve()

REPO_ROOT = _SELF.parents[1]
MARKER = "real_cover"
_MARKER_SPELLING = f"pytest.mark.{MARKER}"

# These lane markers are useful report categories for expensive or
# environment-specific counterparts. They are metadata, not evidence of
# realness; a default-suite counterpart deliberately resolves to no lane.
# test_real_cover_labels.py asserts every reported name is a registered marker.
REAL_LANES = frozenset({"distributed", "e2e", "gpu", "slow_test"})

_LABEL_KWARGS = frozenset({"why", "tracked_in"})
_MODULE_OWNER = "<module>"


@dataclass(frozen=True, slots=True)
class RealCoverLabel:
    """One ``pytest.mark.real_cover(...)`` call found on disk."""

    file: str
    lineno: int
    # The test/class the label decorates, ``"<module>"`` when a ``pytestmark``
    # assignment applies it (inline or through a constant), ``"<via NAME>"`` for a
    # constant reused as a decorator, and None when the label is attached to
    # nothing at all (a dead label).
    owner: str | None
    target: str | None
    why: str | None
    tracked_in: str | None
    unknown_kwargs: tuple[str, ...] = ()
    problem: str | None = None

    @property
    def where(self) -> str:
        return f"{self.file}:{self.lineno}"


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """The lanes a label's target lives in, or why it could not be resolved."""

    lanes: tuple[str, ...] = ()
    problem: str | None = None

    @property
    def lane_label(self) -> str:
        return "+".join(self.lanes) if self.lanes else "-"


def collect_labels(root: Path = REPO_ROOT) -> list[RealCoverLabel]:
    """Every label carried by ``tests/``, ordered by file then line."""

    labels: list[RealCoverLabel] = []
    for path, text in iter_carrier_sources(root):
        labels.extend(_labels_in(path.relative_to(root).as_posix(), text, path))
    return sorted(labels, key=lambda label: (label.file, label.lineno))


def iter_carrier_sources(root: Path = REPO_ROOT) -> Iterator[tuple[Path, str]]:
    """Files that spell the marker, i.e. every file that must yield a label.

    Reading every test file but AST-parsing only the handful that spell the
    marker keeps the scan under 100 ms; AST-walking the whole tree costs ~600 ms.
    """

    for path in sorted((root / "tests").rglob("*.py")):
        if "__pycache__" in path.parts or path == _SELF:
            continue
        text = path.read_text(encoding="utf-8")
        if _MARKER_SPELLING in text:
            yield path, text


def resolve_target(target: str, root: Path = REPO_ROOT) -> ResolvedTarget:
    """Resolve ``path``, ``path::test`` or ``path::Class::test`` to its lanes."""

    file_part, _, rest = target.partition("::")
    path = root / file_part
    if not path.is_file():
        return ResolvedTarget(problem=f"no such file on disk: {file_part}")
    try:
        tree = _parse(path)
    except SyntaxError as exc:  # pragma: no cover - a target file that stops parsing
        return ResolvedTarget(problem=f"{file_part} does not parse: {exc}")

    aliases = _mark_aliases(tree)
    marks = _module_marks(tree, aliases)
    if rest:
        scope: ast.AST = tree
        for part in rest.split("::"):
            node = _child_named(scope, part)
            if node is None:
                return ResolvedTarget(problem=f"{file_part} defines no {part!r}")
            if not _is_collected_name(node):
                return ResolvedTarget(
                    problem=f"{file_part}::{part} is not a name pytest collects",
                )
            marks |= _own_marks(node, aliases)
            scope = node
    else:
        # A bare file means "the whole file is the counterpart", so the lanes it
        # reports are the ones EVERY test in it shares — a union would let one
        # lane-marked test vouch for a file full of CPU tests.
        tests = _collected_children(tree)
        if not tests:
            return ResolvedTarget(problem=f"{file_part} defines no tests to stand in as cover")
        per_test = [marks | _own_marks(node, aliases) for node in tests]
        marks = set.intersection(*per_test)

    # An empty lane tuple is a valid answer, not a problem. A real counterpart can
    # live in the default lane and be stronger evidence for it: tests/trainers/
    # test_ddp.py runs a real gloo process group on every PR, and marking it
    # `distributed` to satisfy a lane check would move it behind --distributed and
    # lose that coverage. Nor was the marker ever proof of realness — a double can
    # wear @pytest.mark.gpu. So the resolver only reports lanes; the guard
    # requires every label to supply `why=` regardless of the reported lane.
    return ResolvedTarget(lanes=tuple(sorted(marks & REAL_LANES)))


# --- label discovery ---------------------------------------------------------


def _labels_in(rel: str, text: str, path: Path) -> list[RealCoverLabel]:
    tree = ast.parse(text, filename=str(path))
    calls = [node for node in ast.walk(tree) if _is_marker_call(node)]
    if not calls:
        return []

    owners: dict[int, str] = {}
    alias_calls: dict[str, list[ast.Call]] = {}
    # A module constant that holds a mark is applied in one of two ways, and both
    # attach it for real: as a bare `@NAME` decorator, or by naming it in a
    # `pytestmark` assignment. Reusing a constant is the repo's own convention for
    # a reason written more than once (the `requires_fp8` precedent), so a label
    # spelled that way must resolve to its owner instead of being reported as
    # unattached.
    decorator_aliases: set[str] = set()
    module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            for decorator in node.decorator_list:
                for call in _marker_calls_in(decorator):
                    owners[id(call)] = node.name
                if isinstance(decorator, ast.Name):
                    decorator_aliases.add(decorator.id)
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            module_level = "pytestmark" in names
            if module_level:
                module_aliases.update(
                    child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)
                )
            found = _marker_calls_in(node.value)
            if not found:
                continue
            if module_level:
                for call in found:
                    owners[id(call)] = _MODULE_OWNER
            for name in names:
                alias_calls.setdefault(name, []).extend(found)

    # Module ownership first: a constant used both ways covers the whole module,
    # which is the wider and therefore the honest attribution.
    for name in module_aliases:
        for call in alias_calls.get(name, ()):
            owners.setdefault(id(call), _MODULE_OWNER)
    for name in decorator_aliases:
        for call in alias_calls.get(name, ()):
            owners.setdefault(id(call), f"<via {name}>")
    return [_label(rel, call, owners.get(id(call))) for call in calls]


def _label(rel: str, call: ast.Call, owner: str | None) -> RealCoverLabel:
    problem: str | None = None
    target: str | None = None
    if not call.args:
        problem = "takes a positional target (a 'path::test' string, or None for a tracked gap)"
    elif len(call.args) > 1:
        problem = "takes exactly one positional target"
    else:
        literal = call.args[0]
        if isinstance(literal, ast.Constant) and (
            literal.value is None or isinstance(literal.value, str)
        ):
            target = literal.value
        else:
            problem = "the target must be a literal string or None, so the guard can read it"

    values: dict[str, str] = {}
    unknown: list[str] = []
    for keyword in call.keywords:
        if keyword.arg not in _LABEL_KWARGS:
            unknown.append(keyword.arg or "**kwargs")
        elif isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            values[keyword.arg] = keyword.value.value
        else:
            problem = problem or f"{keyword.arg}= must be a literal string"
    return RealCoverLabel(
        file=rel,
        lineno=call.lineno,
        owner=owner,
        target=target,
        why=values.get("why"),
        tracked_in=values.get("tracked_in"),
        unknown_kwargs=tuple(unknown),
        problem=problem,
    )


def _is_marker_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == MARKER
        and _is_pytest_mark(node.func.value)
    )


def _marker_calls_in(expr: ast.expr) -> list[ast.Call]:
    """Marker calls anywhere inside an expression (list, tuple, IfExp, param marks)."""

    return [node for node in ast.walk(expr) if _is_marker_call(node)]


# --- lane resolution ---------------------------------------------------------


@functools.cache
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _mark_aliases(tree: ast.Module) -> Mapping[str, frozenset[str]]:
    """Module constants that hold marks, e.g. ``requires_fp8 = pytest.mark.skipif(...)``."""

    aliases: dict[str, frozenset[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = _mark_names(node.value, {})
        if not names:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases[target.id] = frozenset(names)
    return aliases


def _module_marks(tree: ast.Module, aliases: Mapping[str, frozenset[str]]) -> set[str]:
    marks: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            marks |= _mark_names(node.value, aliases)
    return marks


def _own_marks(node: ast.AST, aliases: Mapping[str, frozenset[str]]) -> set[str]:
    decorators = getattr(node, "decorator_list", [])
    return {name for decorator in decorators for name in _mark_names(decorator, aliases)}


def _mark_names(expr: ast.expr, aliases: Mapping[str, frozenset[str]]) -> set[str]:
    """Mark names in an expression; walking penetrates ``X if cond else ()`` and lists."""

    names: set[str] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Attribute) and _is_pytest_mark(node.value):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names |= aliases.get(node.id, frozenset())
    return names


def _is_pytest_mark(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "mark"
        and isinstance(node.value, ast.Name)
        and node.value.id == "pytest"
    )


def _child_named(scope: ast.AST, name: str) -> ast.AST | None:
    for node in getattr(scope, "body", []):
        if (
            isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == name
        ):
            return node
    return None


def _is_collected_name(node: ast.AST) -> bool:
    if isinstance(node, ast.ClassDef):
        return node.name.startswith("Test")
    return getattr(node, "name", "").startswith("test")


def _collected_children(scope: ast.AST) -> list[ast.AST]:
    return [
        node
        for node in getattr(scope, "body", [])
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and _is_collected_name(node)
    ]
