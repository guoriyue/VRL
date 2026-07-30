"""The consumer that keeps ``real_cover`` labels from rotting.

A label that names a real counterpart is a promise about another file. Prose
promises in this repo have already gone stale (``_FakeCuMem``'s docstring
promised a memory-effect twin "when a vLLM-equipped GPU lane exists" while that
twin already sat in the same file), so the promise is machine-checked here: a
label with no reason, a label attached to nothing, a label naming a target that
does not exist, and a tracked gap whose document is gone all fail the suite.

What the guard cannot check is whether a named counterpart is genuinely real --
a double can wear a lane marker, and a real one can run in the default lane. That
argument lives in the mandatory ``why=``, for a reader to weigh.

These checks read the labels off disk with ``tests/real_cover.py`` — the same
parser ``--real-cover-report`` uses — so the guard and the register can never
disagree about what a label says.
"""

from __future__ import annotations

import pytest

from tests.real_cover import (
    MARKER,
    REAL_LANES,
    REPO_ROOT,
    RealCoverLabel,
    ResolvedTarget,
    collect_labels,
    iter_carrier_sources,
    resolve_target,
)

# Read once: the scan costs ~30 ms and every check below wants the same labels.
LABELS = collect_labels()

# Stable fixtures for the resolver's own behaviour. If these move, fix the paths
# here — a silent resolver is worse than a noisy one.
_LANE_TARGET = (
    "tests/generation/execution/test_worker_sleep.py"
    "::test_real_cumem_one_shot_scope_sleep_wake_in_subprocess"
)
_MODULE_LANE_TARGET = (
    "tests/ray/test_ray_actor_pool.py::test_run_actor_jobs_awaits_real_object_refs"
)
_CPU_TARGET = "tests/nn/layers/test_paged_attention_contract.py::test_paged_attention_config_requires_identity"


def test_real_cover_is_a_registered_marker_and_so_is_every_real_lane(pytestconfig) -> None:
    """``--strict-markers`` turns an unregistered marker into a collection ERROR."""

    registered = {entry.split(":", 1)[0]: entry for entry in pytestconfig.getini("markers")}
    assert MARKER in registered, f"register '{MARKER}' in pyproject.toml's markers table"
    assert ":" in registered[MARKER], "the marker entry must document its contract"
    missing = sorted(REAL_LANES - registered.keys())
    assert not missing, f"real lanes that are not registered markers: {missing}"


def test_every_label_is_attached_to_a_test_and_well_formed() -> None:
    """A label nobody carries, or one the parser cannot read, documents nothing."""

    problems = []
    for label in LABELS:
        if label.problem:
            problems.append((label, label.problem))
        if label.unknown_kwargs:
            problems.append((label, f"unknown keyword(s): {', '.join(label.unknown_kwargs)}"))
        if label.owner is None:
            problems.append((label, "not attached to any test, class or pytestmark"))
        if not (label.why or "").strip():
            problems.append((label, "needs why= explaining what blocks the real thing"))
    assert not problems, _report(problems)


def test_every_named_target_exists() -> None:
    """The point of the guard: a dangling counterpart fails here.

    A counterpart in no real lane is accepted. Some run on every PR --
    tests/trainers/test_ddp.py drives a real gloo process group in the default
    lane -- and marking one `distributed` to satisfy a lane check would move it
    behind --distributed and lose that coverage. The lane was never proof of
    realness anyway (a double can wear @pytest.mark.gpu); the mandatory `why=`
    above is what carries the argument.
    """

    problems = []
    for label in LABELS:
        if label.target is None:
            continue
        resolved = resolve_target(label.target)
        if resolved.problem:
            problems.append((label, resolved.problem))
    assert not problems, _report(problems)


def test_every_tracked_gap_points_at_a_document_on_disk() -> None:
    """``real_cover(None, ...)`` claims no counterpart exists; the claim needs an owner."""

    problems = []
    for label in LABELS:
        if label.target is not None:
            continue
        tracked = (label.tracked_in or "").strip()
        if not tracked:
            problems.append((label, "a None target needs tracked_in=<document>"))
        elif not (REPO_ROOT / tracked).exists():
            problems.append((label, f"tracked_in does not exist: {tracked}"))
    assert not problems, _report(problems)


def test_the_scan_understood_every_file_that_mentions_the_marker() -> None:
    """Non-vacuity: zero labels, or a carrier the parser skipped, must be visible."""

    assert LABELS, (
        "no real_cover labels found at all — either the labels were removed or "
        "tests/real_cover.py stopped recognising them"
    )
    labelled = {label.file for label in LABELS}
    silent = [
        path.relative_to(REPO_ROOT).as_posix()
        for path, _ in iter_carrier_sources()
        if path.relative_to(REPO_ROOT).as_posix() not in labelled
    ]
    assert not silent, (
        "these files mention real_cover but produced no parsed label, so the guard "
        f"is not reading them: {silent}"
    )


def test_label_discovery_finds_every_way_a_mark_actually_gets_attached(tmp_path) -> None:
    """Every spelling pytest honours must resolve to an owner, or the guard cries wolf.

    ``pytestmark = SOME_CONSTANT`` applies the mark to the whole module exactly
    like an inline call does, and hoisting a repeated reason into a constant is
    this repo's own convention, so reporting that label as unattached would fail a
    working label with a false diagnosis.

    The marker is spelled through ``MARKER`` instead of written out so this file
    does not become a carrier that the scan then demands a label from.
    """

    mark = f"pytest.mark.{MARKER}"
    module = tmp_path / "tests" / "test_attachment_forms.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        f"""import pytest

VIA_PYTESTMARK_CONSTANT = {mark}("via_pytestmark_constant", why="w")
VIA_DECORATOR_CONSTANT = {mark}("via_decorator_constant", why="w")
NEVER_APPLIED = {mark}("never_applied", why="w")

pytestmark = [{mark}("in_pytestmark_inline", why="w"), VIA_PYTESTMARK_CONSTANT]


@VIA_DECORATOR_CONSTANT
def test_decorated_by_a_constant() -> None: ...


@{mark}("on_a_decorator", why="w")
def test_decorated_inline() -> None: ...
""",
        encoding="utf-8",
    )

    assert {label.target: label.owner for label in collect_labels(tmp_path)} == {
        "in_pytestmark_inline": "<module>",
        "via_pytestmark_constant": "<module>",
        "via_decorator_constant": "<via VIA_DECORATOR_CONSTANT>",
        "on_a_decorator": "test_decorated_inline",
        "never_applied": None,
    }


def test_the_resolver_refuses_targets_that_cannot_discharge_a_label() -> None:
    """A resolver that says yes to everything would make every check above green."""

    assert resolve_target(_LANE_TARGET).lanes == ("gpu",)
    # This one's lane comes from a module-level `pytestmark`, not a decorator, so
    # resolution has to add module marks to the named test's own.
    assert resolve_target(_MODULE_LANE_TARGET).lanes == ("slow_test",)
    assert "no such file" in (resolve_target("tests/not_a_file.py::test_x").problem or "")
    assert "defines no" in (resolve_target(f"{_LANE_TARGET}_typo").problem or "")
    # A default-lane counterpart resolves cleanly with an empty lane tuple; the
    # mandatory `why=` is what argues it is real, not the marker.
    assert resolve_target(_CPU_TARGET) == ResolvedTarget(lanes=())


def test_the_resolver_sees_a_lane_through_a_conditional_pytestmark(tmp_path) -> None:
    """``pytestmark = <mark> if cond else ()`` still applies the mark at runtime.

    ``_mark_names`` walks the assignment expression precisely so an ``IfExp``
    cannot hide a lane. That used to be asserted against two real files that
    spelled their ``pytestmark`` that way; both were rewritten to a plain
    assignment once they stopped needing the guard, which would have left the
    resolver's ``IfExp`` handling with no test at all. A synthetic file keeps the
    behaviour pinned without holding an unrelated module's style hostage.
    """

    module = tmp_path / "tests" / "test_conditional_pytestmark.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import pytest\n\n"
        "pytestmark = pytest.mark.slow_test if pytest is not None else ()\n\n\n"
        "def test_in_the_lane() -> None: ...\n",
        encoding="utf-8",
    )

    target = "tests/test_conditional_pytestmark.py::test_in_the_lane"
    assert resolve_target(target, tmp_path).lanes == ("slow_test",)


@pytest.mark.parametrize(
    "target, expected",
    [
        ("tests/generation/execution/test_chunks_pipelined_cuda.py", ("gpu",)),
        ("tests/nn/layers/test_paged_attention_contract.py", ()),
    ],
)
def test_a_bare_file_target_reports_only_the_lanes_every_test_in_it_shares(
    target: str,
    expected: tuple[str, ...],
) -> None:
    """A union would let one lane-marked test vouch for a file full of CPU tests."""

    assert resolve_target(target).lanes == expected


def _report(problems: list[tuple[RealCoverLabel, str]]) -> str:
    return "\n".join(
        f"\n{label.where}  {label.owner or '<unattached>'}"
        f"\n    -> {label.target if label.target is not None else 'NO REAL COUNTERPART'}"
        f"\n    {problem}"
        for label, problem in problems
    )
