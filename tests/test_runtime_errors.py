"""Cause-chain contracts shared by runtime, rollout, and verdict handling."""

from __future__ import annotations

from vrl.runtime_errors import (
    TerminalRuntimeError,
    failure_identity_cause,
    find_error_cause,
)


def test_error_chain_prefers_explicit_cleanup_root() -> None:
    terminal = TerminalRuntimeError("worker fleet is unusable")
    wrapper = RuntimeError("cleanup also failed")
    wrapper.root_cause = terminal
    wrapper.__cause__ = RuntimeError("less precise chained cause")

    assert find_error_cause(wrapper, TerminalRuntimeError) is terminal
    assert failure_identity_cause(wrapper) is terminal


def test_terminal_identity_stops_before_dependency_cause() -> None:
    terminal = TerminalRuntimeError("stable runtime failure")
    terminal.__cause__ = TimeoutError("dependency timeout type")

    assert failure_identity_cause(terminal) is terminal


def test_error_chain_stops_at_a_cycle() -> None:
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first

    assert failure_identity_cause(first) is second
    assert find_error_cause(first, TerminalRuntimeError) is None
