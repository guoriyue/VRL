"""Cross-layer runtime failures and cycle-safe cause inspection."""

from __future__ import annotations

from typing import TypeVar


class TerminalRuntimeError(Exception):
    """A failure after which the owning runtime must reject further work."""


_ErrorT = TypeVar("_ErrorT", bound=BaseException)


def find_error_cause(
    error: BaseException,
    error_type: type[_ErrorT],
) -> _ErrorT | None:
    """Find the first matching error along explicit cleanup-wrapper causes."""

    for candidate in _error_chain(error):
        if isinstance(candidate, error_type):
            return candidate
    return None


def deepest_error_cause(error: BaseException) -> BaseException:
    """Return the deepest explicit cause without looping on malformed wrappers."""

    return _error_chain(error)[-1]


def _error_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain = [error]
    seen = {id(error)}
    while True:
        current = chain[-1]
        candidate = getattr(current, "root_cause", None)
        if not isinstance(candidate, BaseException):
            candidate = current.__cause__
        if not isinstance(candidate, BaseException) or id(candidate) in seen:
            return tuple(chain)
        chain.append(candidate)
        seen.add(id(candidate))


__all__ = [
    "TerminalRuntimeError",
    "deepest_error_cause",
    "find_error_cause",
]
