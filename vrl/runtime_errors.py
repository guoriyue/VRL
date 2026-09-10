"""Cross-layer runtime failures and cycle-safe cause inspection."""

from __future__ import annotations


class TerminalRuntimeError(RuntimeError):
    """A failure after which the owning runtime must reject further work."""


def find_error_cause[ErrorT: BaseException](
    error: BaseException,
    error_type: type[ErrorT],
) -> ErrorT | None:
    """Find the first matching error along explicit cleanup-wrapper causes."""

    for candidate in _error_chain(error):
        if isinstance(candidate, error_type):
            return candidate
    return None


def root_failure_cause(error: BaseException) -> BaseException:
    """Return the error a restart policy should key on.

    Walks the cause chain through transparent cleanup wrappers. The first
    ``TerminalRuntimeError`` wins: it is the domain-owned failure, so the
    dependency detail stored in its ``__cause__`` is not the answer. Without
    one, the deepest cause is.
    """

    chain = _error_chain(error)
    terminal = next(
        (candidate for candidate in chain if isinstance(candidate, TerminalRuntimeError)),
        None,
    )
    return terminal if terminal is not None else chain[-1]


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
    "find_error_cause",
    "root_failure_cause",
]
