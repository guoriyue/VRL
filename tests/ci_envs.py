"""Centralised CI environment-variable reads, ported from vLLM's tests/ci_envs.py.

These envs only affect a small part of the tests — fix what you need. Access them
as module attributes (lazy reads, evaluated on each access), e.g.
``ci_envs.VRL_RUN_DISTRIBUTED_TESTS``. The structure (``environment_variables`` table +
``__getattr__`` / ``__dir__`` / ``is_set``) is identical to vLLM's; only the
variable names carry the project's ``VRL_`` / ``WM_`` prefixes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VRL_RUN_DISTRIBUTED_TESTS: bool = False
    WM_RUN_REAL_MODEL_TESTS: bool = False
    WM_REAL_MODEL_RL_CASES: str = "cached"

environment_variables: dict[str, Callable[[], Any]] = {
    # Opt-in to tests that need a distributed CI lane.
    "VRL_RUN_DISTRIBUTED_TESTS": lambda: os.getenv("VRL_RUN_DISTRIBUTED_TESTS") == "1",
    # Opt-in to real-weight e2e tests (expensive; needs GPU + real checkpoints).
    "WM_RUN_REAL_MODEL_TESTS": lambda: os.getenv("WM_RUN_REAL_MODEL_TESTS") == "1",
    # Comma-separated e2e case ids to run; "cached" (default) or "all".
    "WM_REAL_MODEL_RL_CASES": lambda: os.getenv("WM_REAL_MODEL_RL_CASES", "cached"),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
