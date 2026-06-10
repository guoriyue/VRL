"""vLLM-style process-safe logging for driver and Ray worker processes.

Ray actors never run the driver's ``logging.basicConfig``, so module loggers
inside workers have no handler and silently drop INFO records — which is why
ad-hoc ``print(flush=True)`` log helpers kept appearing. ``init_logger``
guarantees exactly one stdout handler on the ``"vrl"`` namespace logger in
every process. ``propagate=False`` keeps records from also reaching the root
logger, so a driver that ran ``basicConfig`` does not print them twice.

Level is controlled by the ``VRL_LOG_LEVEL`` env var (default ``INFO``),
mirroring vLLM's ``VLLM_LOGGING_LEVEL``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_ROOT_NAME = "vrl"
_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_initialized = False


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves sys.stdout at emit time, so logs follow
    stream redirection (pytest capsys, Ray output capture) instead of holding
    the stream that existed when the handler was installed."""

    @property
    def stream(self) -> Any:
        return sys.stdout

    @stream.setter
    def stream(self, value: Any) -> None:
        del value  # base __init__ assigns; always resolve live instead


def _ensure_vrl_handler() -> None:
    global _initialized
    if _initialized:
        return
    root = logging.getLogger(_ROOT_NAME)
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(os.environ.get("VRL_LOG_LEVEL", "INFO").upper())
    root.propagate = False
    _initialized = True


def init_logger(name: str) -> logging.Logger:
    """Return a module logger that is visible in driver and Ray workers alike.

    Call at import time: ``logger = init_logger(__name__)``. Idempotent —
    repeated calls in one process install a single handler.
    """

    _ensure_vrl_handler()
    return logging.getLogger(name)


def kv(**fields: Any) -> str:
    """Format structured log fields as ``key=value`` pairs (floats as .3f)."""

    return " ".join(
        f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in fields.items()
    )


__all__ = ["init_logger", "kv"]
