"""Shared validation helpers for Ray generation boundaries."""

from __future__ import annotations

from typing import Any


def require_installed_policy_version(
    *,
    worker_id: str,
    installed: Any,
    expected: int,
) -> None:
    """Require a worker ACK for the expected installed policy version."""

    try:
        installed_version = int(installed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"worker {worker_id!r} returned invalid installed policy version {installed!r}",
        ) from exc
    if installed_version != int(expected):
        raise RuntimeError(
            f"worker {worker_id!r} installed policy version {installed_version}, "
            f"expected {int(expected)}",
        )


__all__ = ["require_installed_policy_version"]
