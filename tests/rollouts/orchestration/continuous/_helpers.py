"""Shared async polling helper for continuous rollout schedule tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def _wait_until(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() >= deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.001)
