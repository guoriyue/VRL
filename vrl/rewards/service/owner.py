"""Dedicated event-loop owner for one reward inference runtime.

HTTP remains responsive while a model performs synchronous GPU work because
the runtime and all of its lifecycle calls stay on this one owner thread.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrl.rewards.inference import (
        RewardInferenceRequest,
        RewardInferenceResult,
        RewardScorer,
    )


class RewardScorerOwner:
    """Run every runtime operation on one dedicated event-loop thread."""

    def __init__(self, runtime: RewardScorer) -> None:
        self._runtime = runtime
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._closing = False
        self._close_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="reward-runtime-owner",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=10):
            raise RuntimeError("reward runtime owner failed to start")

    @property
    def alive(self) -> bool:
        return self._thread.is_alive() and not self._closing

    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]:
        """Submit one score coroutine without blocking the HTTP event loop."""

        if not self.alive:
            raise RuntimeError("reward runtime owner is shutting down")
        completed = threading.Event()

        async def execute() -> list[RewardInferenceResult]:
            try:
                return await self._runtime.score_batch(request)
            finally:
                completed.set()

        future = asyncio.run_coroutine_threadsafe(
            execute(),
            self._loop,
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # Cooperative runtimes stop immediately. A synchronous GPU kernel
            # cannot be preempted, but its eventual result is discarded and no
            # later request can observe it as a successful completion.
            future.cancel()
            await asyncio.shield(asyncio.to_thread(completed.wait))
            raise

    async def close(self) -> None:
        """Shut the runtime down exactly once, then stop its owner loop."""

        await asyncio.to_thread(self._close_blocking)

    def _close_blocking(self) -> None:
        with self._close_lock:
            first_closer = not self._closing
            if first_closer:
                self._closing = True

        if not first_closer:
            self._closed.wait()
            if self._close_error is not None:
                raise self._close_error
            return

        try:
            shutdown = asyncio.run_coroutine_threadsafe(
                self._runtime.shutdown(),
                self._loop,
            )
            shutdown.result()
        except BaseException as error:
            self._close_error = error
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=30)
            if self._thread.is_alive() and self._close_error is None:
                self._close_error = RuntimeError(
                    "reward runtime owner did not stop after shutdown",
                )
            self._closed.set()

        if self._close_error is not None:
            raise self._close_error

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True),
                )
            self._loop.close()


__all__ = ["RewardScorerOwner"]
