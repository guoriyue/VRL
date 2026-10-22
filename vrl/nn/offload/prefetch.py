"""Device copies of host tensor groups, each issued one step ahead of its use.

A :class:`Prefetcher` holds numbered units (mappings of named host tensors)
and hands out device copies of one unit at a time. While unit ``i`` is in use,
the copy of the unit expected next is already under way on a dedicated CUDA
stream, so each transfer hides behind the compute that precedes its use
instead of stalling in front of it. Which unit comes next is read off the
call sequence (``pass_direction``): a layered model's forward pass and the
checkpoint recompute of its backward pass, each from its own end. The copies
are temporaries: the caller holds them for one use, and ``record_stream`` lets
the allocator recycle them once the stream that used them is done.

The copies run at link speed only from pinned host memory
(``vrl.nn.offload.host_pinning``); from pageable memory they are bounded by
the host memcpy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class Prefetcher:
    """Hands out device copies of its units, one unit copied ahead of its use."""

    def __init__(
        self, units: Sequence[Mapping[str, torch.Tensor]], device: torch.device | str
    ) -> None:
        self._units = units
        self._device = torch.device(device)
        self._stream = (
            torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        )
        self._ahead: tuple[int, dict[str, torch.Tensor], torch.cuda.Event | None] | None = None
        self._last: int | None = None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def pass_direction(last: int | None, current: int, count: int) -> int | None:
        """The unit to copy while ``current`` is in use, read off the call order.

        Consecutive ascending calls are a forward pass, descending ones the
        checkpoint recompute of backward; a first call, a repeated call or a
        jump starts a new pass from its end (the last unit runs backward, any
        other forward).
        """

        if last is None or abs(current - last) != 1:
            step = -1 if current == count - 1 else 1
        else:
            step = current - last
        upcoming = current + step
        return upcoming if 0 <= upcoming < count else None

    def acquire(self, index: int) -> dict[str, torch.Tensor]:
        """Device copies of unit ``index``, ready on the current stream; the next unit's copy under way."""

        if self._ahead is not None and self._ahead[0] == index:
            _, copies, ready = self._ahead
            self.hits += 1
        else:
            copies, ready = self._copy(index)
            self.misses += 1
        self._ahead = None
        if ready is not None:
            current = torch.cuda.current_stream(self._device)
            current.wait_event(ready)
            for tensor in copies.values():
                tensor.record_stream(current)
        upcoming = self.pass_direction(self._last, index, len(self._units))
        self._last = index
        if upcoming is not None:
            self._ahead = (upcoming, *self._copy(upcoming))
        return copies

    def _copy(self, index: int) -> tuple[dict[str, torch.Tensor], torch.cuda.Event | None]:
        unit = self._units[index]
        if self._stream is None:
            return {name: tensor.to(self._device) for name, tensor in unit.items()}, None
        # Allocate on the stream that writes, so the caching allocator's reuse
        # rule (same stream, in order) holds; the consumer records its own use.
        with torch.cuda.stream(self._stream):
            copies = {
                name: torch.empty_like(tensor, device=self._device).copy_(
                    tensor, non_blocking=True
                )
                for name, tensor in unit.items()
            }
            ready = torch.cuda.Event()
            ready.record(self._stream)
        return copies, ready


__all__ = ["Prefetcher"]
