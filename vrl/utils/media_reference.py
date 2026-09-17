"""Sample views of Ray-owned media, shared by generation and reward transports.

The reference stays boxed so a driver can forward it without fetching the
media. Ray imports only at resolution in the process that consumes the sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class MediaReference:
    """One sample in an object-store batch; normalization runs at the consumer."""

    # Ray is optional at config import time; this is a ray.ObjectRef at runtime.
    object_ref: Any
    sample_index: int
    value_range: Literal["unit", "tanh"] = "unit"
    # Per-sample payload bytes: bounded queues account for remote media without
    # fetching it. The producer fills this from the decoded sample tensor.
    nbytes: int = 0

    def resolve(self, cache: dict[Any, Any] | None = None) -> Any:
        """Fetch each batch once per scoring request, then select its sample."""

        import ray
        import torch

        if cache is None:
            batch = ray.get(self.object_ref)
        else:
            if self.object_ref not in cache:
                cache[self.object_ref] = ray.get(self.object_ref)
            batch = cache[self.object_ref]
        media = batch[self.sample_index]
        if isinstance(media, torch.Tensor) and media.dtype == torch.uint8:
            return media.float() / 255.0
        if self.value_range == "tanh":
            return ((media + 1.0) * 0.5).clamp(0.0, 1.0)
        return media
