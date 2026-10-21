"""The ``offload`` section of a role, and what it does to the role's trainable roots."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Offload:
    """``model.offload`` / ``actor.offload``: stream the frozen transformer blocks from the host.

    A role without the section keeps its policy resident on the device. With
    it, the frozen weights of every repeated block list stay on the host and
    each block's call runs over device copies of them
    (``vrl.nn.offload.block_streaming``), so a bf16 ~20B policy generates and
    trains on one 32 GB card. The rollout build carries the section to the
    OffloadPass; the training strategies and the offline recipes read it at
    root placement.
    """

    @classmethod
    def from_section(cls, section: Any) -> Offload | None:
        """The runtime form of a parsed ``offload`` section; ``None`` stays resident."""

        if section is None or isinstance(section, cls):
            return section
        if not isinstance(section, Mapping):
            section = section.model_dump()
        return cls(**section)

    def place(self, roots: Mapping[str, nn.Module], *, device: Any) -> int:
        """Stream every host-resident root of ``roots`` on ``device``; return the blocks streamed.

        A root a loader already dispatched (block-partitioned across several
        devices, a native package loading straight to its card) is left alone,
        as resident placement leaves it (``vrl.models.parking.module_on_host``).
        """

        # Imported here: block streaming reads the repeated block lists from
        # vrl.nn.optimization, whose passes read this section.
        from vrl.nn.offload.block_streaming import HostStreaming

        streamed = 0
        for name, root in roots.items():
            if any(parameter.device.type != "cpu" for parameter in root.parameters()):
                continue
            count = len(HostStreaming.install(root, device=device).blocks)
            logger.info("%s: %d frozen blocks streamed from the host", name, count)
            streamed += count
        return streamed


__all__ = ["Offload"]
