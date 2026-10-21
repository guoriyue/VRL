"""Weight residency below the device: the ``offload`` section and block streaming.

Three mechanisms decide where a policy's weights live, at three granularities:
``model.memory.cpu_resident`` keeps whole pipeline components on the host,
``vrl.models.parking`` moves a role's state off the card between phases, and
this package owns the per-forward middle: a role's ``offload`` section
(:class:`Offload`), under which a transformer's frozen blocks stay on the host
and are streamed per block (:class:`HostStreaming`).
"""

from vrl.nn.offload.block_streaming import HostStreaming
from vrl.nn.offload.offload import Offload

__all__ = ["HostStreaming", "Offload"]
