"""Box online reward media into the Ray object store, once per payload.

Both places a generation output leaves a Ray actor use this: the rank actor's
per-batch path and the finalizer's per-request merge. The media stays boxed
until the reward consumer resolves it, so the driver forwards references
without fetching tensors.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.types import GenerationRequest
from vrl.utils.media_reference import MediaReference


def reference_reward_media(output: Any, request: GenerationRequest, *, primary: bool) -> None:
    """Replace ``output.reward_media`` with boxed per-sample object-store views.

    No-op unless the request asked for references. A non-primary rank of a
    multi-rank engine drops its media instead: engine combination keeps only
    the primary payload, so a redundant object-store copy would be wasted.
    """

    if not request.reward_media_refs:
        return
    if not primary:
        output.reward_media = None
        return
    import ray

    media = output.reward_media
    ref = ray.put(media)
    output.reward_media = [
        MediaReference(ref, index, nbytes=sample.numel() * sample.element_size())
        for index, sample in enumerate(media)
    ]


__all__ = ["reference_reward_media"]
