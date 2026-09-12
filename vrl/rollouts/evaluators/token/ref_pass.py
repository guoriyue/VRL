"""Reference (second-pass) forward shared by the token log-prob evaluators.

The three token evaluators recompute a frozen-reference signal the same way:
with a distinct ``ref_model`` use it directly; otherwise reuse ``model`` with
its LoRA adapter disabled. Both reference paths run under ``no_grad``. The denoise evaluators use a
different identity-based convention (``ref_model is model`` -> nullcontext) and
deliberately do not share this helper.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vrl.models.interfaces import ReplayModel


@contextmanager
def reference_model_context(
    model: ReplayModel,
    ref_model: ReplayModel | None,
) -> Iterator[ReplayModel]:
    """Select the reference model and scope its gradient/adapter state.

    Uses the distinct frozen ``ref_model`` when provided; otherwise reuses
    ``model`` inside ``no_grad`` + ``disable_adapter()`` so the adapter-off pass
    produces the reference distribution.
    """

    with torch.no_grad():
        if ref_model is not None:
            yield ref_model
        else:
            with model.disable_adapter():
                yield model
