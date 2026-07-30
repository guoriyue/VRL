"""Reference (second-pass) forward shared by the token log-prob evaluators.

The three token evaluators recompute a frozen-reference signal the same way:
with a distinct ``ref_model`` do a plain forward; otherwise reuse ``model`` with
its LoRA adapter disabled under ``no_grad``. The denoise evaluators use a
different identity-based convention (``ref_model is model`` -> nullcontext) and
deliberately do not share this helper.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from vrl.models.interfaces import ReplayModel


def ref_forward[T](
    model: ReplayModel,
    ref_model: ReplayModel | None,
    run: Callable[[ReplayModel], T],
) -> T:
    """Run ``run`` under the reference-model convention and return its result.

    Uses the distinct frozen ``ref_model`` when provided; otherwise reuses
    ``model`` inside ``no_grad`` + ``disable_adapter()`` so the adapter-off pass
    produces the reference distribution.
    """

    if ref_model is not None:
        return run(ref_model)
    with torch.no_grad(), model.disable_adapter():
        return run(model)
