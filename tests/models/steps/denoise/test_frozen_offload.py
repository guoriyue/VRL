"""DiffusionModelBase frozen-component offload (SPRINT_frozen_component_preservation).

nn.Module.to moves only registered submodules — for diffusion families that is
just the transformer; the diffusers pipeline (with its frozen VAE / text
encoders) is attached unregistered, so it would stay GPU-resident across a
driver-model offload. ``move_frozen_components`` parks those frozen components
alongside the transformer, derived from the pipeline so the set never rots.

The pipeline is a real ``DiffusionPipeline`` (``build_tiny_pipeline_shell``), so
``.components`` carries what diffusers really puts there: a ``None`` optional
slot and a non-module scheduler next to the frozen VAE.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from tests.models.steps.denoise.fixtures import (
    build_tiny_autoencoder_kl,
    build_tiny_pipeline_shell,
    build_tiny_sd3_transformer,
)
from vrl.models.steps.denoise.base import DiffusionModelBase


class _TinyDiffusionModel(DiffusionModelBase):
    """Minimal concrete family: registers only the transformer, like SD3.5."""

    def __init__(self, pipeline: Any) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    # abstractmethod stubs — unused by the offload path
    def encode_prompt(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_sampling(self, *a: Any, **k: Any) -> Any:
        raise NotImplementedError

    def forward_step(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def decode_latents(self, *a: Any, **k: Any) -> Any:
        raise NotImplementedError


def _model() -> _TinyDiffusionModel:
    from diffusers import FlowMatchEulerDiscreteScheduler

    pipeline = build_tiny_pipeline_shell(
        transformer=build_tiny_sd3_transformer(),
        vae=build_tiny_autoencoder_kl(),
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    return _TinyDiffusionModel(pipeline)


def test_move_frozen_components_moves_only_frozen() -> None:
    """The frozen VAE really moves; the transformer and the non-module slots do not.

    ``meta`` is the observable destination: a module that was moved has meta
    parameters afterwards, one that was skipped still has CPU parameters.
    """

    model = _model()
    components = model.pipeline.components
    # What diffusers itself derives: an optional slot left None and a scheduler
    # that is not an nn.Module. Both must be skipped, not crashed on.
    assert [name for name, value in components.items() if value is None] == ["text_encoder"]
    assert [
        name
        for name, value in components.items()
        if value is not None and not isinstance(value, nn.Module)
    ] == ["scheduler"]

    model.move_frozen_components("meta")

    assert next(model.pipeline.vae.parameters()).device.type == "meta"
    assert next(model.transformer.parameters()).device.type == "cpu"


def test_no_pipeline_is_a_safe_no_op() -> None:
    """Replay models / single-file checkpoints expose no pipeline -> empty set."""

    class _NoPipeline(_TinyDiffusionModel):
        @property
        def pipeline(self) -> Any:
            raise RuntimeError("replay model has no pipeline")

    model = _model()
    no_pipe = _NoPipeline(model._pipeline)
    object.__setattr__(no_pipe, "_pipeline", None)
    no_pipe.move_frozen_components("cpu")  # must not raise (moves nothing)
