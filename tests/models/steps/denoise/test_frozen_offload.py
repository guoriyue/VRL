"""Pipeline frozen-component offload (SPRINT_frozen_component_preservation).

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

from types import SimpleNamespace

import pytest
import torch.nn as nn

from tests.models.steps.denoise.fixtures import (
    build_tiny_autoencoder_kl,
    build_tiny_pipeline_shell,
    build_tiny_transformer,
)
from vrl.models.families.sd3_5.model import SD3_5PipelineModel


def _model() -> SD3_5PipelineModel:
    from diffusers import FlowMatchEulerDiscreteScheduler

    pipeline = build_tiny_pipeline_shell(
        transformer=build_tiny_transformer("sd3"),
        vae=build_tiny_autoencoder_kl(),
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    return SD3_5PipelineModel(pipeline=pipeline)


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


@pytest.mark.parametrize("components", [None, [], "vae"])
def test_invalid_pipeline_components_cannot_silently_skip_offload(components) -> None:
    transformer = nn.Linear(2, 2)
    model = SD3_5PipelineModel(
        pipeline=SimpleNamespace(transformer=transformer, components=components)
    )
    with pytest.raises(TypeError, match=r"pipeline\.components must be a mapping"):
        model.move_frozen_components("meta")
    assert next(transformer.parameters()).device.type == "cpu"


def test_frozen_offload_excludes_every_registered_pipeline_component() -> None:
    first, second, vae = (nn.Linear(2, 2) for _ in range(3))
    pipeline = SimpleNamespace(
        transformer=first,
        components={"transformer": first, "second_expert": second, "vae": vae},
    )
    model = SD3_5PipelineModel(pipeline=pipeline)
    model.second_expert = second

    model.move_frozen_components("meta")

    assert next(first.parameters()).device.type == "cpu"
    assert next(second.parameters()).device.type == "cpu"
    assert next(vae.parameters()).device.type == "meta"
