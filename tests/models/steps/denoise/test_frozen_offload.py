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

from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn

from tests.models.steps.denoise.fixtures import (
    build_tiny_autoencoder_kl,
    build_tiny_pipeline_shell,
    build_tiny_sd3_transformer,
)
from vrl.models.steps.denoise.base import DiffusersReplayModelBase, DiffusionModelBase


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


def test_replay_declares_no_frozen_components() -> None:
    class ReplayModel(DiffusersReplayModelBase, _TinyDiffusionModel):
        pass

    replay = ReplayModel(transformer=nn.Linear(2, 2), scheduler=None)
    replay.move_frozen_components("meta")
    assert replay.generation_memory_targets() == {}
    assert next(replay.transformer.parameters()).device.type == "cpu"


@pytest.mark.parametrize("operation", ["move", "targets"])
def test_pipeline_failure_is_not_treated_as_absence(operation) -> None:
    failure = RuntimeError("pipeline failed")

    class BrokenPipeline(_TinyDiffusionModel):
        @property
        def pipeline(self):
            raise failure

    model = BrokenPipeline(_model().pipeline)
    with pytest.raises(RuntimeError, match="pipeline failed") as caught:
        if operation == "move":
            model.move_frozen_components("cpu")
        else:
            model.generation_memory_targets()
    assert caught.value is failure


@pytest.mark.parametrize("components", [None, [], "vae"])
def test_invalid_pipeline_components_cannot_silently_skip_offload(components) -> None:
    transformer = nn.Linear(2, 2)
    model = _TinyDiffusionModel(SimpleNamespace(transformer=transformer, components=components))
    with pytest.raises(TypeError, match=r"pipeline\.components must be a mapping"):
        model.move_frozen_components("meta")
    assert next(transformer.parameters()).device.type == "cpu"


def test_absent_pipeline_requires_no_component_movement() -> None:
    model = _TinyDiffusionModel(SimpleNamespace(transformer=nn.Linear(2, 2)))
    object.__setattr__(model, "_pipeline", None)
    model.move_frozen_components("meta")
    assert next(model.transformer.parameters()).device.type == "cpu"


def test_frozen_offload_excludes_every_registered_pipeline_component() -> None:
    first, second, vae = (nn.Linear(2, 2) for _ in range(3))
    pipeline = SimpleNamespace(
        transformer=first,
        components={"transformer": first, "second_expert": second, "vae": vae},
    )
    model = _TinyDiffusionModel(pipeline)
    model.second_expert = second

    model.move_frozen_components("meta")

    assert next(first.parameters()).device.type == "cpu"
    assert next(second.parameters()).device.type == "cpu"
    assert next(vae.parameters()).device.type == "meta"
