"""Where each diffusion family's classes resolve their shared members.

A family's replay model is its base class and its sampling class adds the
pipeline on top, so a replay model never inherits a pipeline member it would
have to switch off again; the replay test pins that shape. The others pin
MRO facts whose failure is a silent wrong answer rather than an error:

- ``from_build`` on ``DiffusersPipelineModelBase`` is the shared diffusers
  loader; a family not backed by a diffusers pipeline must own its own, and so
  must a replay class the registry loads through ``from_build``.
- The Wan I2V sampling class must run the I2V math the I2V replay model runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from vrl.models.families.registry import FAMILY_REGISTRY, DenoiseFamilyBuild
from vrl.models.steps.denoise import DenoiseModelBase
from vrl.models.steps.denoise.base import DenoisePipelineModelBase, DiffusersPipelineModelBase
from vrl.utils.config import import_from_path


def _owner(cls: type, name: str) -> type | None:
    """The class in ``cls``'s MRO that actually defines ``name``."""

    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


def _denoise_model_classes() -> list[Any]:
    return [
        pytest.param(import_from_path(entry.family_build.model_cls), id=family)
        for family, entry in FAMILY_REGISTRY.items()
        if isinstance(entry.family_build, DenoiseFamilyBuild)
    ]


def _replay_model_classes(*, self_loading: bool | None = None) -> list[Any]:
    classes = {}
    for entry in FAMILY_REGISTRY.values():
        build = entry.family_build
        if not isinstance(build, DenoiseFamilyBuild) or build.replay_cls is None:
            continue
        if self_loading is not None and (build.transformer_classname is None) != self_loading:
            continue
        replay_cls = import_from_path(build.replay_cls)
        classes[replay_cls.__name__] = replay_cls
    return [pytest.param(cls, id=name) for name, cls in classes.items()]


@pytest.mark.parametrize("model_cls", _denoise_model_classes())
def test_family_without_a_diffusers_pipeline_owns_from_build(
    model_cls: type[DenoiseModelBase],
) -> None:
    """Only diffusers-pipeline families may inherit the shared loader."""

    if not issubclass(model_cls, DiffusersPipelineModelBase):
        assert _owner(model_cls, "from_build") is model_cls


@pytest.mark.parametrize("replay_cls", _replay_model_classes(self_loading=True))
def test_self_loading_replay_model_owns_from_build(replay_cls: type) -> None:
    """A replay class the registry loads through ``from_build`` defines its own.

    Inheriting a parent family's loader is a silent wrong model: ``VDNH3Model``
    without its own ``from_build`` would load MiniMax-H3 and skip the graft.
    """

    assert _owner(replay_cls, "from_build") is replay_cls


def test_wan_i2v_pipeline_takes_forward_math_from_i2v_and_plumbing_from_t2v() -> None:
    """Sampling I2V runs the I2V math that the I2V replay model runs.

    ``WanI2VPipelineModel`` takes ``_transformer_inputs`` /
    ``restore_eval_state`` from ``WanI2VModel`` and its constructor, offload
    lifecycle and second-expert sync from ``WanT2VPipelineModel``. If the I2V
    math moved onto the pipeline class, replay would silently run the T2V math.
    """
    from vrl.models.families.wan_2_1.model import (
        WanI2VModel,
        WanI2VPipelineModel,
        WanT2VPipelineModel,
    )

    for name in ("_transformer_inputs", "restore_eval_state"):
        assert _owner(WanI2VPipelineModel, name) is WanI2VModel, name
    for name in ("__init__", "_set_transformer_2", "load_trainable_state", "apply_full_finetune"):
        assert _owner(WanI2VPipelineModel, name) is WanT2VPipelineModel, name


@pytest.mark.parametrize("replay_cls", _replay_model_classes())
def test_replay_models_carry_no_sampling_surface(replay_cls: type) -> None:
    """A replay model is its family's base class, never a sampling class.

    The trainer replays it with only the transformer and scheduler loaded. A
    replay model that subclassed the sampling class would inherit the
    pipeline-backed writer and the sampling methods, and would have to stub
    them out again.
    """

    assert not issubclass(replay_cls, DenoisePipelineModelBase)
