from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.models.families.sd3_5.model import SD3_5Model, SD3_5ReplayModel
from vrl.nn.layers.attention.joint import SD3JointAttentionProcessor


class _Transformer:
    def __init__(self) -> None:
        self.processor = None

    def set_attn_processor(self, processor: object) -> None:
        self.processor = processor


class _WrappedTransformer:
    def __init__(self) -> None:
        self.base_model = SimpleNamespace(model=_Transformer())


def test_sd3_model_installs_vrl_attention_processor_on_pipeline_transformer() -> None:
    """Checks SD3 model installs VRL attention processor on pipeline transformer."""
    transformer = _Transformer()
    model = SD3_5Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )

    assert model.uses_vrl_attention_processor is True
    assert isinstance(transformer.processor, SD3JointAttentionProcessor)


def test_sd3_replay_model_installs_vrl_attention_processor_through_wrapper() -> None:
    """Checks SD3 replay model installs VRL attention processor through wrapper."""
    transformer = _WrappedTransformer()
    model = SD3_5ReplayModel(
        transformer=transformer,
        scheduler=None,
        device=torch.device("cpu"),
    )

    assert model.uses_vrl_attention_processor is True
    assert isinstance(
        transformer.base_model.model.processor,
        SD3JointAttentionProcessor,
    )


def test_sd3_replay_model_reinstalls_processor_after_transformer_replace() -> None:
    """Checks SD3 replay model reinstalls processor after transformer replace."""
    model = SD3_5ReplayModel(
        transformer=_Transformer(),
        scheduler=None,
        device=torch.device("cpu"),
    )
    replacement = _Transformer()

    model._set_transformer(replacement)

    assert model.uses_vrl_attention_processor is True
    assert isinstance(replacement.processor, SD3JointAttentionProcessor)
