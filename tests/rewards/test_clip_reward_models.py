"""Tests for CLIP-backed in-process reward models."""

from __future__ import annotations

import pytest
import torch


def _aesthetic_head_state_dict() -> dict[str, torch.Tensor]:
    layers = (
        (0, 768, 1024),
        (2, 1024, 128),
        (4, 128, 64),
        (6, 64, 16),
        (7, 16, 1),
    )
    state_dict: dict[str, torch.Tensor] = {}
    for index, in_features, out_features in layers:
        state_dict[f"layers.{index}.weight"] = torch.zeros(out_features, in_features)
        state_dict[f"layers.{index}.bias"] = torch.zeros(out_features)
    return state_dict


def test_aesthetic_model_reads_transformers_5_projected_image_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the LAION head receives CLIP's projected Transformers 5 output."""
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    from vrl.rewards.models.aesthetic import AestheticRewardModel

    class _FakeClip(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def get_image_features(self, *, pixel_values: torch.Tensor):
            batch_size = pixel_values.shape[0]
            return BaseModelOutputWithPooling(
                last_hidden_state=torch.zeros(batch_size, 2, 1024),
                pooler_output=torch.ones(batch_size, 768),
            )

    class _FakeProcessor:
        def __call__(self, *, images, return_tensors: str):
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}

    monkeypatch.setattr(
        transformers.CLIPModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: _FakeClip()),
    )
    monkeypatch.setattr(
        transformers.CLIPProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: _FakeProcessor()),
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: _aesthetic_head_state_dict())

    model = AestheticRewardModel({"device": "cpu", "dtype": "float32"})
    model._load()

    scores = model._scorer([object(), object()])

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_pickscore_reads_transformers_5_projected_image_and_text_features() -> None:
    """Checks PickScore unwraps both projected Transformers 5 feature outputs."""
    pytest.importorskip("transformers")
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    from vrl.rewards.models.pickscore import PickScoreRewardModel

    class _FakeProcessor:
        def __call__(self, *, images=None, text=None, **kwargs):
            del kwargs
            if images is not None:
                return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}
            return {"input_ids": torch.zeros(len(text), 2, dtype=torch.long)}

    class _FakeModel:
        logit_scale = torch.tensor(0.0)

        def get_image_features(self, *, pixel_values: torch.Tensor):
            batch_size = pixel_values.shape[0]
            return BaseModelOutputWithPooling(
                last_hidden_state=torch.zeros(batch_size, 2, 1024),
                pooler_output=torch.eye(batch_size),
            )

        def get_text_features(self, *, input_ids: torch.Tensor):
            batch_size = input_ids.shape[0]
            return BaseModelOutputWithPooling(
                last_hidden_state=torch.zeros(batch_size, 2, 1024),
                pooler_output=torch.eye(batch_size),
            )

    model = PickScoreRewardModel({"device": "cpu"})
    model._processor = _FakeProcessor()
    model._model = _FakeModel()

    score = model._score("prompt", [object(), object()])

    assert score == pytest.approx(1.0 / 26.0)
