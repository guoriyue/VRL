"""Placement guards must fail before any model checkpoint is loaded."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.models.families.minimax_h3.test_model_loading import _build
from vrl.models.families.minimax_h3.partitioned_generation import (
    H3GenerationPlacement,
    PartitionedH3GenerationModel,
    build_partitioned_h3_generation_runtime_bundle,
)


@pytest.mark.parametrize(
    "invalid",
    ["overlap", "vae", "negative", "bool", "empty", "cpu", "implicit", "full", "compile", "park"],
)
def test_invalid_generation_placement_fails_before_checkpoint_loading(monkeypatch, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid generation placement reached checkpoint loading")

    monkeypatch.setattr(
        "vrl.models.families.minimax_h3.partitioned_generation.load_partitioned_transformer",
        forbidden,
    )
    placement = H3GenerationPlacement((0, 1), 2, (2, 3), 2, 3)
    overrides = {
        "overlap": {"encoder_root": 0},
        "vae": {"video_vae": 1},
        "negative": {"audio_vae": -1},
        "bool": {"encoder_root": True},
        "empty": {"encoder_layers": ()},
        "park": {"park_encoder_for_decode": "false"},
    }
    placement = replace(placement, **overrides.get(invalid, {}))
    build = replace(
        _build(rollout=True, device={"cpu": "cpu", "implicit": "cuda"}.get(invalid, "cuda:0")),
        model_config={
            "use_lora": invalid != "full",
            "torch_compile": {"enable": invalid == "compile"},
        },
    )
    with pytest.raises(ValueError):
        build_partitioned_h3_generation_runtime_bundle(build, placement)


@pytest.mark.parametrize(
    "method,component", [("decode_latents", "vae"), ("decode_audio", "audio_vae")]
)
@pytest.mark.parametrize("failure", [None, "move", "decode"])
def test_keep_encoder_decode_cleans_vae_without_parking(monkeypatch, method, component, failure):
    from vrl.models.families.minimax_h3.model import MiniMaxH3Model

    moves = []

    class VAE:
        def to(self, device):
            moves.append(str(device))
            if failure == "move" and str(device).startswith("cuda"):
                raise RuntimeError("injected move failure")
            return self

    def forbidden(*args, **kwargs):
        pytest.fail("Keep-encoder decode must not park or redispatch the encoder")

    token = object()

    def decode(self, value):
        assert value is token
        if failure == "decode":
            raise RuntimeError("injected decode failure")
        return token

    monkeypatch.setattr(
        "vrl.models.families.minimax_h3.partitioned_generation.park_partitioned_text_encoder",
        forbidden,
    )
    monkeypatch.setattr(MiniMaxH3Model, method, decode)
    model = object.__new__(PartitionedH3GenerationModel)
    torch.nn.Module.__init__(model)
    object.__setattr__(model, "_pipeline", SimpleNamespace(**{component: VAE()}))
    model.placement = H3GenerationPlacement((0, 1), 2, (2, 3), 3, 2, False)
    if failure:
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            getattr(model, method)(token)
    else:
        assert getattr(model, method)(token) is token
    assert moves == ["cuda:3" if component == "vae" else "cuda:2", "cpu"]
