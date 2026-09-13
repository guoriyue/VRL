"""Placement guards must fail before any model checkpoint is loaded."""

from dataclasses import replace

import pytest

from tests.models.families.minimax_h3.test_model_loading import _build
from vrl.models.families.minimax_h3.partitioned_generation import (
    H3GenerationPlacement,
    build_partitioned_h3_generation_runtime_bundle,
)


@pytest.mark.parametrize(
    "invalid",
    ["overlap", "vae", "negative", "bool", "empty", "cpu", "implicit", "full", "compile"],
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
