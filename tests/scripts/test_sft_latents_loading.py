"""Online recipe adapter tests for the SFT latent shard."""

from __future__ import annotations

from omegaconf import OmegaConf

from vrl.scripts.common.online import _load_sft_latents_from_config


def test_zero_sft_weight_does_not_read_configured_shard(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"sft_weight": 0.0},
            "data": {"sft_latents": str(tmp_path / "missing.pt")},
            "model": {"path": "nvidia/model"},
        },
    )

    assert _load_sft_latents_from_config(cfg, "cosmos-predict2.5") is None
