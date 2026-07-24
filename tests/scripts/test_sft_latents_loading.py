"""Online recipe adapter tests for the SFT latent shard."""

from __future__ import annotations

from types import SimpleNamespace

from vrl.algorithms.grpo.continuous import GRPOConfig
from vrl.scripts.common.online import _load_sft_latents_from_config


def test_zero_sft_weight_does_not_read_configured_shard(tmp_path) -> None:
    # The shard IS configured (a missing path), but sft_weight=0.0 on the typed
    # algorithm config must short-circuit before it is ever loaded.
    built = SimpleNamespace(
        algorithm=GRPOConfig(sft_weight=0.0),
        root=SimpleNamespace(
            data=SimpleNamespace(sft_latents=str(tmp_path / "missing.pt")),
            model=SimpleNamespace(path="nvidia/model", revision=None),
        ),
    )

    assert _load_sft_latents_from_config(built, "cosmos-predict2.5") is None
