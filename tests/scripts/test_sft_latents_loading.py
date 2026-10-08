"""Online recipe adapter tests for the SFT latent shard."""

from __future__ import annotations

from tests.trainers.online._helpers import real_trainer
from vrl.scripts.common.online import _load_sft_latents_from_config


def test_zero_sft_weight_does_not_read_configured_shard(monkeypatch, tmp_path) -> None:
    # The shard IS configured (a missing path), but the built objective's
    # sft_weight=0.0 must short-circuit before it is ever loaded.
    tb = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=("algorithm.sft_weight=0.0", f"data.sft_latents={tmp_path / 'missing.pt'}"),
    )
    built = tb.collector.stack.resolved.built

    assert tb.trainer.algorithm.sft_weight == 0.0
    assert _load_sft_latents_from_config(built, sft_weight=tb.trainer.algorithm.sft_weight) is None
