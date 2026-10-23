"""A denoise batch times its stages into the bound rollout stats accumulator."""

from __future__ import annotations

import asyncio

from tests.tiny_real_stack import tiny_sana_stack
from vrl.utils.stats import recording


def test_each_stage_of_every_batch_lands_in_the_bound_accumulator(monkeypatch, tmp_path) -> None:
    from vrl.rollouts.collector.config import RolloutCollectorConfig
    from vrl.rollouts.collector.requests import GenerationRequestBuilder

    stack = tiny_sana_stack(
        monkeypatch, tmp_path, overrides=("rollout.samples_per_generation_batch=1",)
    )
    builder = GenerationRequestBuilder(
        entry=stack.family, config=RolloutCollectorConfig.from_root(stack.built.root)
    )
    request = builder.build(["a red fox"], 2).request
    asyncio.run(stack.runtime.activate())

    with recording() as stats:
        asyncio.run(stack.runtime.generate(request))

    stages = ("encode", "prepare_latent", "denoise", "decode")
    assert set(stats.phase_seconds) == {f"generation.{stage}_s" for stage in stages}
    assert all(seconds > 0.0 for seconds in stats.phase_seconds.values())
