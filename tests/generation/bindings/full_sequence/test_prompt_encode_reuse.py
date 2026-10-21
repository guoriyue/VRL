"""The batches of one prompt reuse its conditioning instead of re-encoding it."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import torch

from tests.tiny_real_stack import tiny_sana_stack


def test_batches_of_one_prompt_encode_it_once_with_identical_output(monkeypatch, tmp_path) -> None:
    from vrl.rollouts.collector.config import RolloutCollectorConfig
    from vrl.rollouts.collector.requests import GenerationRequestBuilder

    stack = tiny_sana_stack(
        monkeypatch, tmp_path, overrides=("rollout.samples_per_generation_batch=1",)
    )
    builder = GenerationRequestBuilder(
        entry=stack.family, config=RolloutCollectorConfig.from_root(stack.built.root)
    )
    request = builder.build(["a red fox", "a quiet harbor"], 3).request
    asyncio.run(stack.runtime.activate())
    executor = stack.runtime.worker.executor
    encode_prompt = executor.model.encode_prompt
    encoded_prompts: list[object] = []

    def counting_encode(prompt, *args, **kwargs):
        encoded_prompts.append(prompt)
        return encode_prompt(prompt, *args, **kwargs)

    monkeypatch.setattr(executor.model, "encode_prompt", counting_encode)
    reused = asyncio.run(stack.runtime.generate(request))
    assert encoded_prompts == ["a red fox", "a quiet harbor"]

    # The same request (same seeds) with the cached prompt forgotten before
    # every batch: six encodes, and the very same samples.
    forward_batch = executor._forward_batch

    def forgetting_forward(generation_request, batch):
        executor._encoded_prompt = None
        return forward_batch(generation_request, batch)

    monkeypatch.setattr(executor, "_forward_batch", forgetting_forward)
    encoded_prompts.clear()
    fresh = asyncio.run(stack.runtime.generate(replace(request, request_id="re-encoded")))
    assert len(encoded_prompts) == 6
    assert torch.equal(reused.output, fresh.output)
    for name, segment in reused.trajectory.segments.items():
        for key, tensor in segment.tensors.items():
            expected = fresh.trajectory.segments[name].tensors[key].value
            assert torch.equal(tensor.value, expected), (name, key)
