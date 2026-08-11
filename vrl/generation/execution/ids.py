"""Generation sample identity helpers.

Sample identity is minted exactly once, here, from the request: the driver
derives the rows before dispatching batches, and everything downstream joins
on them — batch gatherers check exact coverage against these rows,
``TrajectoryBatch.sample_rows`` records them, and reward scoring keys
``RewardSample.sample_id`` off them (vrl/rollouts/collector/batch_builder.py).
That cross-package join is why the derivation lives in its own neutral module
instead of inside any runtime, and why it must stay deterministic
(tests/generation/execution/test_generation_contracts.py pins this).
"""

from __future__ import annotations

from vrl.generation.types import GenerationRequest, GenerationSampleRow


def build_sample_rows(request: GenerationRequest) -> list[GenerationSampleRow]:
    """Build deterministic sample rows from a generation request."""

    rows: list[GenerationSampleRow] = []
    for prompt_index, request_input in enumerate(request.inputs):
        prompt = request_input.prompt
        prompt_id = f"{request.request_id}:prompt:{prompt_index}"
        for sample_index in range(request.samples_per_prompt):
            sample_id = f"{prompt_id}:sample:{sample_index}"
            rows.append(
                GenerationSampleRow(
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    prompt=prompt,
                    sample_id=sample_id,
                )
            )
    return rows


__all__ = ["build_sample_rows"]
