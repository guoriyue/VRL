"""Pure tests for CausVid's released temporal-chunk policy loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.models.families.causvid.runner import (
    OFFICIAL_CAUSVID_SCHEDULE,
    CausVidCausalRunner,
    CausVidGeometry,
)

# Small architecture fixture: three temporal chunks, two policy transitions each.
TINY_GEOMETRY = CausVidGeometry(
    latent_frames=6,
    latent_channels=1,
    latent_height=2,
    latent_width=2,
    frames_per_chunk=2,
    pixel_frames=21,
    pixel_height=16,
    pixel_width=16,
)
BATCH_SIZE = 2


@dataclass(frozen=True, slots=True)
class _CachedCall:
    chunk_index: int
    timestep: int
    finalize_cache: bool
    cache: object
    input: torch.Tensor
    output: torch.Tensor
    grad_enabled: bool


class _RecordingBackend:
    """Protocol fake whose predictions preserve input/seed dependence."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.cache = object()
        self.cache_allocations = 0
        self.cached_calls: list[_CachedCall] = []
        self.full_prefix_calls = 0

    def new_causal_cache(
        self,
        *,
        batch_size: int,
        geometry: CausVidGeometry,
    ) -> object:
        assert batch_size == BATCH_SIZE
        assert geometry is TINY_GEOMETRY
        self.cache_allocations += 1
        return self.cache

    def predict_x0_cached(
        self,
        noisy_chunk: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        timestep: int,
        cache: Any,
        chunk_index: int,
        geometry: CausVidGeometry,
    ) -> torch.Tensor:
        assert prompt_embeds.shape[0] == BATCH_SIZE
        assert geometry is TINY_GEOMETRY
        # Keep later predictions dependent on earlier sampled transitions.
        prediction = noisy_chunk * 0.25 + float(timestep) / 10_000
        self.cached_calls.append(
            _CachedCall(
                chunk_index=chunk_index,
                timestep=timestep,
                finalize_cache=timestep == OFFICIAL_CAUSVID_SCHEDULE.cache_timestep,
                cache=cache,
                input=noisy_chunk.detach().clone(),
                output=prediction.detach().clone(),
                grad_enabled=torch.is_grad_enabled(),
            ),
        )
        return prediction

    def predict_x0_full_prefix(
        self,
        prefix_and_chunk: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        per_frame_timesteps: torch.Tensor,
        geometry: CausVidGeometry,
    ) -> torch.Tensor:
        del prompt_embeds, per_frame_timesteps, geometry
        self.full_prefix_calls += 1
        return prefix_and_chunk * 0.25


def _noise() -> torch.Tensor:
    values = torch.arange(
        BATCH_SIZE * TINY_GEOMETRY.latent_frames * 4,
        dtype=torch.float32,
    )
    return values.reshape(BATCH_SIZE, *TINY_GEOMETRY.latent_shape) / 100


def _generators(first_seed: int, second_seed: int) -> tuple[torch.Generator, ...]:
    return (
        torch.Generator(device="cpu").manual_seed(first_seed),
        torch.Generator(device="cpu").manual_seed(second_seed),
    )


def _run(
    *seed_pair: int,
) -> tuple[_RecordingBackend, Any]:
    backend = _RecordingBackend()
    runner = CausVidCausalRunner(
        backend,
        geometry=TINY_GEOMETRY,
        schedule=OFFICIAL_CAUSVID_SCHEDULE,
    )
    result = runner.run(
        _noise(),
        prompt_embeds=torch.ones(BATCH_SIZE, 3, 4),
        generators=_generators(seed_pair[0], seed_pair[1]),
    )
    return backend, result


def test_released_call_order_and_chunk_policy_shapes() -> None:
    backend, result = _run(101, 202)

    expected_order = [
        (chunk_index, timestep, finalize)
        for chunk_index in range(TINY_GEOMETRY.temporal_chunk_count)
        for timestep, finalize in (
            (1000, False),
            (757, False),
            (522, False),
            (0, True),
        )
    ]
    actual_order = [
        (call.chunk_index, call.timestep, call.finalize_cache) for call in backend.cached_calls
    ]
    assert actual_order == expected_order
    assert backend.cache_allocations == 1
    assert backend.full_prefix_calls == 0
    assert all(call.cache is backend.cache for call in backend.cached_calls)
    assert not any(call.grad_enabled for call in backend.cached_calls)

    # The t=0 cache write consumes the clean result of the third prediction;
    # it is deliberately not represented as another policy action.
    for chunk_index in range(TINY_GEOMETRY.temporal_chunk_count):
        third_prediction = backend.cached_calls[chunk_index * 4 + 2]
        finalization = backend.cached_calls[chunk_index * 4 + 3]
        torch.testing.assert_close(finalization.input, third_prediction.output)

    chunk_count = TINY_GEOMETRY.temporal_chunk_count
    latent_tail = (
        TINY_GEOMETRY.frames_per_chunk,
        TINY_GEOMETRY.latent_channels,
        TINY_GEOMETRY.latent_height,
        TINY_GEOMETRY.latent_width,
    )
    assert result.observations.shape == (BATCH_SIZE, chunk_count, 2, *latent_tail)
    assert result.actions.shape == (BATCH_SIZE, chunk_count, 2, *latent_tail)
    assert result.old_log_prob.shape == (BATCH_SIZE, chunk_count, 2)
    assert result.timesteps.shape == (BATCH_SIZE, chunk_count, 2)
    assert result.next_sigmas.shape == (BATCH_SIZE, chunk_count, 2)
    assert result.finalized_chunk_latents.shape == (
        BATCH_SIZE,
        chunk_count,
        *latent_tail,
    )
    assert result.latents.shape == (BATCH_SIZE, *TINY_GEOMETRY.latent_shape)
    assert torch.equal(result.timesteps[0, 0], torch.tensor([1000, 757]))
    assert torch.isfinite(result.old_log_prob).all()

    trajectory = result.trajectory_mapping(context={"request_id": "tiny"})
    assert trajectory["temporal_chunk_count"] == chunk_count
    assert trajectory["denoise_transition_count"] == 2
    assert trajectory["mask"].shape == (BATCH_SIZE, chunk_count, 2)
    assert trajectory["context"] == {"request_id": "tiny"}


def test_per_sample_generators_are_repeatable_and_row_local() -> None:
    _, baseline = _run(11, 29)
    _, repeated = _run(11, 29)
    _, second_seed_changed = _run(11, 31)

    torch.testing.assert_close(repeated.actions, baseline.actions, rtol=0, atol=0)
    torch.testing.assert_close(repeated.old_log_prob, baseline.old_log_prob, rtol=0, atol=0)
    torch.testing.assert_close(repeated.latents, baseline.latents, rtol=0, atol=0)

    # Changing row 1's generator cannot perturb row 0, including downstream
    # batches whose transformer inputs depend on earlier sampled transitions.
    torch.testing.assert_close(
        second_seed_changed.actions[0],
        baseline.actions[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        second_seed_changed.latents[0],
        baseline.latents[0],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(second_seed_changed.actions[1], baseline.actions[1])
    assert not torch.equal(second_seed_changed.latents[1], baseline.latents[1])
