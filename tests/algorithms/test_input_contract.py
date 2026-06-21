"""Algorithm input contract: AlgorithmAdapter.validate_inputs.

A single declarative gate replaces the per-loss inline checks each objective
used to hand-roll. An algorithm declares ``required_signal_keys`` (SegmentSignal
fields, signal branch) and/or ``required_data_keys`` (replay-tensor names,
replay branch); the adapter fails fast with available-vs-missing diagnostics,
mirroring verl-omni's ``DiffusionLossFn.validate_inputs``.
"""

from __future__ import annotations

import pytest
import torch

from tests.models.diffusion.fixtures import (
    TINY_WAN_LATENT_SHAPE,
    TINY_WAN_TEXT_DIM,
    TINY_WAN_TEXT_LEN,
)
from vrl.algorithms.diffusion_nft import DiffusionNFT
from vrl.algorithms.grpo.continuous import GRPO
from vrl.algorithms.trajectory import AlgorithmAdapter, AlgorithmInput
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch
from vrl.trajectory.builders import build_diffusion_trajectory

_BATCH = TINY_WAN_LATENT_SHAPE[0]
_LATENT_SHAPE = TINY_WAN_LATENT_SHAPE


def _grpo_signals(*, old_log_prob: torch.Tensor | None) -> TrajectorySignalBatch:
    """A flow-matching signal batch; ``old_log_prob=None`` omits that field."""

    return TrajectorySignalBatch(
        segments={
            "denoise": SegmentSignal(
                name="denoise",
                distribution="flow_matching",
                log_prob=torch.zeros(_BATCH),
                old_log_prob=old_log_prob,
                mask=torch.ones(_BATCH),
            ),
        },
        group_ids=torch.arange(_BATCH),
        primary_segment="denoise",
    )


def _nft_batch(*, latents_clean: torch.Tensor | None) -> RolloutBatch:
    """A real denoise TrajectoryBatch; ``latents_clean=None`` omits that key."""

    request = GenerationRequest(
        request_id="nft-contract",
        family="wan",
        task="t2v",
        prompts=["a test prompt"],
        samples_per_prompt=_BATCH,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=0,
            prompt="a test prompt",
            prompt_id="p0",
            group_id="g0",
            sample_id="s0",
            trajectory_id="t0",
            seed=0,
        )
    ]
    replay_tensors: dict[str, torch.Tensor] = {
        "prompt_embeds": torch.randn(_BATCH, TINY_WAN_TEXT_LEN, TINY_WAN_TEXT_DIM),
    }
    if latents_clean is not None:
        replay_tensors["latents_clean"] = latents_clean
    log_prob = torch.zeros(_BATCH, 1)
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(_BATCH, 1, *_LATENT_SHAPE[1:]),
        actions=torch.zeros(_BATCH, 1, *_LATENT_SHAPE[1:]),
        old_log_prob=log_prob,
        timesteps=torch.full((_BATCH, 1), 500.0),
        kl=torch.zeros(_BATCH, 1),
        replay_tensors=replay_tensors,
        context={"num_frames": 1, "height": 4, "width": 4},
    )
    return RolloutBatch(
        observations=None,
        actions=None,
        rewards=torch.zeros(_BATCH),
        dones=torch.zeros(_BATCH),
        group_ids=torch.zeros(_BATCH, dtype=torch.long),
        context={"num_frames": 1, "height": 4, "width": 4},
        trajectory=trajectory,
    )


def test_signal_branch_reports_missing_and_available() -> None:
    """A GRPO signal missing old_log_prob fails fast, naming missing + available."""

    inputs = AlgorithmInput(
        signals=_grpo_signals(old_log_prob=None),
        advantages=torch.ones(_BATCH),
    )
    with pytest.raises(KeyError) as exc:
        AlgorithmAdapter().compute_loss(GRPO(), inputs)
    msg = str(exc.value)
    assert "Missing signal keys: ['old_log_prob']" in msg
    assert "Available signal keys:" in msg
    # log_prob + mask are present and must show up as available.
    assert "mask" in msg


def test_replay_branch_reports_missing_and_available() -> None:
    """An NFT batch missing latents_clean fails fast, naming missing + available."""

    inputs = AlgorithmInput(
        advantages=torch.zeros(_BATCH),
        group_ids=torch.zeros(_BATCH, dtype=torch.long),
        metadata={"rollout_batch": _nft_batch(latents_clean=None), "timestep_index": 0},
    )
    with pytest.raises(KeyError) as exc:
        AlgorithmAdapter().compute_loss(DiffusionNFT(), inputs)
    msg = str(exc.value)
    assert "Missing data keys: ['latents_clean']" in msg
    assert "Available data keys:" in msg
    assert "prompt_embeds" in msg
    assert "timesteps" in msg


def test_well_formed_inputs_pass_validation() -> None:
    """The gate is silent on the happy path for both branches."""

    adapter = AlgorithmAdapter()
    adapter.validate_inputs(
        GRPO(),
        AlgorithmInput(
            signals=_grpo_signals(old_log_prob=torch.zeros(_BATCH)),
            advantages=torch.ones(_BATCH),
        ),
    )
    adapter.validate_inputs(
        DiffusionNFT(),
        AlgorithmInput(
            advantages=torch.zeros(_BATCH),
            metadata={"rollout_batch": _nft_batch(latents_clean=torch.randn(*_LATENT_SHAPE))},
        ),
    )
