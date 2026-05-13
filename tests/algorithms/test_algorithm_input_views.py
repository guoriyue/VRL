"""Tests for trajectory AlgorithmInput adapters."""

from __future__ import annotations

import pytest
import torch

from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig
from vrl.algorithms.dpo import DiffusionDPOConfig, diffusion_dpo_loss
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.grpo.multisegment import (
    MultiSegmentTokenGRPO,
    MultiSegmentTokenGRPOConfig,
)
from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.algorithms.trajectory import AlgorithmAdapter, AlgorithmInput
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.types import SegmentSignal, SignalBatch, TrajectorySignalBatch


def test_algorithm_input_rejects_legacy_signal_batch() -> None:
    with pytest.raises(TypeError, match="TrajectorySignalBatch"):
        AlgorithmInput(signals=SignalBatch(log_prob=torch.zeros(2)))


def test_algorithm_adapter_matches_continuous_grpo_signal_loss() -> None:
    rewards = torch.tensor([0.0, 1.0])
    group_ids = torch.tensor([0, 0])
    old_log_prob = torch.zeros(2)
    new_log_prob = torch.tensor([0.1, -0.2])
    direct_algorithm = GRPO(GRPOConfig(init_kl_coef=0.0, eps_clip=10.0))
    adapter_algorithm = GRPO(GRPOConfig(init_kl_coef=0.0, eps_clip=10.0))
    direct_advantages = direct_algorithm.compute_advantages_from_tensors(rewards, group_ids)
    legacy = SignalBatch(log_prob=new_log_prob, dist_family="flow_matching")
    direct_loss, direct_metrics = direct_algorithm.compute_signal_loss(
        legacy,
        direct_advantages,
        old_log_prob,
    )

    adapter_loss, adapter_metrics = AlgorithmAdapter().compute_loss(
        adapter_algorithm,
        AlgorithmInput(
            signals=TrajectorySignalBatch(
                segments={
                    "denoise": SegmentSignal(
                        name="denoise",
                        segment="denoise",
                        axis="timestep",
                        axes=("sample",),
                        distribution="flow_matching",
                        log_prob=new_log_prob,
                        old_log_prob=old_log_prob,
                        mask=torch.ones_like(old_log_prob),
                    )
                },
                group_ids=group_ids,
                primary_segment="denoise",
            ),
            rewards=rewards,
            group_ids=group_ids,
        ),
    )

    assert torch.allclose(adapter_loss, direct_loss)
    assert adapter_metrics.loss == direct_metrics.loss
    assert adapter_metrics.policy_loss == direct_metrics.policy_loss


def test_algorithm_adapter_matches_token_grpo_with_mask() -> None:
    old_log_prob = torch.zeros(2, 3)
    new_log_prob = torch.tensor([[0.0, 0.5, 0.5], [0.1, 0.2, 0.3]])
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    direct_algorithm = TokenGRPO(TokenGRPOConfig(init_kl_coef=0.0, eps_clip=10.0))
    adapter_algorithm = TokenGRPO(TokenGRPOConfig(init_kl_coef=0.0, eps_clip=10.0))
    legacy = SignalBatch(
        log_prob=new_log_prob,
        dist_family="categorical",
        aux={"token_mask": mask},
    )
    direct_loss, direct_metrics = direct_algorithm.compute_signal_loss(
        legacy,
        advantages,
        old_log_prob,
    )

    adapter_loss, adapter_metrics = AlgorithmAdapter().compute_loss(
        adapter_algorithm,
        AlgorithmInput(
            signals=TrajectorySignalBatch(
                segments={
                    "image_tokens": SegmentSignal(
                        name="image_tokens",
                        segment="image_tokens",
                        axis="token",
                        axes=("sample", "token"),
                        distribution="categorical",
                        log_prob=new_log_prob,
                        old_log_prob=old_log_prob,
                        mask=mask,
                    )
                },
                group_ids=torch.tensor([0, 0]),
                primary_segment="image_tokens",
            ),
            advantages=advantages,
        ),
    )

    assert torch.allclose(adapter_loss, direct_loss)
    assert adapter_metrics.policy_loss == direct_metrics.policy_loss
    assert adapter_metrics.clip_fraction == direct_metrics.clip_fraction


def test_algorithm_adapter_supports_multisegment_per_segment_advantages() -> None:
    old_initial = torch.zeros(2, 2)
    old_final = torch.zeros(2, 3)
    new_initial = torch.full((2, 2), 0.1)
    new_final = torch.full((2, 3), 0.3)
    adv_by_segment = {
        "initial_image": torch.tensor([1.0, 1.0]),
        "final_image": torch.tensor([-1.0, 1.0]),
    }
    signals = TrajectorySignalBatch(
        segments={
            "initial_image": SegmentSignal(
                name="initial_image",
                segment="initial_image",
                axis="token",
                axes=("sample", "token"),
                distribution="categorical",
                log_prob=new_initial,
                old_log_prob=old_initial,
                mask=torch.ones_like(old_initial),
            ),
            "final_image": SegmentSignal(
                name="final_image",
                segment="final_image",
                axis="token",
                axes=("sample", "token"),
                distribution="categorical",
                log_prob=new_final,
                old_log_prob=old_final,
                mask=torch.ones_like(old_final),
            ),
        },
        group_ids=torch.tensor([0, 0]),
        primary_segment="final_image",
    )
    cfg = MultiSegmentTokenGRPOConfig(
        init_kl_coef=0.0,
        eps_clip=10.0,
        segment_weights={
            "initial_image": 1.0,
            "selfcheck_text": 0.0,
            "final_image": 3.0,
        },
    )

    adapter_loss, adapter_metrics = AlgorithmAdapter().compute_loss(
        MultiSegmentTokenGRPO(cfg),
        AlgorithmInput(signals=signals, advantages=adv_by_segment),
    )

    base = TokenGRPO(TokenGRPOConfig(init_kl_coef=0.0, eps_clip=10.0))
    initial_loss, initial_metrics = base.compute_signal_loss(
        SignalBatch(
            log_prob=new_initial,
            dist_family="categorical",
            aux={"token_mask": torch.ones_like(old_initial)},
        ),
        adv_by_segment["initial_image"],
        old_initial,
    )
    final_loss, final_metrics = base.compute_signal_loss(
        SignalBatch(
            log_prob=new_final,
            dist_family="categorical",
            aux={"token_mask": torch.ones_like(old_final)},
        ),
        adv_by_segment["final_image"],
        old_final,
    )
    expected = (initial_loss + 3.0 * final_loss) / 4.0

    assert torch.allclose(adapter_loss, expected)
    assert adapter_metrics.policy_loss == (
        initial_metrics.policy_loss + 3.0 * final_metrics.policy_loss
    ) / 4.0


def test_algorithm_adapter_matches_diffusion_nft_batch_timestep_loss() -> None:
    advantages = torch.tensor([1.0, -1.0])
    direct_algorithm = DiffusionNFT(DiffusionNFTConfig(kl_beta=0.1))
    adapter_algorithm = DiffusionNFT(DiffusionNFTConfig(kl_beta=0.1))
    direct_loss, direct_metrics = direct_algorithm.compute_batch_timestep_loss(
        _FakeModel(),
        _nft_batch(),
        0,
        advantages,
    )

    adapter_loss, adapter_metrics = AlgorithmAdapter().compute_loss(
        adapter_algorithm,
        AlgorithmInput(
            advantages=advantages,
            metadata={
                "model": _FakeModel(),
                "rollout_batch": _nft_batch(),
                "timestep_index": 0,
            },
        ),
    )

    assert torch.allclose(adapter_loss, direct_loss)
    assert adapter_metrics.loss == direct_metrics.loss
    assert adapter_metrics.policy_loss == direct_metrics.policy_loss


def test_algorithm_adapter_matches_diffusion_dpo_loss() -> None:
    torch.manual_seed(0)
    model_pred = torch.randn(4, 2, 3, 3)
    ref_pred = torch.randn(4, 2, 3, 3)
    target = torch.randn(4, 2, 3, 3)
    config = DiffusionDPOConfig(beta=123.0)

    adapter_loss, adapter_metrics = AlgorithmAdapter().compute_loss(
        config,
        AlgorithmInput(
            metadata={
                "model_pred": model_pred,
                "ref_pred": ref_pred,
                "target": target,
            },
        ),
    )
    expected = diffusion_dpo_loss(model_pred, ref_pred, target, beta=config.beta)

    assert torch.allclose(adapter_loss, expected["loss"])
    assert adapter_metrics.loss == expected["loss"].item()
    assert adapter_metrics.policy_loss == expected["loss"].item()


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.default_scale = torch.nn.Parameter(torch.tensor(0.4))
        self.previous_scale = torch.nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.base_scale = torch.nn.Parameter(torch.tensor(0.2), requires_grad=False)
        self.active = "default"
        self.disabled = False

    def set_adapter(self, name: str) -> None:
        self.active = name
        self.disabled = False

    def disable_adapters(self) -> None:
        self.disabled = True

    def enable_adapters(self) -> None:
        self.disabled = False

    def forward(self, **kwargs):
        hidden_states = kwargs["hidden_states"]
        if self.disabled:
            scale = self.base_scale
        elif self.active == "previous":
            scale = self.previous_scale
        else:
            scale = self.default_scale
        return (hidden_states * scale,)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _FakeTransformer()

    def diffusion_nft_prepare_transformer_input(self, **kwargs):
        return {
            "hidden_states": kwargs["latents"],
            "timestep": kwargs["timestep"],
            "encoder_hidden_states": kwargs["prompt_embeds"],
            "condition_mask": torch.zeros_like(kwargs["latents"][:, :1]),
            "padding_mask": torch.zeros(1, 1, kwargs["height"], kwargs["width"]),
            "return_dict": False,
        }


def _nft_batch() -> RolloutBatch:
    x0 = torch.ones(2, 1, 2, 2, 2)
    return RolloutBatch(
        observations=x0[:, None],
        actions=x0[:, None],
        rewards=torch.tensor([0.0, 1.0]),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        extras={
            "latents_clean": x0,
            "timesteps": torch.tensor([[500.0], [500.0]]),
            "prompt_embeds": torch.zeros(2, 4, 8),
            "diffusion_nft_noise": torch.zeros_like(x0),
        },
        context={"height": 16, "width": 16, "num_frames": 9},
        prompts=["a", "a"],
    )
