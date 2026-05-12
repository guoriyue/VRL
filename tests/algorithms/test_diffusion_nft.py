from __future__ import annotations

import pytest
import torch

from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig
from vrl.rollouts.batch import RolloutBatch


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


class _PeftLikeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.lora_A = torch.nn.ModuleDict(
            {
                "default": torch.nn.Linear(1, 1, bias=False),
                "previous": torch.nn.Linear(1, 1, bias=False),
            }
        )
        self.block.lora_A["default"].weight.data.fill_(3.0)
        self.block.lora_A["previous"].weight.data.fill_(0.0)


class _PeftLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _PeftLikeTransformer()

    def sync_previous_policy_adapter(self, *, decay: float = 0.0) -> None:
        previous = self.transformer.block.lora_A["previous"].weight
        default = self.transformer.block.lora_A["default"].weight
        previous.data.mul_(decay).add_(default.data, alpha=1.0 - decay)


def _batch() -> RolloutBatch:
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


def test_diffusion_nft_loss_backpropagates_through_default_adapter() -> None:
    alg = DiffusionNFT(DiffusionNFTConfig(kl_beta=0.1))
    model = _FakeModel()
    batch = _batch()
    advantages = torch.tensor([1.0, -1.0])

    loss, metrics = alg.compute_batch_timestep_loss(model, batch, 0, advantages)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics.loss > 0
    assert model.transformer.default_scale.grad is not None
    assert model.transformer.default_scale.grad.abs().item() > 0


def test_diffusion_nft_fails_fast_without_required_latents() -> None:
    alg = DiffusionNFT()
    model = _FakeModel()
    batch = _batch()
    del batch.extras["latents_clean"]

    with pytest.raises(RuntimeError, match="latents_clean"):
        alg.compute_batch_timestep_loss(model, batch, 0, torch.ones(2))


def test_diffusion_nft_rejects_signal_loss_path() -> None:
    alg = DiffusionNFT()

    with pytest.raises(RuntimeError, match="does not consume evaluator"):
        alg.compute_signal_loss(None, None, None)  # type: ignore[arg-type]


def test_diffusion_nft_after_optimizer_step_syncs_previous_policy_adapter() -> None:
    alg = DiffusionNFT(DiffusionNFTConfig(weight_copy_decay=0.0))
    model = _PeftLikeModel()

    alg.after_optimizer_step(model, global_step=1)

    assert model.transformer.block.lora_A["previous"].weight.item() == pytest.approx(3.0)


def test_diffusion_nft_after_optimizer_step_requires_policy_sync_hook() -> None:
    alg = DiffusionNFT()

    with pytest.raises(RuntimeError, match="sync_previous_policy_adapter"):
        alg.after_optimizer_step(torch.nn.Module(), global_step=1)
