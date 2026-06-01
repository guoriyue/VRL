"""Loss-correctness tests for DiffusionNFT (algorithms/diffusion_nft.py).

DiffusionNFT runs the ``uses_evaluator=False`` trainer branch — a completely
separate code path from every GRPO test, involving a previous-policy adapter
forward and a reference forward. A sign or normalization error here would train
in reverse with nothing catching it. These tests pin the loss to an independent
re-implementation of the same formula and exercise the adapter-branch wiring.

The trajectory-resolution layer has its own tests, so we patch
``TrajectoryResolver.from_batch`` to feed known replay tensors and keep the focus
on the NFT math itself.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

import vrl.trajectory as trajectory_mod
from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig
from vrl.math.diffusion.nft import normalized_mse


class _FakeTransformer:
    """Transformer whose output depends only on the active adapter state.

    ``mode`` is set by set_adapter / disable_adapters and read at call time, so
    each NFT branch (previous / default / reference) yields a distinct constant.
    """

    def __init__(self, outputs: dict[str, torch.Tensor]) -> None:
        self._outputs = outputs
        self.mode = "default"
        self.calls: list[str] = []

    def set_adapter(self, name: str) -> None:
        self.mode = name

    def disable_adapters(self) -> None:
        self.mode = "reference"

    def enable_adapters(self) -> None:
        self.mode = "default"

    def __call__(self, **_inputs: object) -> tuple[torch.Tensor]:
        self.calls.append(self.mode)
        return (self._outputs[self.mode].clone(),)


class _FakeModel:
    def __init__(self, transformer: _FakeTransformer) -> None:
        self.transformer = transformer
        self.synced_decay: float | None = None

    def diffusion_nft_prepare_transformer_input(self, **_kwargs: object) -> dict[str, object]:
        # Content is irrelevant: the fake transformer ignores its inputs.
        return {"hidden_states": torch.zeros(1)}

    def sync_previous_policy_adapter(self, *, decay: float) -> None:
        self.synced_decay = decay


class _FakeResolver:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    @classmethod
    def for_tensors(cls, tensors: dict[str, torch.Tensor]):
        def _from_batch(_batch: object) -> _FakeResolver:
            return cls(tensors)

        return _from_batch

    def replay_tensor_dict(self, _segment: str | None = None, **_kw: object) -> dict[str, object]:
        return dict(self._tensors)


class _FakeBatch:
    context: ClassVar[dict[str, int]] = {"num_frames": 1, "height": 0, "width": 0}


def _reference_loss(
    *,
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: float,
    forward: torch.Tensor,
    previous: torch.Tensor,
    reference: torch.Tensor,
    advantage: float,
    cfg: DiffusionNFTConfig,
) -> tuple[float, float, float]:
    """Independent re-derivation of the NFT loss for parity checks."""
    xt = (1 - t) * x0 + t * noise
    adv = max(min(advantage, cfg.advantage_high), cfg.advantage_low)
    reward_mix = min(max((adv / cfg.advantage_high) / 2.0 + 0.5, 0.0), 1.0)
    beta = cfg.nft_beta
    positive_pred = beta * forward + (1 - beta) * previous
    negative_pred = (1 + beta) * previous - beta * forward
    positive_x0 = xt - t * positive_pred
    negative_x0 = xt - t * negative_pred
    pos_loss = normalized_mse(positive_x0, x0)
    neg_loss = normalized_mse(negative_x0, x0)
    policy = (reward_mix * pos_loss / beta + (1 - reward_mix) * neg_loss / beta)
    policy_loss = float(policy.mean().item()) * cfg.advantage_high
    kl_loss = float(((forward - reference) ** 2).mean().item())
    return policy_loss + cfg.kl_beta * kl_loss, policy_loss, kl_loss


def test_loss_matches_independent_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DiffusionNFTConfig(nft_beta=1.0, kl_beta=1.0, advantage_high=5.0, advantage_low=-5.0)
    x0 = torch.tensor([[1.0, 2.0]])
    noise = torch.tensor([[0.5, 0.5]])
    t = 0.4
    forward = torch.tensor([[0.1, 0.2]])
    previous = torch.tensor([[0.3, 0.1]])
    reference = torch.tensor([[0.0, 0.0]])

    transformer = _FakeTransformer(
        {"default": forward, "previous": previous, "reference": reference},
    )
    model = _FakeModel(transformer)
    tensors = {
        "latents_clean": x0,
        "prompt_embeds": torch.zeros(1, 3),
        "timesteps": torch.tensor([t]),
        "diffusion_nft_noise": noise,
    }
    monkeypatch.setattr(
        trajectory_mod, "TrajectoryResolver", type(
            "_Patched", (), {"from_batch": staticmethod(_FakeResolver.for_tensors(tensors))},
        ),
    )

    algo = DiffusionNFT(cfg)
    loss, metrics = algo.compute_batch_timestep_loss(
        model, _FakeBatch(), 0, torch.tensor([2.0]),
    )

    exp_loss, exp_policy, exp_kl = _reference_loss(
        x0=x0, noise=noise, t=t, forward=forward, previous=previous,
        reference=reference, advantage=2.0, cfg=cfg,
    )
    assert loss.item() == pytest.approx(exp_loss, rel=1e-5)
    assert metrics.policy_loss == pytest.approx(exp_policy, rel=1e-5)
    assert metrics.kl_penalty == pytest.approx(exp_kl, rel=1e-5)
    # All three adapter branches were actually exercised.
    assert "previous" in transformer.calls
    assert "default" in transformer.calls
    assert "reference" in transformer.calls


def test_nft_beta_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DiffusionNFTConfig(nft_beta=0.0)
    x0 = torch.tensor([[1.0, 2.0]])
    transformer = _FakeTransformer(
        {k: torch.zeros(1, 2) for k in ("default", "previous", "reference")},
    )
    model = _FakeModel(transformer)
    tensors = {
        "latents_clean": x0,
        "prompt_embeds": torch.zeros(1, 3),
        "timesteps": torch.tensor([0.4]),
        "diffusion_nft_noise": torch.zeros(1, 2),
    }
    monkeypatch.setattr(
        trajectory_mod, "TrajectoryResolver", type(
            "_Patched", (), {"from_batch": staticmethod(_FakeResolver.for_tensors(tensors))},
        ),
    )
    algo = DiffusionNFT(cfg)
    with pytest.raises(RuntimeError, match="nft_beta must be > 0"):
        algo.compute_batch_timestep_loss(model, _FakeBatch(), 0, torch.tensor([1.0]))


def test_after_optimizer_step_syncs_previous_adapter() -> None:
    cfg = DiffusionNFTConfig(weight_copy_decay=0.25)
    transformer = _FakeTransformer({"default": torch.zeros(1)})
    model = _FakeModel(transformer)
    DiffusionNFT(cfg).after_optimizer_step(model, global_step=7)
    assert model.synced_decay == pytest.approx(0.25)


def _forward_grad_step_distances(
    *,
    advantage: float,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[float, float, torch.Tensor]:
    """Run one backward pass with a *trainable* forward prediction and report the
    distance of that prediction to the reconstruction target before vs after a
    gradient-descent step. KL is disabled so only the policy term drives the sign.

    With ``nft_beta == 1`` the positive branch sets ``positive_pred == forward``
    and ``positive_x0 == x0`` iff ``forward == noise - x0`` (the flow-matching
    velocity that reconstructs the clean latent). So that ``noise - x0`` is the
    target a *good* sample should be pulled toward.
    """

    cfg = DiffusionNFTConfig(
        nft_beta=1.0, kl_beta=0.0, advantage_high=5.0, advantage_low=-5.0
    )
    x0 = torch.tensor([[1.0, 2.0]])
    noise = torch.tensor([[0.5, 0.5]])
    t = 0.4
    forward = torch.zeros(1, 2, requires_grad=True)
    previous = torch.tensor([[0.3, 0.1]])
    reference = torch.zeros(1, 2)
    transformer = _FakeTransformer(
        {"default": forward, "previous": previous, "reference": reference},
    )
    model = _FakeModel(transformer)
    tensors = {
        "latents_clean": x0,
        "prompt_embeds": torch.zeros(1, 3),
        "timesteps": torch.tensor([t]),
        "diffusion_nft_noise": noise,
    }
    monkeypatch.setattr(
        trajectory_mod, "TrajectoryResolver", type(
            "_Patched", (), {"from_batch": staticmethod(_FakeResolver.for_tensors(tensors))},
        ),
    )
    loss, _ = DiffusionNFT(cfg).compute_batch_timestep_loss(
        model, _FakeBatch(), 0, torch.tensor([advantage]),
    )
    loss.backward()
    target = noise - x0
    before = float((forward.detach() - target).norm().item())
    after = float((forward.detach() - 0.1 * forward.grad - target).norm().item())
    return before, after, forward.grad


def test_positive_advantage_trains_toward_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A good sample (high positive advantage) must pull the forward prediction
    # TOWARD the velocity that reconstructs the clean latent. This catches a
    # conceptual sign flip the parity test cannot (both would re-derive it wrong).
    before, after, grad = _forward_grad_step_distances(advantage=5.0, monkeypatch=monkeypatch)
    assert grad.abs().sum() > 0  # non-degenerate gradient
    assert after < before


def test_negative_advantage_trains_away_from_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bad sample (negative advantage) must push the forward prediction AWAY
    # from the reconstruction target — the opposite sign.
    before, after, _ = _forward_grad_step_distances(advantage=-5.0, monkeypatch=monkeypatch)
    assert after > before
