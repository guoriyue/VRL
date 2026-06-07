"""Loss-correctness tests for DiffusionNFT (algorithms/diffusion_nft.py).

DiffusionNFT runs the ``uses_evaluator=False`` trainer branch — a completely
separate code path from every GRPO test, involving a previous-policy adapter
forward and a reference forward. A sign error here would train in reverse with
nothing catching it.

These tests use real collaborators, not stand-ins: a tiny real ``WanTransformer3DModel``
wrapped in real PEFT LoRA (``default`` + ``previous`` adapters, exactly as the
production model does in ``cosmos/predict2_5/model.py``), and a real
``TrajectoryBatch`` built by the production ``build_diffusion_trajectory``. So the
three NFT branches differ because their LoRA weights genuinely differ, and the
gradient that drives training flows through real attention parameters. The math
is not re-derived anywhere — the tests take one real optimizer step and assert
its *direction*: a good sample must pull the forward prediction toward the
flow-matching velocity that reconstructs the clean latent, a bad sample must push
it away. That catches a sign flip independently of how the loss is written.
"""

from __future__ import annotations

import pytest
import torch

from tests.models.diffusion.fixtures import (
    TINY_WAN_LATENT_SHAPE,
    TINY_WAN_TEXT_DIM,
    TINY_WAN_TEXT_LEN,
    add_lora_adapters,
    build_tiny_wan_transformer,
)
from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.diffusion.cosmos.predict2_5.model import _copy_adapter_weights
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.builders import build_diffusion_trajectory

_BATCH = TINY_WAN_LATENT_SHAPE[0]
_LATENT_SHAPE = TINY_WAN_LATENT_SHAPE
_TEXT_LEN = TINY_WAN_TEXT_LEN
_TEXT_DIM = TINY_WAN_TEXT_DIM


@pytest.mark.parametrize(
    ("global_std", "expected"),
    [
        (
            False,
            torch.tensor(
                [-1.2247449, 0.0, 1.2247449, -1.2247449, 0.0, 1.2247449],
            ),
        ),
        (
            True,
            torch.tensor(
                [-0.5855400, 0.0, 0.5855400, -0.5855400, 0.0, 0.5855400],
            ),
        ),
    ],
)
def test_diffusion_nft_advantages_match_grpo_contract(
    global_std: bool,
    expected: torch.Tensor,
) -> None:
    """DiffusionNFT and GRPO share one group-relative advantage contract."""

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    grpo = GRPO(GRPOConfig(eps=1e-4, global_std=global_std))
    nft = DiffusionNFT(DiffusionNFTConfig(eps=1e-4, global_std=global_std))

    grpo_advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
    nft_advantages = nft.compute_advantages_from_tensors(rewards, group_ids)

    assert torch.allclose(grpo_advantages, expected, atol=1e-6)
    assert torch.allclose(nft_advantages, grpo_advantages, atol=0.0, rtol=0.0)


class _NFTModel:
    """Holds a real PEFT-wrapped Wan DiT and the two methods NFT calls on a model.

    ``transformer`` is a genuine ``WanTransformer3DModel`` carrying real
    ``default`` and ``previous`` LoRA adapters, so ``set_adapter`` /
    ``disable_adapters`` route to actually-different weights. ``sync_previous_policy_adapter``
    delegates to the same ``_copy_adapter_weights`` the production model uses.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        self.transformer = transformer

    def diffusion_nft_prepare_transformer_input(
        self,
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, object]:
        # Real kwargs the Wan transformer consumes; return_dict=False so the
        # forward yields a (sample,) tuple, matching NFT's transformer(...)[0].
        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "return_dict": False,
        }

    def sync_previous_policy_adapter(self, *, decay: float) -> None:
        _copy_adapter_weights(self.transformer, src="default", dst="previous", decay=decay)


def _build_model() -> _NFTModel:
    """A tiny real Wan DiT carrying distinct default/previous LoRA adapters."""

    return _NFTModel(add_lora_adapters(build_tiny_wan_transformer()))


def _build_batch(
    *,
    x0: torch.Tensor,
    noise: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep: float,
) -> RolloutBatch:
    """Build a real RolloutBatch carrying a real denoise TrajectoryBatch."""

    request = GenerationRequest(
        request_id="nft-test",
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
    # Core role-triple tensors are required by the trajectory validator but
    # unused by NFT; the NFT inputs ride along as replay tensors.
    log_prob = torch.zeros(_BATCH, 1)
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(_BATCH, 1, *_LATENT_SHAPE[1:]),
        actions=torch.zeros(_BATCH, 1, *_LATENT_SHAPE[1:]),
        old_log_prob=log_prob,
        timesteps=torch.full((_BATCH, 1), timestep),
        kl=torch.zeros(_BATCH, 1),
        replay_tensors={
            "latents_clean": x0,
            "prompt_embeds": prompt_embeds,
            "diffusion_nft_noise": noise,
        },
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


def _default_forward(model: _NFTModel, xt: torch.Tensor, prompt_embeds: torch.Tensor,
                     timestep: torch.Tensor) -> torch.Tensor:
    """Run the trainable (default-adapter) forward on a known xt."""

    model.transformer.set_adapter("default")
    return model.transformer(
        hidden_states=xt, timestep=timestep, encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]


def _step_distances(*, advantage: float) -> tuple[float, float, torch.Tensor]:
    """One real backward+SGD step on the LoRA weights; report the distance of the
    default-adapter forward prediction to the reconstruction target before vs
    after. KL is off so only the policy term drives the sign.

    With ``nft_beta == 1`` the positive branch sets ``positive_x0 == x0`` iff the
    forward prediction equals ``noise - x0`` (the flow-matching velocity that
    reconstructs the clean latent) — so that is the target a good sample is
    pulled toward, independent of the loss's algebra.
    """

    cfg = DiffusionNFTConfig(nft_beta=1.0, kl_beta=0.0, advantage_high=5.0, advantage_low=-5.0)
    torch.manual_seed(1234)  # fix x0/noise so the gradient-direction check is reproducible
    x0 = torch.randn(_LATENT_SHAPE)
    noise = torch.randn(_LATENT_SHAPE)
    prompt_embeds = torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM)
    timestep = 500.0  # flow t = 0.5 after NFT's /1000 rescale
    t = 0.5
    target = noise - x0
    xt = (1 - t) * x0 + t * noise
    t_raw = torch.full((_BATCH,), timestep)

    model = _build_model()
    batch = _build_batch(x0=x0, noise=noise, prompt_embeds=prompt_embeds, timestep=timestep)

    before = float((_default_forward(model, xt, prompt_embeds, t_raw).detach() - target).norm())

    loss, _ = DiffusionNFT(cfg).compute_batch_timestep_loss(
        model, batch, 0, torch.tensor([advantage]),
    )
    trainable = [p for p in model.transformer.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=50.0)
    opt.zero_grad()
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in trainable if p.grad is not None])
    opt.step()

    after = float((_default_forward(model, xt, prompt_embeds, t_raw).detach() - target).norm())
    return before, after, grad


def test_positive_advantage_trains_toward_reconstruction() -> None:
    # A good sample (high positive advantage) must pull the forward prediction
    # TOWARD the velocity that reconstructs the clean latent.
    """Checks positive advantage trains toward reconstruction."""
    before, after, grad = _step_distances(advantage=5.0)
    assert grad.abs().sum() > 0  # non-degenerate gradient through real LoRA params
    assert after < before


def test_negative_advantage_trains_away_from_reconstruction() -> None:
    # A bad sample (negative advantage) must push the forward prediction AWAY
    # from the reconstruction target — the opposite sign.
    """Checks negative advantage trains away from reconstruction."""
    before, after, _ = _step_distances(advantage=-5.0)
    assert after > before


def test_nft_beta_must_be_positive() -> None:
    """Checks NFT beta must be positive."""
    cfg = DiffusionNFTConfig(nft_beta=0.0)
    model = _build_model()
    batch = _build_batch(
        x0=torch.randn(_LATENT_SHAPE),
        noise=torch.randn(_LATENT_SHAPE),
        prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
        timestep=500.0,
    )
    with pytest.raises(RuntimeError, match="nft_beta must be > 0"):
        DiffusionNFT(cfg).compute_batch_timestep_loss(model, batch, 0, torch.tensor([1.0]))


def test_after_optimizer_step_syncs_previous_adapter() -> None:
    # after_optimizer_step must refresh the previous adapter from the trainable
    # one. With decay=0 the previous weights become an exact copy of default.
    """Checks after optimizer step syncs previous adapter."""
    model = _build_model()
    named = dict(model.transformer.named_parameters())
    a_name = next(n for n in named if ".previous." in n and "lora_A" in n)
    d_name = a_name.replace(".previous.", ".default.")
    # Perturb default so it differs from previous before the sync.
    with torch.no_grad():
        named[d_name].add_(1.0)
    assert not torch.allclose(named[a_name], named[d_name])

    DiffusionNFT(DiffusionNFTConfig(weight_copy_decay=0.0)).after_optimizer_step(
        model, global_step=7,
    )
    assert torch.allclose(named[a_name], named[d_name])
