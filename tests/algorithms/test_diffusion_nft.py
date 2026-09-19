"""Loss-correctness tests for DiffusionNFT (algorithms/diffusion_nft.py).

DiffusionNFT runs the ``uses_evaluator=False`` trainer branch — a completely
separate code path from every GRPO test, involving a previous-policy forward
and a reference forward. A sign error here would train in reverse with nothing
catching it.

These tests use real collaborators, not stand-ins: a tiny real ``WanTransformer3DModel``,
either behind a real PEFT LoRA adapter or trained fully (the objective must not
tell the two apart), and a real ``TrajectoryBatch`` built by the production
``build_diffusion_trajectory``. So the three NFT branches differ because their
weights genuinely differ, and the
gradient that drives training flows through real attention parameters. The math
is not re-derived anywhere — the tests take one real optimizer step and assert
its *direction*: a good sample must pull the forward prediction toward the
flow-matching velocity that reconstructs the clean latent, a bad sample must push
it away. That catches a sign flip independently of how the loss is written.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import (
    TINY_WAN_LATENT_SHAPE,
    TINY_WAN_TEXT_DIM,
    TINY_WAN_TEXT_LEN,
    add_lora_adapters,
    build_tiny_wan_transformer,
)
from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.config.precision import RolePrecision
from vrl.generation.types import DenoiseRequest, GenerationRequest, GenerationSampleRow
from vrl.models.steps.denoise import DenoiseModelBase
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.builders import build_diffusion_trajectory

_BATCH = TINY_WAN_LATENT_SHAPE[0]
_LATENT_SHAPE = TINY_WAN_LATENT_SHAPE
_TEXT_LEN = TINY_WAN_TEXT_LEN
_TEXT_DIM = TINY_WAN_TEXT_DIM
_PRECISION = RolePrecision(
    dtype="fp32",
    float32_precision="ieee",
    outer_autocast=False,
)


@pytest.mark.parametrize("global_std", [False, True])
def test_diffusion_nft_advantages_match_grpo_contract(global_std: bool) -> None:
    """DiffusionNFT and GRPO share one group-relative advantage contract."""

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    grpo = GRPO(GRPOConfig(eps=1e-4, global_std=global_std))
    nft = DiffusionNFT(DiffusionNFTConfig(eps=1e-4, global_std=global_std))

    grpo_advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
    nft_advantages = nft.compute_advantages_from_tensors(rewards, group_ids)

    # The real contract: NFT reuses the GRPO group-relative advantage.
    assert torch.allclose(nft_advantages, grpo_advantages, atol=0.0, rtol=0.0)
    # Independent numeric oracle: the closed-form group-relative advantage for
    # this exact input, hand-derived (NOT via group_relative_advantages, which
    # is the function under test — re-calling it would be a tautology). Both
    # groups have mean-centered rewards [-1, 0, +1] over a population std.
    #   global_std=False: per-group std = sqrt(2/3) = 0.8164966 -> +-1.2247449
    #   global_std=True:  std over [1..6] = sqrt(35/12) = 1.7078251 -> +-0.5855400
    # adv_clip_max=5.0 does not bind. A change to eps placement, the unbiased
    # flag, or the clamp would move these and fail here.
    expected = (
        torch.tensor([-0.5855400, 0.0, 0.5855400, -0.5855400, 0.0, 0.5855400])
        if global_std
        else torch.tensor([-1.2247449, 0.0, 1.2247449, -1.2247449, 0.0, 1.2247449])
    )
    assert torch.allclose(grpo_advantages, expected, atol=1e-6)


class _NFTModel(DenoiseModelBase):
    """Holds a real Wan DiT behind the production policy boundary.

    ``transformer`` is a genuine ``WanTransformer3DModel``; ``previous_policy`` /
    ``reference_policy`` / ``sync_previous_policy`` are the inherited
    ``DenoiseModelBase`` implementations, so this double adds nothing to them.

    The objectives reach the transformer only through the shared replay
    contract (``replay_forward_with_latents`` -> ``restore_eval_state`` ->
    ``forward_step``), so this fake implements exactly that pair the way a
    single-transformer family does: the step's raw grid timestep is the Wan
    timestep, the prompt embeds are the only conditioning.
    """

    precision = _PRECISION

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.device = torch.device("cpu")

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        del batch_context, step_idx
        # ``timesteps`` arrives already sliced to this step: [B] raw grid values.
        return SimpleNamespace(
            latents=latents,
            timesteps=replay_tensors["timesteps"],
            prompt_embeds=replay_tensors["prompt_embeds"],
        )

    def forward_step(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        del step_idx
        prediction = self.transformer(
            hidden_states=state.latents,
            timestep=state.timesteps,
            encoder_hidden_states=state.prompt_embeds,
            return_dict=False,
        )[0]
        return {"noise_pred": prediction}

    def decode_latents(self, latents: Any) -> Any:
        raise NotImplementedError


def _build_model(trainable: str = "lora") -> _NFTModel:
    """A tiny real Wan DiT, trained through a LoRA adapter or fully."""

    transformer = build_tiny_wan_transformer()
    if trainable == "lora":
        add_lora_adapters(transformer)
    else:
        transformer.requires_grad_(True)
    model = _NFTModel(transformer)
    model.attach_reference_policy()
    return model


def _build_batch(
    *,
    x0: torch.Tensor,
    noise: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep: float | tuple[float, ...],
) -> RolloutBatch:
    """Build a real RolloutBatch carrying a real denoise TrajectoryBatch."""

    request = GenerationRequest(
        request_id="nft-test",
        family="wan",
        task="t2v",
        inputs=["a test prompt"],
        samples_per_prompt=_BATCH,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=0,
            prompt="a test prompt",
            sample_id="s0",
        )
    ]
    # Core role-triple tensors are required by the trajectory validator but
    # unused by NFT; the NFT inputs ride along as replay tensors.
    timestep_values = (timestep,) if isinstance(timestep, float) else timestep
    timestep_count = len(timestep_values)
    log_prob = torch.zeros(_BATCH, timestep_count)
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(_BATCH, timestep_count, *_LATENT_SHAPE[1:]),
        actions=torch.zeros(_BATCH, timestep_count, *_LATENT_SHAPE[1:]),
        old_log_prob=log_prob,
        timesteps=torch.tensor(timestep_values).repeat(_BATCH, 1),
        replay_tensors={
            "latents_clean": x0,
            "prompt_embeds": prompt_embeds,
            "diffusion_nft_noise": noise,
        },
        context={"num_frames": 1, "height": 4, "width": 4},
    )
    return RolloutBatch(
        rewards=torch.zeros(_BATCH),
        group_ids=torch.zeros(_BATCH, dtype=torch.long),
        context={"num_frames": 1, "height": 4, "width": 4},
        trajectory=trajectory,
    )


def test_nft_rejects_timestep_index_outside_trajectory() -> None:
    model = _build_model()
    batch = _build_batch(
        x0=torch.randn(_LATENT_SHAPE),
        noise=torch.randn(_LATENT_SHAPE),
        prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
        timestep=(250.0, 500.0),
    )

    with pytest.raises(RuntimeError, match="timestep_index out of range"):
        DiffusionNFT().compute_batch_timestep_loss(
            model,
            batch,
            2,
            torch.ones(_BATCH),
        )


def _default_forward(
    model: _NFTModel, xt: torch.Tensor, prompt_embeds: torch.Tensor, timestep: torch.Tensor
) -> torch.Tensor:
    """Run the trainable (default-adapter) forward on a known xt."""

    if getattr(model.transformer, "peft_config", None):
        model.transformer.set_adapter("default")
    return model.transformer(
        hidden_states=xt,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]


def _step_distances(
    *, advantage: float, trainable: str = "lora"
) -> tuple[float, float, torch.Tensor]:
    """One real backward+SGD step on the LoRA weights; report the distance of the
    default-adapter forward prediction to the reconstruction target before vs
    after. KL is off so only the policy term drives the sign.

    With ``nft_beta == 1`` the positive branch sets ``positive_x0 == x0`` iff the
    forward prediction equals ``noise - x0`` (the flow-matching velocity that
    reconstructs the clean latent) — so that is the target a good sample is
    pulled toward, independent of the loss's algebra.
    """

    cfg = DiffusionNFTConfig(nft_beta=1.0, kl_coef=0.0, advantage_scale=5.0)
    torch.manual_seed(1234)  # fix x0/noise so the gradient-direction check is reproducible
    x0 = torch.randn(_LATENT_SHAPE)
    noise = torch.randn(_LATENT_SHAPE)
    prompt_embeds = torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM)
    timestep = 500.0  # flow t = 0.5 after NFT's /1000 rescale
    t = 0.5
    target = noise - x0
    xt = (1 - t) * x0 + t * noise
    t_raw = torch.full((_BATCH,), timestep)

    model = _build_model(trainable)
    batch = _build_batch(x0=x0, noise=noise, prompt_embeds=prompt_embeds, timestep=timestep)

    before = float((_default_forward(model, xt, prompt_embeds, t_raw).detach() - target).norm())

    loss, _ = DiffusionNFT(cfg).compute_batch_timestep_loss(
        model,
        batch,
        0,
        torch.tensor([advantage]),
    )
    parameters = [p for p in model.transformer.parameters() if p.requires_grad]
    # LoRA weights sit at gaussian init and need a large step to move the
    # forward; the full transformer moves under a small one.
    opt = torch.optim.SGD(parameters, lr=50.0 if trainable == "lora" else 0.05)
    opt.zero_grad()
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in parameters if p.grad is not None])
    opt.step()

    after = float((_default_forward(model, xt, prompt_embeds, t_raw).detach() - target).norm())
    return before, after, grad


def test_nft_returns_only_objective_owned_step_metrics() -> None:
    """The trainer, not the NFT objective, owns batch and optimizer diagnostics."""

    torch.manual_seed(321)
    model = _build_model()
    batch = _build_batch(
        x0=torch.randn(_LATENT_SHAPE),
        noise=torch.randn(_LATENT_SHAPE),
        prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
        timestep=500.0,
    )

    _, metrics = DiffusionNFT(DiffusionNFTConfig()).compute_batch_timestep_loss(
        model,
        batch,
        0,
        torch.full((_BATCH,), 5.0),
    )

    # approx_kl is the historical CSV view of this exact reference-prediction
    # penalty, so it must reuse the computed KL scalar rather than run another MSE.
    assert metrics.update.approx_kl == metrics.kl_penalty
    assert metrics.advantage_mean == 0.0
    assert metrics.grad_norm == 0.0
    assert metrics.adv_zero_rate == 0.0
    assert metrics.adv_saturation == 0.0
    assert metrics.phase_times == {}


@pytest.mark.parametrize("trainable", ["lora", "full"])
def test_positive_advantage_trains_toward_reconstruction(trainable: str) -> None:
    # A good sample (high positive advantage) must pull the forward prediction
    # TOWARD the velocity that reconstructs the clean latent. The full-parameter
    # case also proves backward survives the two in-place policy swaps.
    before, after, grad = _step_distances(advantage=5.0, trainable=trainable)
    assert grad.abs().sum() > 0  # non-degenerate gradient through real params
    assert after < before


@pytest.mark.parametrize("trainable", ["lora", "full"])
def test_negative_advantage_trains_away_from_reconstruction(trainable: str) -> None:
    # A bad sample (negative advantage) must push the forward prediction AWAY
    # from the reconstruction target — the opposite sign.
    before, after, _ = _step_distances(advantage=-5.0, trainable=trainable)
    assert after > before


def test_nft_beta_must_be_positive() -> None:
    cfg = DiffusionNFTConfig(nft_beta=0.0)
    model = _build_model()
    batch = _build_batch(
        x0=torch.randn(_LATENT_SHAPE),
        noise=torch.randn(_LATENT_SHAPE),
        prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
        timestep=500.0,
    )
    with pytest.raises(RuntimeError, match="nft_beta must be > 0"):
        DiffusionNFT(cfg).compute_batch_timestep_loss(
            model,
            batch,
            0,
            torch.tensor([1.0]),
        )


def test_advantage_scale_must_be_positive() -> None:
    """Checks NFT advantage scale must be positive."""

    cfg = DiffusionNFTConfig(advantage_scale=0.0)
    model = _build_model()
    batch = _build_batch(
        x0=torch.randn(_LATENT_SHAPE),
        noise=torch.randn(_LATENT_SHAPE),
        prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
        timestep=500.0,
    )
    with pytest.raises(RuntimeError, match="advantage_scale must be > 0"):
        DiffusionNFT(cfg).compute_batch_timestep_loss(
            model,
            batch,
            0,
            torch.tensor([1.0]),
        )


@pytest.mark.parametrize("trainable", ["lora", "full"])
def test_after_optimizer_step_syncs_the_previous_policy(trainable: str) -> None:
    """The previous policy is the trainable weights as of the last sync; with
    ``weight_copy_decay=0`` a sync makes it an exact copy."""
    model = _build_model(trainable)
    model.sync_previous_policy()
    parameter = next(p for p in model.parameters() if p.requires_grad)
    synced = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1.0)

    with model.previous_policy():
        assert torch.equal(parameter, synced)
    assert torch.equal(parameter, synced + 1.0)

    DiffusionNFT(DiffusionNFTConfig(weight_copy_decay=0.0)).after_optimizer_step(
        model,
        global_step=7,
    )
    with model.previous_policy():
        assert torch.equal(parameter, synced + 1.0)


@pytest.mark.parametrize("trainable", ["lora", "full"])
def test_first_step_invariant_check_passes_when_previous_synced(trainable: str) -> None:
    """Advantage-flip invariant holds with previous freshly synced (lr=0 gate)."""

    model = _build_model(trainable)
    model.sync_previous_policy()
    batch = _build_batch(
        x0=torch.randn(_LATENT_SHAPE),
        noise=torch.randn(_LATENT_SHAPE),
        prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
        timestep=500.0,
    )

    record = DiffusionNFT(DiffusionNFTConfig()).first_step_invariant_check(
        model=model,
        batch=batch,
        advantages=torch.tensor([2.0]),
        timestep_index=0,
    )

    assert record["passed"] is True
    assert record["abs_diff"] <= record["threshold"]
    assert record["loss"] == pytest.approx(record["flipped_loss"], abs=1e-6)


def test_edm_scale_timestep_grid_fails_loudly() -> None:
    """Checks EDM-scale timestep grids cannot silently pass the /1000 heuristic.

    Cosmos Predict2's FlowMatch grid reaches 80000; after /1000 that is t=80,
    and xt = (1-t)*x0 + t*noise would leave the data manifold without a
    single warning — the same failure shape as the sigma-domain incident.
    """
    model = _build_model()
    batch = _build_batch(
        x0=torch.randn(_LATENT_SHAPE),
        noise=torch.randn(_LATENT_SHAPE),
        prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
        timestep=80000.0,
    )

    with pytest.raises(RuntimeError, match="EDM-scale"):
        DiffusionNFT(DiffusionNFTConfig()).compute_batch_timestep_loss(
            model,
            batch,
            0,
            torch.tensor([1.0]),
        )


def test_lr_zero_reward_channel_is_inert() -> None:
    """Checks the NFT analog of the GRPO ratio==1 invariant.

    With the previous policy synced to the trainable one (the lr=0 /
    just-synced state), forward == previous, so positive and negative
    branch losses coincide and the advantage mix cannot move the policy
    loss: flipping the advantage sign must leave the loss bit-identical.
    """
    torch.manual_seed(4321)
    x0 = torch.randn(_LATENT_SHAPE)
    noise = torch.randn(_LATENT_SHAPE)
    prompt_embeds = torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM)
    model = _build_model()
    # Sync previous <- default exactly (decay=0), as after_optimizer_step does.
    nft = DiffusionNFT(DiffusionNFTConfig(weight_copy_decay=0.0, kl_coef=0.0))
    nft.after_optimizer_step(model, global_step=0)
    batch = _build_batch(x0=x0, noise=noise, prompt_embeds=prompt_embeds, timestep=500.0)

    loss_pos, metrics_pos = nft.compute_batch_timestep_loss(
        model,
        batch,
        0,
        torch.tensor([5.0]),
    )
    loss_neg, metrics_neg = nft.compute_batch_timestep_loss(
        model,
        batch,
        0,
        torch.tensor([-5.0]),
    )

    assert metrics_pos.policy_loss == metrics_neg.policy_loss
    assert float(loss_pos) == float(loss_neg)
