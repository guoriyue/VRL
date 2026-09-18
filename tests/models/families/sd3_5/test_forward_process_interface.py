"""SD3.5 on the forward-process objective surface (DiffusionNFT / V-GRPO).

The objectives own no family-specific forward hook: they re-noise the clean
latent and hand it to the shared ``replay_forward_with_latents`` with
classifier-free guidance forced off. So the things SD3.5 must get right are
that this call reaches the real (tiny) ``SD3Transformer2DModel`` exactly as the
conditional ``forward_step`` branch does even when the rollout ran CFG, that
``latents_clean`` is exported, and that the frozen ``previous`` adapter attaches
and syncs through the shared ``DiffusionModelBase`` on a real PEFT transformer.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_SD3_JOINT_DIM,
    TINY_SD3_LATENT_SHAPE,
    TINY_SD3_POOLED_DIM,
    build_tiny_sd3_transformer,
    stamp_model_precision,
)
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.models.families.sd3_5.model import SD3_5Model, SD3_5ReplayModel, SD3SamplingState
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.builders import build_diffusion_trajectory

_TEXT_LEN = 3
_LORA_TARGETS = ["to_q", "to_v"]
_BATCH = TINY_SD3_LATENT_SHAPE[0]
_GUIDANCE = 2.0


def _model(transformer: torch.nn.Module) -> SD3_5Model:
    model = SD3_5Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _conditioning() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(_BATCH, _TEXT_LEN, TINY_SD3_JOINT_DIM),
        torch.randn(_BATCH, TINY_SD3_POOLED_DIM),
    )


def _cfg_rollout_batch(
    *,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_pooled: torch.Tensor,
    timestep: float,
) -> RolloutBatch:
    """A real one-step trajectory recorded under CFG, as the SD3.5 rollout exports it."""

    request = GenerationRequest(
        request_id="sd3-forward-process",
        family="sd3_5",
        task="t2i",
        inputs=["a test prompt"],
        samples_per_prompt=_BATCH,
    )
    rows = [
        GenerationSampleRow(
            prompt_index=0, sample_index=i, prompt="a test prompt", sample_id=f"s{i}"
        )
        for i in range(_BATCH)
    ]
    context = {"guidance_scale": _GUIDANCE, "cfg": True, "height": 64, "width": 64}
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=rows,
        observations=torch.zeros(_BATCH, 1, *TINY_SD3_LATENT_SHAPE[1:]),
        actions=torch.zeros(_BATCH, 1, *TINY_SD3_LATENT_SHAPE[1:]),
        old_log_prob=torch.zeros(_BATCH, 1),
        timesteps=torch.full((_BATCH, 1), timestep),
        kl=torch.zeros(_BATCH, 1),
        replay_tensors={
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled,
            "latents_clean": torch.zeros(TINY_SD3_LATENT_SHAPE),
        },
        context=context,
    )
    return RolloutBatch(
        rewards=torch.zeros(_BATCH),
        group_ids=torch.zeros(_BATCH, dtype=torch.long),
        context=context,
        trajectory=trajectory,
    )


def test_replay_forward_on_caller_latents_can_force_the_conditional_branch() -> None:
    """``classifier_free_guidance=False`` gives exactly the conditional ``forward_step``
    output on the caller's latent even though the trajectory recorded CFG; left at
    the default the rollout's CFG combine is reproduced instead."""
    torch.manual_seed(0)
    model = _model(build_tiny_sd3_transformer())
    prompt_embeds, pooled = _conditioning()
    negative_prompt_embeds, negative_pooled = _conditioning()
    xt = torch.randn(TINY_SD3_LATENT_SHAPE)
    batch = _cfg_rollout_batch(
        prompt_embeds=prompt_embeds,
        pooled=pooled,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled=negative_pooled,
        timestep=500.0,
    )

    def _state(do_cfg: bool) -> SD3SamplingState:
        return SD3SamplingState(
            latents=xt,
            timesteps=torch.tensor([[500.0] * _BATCH]),
            scheduler=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled,
            guidance_scale=_GUIDANCE,
            do_cfg=do_cfg,
        )

    with torch.no_grad():
        conditional = model.replay_forward_with_latents(
            batch, 0, xt, classifier_free_guidance=False
        )["noise_pred"]
        guided = model.replay_forward_with_latents(batch, 0, xt)["noise_pred"]
        via_step_cond = model.forward_step(_state(do_cfg=False), 0)["noise_pred"]
        via_step_cfg = model.forward_step(_state(do_cfg=True), 0)["noise_pred"]

    torch.testing.assert_close(conditional, via_step_cond)
    torch.testing.assert_close(guided, via_step_cfg)
    assert not torch.allclose(conditional, guided)


def test_replay_tensors_carry_the_final_latent_for_the_forward_process_objectives() -> None:
    model = _model(build_tiny_sd3_transformer())
    prompt_embeds, pooled = _conditioning()
    latents = torch.randn(TINY_SD3_LATENT_SHAPE)
    state = SD3SamplingState(
        latents=latents,
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        guidance_scale=1.0,
        do_cfg=False,
    )

    exported = model.export_replay_tensors(state)

    assert torch.equal(exported["latents_clean"], latents)
    assert exported["latents_clean"].requires_grad is False


def _peft_default_only_replay_model() -> SD3_5ReplayModel:
    from peft import LoraConfig, get_peft_model

    base = build_tiny_sd3_transformer()
    base.requires_grad_(False)
    peft_t = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, init_lora_weights="gaussian", target_modules=_LORA_TARGETS),
    )
    return SD3_5ReplayModel(transformer=peft_t, scheduler=None, device="cpu")


def test_previous_policy_adapter_attaches_frozen_and_syncs_through_the_shared_mixin() -> None:
    """The replay model (no pipeline) reaches ``DiffusionModelBase``'s attach/sync:
    a frozen ``previous`` mirror seeded from ``default`` and refreshed on sync."""
    model = _peft_default_only_replay_model()
    model.attach_previous_policy_adapter()

    named = dict(model.transformer.named_parameters())
    previous = {n: p for n, p in named.items() if ".previous." in n}
    assert previous and all(not p.requires_grad for p in previous.values())
    assert any(".default." in n and p.requires_grad for n, p in named.items())
    a_name = next(n for n in previous if "lora_A" in n)
    d_name = a_name.replace(".previous.", ".default.")
    assert torch.allclose(named[a_name], named[d_name])
    with torch.no_grad():
        named[d_name].add_(1.0)
    assert not torch.allclose(named[a_name], named[d_name])

    model.sync_previous_policy_adapter(decay=0.0)

    assert torch.allclose(named[a_name], named[d_name])
    assert model.transformer.active_adapter == "default"
