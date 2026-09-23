"""SD3.5 on the forward-process objective surface (DiffusionNFT / V-GRPO).

The objectives own no family-specific forward hook: they re-noise the clean
latent and hand it to the shared ``replay_forward_with_latents`` with
classifier-free guidance forced off. So the things SD3.5 must get right are
that this call reaches the real (tiny) ``SD3Transformer2DModel`` exactly as the
conditional ``forward_step`` branch does even when the rollout ran CFG, that
``latents_clean`` is exported, and that the frozen ``previous`` adapter attaches
and syncs through the shared ``DenoiseModelBase`` on a real PEFT transformer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.models.steps.denoise.fixtures import (
    TINY_SD3_JOINT_DIM,
    TINY_SD3_LATENT_SHAPE,
    TINY_SD3_POOLED_DIM,
    build_tiny_transformer,
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
    model = _model(build_tiny_transformer("sd3"))
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
    model = _model(build_tiny_transformer("sd3"))
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


def _replay_model(trainable: str) -> SD3_5ReplayModel:
    """A replay model (no pipeline) whose trainable weights are a LoRA adapter or the DiT."""

    base = build_tiny_transformer("sd3")
    if trainable == "lora":
        from peft import LoraConfig, get_peft_model

        base.requires_grad_(False)
        base = get_peft_model(
            base,
            LoraConfig(
                r=4, lora_alpha=8, init_lora_weights="gaussian", target_modules=_LORA_TARGETS
            ),
        )
    else:
        base.requires_grad_(True)
    return SD3_5ReplayModel(transformer=base, scheduler=None, device="cpu")


@pytest.mark.parametrize("trainable", ["lora", "full"])
def test_previous_policy_is_a_snapshot_of_the_trainable_weights(trainable: str) -> None:
    """``DenoiseModelBase`` owns the previous policy for both trainable shapes:
    a snapshot taken at the first sync, swapped in for the forward, refreshed
    on sync; the live weights are untouched outside the context."""
    model = _replay_model(trainable)
    parameter = next(p for p in model.parameters() if p.requires_grad)
    model.sync_previous_policy()
    synced = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1.0)

    with model.previous_policy():
        assert torch.equal(parameter, synced)
    assert torch.equal(parameter, synced + 1.0)

    model.sync_previous_policy(decay=0.5)
    with model.previous_policy():
        assert torch.allclose(parameter, synced + 0.5)
    assert torch.equal(parameter, synced + 1.0)
    if trainable == "lora":
        assert model.transformer.active_adapter == "default"


def test_reference_policy_is_the_base_under_an_adapter_and_a_snapshot_otherwise() -> None:
    from peft.tuners.lora import LoraLayer

    lora = _replay_model("lora")
    lora.attach_reference_policy()
    assert lora._modules.get("_reference_policy") is None
    with lora.reference_policy():
        layers = [m for m in lora.transformer.modules() if isinstance(m, LoraLayer)]
        assert layers and all(layer.disable_adapters for layer in layers)

    full = _replay_model("full")
    with pytest.raises(RuntimeError, match="no reference policy"):
        full.reference_policy()
    full.attach_reference_policy()
    parameter = next(p for p in full.parameters() if p.requires_grad)
    start = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1.0)
    with full.reference_policy():
        assert torch.equal(parameter, start)
    assert torch.equal(parameter, start + 1.0)
