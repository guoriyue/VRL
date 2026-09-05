"""Opt-in real-checkpoint parity for SANA's native and production denoise paths."""

from __future__ import annotations

import os

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.e2e
def test_sana_training_path_matches_native_flow_euler_at_every_step() -> None:
    """Compare the exact training context against an independently built native loop."""

    if os.environ.get("WM_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("set WM_RUN_REAL_MODEL_TESTS=1 to load the pinned SANA checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("SANA real-checkpoint parity requires CUDA")

    from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
    from huggingface_hub import snapshot_download

    from vrl.config.loading import load_config
    from vrl.config.precision import resolve_precision_policy
    from vrl.config.schema import parse_config
    from vrl.generation.types import DenoiseRequest
    from vrl.math.denoise.flow_matching import sde_step_with_logprob
    from vrl.models.families.registry import get_model_family_entry

    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    repo_id = str(cfg.model.path)
    revision = str(cfg.model.revision)
    snapshot = snapshot_download(repo_id, revision=revision, local_files_only=True)
    cfg.model.path = snapshot
    cfg.model.revision = None
    cfg.model.use_lora = False
    root = parse_config(cfg)
    precision = resolve_precision_policy(root)

    device = torch.device("cuda")
    entry = get_model_family_entry("sana")
    build = entry.resolve_model_build(
        root,
        device,
        precision=precision,
        for_rollout=True,
    )
    bundle = entry.build_rollout(build)
    model = bundle.model
    assert model.transformer.dtype is torch.float16
    assert model.pipeline.text_encoder.dtype is torch.bfloat16
    assert model.pipeline.vae.dtype is torch.float32
    assert bundle.precision.dtype == "fp16"
    assert bundle.precision.float32_precision == "ieee"
    assert bundle.precision.outer_autocast is False

    prompt = "a red apple on a blue ceramic plate, studio photo"
    encoded = model.encode_prompt([prompt], [""], guidance_scale=4.5)
    request = DenoiseRequest(
        negative_prompt="",
        width=512,
        height=512,
        frame_count=1,
        num_steps=20,
        guidance_scale=4.5,
        seed=20260712,
    )
    state = model.prepare_sampling(request, encoded)
    reference_latents = state.latents.detach().clone()
    checkpoint_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        snapshot,
        subfolder="scheduler",
        local_files_only=True,
    )
    flow_shift = float(checkpoint_scheduler.config.flow_shift)
    assert flow_shift == 3.0
    reference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        checkpoint_scheduler.config,
        shift=flow_shift,
    )
    reference_scheduler.set_timesteps(request.num_steps, device=device)
    torch.testing.assert_close(state.timesteps, reference_scheduler.timesteps)
    torch.testing.assert_close(state.scheduler.sigmas, reference_scheduler.sigmas)

    prompt_embeds = torch.cat(
        (state.negative_prompt_embeds, state.prompt_embeds),
        dim=0,
    )
    prompt_mask = torch.cat(
        (state.negative_prompt_attention_mask, state.prompt_attention_mask),
        dim=0,
    )

    with torch.inference_mode():
        for step_index, timestep in enumerate(reference_scheduler.timesteps):
            latent_input = torch.cat((reference_latents, reference_latents), dim=0)
            timestep_batch = timestep.expand(latent_input.shape[0])
            reference_noise = model.transformer(
                latent_input.to(torch.float16),
                encoder_hidden_states=prompt_embeds.to(torch.float16),
                encoder_attention_mask=prompt_mask,
                timestep=timestep_batch * model.transformer.config.timestep_scale,
                return_dict=False,
            )[0].float()
            reference_uncond, reference_cond = reference_noise.chunk(2)
            reference_noise = reference_uncond + request.guidance_scale * (
                reference_cond - reference_uncond
            )

            # RuntimeBundle stamps the resolved role policy on the model, and
            # DiffusionModelBase owns the forward autocast boundary.
            production_noise = model.forward_step(state, step_index)["noise_pred"]
            torch.testing.assert_close(
                production_noise.float(),
                reference_noise,
                rtol=0.0,
                atol=0.0,
            )

            reference_latents = reference_scheduler.step(
                reference_noise,
                timestep,
                reference_latents,
                return_dict=False,
            )[0]
            production_step = sde_step_with_logprob(
                state.scheduler,
                production_noise.float(),
                state.timesteps[step_index].unsqueeze(0),
                state.latents.float(),
                deterministic=True,
                step_index=step_index,
            )
            torch.testing.assert_close(
                production_step.prev_sample,
                reference_latents,
                rtol=1e-6,
                atol=1e-6,
            )
            state.latents = production_step.prev_sample

        production_image = model.decode_latents(state.latents)
        reference_image = model.decode_latents(reference_latents)

    torch.testing.assert_close(production_image, reference_image, rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(production_image).all())
