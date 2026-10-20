"""Real-inference tests for the MOSS-Image backbone wrapper.

A real (tiny, cache-free) vendored ``MOSSImageTransformer`` runs through
``forward_step``; the raw-timestep / bool-mask call, the separate-branch CFG
whose negative branch skips a DiT block, the fp32 combine, the pipeline-less
replay restore and the FLUX.2-VAE decode are verified against the model's OWN
outputs, never a re-derived formula.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_MOSS_LATENT_SHAPE,
    TINY_MOSS_TEXT_DIM,
    build_tiny_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.moss_image.model import (
    UNCOND_SKIP_LAYERS,
    MossImageModel,
    MossImagePipelineModel,
    MossImageSamplingState,
)

_GUIDANCE = 3.0


def _model(transformer: torch.nn.Module, **components: object) -> MossImagePipelineModel:
    model = MossImagePipelineModel(
        pipeline=SimpleNamespace(
            transformer=transformer, device=torch.device("cpu"), **components
        ),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(*, do_cfg: bool, text_len: int = 5, negative_len: int = 3) -> MossImageSamplingState:
    torch.manual_seed(1)
    mask = torch.ones(2, text_len, dtype=torch.bool)
    mask[1, text_len - 2 :] = False  # the second row is padded
    negative_mask = torch.ones(2, negative_len, dtype=torch.bool)
    return MossImageSamplingState(
        latents=torch.randn(TINY_MOSS_LATENT_SHAPE),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, text_len, TINY_MOSS_TEXT_DIM),
        prompt_embeds_mask=mask,
        negative_prompt_embeds=torch.randn(2, negative_len, TINY_MOSS_TEXT_DIM)
        if do_cfg
        else None,
        negative_prompt_embeds_mask=negative_mask if do_cfg else None,
        guidance_scale=_GUIDANCE,
        do_cfg=do_cfg,
        uncond_skip_layers=UNCOND_SKIP_LAYERS,
    )


def test_moss_forward_step_single_branch_feeds_the_raw_timestep_and_a_bool_mask() -> None:
    """do_cfg=False: one forward, timestep unscaled, the padded row's mask passed as bool."""
    transformer = build_tiny_transformer("moss_image")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["timestep"], torch.full((2,), 500.0))
    assert calls[0]["encoder_attention_mask"].dtype is torch.bool
    assert torch.equal(calls[0]["encoder_attention_mask"], state.prompt_embeds_mask)
    assert "skip_layers" not in calls[0]
    assert out["noise_pred"].shape == TINY_MOSS_LATENT_SHAPE
    assert out["noise_pred"].dtype is torch.float32


def test_moss_forward_step_skips_the_released_block_only_on_the_negative_branch() -> None:
    """do_cfg=True runs TWO forwards; the negative one carries ``skip_layers=(22,)`` and
    the combine is the plain fp32 ``uncond + g * (cond - uncond)`` on the real outputs."""
    transformer = build_tiny_transformer("moss_image")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    assert len(calls) == 2
    assert "skip_layers" not in calls[0]
    assert calls[1]["skip_layers"] == UNCOND_SKIP_LAYERS
    assert calls[1]["encoder_hidden_states"].shape[1] == 3
    cond = model.forward_step(replace(state, do_cfg=False), 0)["noise_pred"]
    with torch.no_grad():
        uncond_skipped = transformer(
            state.latents,
            state.timesteps,
            state.negative_prompt_embeds,
            state.negative_prompt_embeds_mask,
            skip_layers=UNCOND_SKIP_LAYERS,
            return_dict=False,
        )[0]
        uncond_full = transformer(
            state.latents,
            state.timesteps,
            state.negative_prompt_embeds,
            state.negative_prompt_embeds_mask,
            return_dict=False,
        )[0]
    torch.testing.assert_close(
        out["noise_pred"], uncond_skipped + _GUIDANCE * (cond - uncond_skipped)
    )
    # The skip is observable: the block it removes changes the negative prediction.
    assert not torch.allclose(uncond_skipped, uncond_full)


def test_moss_replay_model_restores_state_without_a_pipeline() -> None:
    """The pipeline-less replay model rebuilds the CFG state from trajectory tensors and
    context and reproduces the rollout model's forward on the same weights."""
    transformer = build_tiny_transformer("moss_image")
    rollout = _model(transformer)
    replay = MossImageModel(transformer=transformer, scheduler=None, device=torch.device("cpu"))
    stamp_model_precision(replay)
    state = _state(do_cfg=True)

    restored = replay.restore_eval_state(
        {
            **rollout.export_replay_tensors(state),
            # [B, num_steps] per-sample timesteps; the step under replay is index 1.
            "timesteps": torch.tensor([[900.0, 500.0], [900.0, 500.0]]),
        },
        rollout.export_batch_context(state),
        state.latents,
        1,
    )

    assert restored.do_cfg is True
    assert restored.uncond_skip_layers == UNCOND_SKIP_LAYERS
    assert restored.prompt_embeds_mask.dtype is torch.bool
    torch.testing.assert_close(
        replay.forward_step(restored, 0)["noise_pred"],
        rollout.forward_step(state, 0)["noise_pred"],
    )


def test_moss_replay_without_a_negative_pair_runs_the_conditional_branch_only() -> None:
    """A no-CFG rollout exports no negative tensors, so replay restores do_cfg=False
    even though the context's cfg flag is read."""
    transformer = build_tiny_transformer("moss_image")
    rollout = _model(transformer)
    state = _state(do_cfg=False)
    exported = rollout.export_replay_tensors(state)

    assert "negative_prompt_embeds" not in exported
    restored = MossImageModel(
        transformer=transformer, scheduler=None, device=torch.device("cpu")
    ).restore_eval_state(
        {**exported, "timesteps": torch.tensor([[500.0], [500.0]])},
        rollout.export_batch_context(state),
        state.latents,
        0,
    )
    assert restored.do_cfg is False
    assert torch.equal(exported["latents_clean"], state.latents)


def test_moss_decode_unfolds_and_denormalizes_through_a_real_flux2_vae() -> None:
    """``decode_latents`` reads the VAE's batch-norm statistics: the DiT's 16 latent
    channels unfold to the VAE's 4 at twice the grid, decode to RGB at 16 px per
    latent cell, and shifted running statistics change the picture."""
    from diffusers import AutoencoderKLFlux2
    from diffusers.image_processor import VaeImageProcessor

    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        decoder_block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=TINY_MOSS_LATENT_SHAPE[1] // 4,
        norm_num_groups=4,
        sample_size=64,
        patch_size=(2, 2),
        mid_block_add_attention=False,
    ).eval()
    model = _model(
        build_tiny_transformer("moss_image"),
        vae=vae,
        image_processor=VaeImageProcessor(vae_scale_factor=16),
    )
    latents = torch.randn(TINY_MOSS_LATENT_SHAPE)

    image = model.decode_latents(latents)

    batch, _, height, width = TINY_MOSS_LATENT_SHAPE
    assert image.shape == (batch, 3, 16 * height, 16 * width)
    assert image.min() >= 0.0 and image.max() <= 1.0
    vae.bn.running_mean.add_(1.0)
    assert not torch.allclose(model.decode_latents(latents), image)


def test_moss_scheduler_loads_by_import_path_onto_the_training_grid() -> None:
    """The replay loader resolves the vendored scheduler by path and sets the released
    shift-5 schedule: ``num_steps`` distinct timesteps, descending from 1000, every
    one on the 1000-point training grid, matching the rollout's own set_timesteps."""
    import tempfile

    from vrl.config.precision import RolePrecision
    from vrl.models.families.moss_image.model import SCHEDULER_CLS
    from vrl.models.families.moss_image.pipeline_moss_image import MOSSImageScheduler
    from vrl.models.interfaces.runtime import ModelBuild
    from vrl.models.loader import load_diffusers_scheduler

    with tempfile.TemporaryDirectory() as tmp:
        MOSSImageScheduler(shift=5.0).save_pretrained(f"{tmp}/scheduler")
        replay = load_diffusers_scheduler(
            ModelBuild(
                model_name_or_path=tmp,
                revision=None,
                device="cpu",
                parameter_dtype=torch.float32,
                family="moss_image",
                precision=RolePrecision("fp32", "ieee", outer_autocast=False),
                sampling_config={"num_steps": 10},
            ),
            SCHEDULER_CLS,
        )

    assert type(replay) is MOSSImageScheduler
    assert replay.config.shift == 5.0
    timesteps = replay.timesteps
    assert timesteps.shape == (10,)
    assert timesteps[0] == 1000.0
    assert torch.all(timesteps[1:] < timesteps[:-1])
    torch.testing.assert_close(timesteps, timesteps.round())
    rollout = MOSSImageScheduler(shift=5.0)
    rollout.set_timesteps(10)
    torch.testing.assert_close(rollout.sigmas, replay.sigmas)


def test_moss_replay_through_a_built_trajectory_matches_the_rollout_forward() -> None:
    """The exported bool masks and the skip indices survive the real trajectory build:
    ``replay_forward`` on the stored step reproduces the rollout ``forward_step``."""
    from vrl.generation.types import GenerationRequest, GenerationSampleRow
    from vrl.rollouts.batch import RolloutBatch
    from vrl.trajectory.builders import build_diffusion_trajectory

    transformer = build_tiny_transformer("moss_image")
    rollout = _model(transformer)
    replay = MossImageModel(transformer=transformer, scheduler=None, device=torch.device("cpu"))
    stamp_model_precision(replay)
    state = _state(do_cfg=True)
    batch = TINY_MOSS_LATENT_SHAPE[0]
    context = rollout.export_batch_context(state)
    trajectory = build_diffusion_trajectory(
        request=GenerationRequest(
            request_id="moss-replay",
            family="moss_image",
            task="t2i",
            inputs=["a test prompt"],
            samples_per_prompt=batch,
        ),
        sample_rows=[
            GenerationSampleRow(
                prompt_index=0, sample_index=i, prompt="a test prompt", sample_id=f"s{i}"
            )
            for i in range(batch)
        ],
        observations=state.latents.unsqueeze(1),
        actions=torch.zeros(batch, 1, *TINY_MOSS_LATENT_SHAPE[1:]),
        old_log_prob=torch.zeros(batch, 1),
        timesteps=torch.full((batch, 1), 500.0),
        replay_tensors=rollout.export_replay_tensors(state),
        context=context,
    )
    rollout_batch = RolloutBatch(
        rewards=torch.zeros(batch),
        group_ids=torch.zeros(batch, dtype=torch.long),
        context=context,
        trajectory=trajectory,
    )

    with torch.no_grad():
        replayed = replay.replay_forward(rollout_batch, 0)["noise_pred"]
        sampled = rollout.forward_step(state, 0)["noise_pred"]

    torch.testing.assert_close(replayed, sampled)
