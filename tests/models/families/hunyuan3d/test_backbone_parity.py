"""Real-inference tests for the Hunyuan3D image-to-shape wrapper.

A real (tiny, cache-free) vendored ``HunYuanDiTPlain`` runs through
``forward_step``; the clock mirroring and velocity sign, the one-call batched
CFG against the zero embedding, the pipeline-less replay restore, the uniform
scheduler grid against upstream's reversed one, the occupancy renderer, and the
released checkpoint layout (yaml + ckpt, loaded end to end into a real tiny
DiT / ShapeVAE / DINOv2 stack) are verified against the models' OWN outputs.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from tests.models.steps.denoise.fixtures import (
    TINY_HUNYUAN3D_CONTEXT_DIM,
    TINY_HUNYUAN3D_CONTEXT_LEN,
    TINY_HUNYUAN3D_LATENT_SHAPE,
    build_tiny_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.generation.types import DenoiseRequest
from vrl.models.families.hunyuan3d.model import (
    Hunyuan3DFlowScheduler,
    Hunyuan3DModel,
    Hunyuan3DPipelineModel,
    Hunyuan3DSamplingState,
)
from vrl.models.families.hunyuan3d.render import BOUNDS, render_occupancy_views

_GUIDANCE = 5.0
_BATCH = TINY_HUNYUAN3D_LATENT_SHAPE[0]


def _model(transformer: torch.nn.Module, **components: object) -> Hunyuan3DPipelineModel:
    model = Hunyuan3DPipelineModel(
        pipeline=SimpleNamespace(
            transformer=transformer, device=torch.device("cpu"), **components
        ),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(*, do_cfg: bool) -> Hunyuan3DSamplingState:
    torch.manual_seed(1)
    return Hunyuan3DSamplingState(
        latents=torch.randn(TINY_HUNYUAN3D_LATENT_SHAPE),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        image_embeds=torch.randn(_BATCH, TINY_HUNYUAN3D_CONTEXT_LEN, TINY_HUNYUAN3D_CONTEXT_DIM),
        guidance_scale=_GUIDANCE,
        do_cfg=do_cfg,
        height=32,
        width=32,
        num_views=2,
    )


def test_hunyuan3d_forward_step_mirrors_the_clock_and_negates_the_velocity() -> None:
    """do_cfg=False: one forward at upstream time ``1 - sigma`` (500 -> 0.5), conditioned
    on the image tokens, and the reported noise_pred is the negated DiT output."""
    transformer = build_tiny_transformer("hunyuan3d")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=False)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["t"], torch.full((_BATCH,), 0.5))
    assert torch.equal(calls[0]["contexts"]["main"], state.image_embeds)
    with torch.no_grad():
        raw = transformer(
            x=state.latents, t=torch.full((_BATCH,), 0.5), contexts={"main": state.image_embeds}
        )
    torch.testing.assert_close(out["noise_pred"], -raw)
    assert out["noise_pred"].shape == TINY_HUNYUAN3D_LATENT_SHAPE


def test_hunyuan3d_cfg_is_one_batched_forward_against_the_zero_embedding() -> None:
    """do_cfg=True runs ONE call on 2B rows (cond rows, then all-zero uncond rows) and
    combines ``uncond + s * (cond - uncond)`` from the real branch outputs."""
    transformer = build_tiny_transformer("hunyuan3d")
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = _state(do_cfg=True)

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    assert calls[0]["x"].shape[0] == 2 * _BATCH
    assert torch.equal(calls[0]["contexts"]["main"][_BATCH:], torch.zeros_like(state.image_embeds))
    cond = model.forward_step(replace(state, do_cfg=False), 0)["noise_pred"]
    uncond = model.forward_step(
        replace(state, do_cfg=False, image_embeds=torch.zeros_like(state.image_embeds)), 0
    )["noise_pred"]
    assert not torch.allclose(cond, uncond)
    torch.testing.assert_close(out["noise_pred"], uncond + _GUIDANCE * (cond - uncond))


def test_hunyuan3d_replay_model_restores_state_without_a_pipeline() -> None:
    """The pipeline-less replay model rebuilds the CFG state from the stored image tokens
    and context (no unconditional tensor is stored) and reproduces the rollout forward."""
    transformer = build_tiny_transformer("hunyuan3d")
    rollout = _model(transformer)
    replay = Hunyuan3DModel(transformer=transformer, scheduler=None, device=torch.device("cpu"))
    stamp_model_precision(replay)
    state = _state(do_cfg=True)
    exported = rollout.export_replay_tensors(state)

    assert set(exported) == {"image_embeds", "latents_clean"}
    restored = replay.restore_eval_state(
        {**exported, "timesteps": torch.tensor([[900.0, 500.0]] * _BATCH)},
        rollout.export_batch_context(state),
        state.latents,
        1,
    )
    assert restored.do_cfg is True
    torch.testing.assert_close(
        replay.forward_step(restored, 0)["noise_pred"],
        rollout.forward_step(state, 0)["noise_pred"],
    )


def test_hunyuan3d_scheduler_grid_mirrors_upstreams_reversed_clock() -> None:
    """Upstream steps ``linspace(0, 1, N + 1)`` noise-to-data with an appended terminal 1
    (an empty last step); on the shared convention that is sigma ``1 - t``: N real
    steps from 1 down to 1/N and the terminal 0."""
    steps = 10
    scheduler = Hunyuan3DFlowScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(steps)

    upstream_sigmas = torch.cat([torch.linspace(0.0, 1.0, steps + 1), torch.ones(1)])
    mirrored = 1.0 - upstream_sigmas
    torch.testing.assert_close(scheduler.sigmas, mirrored[: steps + 1])
    assert scheduler.sigmas[-1] == 0.0
    torch.testing.assert_close(scheduler.timesteps, mirrored[:steps] * 1000.0)
    assert torch.all(scheduler.timesteps[1:] < scheduler.timesteps[:-1])


def test_render_occupancy_views_shades_a_sphere_and_leaves_misses_white() -> None:
    """A synthetic sphere (logit = r - |p|) renders as a shaded disc: its area matches
    the sphere's cross-section, corners miss and read white, the centre is lit, and
    every orbit view of a sphere is the same picture."""
    resolution, radius, size = 32, 0.5, 48
    axis = torch.linspace(-BOUNDS, BOUNDS, resolution + 1)
    xs, ys, zs = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = (radius - torch.sqrt(xs**2 + ys**2 + zs**2)).unsqueeze(0)

    frames = render_occupancy_views(grid, num_views=3, height=size, width=size)

    assert frames.shape == (1, 3, 3, size, size)
    assert frames.min() >= 0.0 and frames.max() <= 1.0
    front = frames[0, 0, 0]
    assert front[0, 0] == 1.0 and front[-1, -1] == 1.0
    assert front[size // 2, size // 2] < 1.0
    hit_fraction = (front < 1.0).float().mean().item()
    expected = math.pi * radius**2 / (2 * BOUNDS) ** 2
    assert abs(hit_fraction - expected) < 0.04
    # Every orbit view of a sphere is the same disc up to edge aliasing of the
    # discretized volume: silhouettes overlap almost entirely, shading agrees.
    other = frames[0, 1, 0]
    overlap = ((front < 1.0) & (other < 1.0)).float().sum() / (
        (front < 1.0) | (other < 1.0)
    ).float().sum()
    assert overlap > 0.97
    assert abs(front.mean().item() - other.mean().item()) < 0.01


# ---- the released checkpoint layout, end to end on tiny real components ----------

_TINY_DINO = {
    "model_type": "dinov2",
    "hidden_size": TINY_HUNYUAN3D_CONTEXT_DIM,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "image_size": 28,
    "patch_size": 14,
    "num_channels": 3,
    "layerscale_value": 1.0,
}


def _tiny_checkpoint_dir(tmp_path):
    """Write ``hunyuan3d-dit-v2-1`` / ``hunyuan3d-vae-v2-1`` the way the Hub repo lays
    them out: a yaml per component and a ``model.fp16.ckpt`` (the DiT ckpt nests
    ``model`` and ``conditioner``; the VAE ckpt is flat and carries encoder weights)."""
    from tests.models.steps.denoise.fixtures import _TINY_TRANSFORMERS
    from vrl.models.families.hunyuan3d.upstream import ShapeVAE, SingleImageEncoder

    _, dit_params = _TINY_TRANSFORMERS["hunyuan3d"]
    vae_params = dict(
        num_latents=TINY_HUNYUAN3D_LATENT_SHAPE[1],
        embed_dim=TINY_HUNYUAN3D_LATENT_SHAPE[2],
        width=32,
        heads=4,
        num_decoder_layers=1,
        num_encoder_layers=1,
        num_freqs=2,
        include_pi=False,
        qkv_bias=False,
        qk_norm=True,
        scale_factor=1.25,
        pc_size=8,
        point_feats=4,
    )
    conditioner_params = {
        "main_image_encoder": {
            "type": "DinoImageEncoder",
            "kwargs": {"config": _TINY_DINO, "image_size": 28, "use_cls_token": True},
        }
    }
    torch.manual_seed(0)
    dit = build_tiny_transformer("hunyuan3d")
    vae = ShapeVAE(**vae_params)
    conditioner = SingleImageEncoder(**conditioner_params)

    dit_dir = tmp_path / "hunyuan3d-dit-v2-1"
    vae_dir = tmp_path / "hunyuan3d-vae-v2-1"
    dit_dir.mkdir()
    vae_dir.mkdir()
    dit_config = {
        "model": {
            "target": "hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain",
            "params": dit_params,
        },
        "vae": {"target": "hy3dshape.models.autoencoders.ShapeVAE", "params": vae_params},
        "conditioner": {
            "target": "hy3dshape.models.conditioner.SingleImageEncoder",
            "params": conditioner_params,
        },
        "scheduler": {
            "target": "hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler",
            "params": {"num_train_timesteps": 1000},
        },
        "image_processor": {
            "target": "hy3dshape.preprocessors.ImageProcessorV2",
            "params": {"size": 28, "border_ratio": 0.15},
        },
    }
    (dit_dir / "config.yaml").write_text(yaml.safe_dump(dit_config))
    (vae_dir / "config.yaml").write_text(
        yaml.safe_dump({"target": "hy3dshape.models.ShapeVAE", "params": vae_params})
    )
    torch.save(
        {"model": dit.state_dict(), "conditioner": conditioner.state_dict()},
        dit_dir / "model.fp16.ckpt",
    )
    vae_state = {
        **vae.state_dict(),
        "encoder.input_proj.weight": torch.zeros(2, 2),
        "pre_kl.weight": torch.zeros(8, 32),
        "pre_kl.bias": torch.zeros(8),
    }
    torch.save(vae_state, vae_dir / "model.fp16.ckpt")
    return tmp_path, dit, vae, conditioner


def _build(path, *, rollout: bool):
    from vrl.config.precision import RolePrecision
    from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions

    return ModelBuild(
        model_name_or_path=str(path),
        revision=None,
        device="cpu",
        parameter_dtype=torch.float32,
        family="hunyuan3d",
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        sampling_config={"num_steps": 3},
        rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.float32) if rollout else None,
    )


def _cutout() -> Image.Image:
    """A 40x40 RGBA image with an opaque red square on a transparent background."""
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    rgba[10:30, 12:34] = (200, 30, 30, 255)
    return Image.fromarray(rgba, mode="RGBA")


def test_hunyuan3d_from_build_runs_the_released_layout_end_to_end(tmp_path) -> None:
    """``from_build`` reads the yaml + ckpt layout (nested DiT ckpt, flat VAE ckpt with
    encoder weights skipped), freezes everything but the DiT, and the loaded stack runs
    image encode -> prepare -> forward_step -> occupancy decode -> rendered views."""
    root, dit, _, _ = _tiny_checkpoint_dir(tmp_path)

    model = Hunyuan3DPipelineModel.from_build(_build(root, rollout=True))
    stamp_model_precision(model)

    for name, tensor in dit.state_dict().items():
        torch.testing.assert_close(model.transformer.state_dict()[name], tensor)
    assert not any(p.requires_grad for p in model.pipeline.vae.parameters())
    assert not any(p.requires_grad for p in model.pipeline.conditioner.parameters())
    assert model.pipeline.vae.scale_factor == 1.25

    encoded = model.encode_prompt(
        "unused text", None, reference_image=_cutout(), guidance_scale=5.0
    )
    assert encoded["image_embeds"].shape == (
        1,
        TINY_HUNYUAN3D_CONTEXT_LEN,
        TINY_HUNYUAN3D_CONTEXT_DIM,
    )
    request = DenoiseRequest(
        width=32, height=32, frame_count=2, num_steps=3, guidance_scale=5.0, seed=1
    )
    state = model.prepare_sampling(request, encoded)
    assert state.latents.shape == (1, *TINY_HUNYUAN3D_LATENT_SHAPE[1:])
    assert len(state.timesteps) == 3 and state.do_cfg is True
    torch.testing.assert_close(state.timesteps, torch.tensor([1000.0, 2000.0 / 3, 1000.0 / 3]))
    noise_pred = model.forward_step(state, 0)["noise_pred"]
    assert noise_pred.shape == state.latents.shape
    frames = model.decode_latents(state.latents)
    assert frames.shape == (1, 2, 3, 32, 32)
    assert frames.min() >= 0.0 and frames.max() <= 1.0
    with pytest.raises(ValueError, match="reference image"):
        model.encode_prompt("unused text", None)


def test_hunyuan3d_replay_build_loads_only_the_dit_and_can_stay_a_skeleton(tmp_path) -> None:
    """The replay loader needs neither VAE nor conditioner files, fills the DiT from
    the nested ckpt, and with ``materialize_weights=False`` yields a meta skeleton."""
    root, dit, _, _ = _tiny_checkpoint_dir(tmp_path)
    (root / "hunyuan3d-vae-v2-1" / "model.fp16.ckpt").unlink()

    replay = Hunyuan3DModel.from_build(_build(root, rollout=False))
    for name, tensor in dit.state_dict().items():
        torch.testing.assert_close(replay.transformer.state_dict()[name], tensor)
    assert len(replay.scheduler.timesteps) == 3

    skeleton = Hunyuan3DModel.from_build(
        replace(_build(root, rollout=False), materialize_weights=False)
    )
    assert all(p.is_meta for p in skeleton.transformer.parameters())
    assert {n: tuple(t.shape) for n, t in skeleton.transformer.state_dict().items()} == {
        n: tuple(t.shape) for n, t in dit.state_dict().items()
    }
