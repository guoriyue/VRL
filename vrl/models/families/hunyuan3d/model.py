"""Hunyuan3D 2.1 image-to-shape model (Tencent, shape stage only).

The generation helper flow mirrors every diffusion family, with an image in
place of the prompt and rendered views in place of the decoded picture:

    encode_prompt (DINOv2 of the reference image) -> prepare_sampling ->
    forward_step xN -> decode_latents (occupancy grid -> shaded views)

``Hunyuan3DPipelineModel`` runs that flow. ``Hunyuan3DModel`` — the DiT, its
scheduler and ``forward_step`` — is what replay runs; the pipeline class adds
the DINOv2 conditioner, the ShapeVAE decoder and the renderer on top of it.

Hunyuan3D specifics vs SD3 (the reference family):
- Nothing is in diffusers: the 3.3B MoE DiT (``HunYuanDiTPlain``), the ShapeVAE
  and the conditioner are vendored in ``upstream.py`` and loaded from the released
  ``model.fp16.ckpt`` files by ``from_build`` (one yaml config per component).
- Latents are a SET of 4096 tokens x 64 channels (``[B, 4096, 64]``, VecSet), with
  no spatial grid; ``sampling.height/width`` size the rendered views instead.
- Conditioning is one RGBA image, recentered and encoded by DINOv2-L into
  ``[B, 1370, 1024]`` tokens; the unconditional branch is the all-zero embedding,
  so replay stores only the conditional tokens and a CFG flag.
- CFG runs as ONE batched forward (cond rows then uncond rows), as upstream.
- REVERSED CLOCK: upstream's DiT takes ``t in [0, 1]`` with ``t = 0`` pure noise
  and predicts ``data - noise``. The shared SDE math (``sigma = 1`` noise,
  velocity ``noise - data``) is the mirror image, so ``forward_step`` feeds
  ``1 - sigma`` and returns the NEGATED prediction; the mixture ``x_t`` is the
  same tensor under both clocks, so latents and log-probs need no conversion.
- Schedule: upstream samples ``N + 1`` uniform points from noise to data whose
  last Euler step has zero length. ``Hunyuan3DFlowScheduler`` is that grid on
  the shared convention — ``N`` uniform sigmas from 1 down to ``1/N`` plus the
  terminal 0 — so ``sampling.num_steps`` counts real transformer steps.
- The decoded shape is rendered to ``num_frames`` orthographic views (a frame
  sequence, task ``i23d``), which frame-based rewards such as the DINOv2 target
  similarity against the input image consume.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler

from vrl.generation.types import DenoiseRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersModelBase,
    DiffusersPipelineModelBase,
    GuidedDenoiseSamplingStateBase,
)
from vrl.models.steps.denoise.common import combine_cfg, expand_batch_timestep, pack_eval_timestep

DIT_SUBFOLDER = "hunyuan3d-dit-v2-1"
VAE_SUBFOLDER = "hunyuan3d-vae-v2-1"
# Upstream's released sampling recipe renders at this many views for the
# frame-based rewards; the dense occupancy grid has this many cells per axis
# (upstream exports meshes at 384; 128 is the render-only budget).
VOLUME_RESOLUTION = 128


@dataclass
class Hunyuan3DSamplingState(GuidedDenoiseSamplingStateBase):
    """Private Hunyuan3D sampling state. Engine MUST NOT introspect."""

    image_embeds: torch.Tensor
    do_cfg: bool
    height: int
    width: int
    num_views: int


# -- checkpoint loading -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hunyuan3DCheckpoint:
    """The released shape checkpoint's two directories, resolved to local files."""

    dit_config: dict[str, Any]
    dit_checkpoint: Path
    vae_config: dict[str, Any]
    vae_checkpoint: Path


def resolve_checkpoint(build: ModelBuild, *, with_vae: bool) -> Hunyuan3DCheckpoint:
    """Locate the DiT (and VAE) yaml + ``model.fp16.ckpt`` of ``build.model_name_or_path``.

    A local directory laid out like the Hub repo is used as is; a repo id is
    fetched at the pinned revision (only the shape subfolders, never the paint
    stage) into the Hugging Face cache.
    """

    from vrl.utils.hf_hub import HuggingFaceRepoRevision

    path = Path(str(build.model_name_or_path)).expanduser()
    if path.is_dir():
        root = path
    else:
        patterns = [f"{DIT_SUBFOLDER}/*"] + ([f"{VAE_SUBFOLDER}/*"] if with_vae else [])
        root = HuggingFaceRepoRevision(
            str(build.model_name_or_path), build.revision or "main"
        ).download(
            local_files_only=bool(build.pretrained_kwargs.get("local_files_only", False)),
            allow_patterns=patterns,
        )

    def component(subfolder: str) -> tuple[dict[str, Any], Path]:
        folder = root / subfolder
        config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        for name in ("model.fp16.ckpt", "model.ckpt"):
            if (folder / name).is_file():
                return config, folder / name
        raise FileNotFoundError(f"{folder} has no model.fp16.ckpt / model.ckpt")

    dit_config, dit_checkpoint = component(DIT_SUBFOLDER)
    if with_vae:
        vae_config, vae_checkpoint = component(VAE_SUBFOLDER)
    else:
        vae_config, vae_checkpoint = dit_config["vae"], dit_checkpoint
    return Hunyuan3DCheckpoint(
        dit_config=dit_config,
        dit_checkpoint=dit_checkpoint,
        vae_config=vae_config,
        vae_checkpoint=vae_checkpoint,
    )


def _load_state(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_dit(
    checkpoint: Hunyuan3DCheckpoint,
    *,
    dtype: torch.dtype,
    materialize_weights: bool = True,
) -> Any:
    """The DiT from its yaml ``model.params`` and the ``model`` entry of the ckpt.

    Built on the meta device and filled by ``assign`` so the 3.3B parameters are
    never initialized in fp32 host memory first. ``materialize_weights=False``
    returns the meta skeleton for a sharded strategy to fill.
    """

    from vrl.models.families.hunyuan3d.upstream import HunYuanDiTPlain

    with torch.device("meta"):
        dit = HunYuanDiTPlain(**checkpoint.dit_config["model"]["params"])
    if materialize_weights:
        state = _load_state(checkpoint.dit_checkpoint)
        state = state.get("model", state)
        dit.load_state_dict(state, strict=True, assign=True)
    return dit.to(dtype=dtype)


def load_vae(checkpoint: Hunyuan3DCheckpoint) -> Any:
    """The ShapeVAE decoder; the encoder weights the ckpt also carries are skipped."""

    from vrl.models.families.hunyuan3d.upstream import ShapeVAE

    vae = ShapeVAE(**checkpoint.vae_config["params"])
    state = _load_state(checkpoint.vae_checkpoint)
    state = state.get("vae", state)
    missing, unexpected = vae.load_state_dict(state, strict=False)
    # The released ckpt also carries the encoder (``encoder.*``) and its posterior
    # projection (``pre_kl``); neither exists on the decoder-only module.
    stray = [key for key in unexpected if not key.startswith(("encoder.", "pre_kl."))]
    if missing or stray:
        raise RuntimeError(
            f"ShapeVAE checkpoint mismatch: missing={sorted(missing)[:5]} unexpected={stray[:5]}"
        )
    return vae.eval()


def load_conditioner(checkpoint: Hunyuan3DCheckpoint) -> Any:
    """The DINOv2 conditioner from the ckpt's ``conditioner`` entry (upstream builds
    the encoder from config and relies on these weights)."""

    from vrl.models.families.hunyuan3d.upstream import SingleImageEncoder

    conditioner = SingleImageEncoder(**checkpoint.dit_config["conditioner"]["params"])
    state = _load_state(checkpoint.dit_checkpoint)
    if "conditioner" not in state:
        raise RuntimeError(
            f"{checkpoint.dit_checkpoint} carries no 'conditioner' weights; the DINOv2 "
            "encoder would be random"
        )
    conditioner.load_state_dict(state["conditioner"], strict=True)
    return conditioner.eval()


class Hunyuan3DFlowScheduler(FlowMatchEulerDiscreteScheduler):
    """Uniform noise-to-data Euler grid of the Hunyuan3D 2.1 shape pipeline.

    Diffusers' own grid for ``N`` steps ends at ``sigma_min`` (1e-3), not at
    upstream's ``1/N``; the shared denoise code only calls
    ``set_timesteps(num_steps)``, so the grid is chosen here. The shared SDE
    math only reads ``sigmas`` / ``timesteps`` / ``index_for_timestep``, all
    inherited unchanged.
    """

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 1:
                raise ValueError("Hunyuan3DFlowScheduler needs a positive step count")
            sigmas = [1.0 - i / num_inference_steps for i in range(num_inference_steps)]
        super().set_timesteps(device=device, sigmas=list(sigmas), mu=mu)


def build_scheduler(checkpoint: Hunyuan3DCheckpoint) -> Any:
    """The uniform-grid scheduler with the yaml's ``num_train_timesteps``."""

    params = dict(checkpoint.dit_config["scheduler"].get("params") or {})
    return Hunyuan3DFlowScheduler(num_train_timesteps=int(params.get("num_train_timesteps", 1000)))


@dataclass
class Hunyuan3DComponents:
    """The shape-stage modules, on the attribute surface ``DiffusersPipelineModelBase`` reads."""

    transformer: Any
    vae: Any
    conditioner: Any
    image_processor: Any
    scheduler: Any

    @property
    def components(self) -> dict[str, Any]:
        """Every module, for the frozen-component freeze / placement / offload rules."""

        return {"transformer": self.transformer, "vae": self.vae, "conditioner": self.conditioner}


# -- models ------------------------------------------------------------------


class Hunyuan3DModel(DiffusersModelBase):
    """Hunyuan3D shape DiT, scheduler and per-step math: the model replay runs."""

    @classmethod
    def from_build(cls, build: ModelBuild) -> Hunyuan3DModel:
        """DiT + scheduler from the released checkpoint (no VAE, no conditioner)."""

        build.require_replay()
        checkpoint = resolve_checkpoint(build, with_vae=False)
        model = cls(
            transformer=load_dit(
                checkpoint,
                dtype=build.parameter_dtype,
                materialize_weights=build.materialize_weights,
            ),
            scheduler=build_scheduler(checkpoint),
            device=build.device,
        )
        num_steps = build.num_steps
        if num_steps is not None:
            model.set_num_steps(num_steps)
        return model

    def forward_step(
        self,
        state: Hunyuan3DSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """One DiT forward on upstream's clock, reported on the shared one.

        ``t_upstream = 1 - sigma`` and ``noise_pred = -(data - noise prediction)``;
        under CFG the two branches share one batched call (cond rows first).
        """
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        sigma = expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=torch.float32)
        upstream_time = (1.0 - sigma / 1000.0).to(td)
        cond = state.image_embeds.to(td)
        if state.do_cfg:
            latent_input = torch.cat([latent_input, latent_input], dim=0)
            upstream_time = torch.cat([upstream_time, upstream_time], dim=0)
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        prediction = self.transformer(x=latent_input, t=upstream_time, contexts={"main": cond})
        if state.do_cfg:
            pred_cond, pred_uncond = prediction.chunk(2, dim=0)
            prediction = combine_cfg(pred_cond, pred_uncond, guidance_scale=state.guidance_scale)
        return {"noise_pred": (-prediction).to(td)}

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Hunyuan3DSamplingState:
        """Rebuild Hunyuan3DSamplingState from a batch slice for the eval forward path."""
        ts = replay_tensors["timesteps"]
        return Hunyuan3DSamplingState(
            latents=latents,
            timesteps=pack_eval_timestep(ts, step_idx),
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            image_embeds=replay_tensors["image_embeds"],
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=bool(batch_context["cfg"]),
            height=int(batch_context["height"]),
            width=int(batch_context["width"]),
            num_views=int(batch_context["num_views"]),
        )


class Hunyuan3DPipelineModel(DiffusersPipelineModelBase, Hunyuan3DModel):
    """Hunyuan3D shape DiT plus the conditioner, ShapeVAE decoder and renderer sampling needs."""

    def __init__(self, *, pipeline: Any, device: Any = None) -> None:
        super().__init__(pipeline=pipeline, device=device)
        # decode_latents only receives the latent set; prepare_sampling records the
        # view count and canvas it must render (one model instance runs prepare ->
        # denoise -> decode sequentially per batch).
        self._decode_views = 4
        self._decode_height = 256
        self._decode_width = 256

    @classmethod
    def from_build(cls, build: ModelBuild) -> Hunyuan3DPipelineModel:
        """Load the shape stage's three modules and apply the shared freeze / placement rule.

        The DiT stays where the loader put it (the rollout builder places the
        trainable root); the conditioner takes the rollout encoder dtype and the
        ShapeVAE decodes in fp32 like every family's VAE.
        """

        from vrl.models.families.hunyuan3d.upstream import ImageProcessorV2

        checkpoint = resolve_checkpoint(build, with_vae=True)
        prompt_encoder_dtype, _ = cls._pipeline_load_dtypes(build)
        processor_params = dict(checkpoint.dit_config["image_processor"].get("params") or {})
        components = Hunyuan3DComponents(
            transformer=load_dit(checkpoint, dtype=build.parameter_dtype),
            vae=load_vae(checkpoint),
            conditioner=load_conditioner(checkpoint),
            image_processor=ImageProcessorV2(**processor_params),
            scheduler=build_scheduler(checkpoint),
        )
        cls.freeze_pipeline_components(components)
        cpu_resident = cls.place_pipeline_components(
            components, build, prompt_encoder_dtype=prompt_encoder_dtype
        )
        model = cls(pipeline=components, device=build.device)
        model._cpu_resident = cpu_resident
        return model

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        *,
        reference_image: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode the reference image with DINOv2; the text prompt is not an input.

        Upstream's preprocessing (RGBA recenter, white background, 512 px) runs
        on the image, then the conditioner's main encoder. The unconditional
        branch is the all-zero embedding and is rebuilt at forward time.
        """
        del prompt
        self._reject_unsupported_negative_prompt(negative_prompt)
        if reference_image is None:
            raise ValueError("Hunyuan3D conditions on one reference image per prompt")
        pipe = self.pipeline
        image = pipe.image_processor(reference_image)["image"]
        conditioner = pipe.conditioner
        encoder_dtype = next(conditioner.parameters()).dtype
        with torch.no_grad():
            embeds = conditioner(image=image.to(self.device, dtype=encoder_dtype))["main"]
        return {"image_embeds": embeds.to(self.device, dtype=self._transformer_dtype())}

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Hunyuan3DSamplingState:
        """Build the per-request VecSet-latent SamplingState for a denoise loop."""
        del kwargs
        from diffusers.utils.torch_utils import randn_tensor

        pipe = self.pipeline
        device = self.device
        image_embeds = encoded["image_embeds"]

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        shape = (image_embeds.shape[0], *pipe.vae.latent_shape)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

        self._decode_views = int(request.frame_count)
        self._decode_height = int(request.height)
        self._decode_width = int(request.width)

        return Hunyuan3DSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            image_embeds=image_embeds,
            guidance_scale=request.guidance_scale,
            do_cfg=request.guidance_scale > 1.0,
            height=int(request.height),
            width=int(request.width),
            num_views=int(request.frame_count),
        )

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: Hunyuan3DSamplingState) -> dict[str, Any]:
        """Project Hunyuan3D sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "height": state.height,
            "width": state.width,
            "num_views": state.num_views,
        }

    def export_replay_tensors(self, state: Hunyuan3DSamplingState) -> dict[str, Any]:
        """Project Hunyuan3D sampling state into per-sample trajectory tensors.

        The unconditional embedding is zeros, so only the conditional tokens are
        stored; ``cfg`` in the context says whether replay rebuilds the branch.
        """
        return {
            "image_embeds": state.image_embeds,
            "latents_clean": state.latents.detach(),
        }

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VecSet latents -> ``num_views`` shaded views ``[B, V, 3, H, W]``.

        The ShapeVAE answers occupancy on a dense ``VOLUME_RESOLUTION`` grid and
        the renderer marches orthographic rays through it; the view count and
        canvas come from the last ``prepare_sampling`` (one model instance runs
        prepare -> denoise -> decode sequentially per batch).
        """
        from vrl.models.families.hunyuan3d.render import (
            query_occupancy_grid,
            render_occupancy_views,
        )

        with torch.no_grad():
            grid = query_occupancy_grid(self.pipeline.vae, latents, resolution=VOLUME_RESOLUTION)
            return render_occupancy_views(
                grid,
                num_views=self._decode_views,
                height=self._decode_height,
                width=self._decode_width,
            )


__all__ = [
    "DIT_SUBFOLDER",
    "VAE_SUBFOLDER",
    "VOLUME_RESOLUTION",
    "Hunyuan3DCheckpoint",
    "Hunyuan3DComponents",
    "Hunyuan3DFlowScheduler",
    "Hunyuan3DModel",
    "Hunyuan3DPipelineModel",
    "Hunyuan3DSamplingState",
    "build_scheduler",
    "load_conditioner",
    "load_dit",
    "load_vae",
    "resolve_checkpoint",
]
