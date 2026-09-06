"""CausVid causal-Wan model and differentiable grouped replay.

Heavy upstream imports are deliberately confined to ``from_build``.  Importing
``causvid.models.wan.causal_model`` compiles FlexAttention at module import, so
ordinary registry/config/test imports must never transitively import it.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.machinery
import importlib.util
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from vrl.generation.bindings.chunk_autoregressive_denoise import (
    ChunkAutoregressiveDenoiseResult,
)
from vrl.math.denoise.renoise import renoise_step_with_logprob
from vrl.models.checkpoint_identity import require_checkpoint_source_member
from vrl.models.families.causvid.config import (
    CAUSVID_CHECKPOINT_FILE,
    CAUSVID_SOURCE_REVISION,
)
from vrl.models.families.causvid.runner import (
    OFFICIAL_CAUSVID_GEOMETRY,
    OFFICIAL_CAUSVID_SCHEDULE,
    CausVidCausalRunner,
    CausVidGeometry,
    CausVidSchedule,
)
from vrl.models.interfaces import (
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
)
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.precision import model_autocast
from vrl.models.source_integrity import runtime_source_tree_sha256
from vrl.models.steps.denoise.base import DiffusionModelBase
from vrl.models.steps.denoise.common.lora import LoraModelMixin
from vrl.utils.artifacts import sha256_file

# Immutable upstream/checkpoint identifiers are genuine external boundaries.
CAUSVID_CHECKPOINT_REPOSITORY = "tianweiy/CausVid"
CAUSVID_CHECKPOINT_REVISION = "b545eb2728fc9d1515023a270b847f7b24b3aa89"
CAUSVID_CHECKPOINT_SHA256 = "8b75a848a525a6ab0b8cefa7f2a3997aefb581eb5824ad60bafb5337e81e521c"
CAUSVID_BASE_REPOSITORY = "Wan-AI/Wan2.1-T2V-1.3B"
CAUSVID_RUNTIME_SOURCE_SHA256 = "fe5e028a84beb2799adea1952ab428be0a1f82b90ee4c552874c81c67c2820b1"
_CAUSVID_RUNTIME_SOURCE_GLOBS = ("causvid/**/*.py",)
_CAUSVID_RUNTIME_SOURCE_EXCLUDES = ("causvid/evaluation",)


@dataclass(frozen=True, slots=True)
class CausVidResolvedArtifacts:
    """Pinned local paths used to construct one CausVid runtime."""

    base_model_dir: Path
    checkpoint_file: Path


@dataclass(slots=True)
class CausVidCache:
    """One request's mutable self- and cross-attention state."""

    kv_cache: list[dict[str, torch.Tensor]]
    crossattn_cache: list[dict[str, Any]]


class _OfficialCausVidBackend:
    """Explicit-path adapter around the pinned upstream causal Wan modules."""

    def __init__(
        self,
        *,
        transformer: Any,
        parameter_dtype: torch.dtype,
        device: Any,
        text_encoder: Any | None = None,
        tokenizer: Any | None = None,
        vae: Any | None = None,
    ) -> None:
        self.transformer = transformer
        self._causal_core = transformer
        self._parameter_dtype = parameter_dtype
        self._device = torch.device(device)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self._block_masks: dict[tuple[int, int, int], Any] = {}

        self.num_layers = len(transformer.blocks)
        self.num_heads = int(transformer.num_heads)
        self.head_dim = int(transformer.dim) // self.num_heads
        self.patch_size = tuple(int(value) for value in transformer.patch_size)
        self.text_len = int(transformer.text_len)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._parameter_dtype

    def set_transformer(self, transformer: Any) -> None:
        """Update the PEFT/compile forward shell while retaining the causal core."""

        self.transformer = transformer

    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("CausVid replay backend does not own a prompt encoder")
        ids, mask = self.tokenizer(
            prompts,
            return_mask=True,
            add_special_tokens=True,
        )
        encoder_device = next(self.text_encoder.parameters()).device
        ids = ids.to(encoder_device)
        mask = mask.to(encoder_device)
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            context = self.text_encoder(ids, mask)
        for row, sequence_length in zip(context, sequence_lengths, strict=True):
            row[int(sequence_length) :] = 0
        return context.to(device=self.device, dtype=self.dtype)

    def new_causal_cache(
        self,
        *,
        batch_size: int,
        geometry: CausVidGeometry,
    ) -> CausVidCache:
        frame_seq_length = self._frame_seq_length(geometry)
        token_capacity = geometry.latent_frames * frame_seq_length
        kv_cache = [
            {
                "k": torch.zeros(
                    (batch_size, token_capacity, self.num_heads, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                ),
                "v": torch.zeros(
                    (batch_size, token_capacity, self.num_heads, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                ),
            }
            for _ in range(self.num_layers)
        ]
        crossattn_cache = [
            {
                "k": torch.zeros(
                    (batch_size, self.text_len, self.num_heads, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                ),
                "v": torch.zeros(
                    (batch_size, self.text_len, self.num_heads, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                ),
                "is_init": False,
            }
            for _ in range(self.num_layers)
        ]
        return CausVidCache(kv_cache=kv_cache, crossattn_cache=crossattn_cache)

    def predict_x0_cached(
        self,
        noisy_chunk: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        timestep: int,
        cache: CausVidCache,
        chunk_index: int,
        geometry: CausVidGeometry,
    ) -> torch.Tensor:
        batch_size, chunk_frames = int(noisy_chunk.shape[0]), int(noisy_chunk.shape[1])
        timestep_tensor = torch.full(
            (batch_size, chunk_frames),
            int(timestep),
            device=noisy_chunk.device,
            dtype=torch.long,
        )
        frame_seq_length = self._frame_seq_length(geometry)
        current_start = chunk_index * geometry.frames_per_chunk * frame_seq_length
        current_end = current_start + chunk_frames * frame_seq_length
        flow = self.transformer(
            noisy_chunk.permute(0, 2, 1, 3, 4),
            t=timestep_tensor,
            context=prompt_embeds,
            seq_len=geometry.latent_frames * frame_seq_length,
            kv_cache=cache.kv_cache,
            crossattn_cache=cache.crossattn_cache,
            current_start=current_start,
            current_end=current_end,
        ).permute(0, 2, 1, 3, 4)
        return self._flow_to_x0(flow, noisy_chunk, timestep_tensor)

    def predict_x0_full_prefix(
        self,
        prefix_and_chunk: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        per_frame_timesteps: torch.Tensor,
        geometry: CausVidGeometry,
    ) -> torch.Tensor:
        """Differentiable block-causal forward with no mutable K/V cache."""

        frame_count = int(prefix_and_chunk.shape[1])
        frame_seq_length = self._frame_seq_length(geometry)
        mask_key = (frame_count, geometry.latent_height, geometry.latent_width)
        self._causal_core.num_frame_per_block = geometry.frames_per_chunk
        self._causal_core.block_mask = self._block_masks.get(mask_key)
        flow = self.transformer(
            prefix_and_chunk.permute(0, 2, 1, 3, 4),
            t=per_frame_timesteps,
            context=prompt_embeds,
            seq_len=frame_count * frame_seq_length,
            kv_cache=None,
        ).permute(0, 2, 1, 3, 4)
        if self._causal_core.block_mask is not None:
            self._block_masks[mask_key] = self._causal_core.block_mask
        return self._flow_to_x0(flow, prefix_and_chunk, per_frame_timesteps)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("CausVid replay backend does not own a VAE")
        vae_dtype = next(self.vae.parameters()).dtype
        channel_first = latents.permute(0, 2, 1, 3, 4).to(dtype=vae_dtype)
        mean = torch.tensor(
            [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ],
            device=channel_first.device,
            dtype=vae_dtype,
        )
        std = torch.tensor(
            [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ],
            device=channel_first.device,
            dtype=vae_dtype,
        )
        scale = [mean, 1.0 / std]
        with torch.no_grad():
            decoded = [
                self.vae.decode(row.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                for row in channel_first
            ]
        # Upstream VAE rows are [C,T,H,W]; VRL's generation/reward wire contract
        # is [B,C,T,H,W], so keep the stacked channel-first layout.
        video = torch.stack(decoded, dim=0)
        return (video * 0.5 + 0.5).clamp_(0, 1)

    def move_frozen_components(self, device: Any) -> None:
        if self.text_encoder is not None:
            self.text_encoder.to(device)
        if self.vae is not None:
            self.vae.to(device)

    def _frame_seq_length(self, geometry: CausVidGeometry) -> int:
        spatial_patch_area = self.patch_size[1] * self.patch_size[2]
        return geometry.latent_height * geometry.latent_width // spatial_patch_area

    @staticmethod
    def _sigma_for_timesteps(timesteps: torch.Tensor) -> torch.Tensor:
        # Reproduce upstream FlowMatchScheduler's float32 table and nearest-index
        # lookup exactly (1000 training entries, shift=8, sigma_min=0).
        base = torch.linspace(1.0, 0.0, 1001, device=timesteps.device, dtype=torch.float32)[:-1]
        sigmas = 8.0 * base / (1.0 + 7.0 * base)
        table_timesteps = sigmas * 1000.0
        indices = torch.argmin(
            (table_timesteps.double().view(1, 1, -1) - timesteps.double().unsqueeze(-1)).abs(),
            dim=-1,
        )
        return sigmas[indices]

    def _flow_to_x0(
        self,
        flow: torch.Tensor,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (
            self._sigma_for_timesteps(timesteps)
            .double()
            .view(
                int(noisy.shape[0]),
                int(noisy.shape[1]),
                1,
                1,
                1,
            )
        )
        # Mirror upstream WanDiffusionWrapper exactly: the schedule originated
        # in float32, then sigma/flow/x_t are promoted to float64 for x0 math.
        return (noisy.double() - sigma * flow.double()).to(flow.dtype)


class _CausVidPolicyModel(LoraModelMixin, DiffusionModelBase):
    """Shared trainable causal transformer surface for rollout and replay."""

    # Grouped replay: every [chunk, transition] action is replayed in ONE call,
    # so there is no timestep axis to select (unlike the denoise-family default
    # this class inherits).
    replay_indexes_timesteps: ClassVar[bool] = False

    _lora_default_init_weights: Any = True

    def __init__(
        self,
        *,
        backend: Any,
        geometry: CausVidGeometry = OFFICIAL_CAUSVID_GEOMETRY,
        schedule: CausVidSchedule = OFFICIAL_CAUSVID_SCHEDULE,
        generation_enabled: bool,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_backend", backend)
        self.transformer = backend.transformer
        self.geometry = geometry
        self._schedule = schedule
        self._generation_enabled = generation_enabled

    @property
    def device(self) -> Any:
        return self._backend.device

    @property
    def scheduler(self) -> CausVidSchedule:
        # Grouped CausVid replay owns a discrete causal recipe, not a diffusers
        # scheduler. RuntimeBundle still exposes it at the standard boundary.
        return self._schedule

    @property
    def raw_handle(self) -> Any:
        return self._backend if self._generation_enabled else None

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        setter = getattr(self._backend, "set_transformer", None)
        if callable(setter):
            setter(transformer)
        else:
            self._backend.transformer = transformer

    def enable_gradient_checkpointing(self) -> None:
        core = getattr(self._backend, "_causal_core", self.transformer)
        enabler = getattr(core, "enable_gradient_checkpointing", None)
        if not callable(enabler):
            raise AttributeError("CausVid transformer has no gradient-checkpointing hook")
        enabler()

    def set_num_steps(self, n: int) -> None:
        if int(n) != 3:
            raise ValueError("released CausVid support requires sampling.num_steps=3")

    # These methods preserve the denoise-family ABC shape. The custom causal
    # executor owns generation as one grouped trajectory, so generic single-step
    # prepare/forward entry points are intentionally unreachable.
    def prepare_sampling(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CausVid sampling must use CausVidChunkExecutor")

    def forward_step(self, state: Any, step_idx: int) -> dict[str, Any]:
        del state, step_idx
        raise RuntimeError("CausVid replay is trajectory-grouped, not single-step")

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if not self._generation_enabled:
            raise RuntimeError(f"{type(self).__name__} cannot encode prompts")
        if negative_prompt not in (None, "", []):
            raise ValueError("released CausVid does not support negative prompts or CFG")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        return {"prompt_embeds": self._backend.encode_prompts(prompts)}

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if not self._generation_enabled:
            raise RuntimeError(f"{type(self).__name__} cannot decode latents")
        return self._backend.decode_latents(latents)

    def move_frozen_components(self, device: Any) -> None:
        mover = getattr(self._backend, "move_frozen_components", None)
        if callable(mover):
            mover(device)

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        """Replay all ``[chunk, transition]`` actions in one grouped call."""

        self.reject_replay_timestep_selection(timestep_idx)
        self.reject_unsupported_replay_segments(request)
        from vrl.trajectory import TrajectoryResolver

        replay = TrajectoryResolver.from_batch(batch).replay_tensor_dict(
            "denoise",
            device=self.device,
        )
        log_probs = self.replay_log_probs(
            observations=replay["observations"],
            actions=replay["actions"],
            timesteps=replay["timesteps"],
            next_sigmas=replay["next_sigmas"],
            finalized_chunk_latents=replay["finalized_chunk_latents"],
            prompt_embeds=replay["prompt_embeds"],
        )
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values={"log_probs": log_probs},
                ),
            },
        )

    def replay_log_probs(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        next_sigmas: torch.Tensor,
        finalized_chunk_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiably score the full causal trajectory as ``[B,C,S]``.

        Each current chunk is recomputed together with every finalized earlier
        chunk under upstream block-causal attention and ``kv_cache=None``.  This
        avoids autograd-unsafe mutable K/V state and exactly reproduces the clean
        prefix boundary.  The temporary transformer work is intentionally
        ``O(C^2)`` in temporal chunks; optimizing it requires a differentiable
        cache design and is not part of this correctness-first implementation.
        """

        batch_size, chunk_count, transition_count = self._validate_replay_tensors(
            observations=observations,
            actions=actions,
            timesteps=timesteps,
            next_sigmas=next_sigmas,
            finalized_chunk_latents=finalized_chunk_latents,
            prompt_embeds=prompt_embeds,
        )
        frames_per_chunk = self.geometry.frames_per_chunk
        per_chunk: list[torch.Tensor] = []
        prefix_chunks = finalized_chunk_latents.detach()
        prompt_embeds = prompt_embeds.detach()

        for chunk_index in range(chunk_count):
            if chunk_index:
                prefix = prefix_chunks[:, :chunk_index].reshape(
                    batch_size,
                    chunk_index * frames_per_chunk,
                    *observations.shape[-3:],
                )
            else:
                prefix = observations.new_empty(
                    (batch_size, 0, *observations.shape[-3:]),
                )
            per_transition: list[torch.Tensor] = []
            for transition_index in range(transition_count):
                current = observations[:, chunk_index, transition_index]
                full_prefix = torch.cat((prefix, current), dim=1)
                current_timestep = timesteps[:, chunk_index, transition_index]
                per_frame_timestep = torch.cat(
                    (
                        torch.zeros(
                            (batch_size, int(prefix.shape[1])),
                            device=current.device,
                            dtype=current_timestep.dtype,
                        ),
                        current_timestep.view(batch_size, 1).expand(
                            batch_size,
                            frames_per_chunk,
                        ),
                    ),
                    dim=1,
                )
                with self._autocast_context():
                    full_x0 = self._backend.predict_x0_full_prefix(
                        full_prefix,
                        prompt_embeds=prompt_embeds,
                        per_frame_timesteps=per_frame_timestep,
                        geometry=self.geometry,
                    )
                current_x0 = full_x0[:, -frames_per_chunk:]
                score = renoise_step_with_logprob(
                    current_x0,
                    next_sigmas[:, chunk_index, transition_index],
                    next_sample=actions[:, chunk_index, transition_index],
                    math_dtype=torch.float32,
                )
                per_transition.append(score.log_prob)
            per_chunk.append(torch.stack(per_transition, dim=1))
        return torch.stack(per_chunk, dim=1)

    def _autocast_context(self) -> Any:
        # RuntimeBundle always stamps precision. Direct fake-backend unit tests
        # intentionally construct the lightweight model without bundle assembly.
        if getattr(self, "precision", None) is None:
            return contextlib.nullcontext()
        return model_autocast(self, self.device)

    def _validate_replay_tensors(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        next_sigmas: torch.Tensor,
        finalized_chunk_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> tuple[int, int, int]:
        if observations.ndim != 7:
            raise ValueError("CausVid observations must be [B,C,S,F,channels,H,W]")
        batch_size, chunk_count, transition_count = map(int, observations.shape[:3])
        expected_prefix = (batch_size, chunk_count, transition_count)
        for name, value in (
            ("actions", actions),
            ("timesteps", timesteps),
            ("next_sigmas", next_sigmas),
        ):
            actual = tuple(int(length) for length in value.shape[:3])
            if actual != expected_prefix:
                raise ValueError(f"{name} leading shape must be {expected_prefix}, got {actual}")
        if tuple(actions.shape) != tuple(observations.shape):
            raise ValueError("CausVid actions shape must match observations")
        expected_chunks = self.geometry.temporal_chunk_count
        if chunk_count != expected_chunks or transition_count != self._schedule.transition_count:
            raise ValueError(
                "CausVid replay policy axes do not match its runtime recipe: "
                f"got chunks={chunk_count}, transitions={transition_count}; "
                f"expected {expected_chunks}, {self._schedule.transition_count}",
            )
        expected_finalized = (
            batch_size,
            chunk_count,
            self.geometry.frames_per_chunk,
            self.geometry.latent_channels,
            self.geometry.latent_height,
            self.geometry.latent_width,
        )
        if tuple(finalized_chunk_latents.shape) != expected_finalized:
            raise ValueError(
                "finalized_chunk_latents shape must be "
                f"{expected_finalized}, got {tuple(finalized_chunk_latents.shape)}",
            )
        if int(prompt_embeds.shape[0]) != batch_size:
            raise ValueError("prompt_embeds must be sample-aligned")
        return batch_size, chunk_count, transition_count


class CausVidModel(_CausVidPolicyModel):
    """Full CausVid rollout model: prompt encoder, causal policy, and VAE."""

    def __init__(
        self,
        *,
        backend: Any,
        geometry: CausVidGeometry = OFFICIAL_CAUSVID_GEOMETRY,
        schedule: CausVidSchedule = OFFICIAL_CAUSVID_SCHEDULE,
    ) -> None:
        super().__init__(
            backend=backend,
            geometry=geometry,
            schedule=schedule,
            generation_enabled=True,
        )

    @classmethod
    def from_build(cls, build: ModelBuild) -> CausVidModel:
        build.require_rollout()
        _validate_released_build(build)
        backend = _load_official_backend(build, generation=True)
        return cls(backend=backend)

    def generate_chunk_autoregressive(
        self,
        *,
        request: Any,
        batch: Any,
    ) -> ChunkAutoregressiveDenoiseResult:
        """Generate one transport sample batch through seven temporal chunks."""

        self._validate_generation_request(request)
        batch_size = int(batch.sample_count)
        prompt = request.inputs[batch.prompt_index].prompt
        encoded = self.encode_prompt([prompt] * batch_size)
        generators = self._sample_generators(request, batch)
        noise_rows = [
            torch.randn(
                self.geometry.latent_shape,
                generator=generator,
                device=self.device,
                dtype=self._backend.dtype,
            )
            for generator in generators
        ]
        initial_noise = torch.stack(noise_rows, dim=0)
        run = CausVidCausalRunner(
            self._backend,
            geometry=self.geometry,
            schedule=self._schedule,
        ).run(
            initial_noise,
            prompt_embeds=encoded["prompt_embeds"],
            generators=generators,
        )
        video = self.decode_latents(run.latents)
        if isinstance(video, torch.Tensor) and video.is_floating_point():
            from vrl.utils.media import to_uint8

            video = to_uint8(video)
        mapping = run.trajectory_mapping(context={})
        return ChunkAutoregressiveDenoiseResult(
            batch=batch,
            output=video,
            **mapping,
        )

    def _validate_generation_request(self, request: Any) -> None:
        sampling = dict(request.sampling)
        expected = {
            "width": self.geometry.pixel_width,
            "height": self.geometry.pixel_height,
            "num_frames": self.geometry.pixel_frames,
            "fps": self.geometry.fps,
            "num_steps": 3,
        }
        for key, expected_value in expected.items():
            actual = int(sampling.get(key, expected_value))
            if actual != expected_value:
                raise ValueError(
                    f"released CausVid requires sampling.{key}={expected_value}, got {actual}",
                )
        guidance_scale = float(sampling.get("guidance_scale", 1.0))
        if guidance_scale != 1.0:
            raise ValueError("released CausVid requires guidance_scale=1 (no CFG)")
        negative_prompt = sampling.get("negative_prompt")
        if negative_prompt not in (None, ""):
            raise ValueError("released CausVid does not support a negative prompt")

    def _sample_generators(self, request: Any, batch: Any) -> tuple[torch.Generator, ...]:
        seed = request.sampling.get("seed")
        generators: list[torch.Generator] = []
        for local_index in range(int(batch.sample_count)):
            generator = torch.Generator(device=torch.device(self.device))
            if seed is None:
                generator.seed()
            else:
                flat_sample_index = (
                    int(batch.prompt_index) * int(request.samples_per_prompt)
                    + int(batch.sample_start)
                    + local_index
                )
                generator.manual_seed(int(seed) + flat_sample_index)
            generators.append(generator)
        return tuple(generators)


class CausVidReplayModel(_CausVidPolicyModel):
    """Transformer-only trainer model for grouped differentiable replay."""

    def __init__(
        self,
        *,
        backend: Any,
        geometry: CausVidGeometry = OFFICIAL_CAUSVID_GEOMETRY,
        schedule: CausVidSchedule = OFFICIAL_CAUSVID_SCHEDULE,
    ) -> None:
        super().__init__(
            backend=backend,
            geometry=geometry,
            schedule=schedule,
            generation_enabled=False,
        )

    @classmethod
    def from_build(cls, build: ModelBuild) -> CausVidReplayModel:
        build.require_replay()
        _validate_released_build(build)
        backend = _load_official_backend(build, generation=False)
        return cls(backend=backend)


def _validate_released_build(build: ModelBuild) -> None:
    model_config = build.model_config or {}
    _require_noncommercial_license(model_config)
    sampling = build.sampling_config or {}
    expected_sampling = {
        "width": OFFICIAL_CAUSVID_GEOMETRY.pixel_width,
        "height": OFFICIAL_CAUSVID_GEOMETRY.pixel_height,
        "num_frames": OFFICIAL_CAUSVID_GEOMETRY.pixel_frames,
        "fps": OFFICIAL_CAUSVID_GEOMETRY.fps,
        "num_steps": 3,
    }
    for key, expected in expected_sampling.items():
        if key in sampling and int(sampling[key]) != expected:
            raise ValueError(
                f"released CausVid initially supports sampling.{key}={expected}; "
                f"got {sampling[key]!r}",
            )
    if "guidance_scale" in sampling and float(sampling["guidance_scale"]) != 1.0:
        raise ValueError("released CausVid requires sampling.guidance_scale=1")
    if sampling.get("negative_prompt") not in (None, ""):
        raise ValueError("released CausVid does not support negative prompts")


def _require_noncommercial_license(model_config: Mapping[str, Any]) -> None:
    accepted = model_config.get("accept_noncommercial_license")
    if accepted is not True:
        raise ValueError(
            "CausVid released weights are CC BY-NC-SA 4.0 and restricted to "
            "non-commercial use. Set model.accept_noncommercial_license=true "
            "to acknowledge those terms before source/weight resolution.",
        )


def _load_official_backend(build: ModelBuild, *, generation: bool) -> _OfficialCausVidBackend:
    artifacts = _resolve_artifacts(build)

    # Importing this module triggers upstream's FlexAttention compile; this is
    # intentionally inside the explicit heavy-load path and nowhere else.
    from causvid.models.wan.causal_model import CausalWanModel

    transformer = CausalWanModel.from_pretrained(str(artifacts.base_model_dir))
    transformer.num_frame_per_block = OFFICIAL_CAUSVID_GEOMETRY.frames_per_chunk
    generator_state = _load_generator_state_dict(artifacts.checkpoint_file)
    transformer.load_state_dict(generator_state, strict=True)
    transformer.eval().to(build.device, dtype=build.parameter_dtype)

    text_encoder = None
    tokenizer = None
    vae = None
    if generation:
        from causvid.models.wan.wan_base.modules.t5 import umt5_xxl
        from causvid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
        from causvid.models.wan.wan_base.modules.vae import _video_vae

        base = artifacts.base_model_dir
        prompt_dtype = build.require_rollout().prompt_encoder_dtype
        text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )
        text_state = torch.load(
            base / "models_t5_umt5-xxl-enc-bf16.pth",
            map_location="cpu",
            weights_only=True,
        )
        text_encoder.load_state_dict(text_state, strict=True)
        text_encoder.to(build.device, dtype=prompt_dtype)
        tokenizer = HuggingfaceTokenizer(
            name=str(base / "google" / "umt5-xxl"),
            seq_len=512,
            clean="whitespace",
        )
        vae = (
            _video_vae(
                pretrained_path=str(base / "Wan2.1_VAE.pth"),
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
            .to(build.device, dtype=torch.float32)
        )

    return _OfficialCausVidBackend(
        transformer=transformer,
        parameter_dtype=build.parameter_dtype,
        device=build.device,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
    )


def _resolve_artifacts(build: ModelBuild) -> CausVidResolvedArtifacts:
    model_config = build.model_config or {}
    _require_noncommercial_license(model_config)
    source_root = _resolve_source_root(model_config)
    # Fail on a missing/wrong editable install before either multi-gigabyte
    # checkpoint resolver is allowed to touch the Hub cache.
    _require_pinned_source_import(source_root)
    _require_causvid_flash_attention()
    base_model_dir = _resolve_base_model(model_config)
    checkpoint = _resolve_checkpoint(build)
    return CausVidResolvedArtifacts(
        base_model_dir=base_model_dir,
        checkpoint_file=checkpoint,
    )


def _resolve_source_root(model_config: Mapping[str, Any]) -> Path:
    requested_revision = str(
        model_config.get("causvid_source_revision", CAUSVID_SOURCE_REVISION),
    )
    if requested_revision != CAUSVID_SOURCE_REVISION:
        raise ValueError(
            "CausVid source revision mismatch: the adapter supports only audited "
            f"revision {CAUSVID_SOURCE_REVISION}, got {requested_revision}",
        )
    configured = model_config.get("causvid_source_path")
    if configured:
        source_root = _resolve_configured_path(str(configured))
    else:
        repo_default = Path(__file__).resolve().parents[4] / "third_party" / "CausVid"
        if repo_default.is_dir():
            source_root = repo_default
        else:
            spec = importlib.util.find_spec("causvid")
            if spec is None or spec.origin is None:
                raise FileNotFoundError(
                    "CausVid source checkout not found. Initialize third_party/CausVid "
                    "or set model.causvid_source_path to a pinned checkout.",
                )
            source_root = Path(spec.origin).resolve().parent.parent
    if not (source_root / "causvid").is_dir():
        raise FileNotFoundError(f"CausVid source root has no causvid package: {source_root}")
    revision = _verified_source_revision(source_root)
    if revision != requested_revision:
        raise ValueError(
            "CausVid source revision mismatch: "
            f"expected {requested_revision}, got {revision} at {source_root}",
        )
    return source_root


def _resolve_base_model(model_config: Mapping[str, Any]) -> Path:
    reference = str(model_config.get("base_model_path", CAUSVID_BASE_REPOSITORY))
    local = _resolve_configured_path(reference)
    revision_value = model_config.get("base_model_revision")
    revision = str(revision_value) if revision_value else None
    if local.is_dir():
        return local
    if revision is None:
        raise ValueError(
            "remote CausVid base_model_path requires immutable model.base_model_revision",
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=reference, revision=revision))


def _resolve_checkpoint(build: ModelBuild) -> Path:
    model_config = build.model_config or {}
    reference = str(build.model_name_or_path)
    relative_file = str(model_config.get("checkpoint_file", CAUSVID_CHECKPOINT_FILE))
    local = _resolve_configured_path(reference)
    revision = build.revision
    if local.is_file():
        checkpoint = local
    elif local.is_dir():
        relative_file = require_checkpoint_source_member(
            relative_file,
            field_name="model.checkpoint_file",
        )
        checkpoint = local / relative_file
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CausVid checkpoint file not found: {checkpoint}")
    else:
        relative_file = require_checkpoint_source_member(
            relative_file,
            field_name="model.checkpoint_file",
        )
        if revision is None and reference == CAUSVID_CHECKPOINT_REPOSITORY:
            revision = CAUSVID_CHECKPOINT_REVISION
        if revision is None:
            raise ValueError(
                "remote CausVid checkpoint repository requires immutable model.revision",
            )
        from huggingface_hub import hf_hub_download

        checkpoint = Path(
            hf_hub_download(
                repo_id=reference,
                filename=relative_file,
                revision=revision,
            ),
        )

    expected_sha = model_config.get("checkpoint_sha256")
    if (
        expected_sha is None
        and reference == CAUSVID_CHECKPOINT_REPOSITORY
        and relative_file == CAUSVID_CHECKPOINT_FILE
    ):
        expected_sha = CAUSVID_CHECKPOINT_SHA256
    if expected_sha is not None:
        actual_sha = sha256_file(checkpoint)
        if actual_sha != str(expected_sha).lower():
            raise ValueError(
                f"CausVid checkpoint SHA256 mismatch for {checkpoint}: "
                f"expected {expected_sha}, got {actual_sha}",
            )
    return checkpoint


def _load_generator_state_dict(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "generator" not in payload:
        raise ValueError("CausVid model.pt must contain a top-level 'generator' state dict")
    generator = payload["generator"]
    if not isinstance(generator, Mapping) or not generator:
        raise ValueError("CausVid model.pt['generator'] must be a non-empty state dict")
    keys = tuple(generator)
    invalid = [key for key in keys if not isinstance(key, str) or not key.startswith("model.")]
    if invalid:
        preview = invalid[:3]
        raise ValueError(
            "CausVid generator keys must all use the official single 'model.' prefix; "
            f"invalid examples: {preview}",
        )
    stripped = {str(key)[len("model.") :]: value for key, value in generator.items()}
    if len(stripped) != len(generator):
        raise ValueError("CausVid generator key normalization produced duplicate keys")
    return stripped


def _resolve_configured_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    repo_relative = Path(__file__).resolve().parents[4] / path
    return repo_relative if repo_relative.exists() else path


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if result.returncode or len(revision) != 40:
        raise ValueError(
            f"cannot verify immutable CausVid source revision at {path}: "
            f"{result.stderr.strip() or 'not a Git checkout'}",
        )
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode or status.stdout.strip():
        detail = status.stdout.strip() or status.stderr.strip()
        raise ValueError(
            f"CausVid source checkout at {path} has tracked modifications: {detail}",
        )
    return revision


def _verified_source_revision(path: Path) -> str:
    """Verify a checkout locally or its exact executable subset in a Ray archive."""

    if (path / ".git").exists():
        return _git_revision(path)

    actual_digest = runtime_source_tree_sha256(
        path,
        include_globs=_CAUSVID_RUNTIME_SOURCE_GLOBS,
        exclude_prefixes=_CAUSVID_RUNTIME_SOURCE_EXCLUDES,
    )
    if actual_digest != CAUSVID_RUNTIME_SOURCE_SHA256:
        raise ValueError(
            "CausVid packaged runtime source digest mismatch: expected "
            f"{CAUSVID_RUNTIME_SOURCE_SHA256}, got {actual_digest} at {path}",
        )
    return CAUSVID_SOURCE_REVISION


def _require_pinned_source_import(source_root: Path) -> None:
    """Require the environment import to resolve to the verified checkout.

    Mutating ``sys.path`` here would not repair an already imported, different
    ``causvid`` module and could silently mix two source revisions. The vendored
    third-party editable install is the ownership boundary: verify it resolves
    to this checkout before importing the compile-heavy causal model.
    """

    expected_package = (source_root / "causvid").resolve()
    packaged_source = not (source_root / ".git").exists()
    if packaged_source:
        # This private function normally follows _resolve_source_root; verify
        # again so direct callers cannot bind an unverified namespace package.
        _verified_source_revision(source_root)
    spec = importlib.util.find_spec("causvid")
    locations = getattr(spec, "submodule_search_locations", None) or ()
    filesystem_locations = {
        Path(location).resolve() for location in locations if Path(location).is_dir()
    }
    if filesystem_locations == {expected_package}:
        pass
    elif not packaged_source:
        raise ImportError(
            "the pinned CausVid checkout is not importable as the active source; "
            "install the vendored packages with `pip install -e third_party`. Expected "
            f"{expected_package}, got {sorted(map(str, filesystem_locations))}",
        )
    else:
        loaded = tuple(
            name for name in sys.modules if name == "causvid" or name.startswith("causvid.")
        )
        if loaded:
            raise ImportError(
                "cannot bind the verified packaged CausVid source after another "
                f"causvid package was imported: {loaded[:3]}",
            )
        # CausVid is an upstream namespace package with no __init__.py. Bind its
        # already-verified archive root explicitly, without mutating sys.path or
        # allowing another installed revision to win import resolution.
        package_spec = importlib.machinery.PathFinder.find_spec(
            "causvid",
            [str(source_root.resolve())],
        )
        package_locations = getattr(package_spec, "submodule_search_locations", None) or ()
        if package_spec is None or {
            Path(location).resolve() for location in package_locations if Path(location).is_dir()
        } != {expected_package}:
            raise ImportError(
                f"verified packaged CausVid namespace is not importable at {expected_package}",
            )
        sys.modules["causvid"] = importlib.util.module_from_spec(package_spec)

    for module_name, module in tuple(sys.modules.items()):
        if module_name != "causvid" and not module_name.startswith("causvid."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        resolved = Path(module_file).resolve()
        if not resolved.is_relative_to(expected_package):
            raise ImportError(
                f"already-imported {module_name} came from unpinned path {resolved}; "
                "restart the process with only the vendored CausVid checkout installed",
            )


def _require_causvid_flash_attention() -> None:
    """Require the backend called directly by upstream Wan cross-attention."""

    failures: list[str] = []
    for module_name in ("flash_attn_interface", "flash_attn"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        except Exception as error:
            failures.append(f"{module_name}: {error}")
        else:
            return
    detail = f" Unusable installs: {'; '.join(failures)}" if failures else ""
    raise ImportError(
        "CausVid's pinned Wan cross-attention requires FlashAttention 2 or 3. "
        "Install a CUDA-compatible build before resolving weights, for example "
        "`uv pip install --no-build-isolation flash-attn`." + detail,
    )


__all__ = [
    "CAUSVID_BASE_REPOSITORY",
    "CAUSVID_CHECKPOINT_FILE",
    "CAUSVID_CHECKPOINT_REPOSITORY",
    "CAUSVID_CHECKPOINT_REVISION",
    "CAUSVID_CHECKPOINT_SHA256",
    "CAUSVID_RUNTIME_SOURCE_SHA256",
    "CAUSVID_SOURCE_REVISION",
    "CausVidCache",
    "CausVidModel",
    "CausVidReplayModel",
    "CausVidResolvedArtifacts",
]
