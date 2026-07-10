"""Shared diffusion model base for RL runtimes.

The public trainer-facing replay interface is ``vrl.models.interfaces.ReplayModel``.
This base class only factors shared diffusion model behavior: generation
primitives, replay-state projection helpers, and trainable transformer weight
loading for diffusion families.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult
from vrl.models.utils import (
    TrainableStateSlots,
    activate_adapter_on,
    disable_adapter_on,
    load_weights_into,
)
from vrl.trajectory.device import move_value_to_device


@dataclass
class DiffusionSamplingStateBase:
    """Engine-contract fields shared by every family's private sampling state.

    The chunk executor only ever touches ``latents`` (read/write),
    ``timesteps`` and ``scheduler`` — nothing else. Every other field a
    family declares in its subclass is private to its own ``forward_step``
    / replay path and MUST NOT be introspected by the engine.
    ``guidance_scale`` is engine-invisible but present in all 17 families,
    so it is lifted here purely for dedup.
    """

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    guidance_scale: float


class DiffusionModelBase(nn.Module, ABC):
    """Shared model base for diffusion families on the RL path."""

    # Some upstream diffusion-RL recipes intentionally keep LoRA replay outside
    # autocast. The trainer reads this flag when choosing the replay context.
    disable_train_autocast: bool = False

    async def load(self) -> None:
        """Load heavy modules. Default no-op for adapters constructed eagerly."""
        return None

    @abstractmethod
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt and optional negative prompt into embedding tensors."""

    @abstractmethod
    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Build a private per-family sampling state for the denoise loop."""

    @abstractmethod
    def forward_step(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Run one transformer forward without stepping the scheduler."""

    def forward(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Run one trainable denoise transformer step."""

        return self.forward_step(state, step_idx)

    @abstractmethod
    def decode_latents(self, latents: Any) -> Any:
        """Decode latents to a frame tensor."""

    def export_batch_context(self, state: Any) -> dict[str, Any]:
        """Project private sampling state into shared trajectory context."""
        raise NotImplementedError

    def export_replay_tensors(self, state: Any) -> dict[str, Any]:
        """Project private sampling state into per-sample trajectory tensors."""
        raise NotImplementedError

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        """Rebuild private sampling state for trainer replay."""
        raise NotImplementedError

    def prepare_replay(self, spec: Any) -> None:
        """Family hook run once by the replay builder right after construction.

        Default no-op. FLUX overrides it to set its dynamic-shift replay
        timesteps (the replay scheduler must carry the same mu-shifted schedule
        the rollout used); the spec carries the sampling block it derives from.
        """
        del spec
        return None

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        """Rebuild diffusion sampling state and run one replay forward."""
        del request
        replay_tensors, batch_context, latents = self._replay_inputs_for_step(
            batch,
            timestep_idx,
        )
        state = self.restore_eval_state(
            replay_tensors,
            batch_context,
            latents,
            timestep_idx,
        )
        values = self.forward(state, self._replay_forward_step_index(timestep_idx))
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values=dict(values),
                ),
            },
        )

    def replay_forward_with_latents(
        self,
        batch: Any,
        timestep_idx: int,
        latents: Any,
    ) -> dict[str, Any]:
        """One replay-shaped forward on CALLER latents at the step's conditioning.

        Same state rebuild as ``replay_forward`` — the trajectory's prompt
        conditioning and the step's own timestep — but the model input is the
        caller's tensor instead of the stored x_t. This is the GRPO
        diffusion-loss regularizer's entry: it noises CLEAN fine-tuning
        latents and needs the model's prediction for them under the exact
        schedule replay uses (no second sigma-domain conversion path).
        """

        replay_tensors, batch_context, _ = self._replay_inputs_for_step(
            batch,
            timestep_idx,
        )
        state = self.restore_eval_state(
            replay_tensors,
            batch_context,
            latents,
            timestep_idx,
        )
        return dict(
            self.forward(
                state,
                self._replay_forward_step_index(timestep_idx),
            ),
        )

    def _replay_forward_step_index(self, timestep_idx: int) -> int:
        """Map a trajectory step to the index the rebuilt family state expects.

        Most families rebuild a one-step state and therefore forward at index
        zero. Cosmos keeps the full scheduler sigma table in replay state, so
        its shared protocol mixin returns the real trajectory index instead.
        Both normal replay and caller-latent replay use this hook; otherwise the
        same noisy input could be evaluated under two different sigma values.
        """

        del timestep_idx
        return 0

    def _replay_inputs_for_step(
        self,
        batch: Any,
        timestep_idx: int,
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        """Resolve only the current denoise step's replay tensors on model device."""

        from vrl.trajectory import TrajectoryResolver

        try:
            device = self.device
        except Exception:
            device = None
        replay_tensors = TrajectoryResolver.from_batch(batch).replay_tensor_dict(
            "denoise",
            axis="denoise",
            axis_index=timestep_idx,
            device=device,
        )
        latents = move_value_to_device(batch.observations[:, timestep_idx], device)
        return replay_tensors, dict(batch.context), latents

    def _require_transformer(self) -> Any:
        """Return the registered trainable transformer."""

        transformer = getattr(self, "transformer", None)
        if transformer is None:
            raise RuntimeError(
                f"{type(self).__name__} has no registered trainable transformer",
            )
        return transformer

    def _transformer_dtype(self) -> torch.dtype:
        """Return the dtype of the current trainable transformer."""

        transformer = self._require_transformer()
        dtype = getattr(transformer, "dtype", None)
        if dtype is not None:
            return dtype
        try:
            return next(transformer.parameters()).dtype
        except StopIteration as exc:
            raise RuntimeError(
                f"{type(self).__name__} transformer has no parameters to infer dtype",
            ) from exc

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        """Disable LoRA/adapters, or return a no-op context when absent."""

        return disable_adapter_on(self._require_transformer())

    def activate_adapter(self, name: str) -> contextlib.AbstractContextManager[None]:
        """Activate the named LoRA/PEFT adapter for a forward pass.

        The named-adapter counterpart to :meth:`disable_adapter`; restores the
        ``"default"`` adapter on exit and switches on the module behind any
        DDP / compile wrapper. Centralizing it here keeps algorithms (e.g. the
        DiffusionNFT previous-policy branch) off ``transformer.set_adapter``
        directly, so adapter control has one boundary that owns the model.
        """

        return activate_adapter_on(self._require_transformer(), name)

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load trainable transformer weights from ``transformer.*`` sync keys."""

        transformer = self._require_transformer()
        return load_weights_into(
            transformer,
            state_dict,
            prefix="transformer",
            label=type(transformer).__name__,
        )

    # -- versioned trainable-state slots (non-draining weight sync) ---------
    # Diffusion families support versioned slots generically: activation reuses
    # ``load_trainable_state`` to copy a retained version onto the live model, so
    # the same flat ``transformer.*`` payload format works for single-transformer
    # (sd3/cosmos) and multi-transformer (wan) families without per-family code.
    # See SPRINT_shadow_model_weight_sync.md.
    supports_versioned_trainable_state: bool = True

    def _versioned_state_slots(self) -> TrainableStateSlots:
        slots = getattr(self, "_trainable_state_slots", None)
        if slots is None:
            slots = TrainableStateSlots()
            self._trainable_state_slots = slots
        return slots

    def install_trainable_state(
        self,
        version: int,
        state_dict: Mapping[str, Any] | None,
    ) -> None:
        """Retain ``state_dict`` under ``version`` without touching live weights.

        Unlike ``load_trainable_state`` (which overwrites the live model), this
        only stashes the payload so an in-flight request stamped with an older
        version can still be activated after the trainer advances.
        """

        self._versioned_state_slots().install(version, state_dict)

    def has_trainable_state(self, version: int) -> bool:
        return self._versioned_state_slots().has(version)

    def activate_trainable_state(self, version: int) -> None:
        """Make slot ``version`` the live trainable state (idempotent).

        Skips the reload when ``version`` is already active so a request whose
        chunks share one version pays the copy at most once.
        """

        if getattr(self, "_active_slot_version", None) == int(version):
            return
        self.load_trainable_state(self._versioned_state_slots().get(version))
        self._active_slot_version = int(version)

    @classmethod
    def from_spec(cls, spec: Any) -> DiffusionModelBase:  # pragma: no cover (abstract)
        """Load the backend from a runtime spec."""
        raise NotImplementedError

    def apply_lora(self, spec: Any) -> None:  # pragma: no cover (default no-op)
        raise NotImplementedError

    def apply_full_finetune(self) -> None:  # pragma: no cover (default no-op)
        raise NotImplementedError

    def torch_compile_transformer(self, mode: str) -> None:
        """Apply torch.compile to the family transformer in-place."""

        self._set_transformer(
            torch.compile(self.transformer, mode=mode, fullgraph=False),
        )

    def quantize_rollout_fp8(self, recipe: str = "rowwise") -> list[str]:
        """Swap the transformer's big policy GEMMs to fp8 in place (rollout only).

        Replaces the large attention/MLP ``nn.Linear`` modules with ``Fp8Linear``,
        leaving embeddings / the noise-pred head / norm-feeding linears in bf16.
        Default ``rowwise`` (torch, validated); ``blockwise`` opts into vLLM's
        faster block kernel (more GPU memory). Returns the dotted paths quantized.
        This is a generation/rollout-only optimization; the trainer's replay forward
        keeps its bf16/fp32 master and is never quantized. Call before
        ``torch_compile_transformer`` so inductor sees the fp8 modules.
        """

        from vrl.nn.quantization import swap_linears_to_fp8

        return swap_linears_to_fp8(self.transformer, recipe=recipe)

    def set_num_steps(self, n: int) -> None:  # pragma: no cover
        raise NotImplementedError

    @property
    def trainable_modules(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    @property
    def scheduler(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    @property
    def raw_handle(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    def generation_memory_targets(self) -> dict[str, Any]:
        """Named modules the generation memory policy may configure.

        Every diffusers-backed family carries its VAE on ``pipeline.vae``;
        Anima (single-file checkpoint, no pipeline) carries ``self.vae``.
        Replay models raise on ``pipeline`` and own no VAE — they expose no
        targets, and the policy fails loud if config still asks for one.
        """

        try:
            vae = self.pipeline.vae
        except (AttributeError, RuntimeError):
            vae = getattr(self, "vae", None)
        return {} if vae is None else {"vae_decode": vae}

    def move_frozen_components(self, device: Any) -> None:
        """Move frozen pipeline components (VAE / text encoders) onto ``device``.

        ``nn.Module.to`` moves only *registered* submodules — for these families
        just the transformer — but the diffusers pipeline is attached
        unregistered (``object.__setattr__``), so its frozen VAE / text encoders
        stay resident unless moved explicitly. This is the offload-and-restore
        discipline for non-trainable components: parking them on CPU during the
        rollout window frees GPU without discarding and reloading them from disk.
        Device-only — dtype is preserved (frozen VAE stays fp32, encoders keep
        their frozen dtype).

        The set is derived from the diffusers pipeline — every nn.Module
        component except the trainable transformer — so it tracks whatever
        ``from_spec`` froze instead of a hand-kept name list. Families that attach
        no diffusers pipeline (single-file checkpoints, replay models) move
        nothing.
        """

        try:
            pipeline = self.pipeline
        except (AttributeError, RuntimeError):
            return
        components = getattr(pipeline, "components", None)
        if not isinstance(components, Mapping):
            return
        transformer = getattr(self, "transformer", None)
        moved: set[int] = set()
        for module in components.values():
            if not isinstance(module, nn.Module) or module is transformer:
                continue
            if id(module) not in moved:
                moved.add(id(module))
                module.to(device)


def diffusers_pipeline_dtypes(
    spec: Any,
    model_dtype: torch.dtype,
) -> tuple[torch.dtype, dict[str, Any]]:
    """Resolve the frozen-module dtype + ``from_pretrained`` kwargs for a family.

    Frozen text encoders / VAE follow the ``frozen`` precision axis. Its
    default already encodes the Flow-GRPO SD3 contract (fp16 when the denoiser
    runs fp32, else the model dtype), so ``spec.frozen_dtype`` is authoritative
    when present; fall back for bare/test specs. Returns ``(frozen_dtype,
    load_kwargs)`` where ``load_kwargs`` carries the per-component
    ``torch_dtype`` mapping diffusers expects.
    """

    frozen_dtype = getattr(spec, "frozen_dtype", None)
    if frozen_dtype is None:
        frozen_dtype = torch.float16 if model_dtype == torch.float32 else model_dtype
    load_kwargs: dict[str, Any] = {}
    if model_dtype == torch.float32 and frozen_dtype != torch.float32:
        load_kwargs["torch_dtype"] = {
            "transformer": torch.float32,
            "vae": torch.float32,
            "default": frozen_dtype,
        }
    elif model_dtype != torch.float32:
        load_kwargs["torch_dtype"] = model_dtype
    return frozen_dtype, load_kwargs


class DiffusersPipelineModelBase(DiffusionModelBase):
    """Shared shape for families backed by ONE diffusers pipeline + ONE
    trainable transformer (sd3_5, flux, qwen_image, cosmos, wan's primary).

    Factors the members that were byte-identical across those families:
    pipeline/device/scheduler/raw_handle access, transformer swap, the
    single-transformer trainable map, full-finetune, and scheduler timestep
    init. A family overrides only where it genuinely differs (sd3's attention
    processor reinstall on ``_set_transformer``, wan's multi-transformer
    ``trainable_modules``/LoRA, Predict2.5's NFT full-finetune guard).
    Families NOT backed by a diffusers pipeline (echo's LTX wrapper, anima's
    single-file checkpoint) stay on ``DiffusionModelBase`` directly.
    """

    def __init__(self, *, pipeline: Any, device: Any = None) -> None:
        super().__init__()
        # Bypass nn.Module attribute registration: the pipeline is a frozen
        # container, not a trainable submodule.
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self._device = device

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    @property
    def raw_handle(self) -> Any:
        return self.pipeline

    def apply_full_finetune(self) -> None:
        """Mark the transformer fully trainable (no-LoRA path)."""
        self.transformer.requires_grad_(True)
        self.transformer.to(self.device)

    def set_num_steps(self, n: int) -> None:
        """Initialize the scheduler timesteps for sampling.

        Dynamic-shifting FlowMatch schedulers (FLUX, Qwen-Image) derive their
        schedule from a resolution-dependent ``mu`` unknown at build time, so
        the real set is deferred to ``prepare_sampling``; static schedulers
        are set eagerly. Reads ``self.scheduler`` (not ``pipeline.scheduler``)
        so pipeline-less replay subclasses can set their replay scheduler too.
        """
        scheduler = self.scheduler
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            return
        scheduler.set_timesteps(n, device=self.device)


class ReplayRolloutStubs:
    """Rollout-only surface stubs shared by replay models.

    Replay models load only the modules needed to recompute log-probs, so the
    rollout-side ABC methods are unreachable by construction. They raise with
    the concrete class name here instead of each family re-writing the stub.
    """

    def encode_prompt(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(f"{type(self).__name__} cannot encode prompts")

    def prepare_sampling(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{type(self).__name__} cannot run rollout sampling")

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{type(self).__name__} cannot decode latents")


class DiffusersReplayModelBase(ReplayRolloutStubs):
    """Shared shape for transformer-only replay models (no diffusers pipeline).

    Factors the byte-identical members of the per-family ``*ReplayModel``
    classes: the transformer/scheduler/device ctor, the no-pipeline guard, the
    transformer swap, and the scheduler/raw_handle accessors. A family
    overrides only where it genuinely differs (sd3_5 reinstalls its attention
    processor on ``_set_transformer``; flux/mochi/pixart_sigma re-standardize
    their replay scheduler in ``prepare_replay``). ``Cosmos3ReplayModel`` stays
    on ``ReplayRolloutStubs`` directly — it wraps a pipeline SHELL, not a bare
    transformer, and reads ``self.pipeline``.
    """

    def __init__(self, *, transformer: Any, scheduler: Any, device: Any = None) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self._scheduler = scheduler
        self._device = device

    @property
    def pipeline(self) -> Any:
        raise RuntimeError(f"{type(self).__name__} does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def raw_handle(self) -> Any:
        return None


__all__ = [
    "DiffusersPipelineModelBase",
    "DiffusersReplayModelBase",
    "DiffusionModelBase",
    "diffusers_pipeline_dtypes",
]
