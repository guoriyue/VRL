"""Shared diffusion model base for RL runtimes.

The public trainer-facing replay interface is ``vrl.models.interfaces.ReplayModel``.
This base class only factors shared diffusion model behavior: generation
primitives, replay-state projection helpers, and trainable transformer weight
loading for diffusion families.
"""

from __future__ import annotations

import contextlib
import functools
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from vrl.generation.types import VideoGenerationRequest
from vrl.models.interfaces import (
    ReplayRequest,
    ReplayRequestContract,
    ReplayResult,
    ReplaySegmentResult,
)
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.peft_adapter import activate_adapter_on, disable_adapter_on
from vrl.models.precision import model_autocast
from vrl.models.weight_utils import (
    TrainableStateSlots,
    load_weights_into,
    validate_weights_for,
)
from vrl.nn.quantization.targeting import DEFAULT_EXCLUDE


@dataclass
class DiffusionSamplingStateBase:
    """Engine-contract fields shared by every family's private sampling state.

    The batch executor only ever touches ``latents`` (read/write),
    ``timesteps`` and ``scheduler`` — nothing else. Every other field a
    family declares in its subclass is private to its own ``forward_step``
    / replay path and MUST NOT be introspected by the engine.
    """

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any


@dataclass
class GuidedDiffusionSamplingStateBase(DiffusionSamplingStateBase):
    """Private state shared by families whose forward/replay path reads guidance."""

    guidance_scale: float


def _forward_step_with_autocast(fn: Any) -> Any:
    """Apply the selected role dtype at every diffusion forward boundary."""

    @functools.wraps(fn)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with model_autocast(self, self.device):
            return fn(self, *args, **kwargs)

    wrapped.__vrl_outer_autocast__ = True
    return wrapped


class DiffusionModelBase(ReplayRequestContract, nn.Module, ABC):
    """Shared model base for diffusion families on the RL path."""

    replay_segments: ClassVar[tuple[str, ...]] = ("denoise",)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Run every concrete ``forward_step`` under its role precision.

        RuntimeBundle stamps the selected role precision on the model. Wrapping
        ``forward_step`` here means rollout,
        replay (``replay_forward`` and
        ``replay_forward_with_latents`` both funnel through ``self.forward``),
        and direct script calls all receive the same boundary without wrapping
        each call site in ``model_autocast`` by hand.
        """

        super().__init_subclass__(**kwargs)
        fn = cls.__dict__.get("forward_step")
        if fn is not None and not getattr(fn, "__vrl_outer_autocast__", False):
            cls.forward_step = _forward_step_with_autocast(fn)

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

    def diffusion_pretraining_prediction(self, values: dict[str, Any]) -> Any:
        """Return the model output expressed in its pretraining target domain.

        Most families expose that value as the finalized ``noise_pred`` used by
        their scheduler. Families whose CFG branch values have different
        semantics override this seam instead of making the trainer infer model
        math from generic dictionary keys.
        """

        return values["noise_pred"]

    def _reject_unsupported_negative_prompt(
        self,
        negative_prompt: str | list[str] | None,
    ) -> None:
        """Fail when a single-branch family receives ignored conditioning."""

        if negative_prompt not in (None, "", []):
            raise ValueError(f"{type(self).__name__} does not support negative prompts")

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

    def prepare_replay(self, build: ModelBuild) -> None:
        """Family hook run once by the replay builder right after construction.

        Default no-op. FLUX overrides it to set its dynamic-shift replay
        timesteps (the replay scheduler must carry the same mu-shifted schedule
        the rollout used); the build carries the sampling block it derives from.
        """
        del build
        return None

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        """Rebuild diffusion sampling state and run one replay forward."""
        self.reject_unsupported_replay_segments(request)
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
        resolver = TrajectoryResolver.from_batch(batch)
        replay_tensors = resolver.replay_tensor_dict(
            "denoise",
            axis="denoise",
            axis_index=timestep_idx,
            device=device,
        )
        latents = replay_tensors[resolver.role_tensor("denoise", "observation").name]
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
        return load_weights_into(transformer, state_dict, prefix="transformer")

    def validate_trainable_state(self, state_dict: Mapping[str, Any]) -> None:
        """Validate a sync payload without mutating the active policy."""

        transformer = self._require_transformer()
        validate_weights_for(transformer, state_dict, prefix="transformer")

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

        if state_dict is not None:
            self.validate_trainable_state(state_dict)
        self._versioned_state_slots().install(version, state_dict)

    def has_trainable_state(self, version: int) -> bool:
        return self._versioned_state_slots().has(version)

    def activate_trainable_state(self, version: int) -> None:
        """Make slot ``version`` the live trainable state (idempotent).

        Skips the reload when ``version`` is already active so a request whose
        batches share one version pays the copy at most once.
        """

        if getattr(self, "_active_slot_version", None) == int(version):
            return
        self.load_trainable_state(self._versioned_state_slots().get(version))
        self._active_slot_version = int(version)

    @classmethod
    def from_build(cls, build: ModelBuild) -> DiffusionModelBase:  # pragma: no cover (abstract)
        """Load the backend from a runtime build."""
        raise NotImplementedError

    def apply_lora(self, build: ModelBuild) -> None:  # pragma: no cover (default no-op)
        raise NotImplementedError

    def apply_full_finetune(self, build: ModelBuild) -> None:
        """Mark the transformer fully trainable (no-LoRA path)."""

        transformer = self._require_transformer()
        transformer.requires_grad_(True)
        if not build.defer_trainable_device_move:
            transformer.to(self.device)

    def _set_transformer(self, transformer: Any) -> None:
        """Register a replacement trainable transformer (LoRA wrap, compile, …).

        Default: ``self.transformer`` is the only handle. Families that keep a
        second reference to the same module — a diffusers pipeline, Echo's LTX
        wrapper, CausVid's backend — override to keep both in sync, and a
        pipeline-less replay subclass of such a family must override BACK to
        this default (its ``pipeline`` raises).
        """

        self.transformer = transformer

    def torch_compile_transformer(self, mode: str) -> None:
        """Compile every rollout policy core in place.

        Walks ``policy_cores`` rather than ``self.transformer`` so a multi-expert
        family compiles each expert it samples through, with no per-family
        override.
        """

        for name, module in self.policy_cores.items():
            self.set_module_root(name, torch.compile(module, mode=mode, fullgraph=False))

    def set_module_root(self, name: str, module: Any) -> None:
        """Write a replaced module root back to EVERY handle that references it.

        Anything that replaces a root rather than mutating it in place goes
        through here: ``torch.compile`` on the rollout side, FSDP/DDP wrapping on
        the trainer side. A family that keeps a second reference to the same
        module (a diffusers pipeline, Echo's LTX wrapper) must update both, or
        sampling silently keeps using the unwrapped copy.

        ``name`` is a key of ``policy_cores`` or ``trainable_modules``. Those are
        two different SELECTIONS of roots — rollout samples through every expert,
        the trainer updates only some — but writing one back is the same
        operation either way, so there is one method rather than a per-caller
        spelling. Single-root families need no override; a multi-root family
        overrides this once and both callers are served.
        """

        if name != "transformer":
            raise ValueError(
                f"{type(self).__name__} has no module root {name!r}; a family with "
                "more than one root must override set_module_root",
            )
        self._set_transformer(module)

    @property
    def policy_cores(self) -> dict[str, Any]:
        """Module roots the rollout optimization passes walk, keyed for logs.

        Deliberately NOT ``trainable_modules``: a multi-expert family filters
        that by which experts the trainer updates, while every expert that runs
        during sampling must be optimized — otherwise the halves of one
        trajectory execute at different precisions. Override this (not the
        individual passes) when a family owns more than one rollout module.
        """

        return {"transformer": self._require_transformer()}

    # Structural exclusions only (norms, embeddings, the noise-pred head): a DiT
    # has no vocabulary head, so the diffusion default is the shared base set.
    quantization_exclude: ClassVar[tuple[str, ...]] = DEFAULT_EXCLUDE

    def set_num_steps(self, n: int) -> None:  # pragma: no cover
        raise NotImplementedError

    @property
    def trainable_modules(self) -> dict[str, Any]:
        """Named modules the trainer optimizes and weight sync targets.

        One transformer is the family-wide shape; Wan (dual expert) and MAGI
        (no trainable module at all) are the only overrides.
        """

        return {"transformer": self._require_transformer()}

    @property
    def adapter_roots(self) -> dict[str, Any]:
        """Trainable roots whose PEFT adapter can be published as an artifact.

        Keyed by checkpoint root name, so a multi-expert family (Wan) exports
        one namespaced artifact per root instead of overwriting a single path.

        The ``save_pretrained`` filter belongs HERE and not in the checkpoint
        exporter: on the diffusion side a root that cannot ``save_pretrained``
        is a family that simply publishes no adapter artifact — ``checkpoint.pt``
        still resumes it — whereas the AR side has exactly one root and must
        fail loudly if it turns out unexportable. Centralizing the filter would
        downgrade that AR raise to a silent no-export.
        """

        return {
            name: module
            for name, module in self.trainable_modules.items()
            if hasattr(module, "save_pretrained")
        }

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
        Custom backends whose VAE does not implement this memory protocol keep
        it behind a family-specific attribute and expose no target. Replay
        models likewise own no VAE. In either case, the policy fails loud if
        config still asks for one.
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
        ``from_build`` froze instead of a hand-kept name list. Families that attach
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
    build: ModelBuild,
    model_dtype: torch.dtype,
) -> tuple[torch.dtype, dict[str, Any]]:
    """Resolve prompt-encoder dtype plus pipeline load kwargs for a family.

    ``build.rollout.prompt_encoder_dtype`` is authoritative when present; bare
    test builds retain the historical fallback. VAEs are not controlled by this
    option: rollout families explicitly keep them in fp32 for decode fidelity.
    The initial ``from_pretrained`` mapping preserves that fp32 VAE boundary
    while avoiding an all-fp32 prompt-encoder load peak.
    """

    rollout = getattr(build, "rollout", None)
    prompt_encoder_dtype = getattr(rollout, "prompt_encoder_dtype", None)
    if prompt_encoder_dtype is None:
        prompt_encoder_dtype = torch.float16 if model_dtype == torch.float32 else model_dtype
    # Full-pipeline rollout and component-only replay must resolve the same
    # immutable Hub snapshot; otherwise parity can compare different weights.
    load_kwargs: dict[str, Any] = build.pretrained_kwargs
    if model_dtype == torch.float32 and prompt_encoder_dtype != torch.float32:
        load_kwargs["torch_dtype"] = {
            "transformer": torch.float32,
            "vae": torch.float32,
            "default": prompt_encoder_dtype,
        }
    elif model_dtype != torch.float32:
        load_kwargs["torch_dtype"] = model_dtype
    return prompt_encoder_dtype, load_kwargs


class DiffusersPipelineModelBase(DiffusionModelBase):
    """Shared shape for families backed by ONE diffusers pipeline + ONE
    trainable transformer (sd3_5, flux, qwen_image, cosmos, wan's primary).

    Factors the members that were byte-identical across those families:
    pipeline/device/scheduler/raw_handle access, frozen encoder device discovery,
    distilled-guidance detection, transformer swap, scheduler timestep init, and
    ``from_build`` — the pipeline load plus the freeze/placement of VAE and
    prompt encoders, driven by the three class declarations below. A family
    overrides only where it genuinely differs (FLUX's dual-encoder discovery and
    NFT guard, SANA's scheduler swap, wan's multi-transformer
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

    def _encoder_device(self) -> Any:
        """Return the device that owns the pipeline's primary text encoder."""

        encoder = getattr(self.pipeline, "text_encoder", None)
        if encoder is not None:
            try:
                return next(encoder.parameters()).device
            except StopIteration:
                pass
        return self.device

    @property
    def _guidance_embeds(self) -> bool:
        """Whether the checkpoint embeds a distilled-guidance scalar."""

        return bool(getattr(self.transformer.config, "guidance_embeds", False))

    @property
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    @property
    def raw_handle(self) -> Any:
        return self.pipeline

    # -- backend ownership -------------------------------------------------
    # The three facts that differ between families whose loading is otherwise
    # byte-identical. Declaring them (instead of copying ``from_build``) is how a
    # family opts into the shared loader below.

    # Resolved with ``getattr(diffusers, name)`` — the same by-name mechanism
    # ``vrl.models.loader.load_diffusers_transformer`` uses for components.
    _pipeline_classname: str = ""
    _frozen_encoder_names: tuple[str, ...] = ("text_encoder",)
    # Deliberately has NO default: whether the prompt encoder co-resides with the
    # transformer or is parked on CPU is a per-family memory decision, and the
    # comment that justifies it needs an anchor on the family itself.
    _prompt_encoder_on_cpu: bool

    @classmethod
    def from_build(cls, build: ModelBuild) -> DiffusersPipelineModelBase:
        """Load the family pipeline and freeze everything but the transformer."""

        import diffusers

        if not cls._pipeline_classname:
            raise NotImplementedError(
                f"{cls.__name__} must declare _pipeline_classname (and the two "
                "sibling declarations) or override from_build",
            )
        pipeline_cls = getattr(diffusers, cls._pipeline_classname)
        # Prompt encoders follow the rollout-only precision policy; the VAE stays
        # family-owned fp32 below because decode fidelity is a separate concern.
        prompt_encoder_dtype, load_kwargs = diffusers_pipeline_dtypes(
            build,
            build.parameter_dtype,
        )
        pipeline = pipeline_cls.from_pretrained(
            build.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        encoder_device = "cpu" if cls._prompt_encoder_on_cpu else build.device
        for name in cls._frozen_encoder_names:
            encoder = getattr(pipeline, name, None)
            if encoder is not None:
                encoder.requires_grad_(False)
                encoder.to(encoder_device, dtype=prompt_encoder_dtype)
        pipeline.vae.to(build.device, dtype=torch.float32)
        return cls(
            pipeline=pipeline,
            device=build.device,
        )

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
    overrides only where it genuinely differs (flux/mochi/pixart_sigma
    re-standardize their replay scheduler in ``prepare_replay``).
    ``Cosmos3ReplayModel`` stays on ``ReplayRolloutStubs`` directly — it wraps a
    pipeline SHELL, not a bare transformer, and reads ``self.pipeline``.
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
    "DiffusionSamplingStateBase",
    "GuidedDiffusionSamplingStateBase",
    "diffusers_pipeline_dtypes",
]
