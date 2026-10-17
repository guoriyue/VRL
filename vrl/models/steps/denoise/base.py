"""Shared denoising model bases for diffusion and flow-matching RL runtimes.

``DenoiseModelBase`` is the trainable model the trainer replays: one denoise
step, replay-state projection, trainable roots and weight loading. A family's
sampling class adds prompt encoding, sampling state and decoding on top of it
(``DenoisePipelineModelBase``). For diffusers families ``DiffusersModelBase``
holds the transformer and scheduler, and ``DiffusersPipelineModelBase`` adds
the pipeline. The public trainer-facing replay interface is
``vrl.models.interfaces.ReplayModel``.
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

from vrl.generation.types import DenoiseRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.peft_adapter import has_lora_adapter, temporarily_disable_lora
from vrl.models.precision import model_autocast
from vrl.models.weight_snapshot import TrainableWeightsSnapshot
from vrl.models.weight_utils import (
    TrainableStateSlots,
    load_weights_into,
    require_weights_for,
)
from vrl.nn.optimization.regional_compile import compile_repeated_blocks
from vrl.nn.quantization.targeting import DEFAULT_EXCLUDE
from vrl.utils.validation import require_int


@dataclass
class DenoiseSamplingStateBase:
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
class GuidedDenoiseSamplingStateBase(DenoiseSamplingStateBase):
    """Private state shared by families whose forward/replay path reads guidance."""

    guidance_scale: float


class DenoiseModelBase(nn.Module, ABC):
    """Shared denoising and replay operations for diffusion and flow-matching families."""

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
            # Apply the selected role dtype at every diffusion forward boundary.
            @functools.wraps(fn)
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                with model_autocast(self, self.device):
                    return fn(self, *args, **kwargs)

            wrapped.__vrl_outer_autocast__ = True
            cls.forward_step = wrapped

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
        """``nn.Module.forward``: one trainable denoise step, so module hooks fire around it."""

        return self.forward_step(state, step_idx)

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

        Default no-op. ``DiffusersModelBase`` rebuilds a dynamic-shift replay
        grid here; other families re-standardize their replay scheduler.
        """
        del build
        return None

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
    ) -> dict[str, Any]:
        """Rebuild diffusion sampling state and run one replay forward."""
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
        return dict(self.forward(state, self._replay_forward_step_index(timestep_idx)))

    def replay_forward_with_latents(
        self,
        batch: Any,
        timestep_idx: int,
        latents: Any,
        *,
        classifier_free_guidance: bool | None = None,
    ) -> dict[str, Any]:
        """One replay-shaped forward on CALLER latents at the step's conditioning.

        Same state rebuild as ``replay_forward`` — the trajectory's prompt
        conditioning and the step's own timestep — but the model input is the
        caller's tensor instead of the stored x_t. Objectives that re-noise the
        CLEAN latent themselves (the GRPO diffusion-loss regularizer, the
        forward-process objectives) enter here so their prediction comes from
        the exact schedule replay uses (no second sigma-domain conversion path).

        ``classifier_free_guidance`` overrides the trajectory's recorded CFG
        setting: ``False`` runs only the conditional branch, which is what an
        objective evaluating the raw denoiser wants; ``None`` keeps the rollout's
        own setting.
        """

        replay_tensors, batch_context, _ = self._replay_inputs_for_step(
            batch,
            timestep_idx,
        )
        if classifier_free_guidance is not None:
            batch_context["cfg"] = bool(classifier_free_guidance)
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

        from vrl.trajectory.reader import TrajectoryReader

        device = self.device
        reader = TrajectoryReader.from_batch(batch)
        replay_tensors = reader.replay_tensor_dict(
            "denoise",
            axis="denoise",
            axis_index=timestep_idx,
            device=device,
        )
        latents = replay_tensors[reader.role_tensor("denoise", "observation").name]
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

        return temporarily_disable_lora(self._require_transformer())

    # -- policies other than the trainable one ---------------------------
    # The KL reference uses the base model with LoRA disabled, or a snapshot
    # for full fine-tuning. Objectives use the same policy interface in either
    # case.

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _has_adapter(self) -> bool:
        return any(has_lora_adapter(module) for module in self.trainable_modules.values())

    def attach_reference_policy(self) -> None:
        """Pin the current trainable weights as the KL reference.

        With LoRA attached, the reference is the base model with LoRA disabled,
        so nothing is copied. This excludes any warm-started LoRA weights.
        A full fine-tune keeps a snapshot of the trainable weights as they
        stood when this method was called, before training.
        """

        if self._has_adapter():
            return
        self.add_module(
            "_reference_policy", TrainableWeightsSnapshot(self._trainable_parameters())
        )

    def reference_policy(self) -> contextlib.AbstractContextManager[None]:
        """Use the LoRA-disabled base model or the saved full-fine-tuning reference.

        The full-fine-tuning case swaps the trainable weights in place: run it
        before the trainable forward of the same loss, never between that
        forward and its backward.
        """

        snapshot = self._modules.get("_reference_policy")
        if snapshot is not None:
            return snapshot.active()
        if self._has_adapter():
            return self.disable_adapter()
        raise RuntimeError(
            f"{type(self).__name__} has no reference policy: a full fine-tune must "
            "attach_reference_policy() before training starts",
        )

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load trainable transformer weights from ``transformer.*`` sync keys."""

        transformer = self._require_transformer()
        return load_weights_into(transformer, state_dict, prefix="transformer")

    def validate_trainable_state(self, state_dict: Mapping[str, Any]) -> None:
        """Validate a sync payload without mutating the active policy."""

        transformer = self._require_transformer()
        require_weights_for(transformer, state_dict, prefix="transformer")

    # -- versioned trainable-state slots (non-draining weight sync) ---------
    # Diffusion families support versioned slots generically: activation reuses
    # ``load_trainable_state`` to copy a retained version onto the live model, so
    # the same flat ``transformer.*`` payload format works for single-transformer
    # (sd3/cosmos) and multi-transformer (wan) families without per-family code.
    # The family registry declares the capability; see SPRINT_shadow_model_weight_sync.md.

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

        version = require_int(version, path="policy version", minimum=0)
        if getattr(self, "_active_slot_version", None) == version:
            return
        self.load_trainable_state(self._versioned_state_slots().get(version))
        self._active_slot_version = version

    @classmethod
    def from_build(cls, build: ModelBuild) -> DenoiseModelBase:  # pragma: no cover (abstract)
        """Load the backend from a runtime build."""
        raise NotImplementedError

    # ── LoRA ──────────────────────────────────────────────────────────
    # One attach implementation over ``trainable_modules`` + ``set_module_root``.
    # Family schemas resolve adapter defaults before this shared PEFT sequence.
    # Attach never places the roots: the rollout builder moves them after the
    # quantization swap (``build_denoise_runtime_bundle``), and the training
    # strategy's ``prepare_model`` owns replay placement.

    def apply_lora(self, build: ModelBuild) -> None:
        """Wrap every trainable root with a PEFT LoRA adapter per ``model.lora``.

        The parameter dtype is fixed at load and the roots stay where the
        loader put them. LoRA must attach before the quantization swap (which
        only wraps plain ``nn.Linear``); the rollout builder moves the compact
        policy afterwards, so the full base model never lands on one GPU
        before its memory-saving transform owns it.
        """

        from peft import LoraConfig, get_peft_model

        from vrl.models.peft_adapter import load_trainable_lora_adapter

        lora_config = build.require_lora_config()
        roots = self.trainable_modules
        if not roots:
            raise RuntimeError(f"{type(self).__name__} exposes no trainable module for LoRA")
        lora_path = lora_config.path
        if lora_path and len(roots) != 1:
            raise ValueError(
                "model.lora.path can only resume one trainable root; "
                f"{type(self).__name__} trains {sorted(roots)}",
            )
        for name, module in roots.items():
            module.requires_grad_(False)
            if lora_path:
                wrapped = load_trainable_lora_adapter(
                    module,
                    lora_path,
                    expected_rank=lora_config.rank,
                    expected_alpha=lora_config.alpha,
                    expected_dropout=lora_config.dropout,
                    expected_target_modules=lora_config.target_modules,
                    adapter_name="default",
                    autocast_adapter_dtype=lora_config.autocast_adapter_dtype,
                )
                wrapped.set_adapter("default")
            else:
                wrapped = get_peft_model(
                    module,
                    LoraConfig(
                        r=lora_config.rank,
                        lora_alpha=lora_config.alpha,
                        lora_dropout=lora_config.dropout,
                        init_lora_weights=lora_config.init_lora_weights,
                        target_modules=lora_config.target_modules,
                    ),
                    autocast_adapter_dtype=lora_config.autocast_adapter_dtype,
                )
            if lora_config.parameter_dtype == "float32":
                for parameter in wrapped.parameters():
                    if parameter.requires_grad:
                        parameter.data = parameter.data.to(dtype=torch.float32)
            self.set_module_root(name, wrapped)

    def apply_full_finetune(self, build: ModelBuild) -> None:
        """Mark the transformer fully trainable (no-LoRA path)."""

        del build
        self._require_transformer().requires_grad_(True)

    def _set_transformer(self, transformer: Any) -> None:
        """Register a replacement trainable transformer (LoRA wrap, compile, …).

        Default: ``self.transformer`` is the only handle. Classes that keep a
        second reference to the same module — the diffusers pipeline base,
        CausVid's backend — override to keep both in sync.
        """

        self.transformer = transformer

    def torch_compile_transformer(self, mode: str, *, regional: bool = False) -> None:
        """Compile every rollout policy core in place.

        Walks ``policy_cores`` rather than ``self.transformer`` so a multi-expert
        family compiles each expert it samples through, with no per-family
        override. ``regional`` compiles each repeated transformer block instead
        of the root (one trace shared by every block, so cold compile and
        recompiles cost one block rather than the whole graph); the root keeps
        its type, so no handle needs writing back.
        """

        for name, module in self.policy_cores.items():
            if regional:
                compile_repeated_blocks(module, mode=mode)
            else:
                self.set_module_root(name, torch.compile(module, mode=mode, fullgraph=False))

    def set_module_root(self, name: str, module: Any) -> None:
        """Write a replaced module root back to EVERY handle that references it.

        Anything that replaces a root rather than mutating it in place goes
        through here: ``torch.compile`` on the rollout side, FSDP/DDP wrapping on
        the trainer side. A family that keeps a second reference to the same
        module (a diffusers pipeline) must update both, or
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


class DenoisePipelineModelBase(DenoiseModelBase):
    """A denoise model that can also sample: prompt encoding, sampling state, decode.

    The trainer replays the model class alone. Generation also needs what
    surrounds it — prompt encoders, latent preparation, a decoder — so a
    family's sampling class adds these to its model class, instead of the model
    class carrying them and a replay subclass stubbing them out again.
    """

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
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Build a private per-family sampling state for the denoise loop.

        The family draws the starting latent from ``request.seed``. Which rows
        share a start is the rollout layer's decision (per-sample seeds on the
        request); the family only executes it.
        """

    @abstractmethod
    def decode_latents(self, latents: Any) -> Any:
        """Decode latents to a frame tensor."""

    def export_batch_context(self, state: Any) -> dict[str, Any]:
        """Project private sampling state into shared trajectory context."""
        raise NotImplementedError

    def export_replay_tensors(self, state: Any) -> dict[str, Any]:
        """Project private sampling state into per-sample trajectory tensors.

        Every family exports ``latents_clean`` as ``state.latents.detach()``:
        the final latent exactly as decoded, which forward-process objectives
        (DiffusionNFT, V-GRPO) re-noise. The action buffer may have rounded
        that step to a lower dtype, so the export reads the state, not the
        buffer.
        """
        raise NotImplementedError

    def _reject_unsupported_negative_prompt(
        self,
        negative_prompt: str | list[str] | None,
    ) -> None:
        """Fail when a single-branch family receives ignored conditioning."""

        if negative_prompt not in (None, "", []):
            raise ValueError(f"{type(self).__name__} does not support negative prompts")

    def generation_memory_targets(self) -> dict[str, Any]:
        """Named modules the generation memory policy may configure.

        None by default: a sampling model whose VAE does not implement this
        memory protocol keeps it behind a family-specific attribute, and the
        policy fails loud if config still asks for one. The diffusers pipeline
        base exposes ``pipeline.vae``.
        """

        return {}


class DiffusersModelBase(DenoiseModelBase):
    """A diffusers transformer and its scheduler: the model replay runs.

    Holds what a diffusers family's per-step math reads — device, scheduler,
    distilled-guidance detection, the timestep grid (static or resolution
    shifted) — and nothing about prompt encoders or the VAE. The replay builder
    constructs it from the transformer and scheduler alone; a family's sampling
    class adds the pipeline through ``DiffusersPipelineModelBase``.
    """

    def __init__(self, *, transformer: Any, scheduler: Any, device: Any = None) -> None:
        super().__init__()
        self.transformer = transformer
        self._scheduler = scheduler
        self._device = device

    @property
    def device(self) -> Any:
        return self._device

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def _guidance_embeds(self) -> bool:
        """Whether the checkpoint embeds a distilled-guidance scalar."""

        return bool(getattr(self.transformer.config, "guidance_embeds", False))

    def set_num_steps(self, n: int) -> None:
        """Initialize the scheduler timesteps for sampling.

        Dynamic-shifting FlowMatch schedulers (FLUX, Qwen-Image) derive their
        schedule from a resolution-dependent ``mu`` unknown at build time, so
        the real set is deferred to ``prepare_sampling``; static schedulers
        are set eagerly. Reads ``self.scheduler``, which a sampling class
        resolves to its pipeline's scheduler.
        """
        scheduler = self.scheduler
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            return
        scheduler.set_timesteps(n, device=self.device)

    def prepare_replay(self, build: ModelBuild) -> None:
        """Give a dynamic-shifting replay scheduler the rollout's timestep grid.

        ``set_num_steps`` defers dynamic-shifting schedulers because ``mu``
        depends on the packed latent length. Rollout sets the grid in
        ``prepare_sampling``; replay never samples, but the SDE log-prob math
        indexes ``scheduler.sigmas`` by timestep, so the grid must exist and
        match. Resolution is fixed per run, so the replay rebuilds it from the
        sampling block through the family's ``packed_token_count``.
        """
        if not getattr(self.scheduler.config, "use_dynamic_shifting", False):
            return
        sampling = build.sampling_config or {}
        num_steps = build.num_steps
        height, width = sampling.get("height"), sampling.get("width")
        if num_steps is None or not height or not width:
            return
        self._set_dynamic_timesteps(
            num_steps, self.packed_token_count(int(height), int(width)), build.device
        )

    def packed_token_count(self, height: int, width: int) -> int:
        """Packed latent length of one ``height x width`` image (dynamic-shift families)."""
        raise NotImplementedError(
            f"{type(self).__name__} runs a dynamic-shifting scheduler and must declare "
            "packed_token_count so its replay rebuilds the rollout timestep grid"
        )

    def _set_dynamic_timesteps(self, num_steps: int, image_seq_len: int, device: Any) -> Any:
        """Set the resolution-shifted grid exactly as the family's diffusers pipeline does."""
        raise NotImplementedError(
            f"{type(self).__name__} runs a dynamic-shifting scheduler and must declare "
            "_set_dynamic_timesteps"
        )


class DiffusersPipelineModelBase(DenoisePipelineModelBase, DiffusersModelBase):
    """A diffusers model plus the pipeline that samples with it.

    A family's sampling class lists this before its model class. The pipeline
    owns the scheduler and a second handle to the transformer, so this class
    reads the scheduler and device back from it and writes a replaced
    transformer to both handles. It also owns what only sampling needs: frozen
    encoder device discovery, the VAE memory target, moving the frozen
    components, and ``from_build`` — the pipeline load plus the freeze and
    placement of the VAE and prompt encoders. A family overrides only where it
    genuinely differs (FLUX's dual-encoder discovery, SANA's scheduler swap).
    Families not backed by a diffusers pipeline (CausVid's backend) build on ``DenoisePipelineModelBase`` directly.
    """

    def __init__(self, *, pipeline: Any, device: Any = None) -> None:
        # The pipeline owns the scheduler (``scheduler`` reads it back), so
        # ``DiffusersModelBase``'s transformer-and-scheduler constructor does
        # not run. The transformer is registered as the trainable child; the
        # pipeline is attached unregistered, as a frozen container.
        DenoiseModelBase.__init__(self)
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
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    def generation_memory_targets(self) -> dict[str, Any]:
        """The pipeline's VAE, when it ships one, is the decode-memory target."""

        vae = getattr(self.pipeline, "vae", None)
        return {} if vae is None else {"vae_decode": vae}

    def move_frozen_components(self, device: Any) -> None:
        """Move frozen pipeline components (VAE / text encoders) onto ``device``.

        ``nn.Module.to`` moves only *registered* submodules — for these families
        just the transformer — but the pipeline is attached unregistered, so its
        frozen VAE / text encoders stay resident unless moved explicitly. This
        is the offload-and-restore discipline for non-trainable components:
        parking them on CPU during the rollout window frees GPU without
        discarding and reloading them from disk. Device-only — dtype is
        preserved (frozen VAE stays fp32, encoders keep their frozen dtype).

        The set is derived from the pipeline — every nn.Module component not
        already registered on this model — so it tracks whatever ``from_build``
        froze instead of a hand-kept name list.
        """

        components = getattr(self.pipeline, "components", None)
        if not isinstance(components, Mapping):
            raise TypeError("diffusion pipeline.components must be a mapping for frozen offload")
        registered = {id(module) for module in self.modules()}
        moved: set[int] = set()
        for name, module in components.items():
            if not isinstance(module, nn.Module) or id(module) in registered:
                continue
            if name in self._cpu_resident:
                continue
            if id(module) not in moved:
                moved.add(id(module))
                module.to(device)

    # -- backend ownership -------------------------------------------------
    # The pipeline class comes from the checkpoint's ``model_index.json``
    # (``DiffusionPipeline.from_pretrained``); every module component but the
    # transformer is frozen, so a family declares nothing for the loader.
    # Component names ``from_build`` left on the host (``model.memory.cpu_resident``);
    # parking's wake never moves them back onto the GPU.
    _cpu_resident: frozenset[str] = frozenset()

    @classmethod
    def _pipeline_load_dtypes(
        cls,
        build: ModelBuild,
    ) -> tuple[torch.dtype, dict[str, Any]]:
        """Resolve prompt-encoder dtype plus pipeline load kwargs.

        ``build.rollout.prompt_encoder_dtype`` is authoritative when present; bare
        test builds retain the historical fallback. Every component but the
        transformer is frozen, so the frozen dtype is the load default and the
        two named exceptions are the trainable transformer and the VAE, which
        rollout families keep in fp32 for decode fidelity. Mapping them at
        ``from_pretrained`` avoids an all-fp32 prompt-encoder load peak.
        """

        rollout = getattr(build, "rollout", None)
        prompt_encoder_dtype = getattr(rollout, "prompt_encoder_dtype", None)
        if prompt_encoder_dtype is None:
            prompt_encoder_dtype = (
                torch.float16 if build.parameter_dtype == torch.float32 else build.parameter_dtype
            )
        # Full-pipeline rollout and component-only replay must resolve the same
        # immutable Hub snapshot; otherwise parity can compare different weights.
        load_kwargs: dict[str, Any] = build.pretrained_kwargs
        load_kwargs["torch_dtype"] = {
            "default": prompt_encoder_dtype,
            "transformer": build.parameter_dtype,
            "vae": torch.float32,
        }
        return prompt_encoder_dtype, load_kwargs

    @staticmethod
    def freeze_pipeline_components(
        pipeline: Any,
        *,
        trainable: tuple[str, ...] = ("transformer",),
    ) -> None:
        """Disable gradients for non-trainable modules without moving or casting them."""

        components = pipeline.components
        if not isinstance(components, Mapping):
            raise TypeError("diffusion pipeline.components must be a mapping")
        for name, module in components.items():
            if name not in trainable and isinstance(module, nn.Module):
                module.requires_grad_(False)

    @staticmethod
    def place_pipeline_components(
        pipeline: Any,
        build: ModelBuild,
        *,
        prompt_encoder_dtype: torch.dtype,
        trainable: tuple[str, ...] = ("transformer",),
    ) -> frozenset[str]:
        """Set devices and dtypes for non-trainable modules without changing gradients.

        The rule every pipeline-backed loader shares: the VAE is a fp32
        fidelity boundary, every other frozen module takes the rollout
        encoder dtype, and a component named by ``model.memory.cpu_resident``
        stays on the host (it runs there; ``encode_prompt`` reads its device
        back). Returns the cpu-resident set for the model to remember, so
        parking's wake leaves those components alone.
        """

        components = pipeline.components
        if not isinstance(components, Mapping):
            raise TypeError("diffusion pipeline.components must be a mapping")
        memory = build.generation_memory
        cpu_resident = frozenset(() if memory is None else memory.cpu_resident)
        unknown = sorted(
            name for name in cpu_resident if not isinstance(components.get(name), nn.Module)
        )
        if unknown:
            raise ValueError(
                f"model.memory.cpu_resident names component(s) {unknown} that "
                f"{type(pipeline).__name__} does not ship as modules",
            )
        for name, module in components.items():
            if name in trainable or not isinstance(module, nn.Module):
                continue
            module.to(
                "cpu" if name in cpu_resident else build.device,
                dtype=torch.float32 if name == "vae" else prompt_encoder_dtype,
            )
        return cpu_resident

    @classmethod
    def from_build(cls, build: ModelBuild) -> DiffusersPipelineModelBase:
        """Load the family pipeline and freeze every module but the transformer."""

        from diffusers import DiffusionPipeline

        prompt_encoder_dtype, load_kwargs = cls._pipeline_load_dtypes(build)
        pipeline = DiffusionPipeline.from_pretrained(
            build.model_name_or_path,
            **load_kwargs,
        )
        cls.freeze_pipeline_components(pipeline)
        cpu_resident = cls.place_pipeline_components(
            pipeline,
            build,
            prompt_encoder_dtype=prompt_encoder_dtype,
        )
        model = cls(
            pipeline=pipeline,
            device=build.device,
        )
        model._cpu_resident = cpu_resident
        return model


__all__ = [
    "DenoiseModelBase",
    "DenoisePipelineModelBase",
    "DenoiseSamplingStateBase",
    "DiffusersModelBase",
    "DiffusersPipelineModelBase",
    "GuidedDenoiseSamplingStateBase",
]
