"""Wan 2.1 t2v diffusers-backed model.

Diffusion implementation for Wan T2V on the RL path. The generation helper
flow is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

``WanT2VPipelineModel`` and ``WanI2VPipelineModel`` run that flow.
``WanT2VModel`` and ``WanI2VModel`` — the expert transformers, their scheduler
and ``forward_step`` — are what replay runs; the pipeline classes add the text
and image encoders, latent preparation, VAE decode and the offload lifecycle on
top of them.

The collector owns the scheduler step / SDE step. ``forward_step`` does only
one transformer forward (with optional batched CFG concat) and returns noise
predictions.

Per-family ``WanT2VSamplingState`` is private to this file. The engine /
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``, plus the embeds the eval path
re-builds explicitly).

Differences from SD3:
- ``prompt_embeds`` only (no pooled/CLIP); transformer signature lacks
  ``pooled_projections``.
- 5D latents ``[B, C, T, H, W]`` (Wan VAE temporal axis).
- VAE decode applies Wan-specific per-channel ``latents_mean`` /
  ``latents_std`` denormalization (over ``z_dim``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.families.wan_2_1.config import (
    WanTransformerName,
    normalize_wan_boundary_ratio,
    normalize_wan_trainable_transformers,
    wan_topology_from_build,
)
from vrl.models.interfaces.runtime import ModelBuild, PipelineOffloadMode
from vrl.models.peft_adapter import temporarily_disable_lora
from vrl.models.steps.denoise import (
    DiffusersModelBase,
    DiffusersPipelineModelBase,
    GuidedDenoiseSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    DenoiseBackboneCaller,
    DenoiseBackboneInput,
    DenoiseBackboneRunnerBase,
    DenoiseBranch,
    expand_batch_timestep,
    expand_tensor_to_batch,
    pack_eval_timestep,
)
from vrl.models.weight_utils import load_weights_into, require_weights_for

logger = logging.getLogger(__name__)


class _PipelineOffloadFailure(Enum):
    """Terminal hook failure, distinct from the shared successful residency modes."""

    BROKEN = "broken"


@dataclass
class WanT2VSamplingState(GuidedDenoiseSamplingStateBase):
    """Private Wan T2V sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    do_cfg: bool
    boundary_ratio: float | None = None
    num_train_timesteps: int | None = None
    # Host copy of ``timesteps``, filled on first expert routing; see
    # ``WanT2VPipelineModel._transformer_for_step``.
    host_timesteps: torch.Tensor | None = field(default=None, init=False, repr=False)


@dataclass
class WanI2VSamplingState(GuidedDenoiseSamplingStateBase):
    """Private Wan I2V sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    image_embeds: torch.Tensor | None
    condition: torch.Tensor
    do_cfg: bool
    boundary_ratio: float | None = None
    num_train_timesteps: int | None = None
    # Host copy of ``timesteps``, filled on first expert routing; see
    # ``WanT2VPipelineModel._transformer_for_step``.
    host_timesteps: torch.Tensor | None = field(default=None, init=False, repr=False)


class WanT2VModel(DiffusersModelBase, DenoiseBackboneRunnerBase):
    """Wan T2V transformers, scheduler and per-step math: the model replay runs.

    Implements the backbone-runner protocol itself (batched CFG over the T5
    text embeds); the I2V model overrides ``build_branch`` with its
    channel-wise conditioning concat. A Wan 2.2 dual-stage checkpoint carries
    a second expert, ``transformer_2``, which ``forward_step`` routes to by
    timestep against ``boundary_ratio``.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DenoiseBackboneInput,
        branch: str,
    ) -> DenoiseBranch:
        """Map Wan transformer kwargs into the shared backbone contract."""
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = request.negative_prompt_embeds
        return DenoiseBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
        )

    def __init__(
        self,
        *,
        transformer: Any,
        scheduler: Any,
        device: Any = None,
        transformer_2: Any = None,
        boundary_ratio: float | None = None,
        trainable_transformers: tuple[WanTransformerName, ...] | None = None,
    ) -> None:
        super().__init__(transformer=transformer, scheduler=scheduler, device=device)
        self.transformer_2 = transformer_2
        self._init_wan_state(
            boundary_ratio=boundary_ratio,
            trainable_transformers=trainable_transformers,
            expert_lifecycle_profiling=False,
        )

    def prepare_replay(self, build: ModelBuild) -> None:
        """Finish the multi-transformer replay setup with the build in hand.

        The generic replay loader constructs only the primary transformer +
        scheduler; Wan2.2 dual-stage checkpoints carry a second transformer
        whose presence is decided by the canonical build topology. Late-load it
        here and apply the already-normalized policy ownership.
        """
        boundary_ratio, trainable_transformers = wan_topology_from_build(build)
        if boundary_ratio is not None and self.transformer_2 is None:
            from vrl.models.loader import load_diffusers_transformer

            self.transformer_2 = load_diffusers_transformer(
                build,
                "WanTransformer3DModel",
                subfolder="transformer_2",
            )
        self._boundary_ratio = boundary_ratio
        self._trainable_transformer_names = trainable_transformers
        self._expert_lifecycle_profiling = bool(
            (build.model_config or {}).get("expert_lifecycle_profiling", False),
        )
        for module in self._wan_transformers().values():
            module.requires_grad_(False)

    def _set_transformer_2(self, transformer: Any) -> None:
        """The second expert's writer; the pipeline class also syncs the pipeline's handle."""
        self.transformer_2 = transformer

    def _init_wan_state(
        self,
        *,
        boundary_ratio: float | None,
        trainable_transformers: tuple[WanTransformerName, ...] | None,
        expert_lifecycle_profiling: bool,
    ) -> None:
        """The Wan-owned state every constructor must establish.

        The model's constructor and the pipeline class's set up different
        module roots, but the expert topology and profiling state are the same
        object graph; one initializer keeps a field added here from being
        missed by the other constructor.

        ``trainable_transformers`` is the canonical tuple from
        ``wan_topology_from_build``; None selects the topology default.
        """

        self._boundary_ratio = boundary_ratio
        self._trainable_transformer_names = (
            normalize_wan_trainable_transformers(None, dual_stage=boundary_ratio is not None)
            if trainable_transformers is None
            else tuple(trainable_transformers)
        )
        self._expert_lifecycle_profiling = bool(expert_lifecycle_profiling)
        self._last_expert_name: str | None = None
        for module in self._wan_transformers().values():
            module.requires_grad_(False)

    @property
    def boundary_ratio(self) -> float | None:
        return self._boundary_ratio

    def _wan_transformers(self) -> dict[str, Any]:
        modules = {"transformer": self.transformer}
        if self.transformer_2 is not None:
            modules["transformer_2"] = self.transformer_2
        return modules

    @property
    def policy_cores(self) -> dict[str, Any]:
        """Both experts, not just ``self.transformer`` and not ``trainable_modules``.

        A dual-stage config that trains only one expert still SAMPLES through
        both, so every optimization pass must reach both: otherwise the two
        halves of one trajectory run at different precisions and the
        rollout-vs-replay drift differs across the boundary.
        """

        return self._wan_transformers()

    def set_module_root(self, name: str, module: Any) -> None:
        """Route a replaced expert to its writer.

        One override serves every caller: rollout compile walks all experts,
        FSDP/DDP walks only the trained ones, and both land here. The pipeline
        class's writers also update the pipeline's handles.
        """

        self._set_wan_transformer(name, module)

    @property
    def trainable_modules(self) -> dict[str, Any]:
        modules = self._wan_transformers()
        return {name: modules[name] for name in self._trainable_transformer_names}

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            for module in self.trainable_modules.values():
                stack.enter_context(temporarily_disable_lora(module))
            yield

    def apply_full_finetune(self, build: ModelBuild) -> None:
        """Mark every trained expert trainable and leave the storage dtype alone.

        FSDP normalizes the dtype right before sharding; touching the full
        parameter set here would defeat its block-wise construction.
        """
        del build
        for module in self.trainable_modules.values():
            module.requires_grad_(True)

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        return self._load_validated_wan_trainable_state(
            self._validate_wan_trainable_state(state_dict),
        )

    def _load_validated_wan_trainable_state(
        self,
        validated: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        results = {}
        for name, module in self.trainable_modules.items():
            target = getattr(module, "_orig_mod", module)
            prefixed = {f"{name}.{key}": value for key, value in validated[name].items()}
            results[name] = load_weights_into(target, prefixed, prefix=name)
        return results

    def validate_trainable_state(self, state_dict: Mapping[str, Any]) -> None:
        """Reject partial dual-expert payloads before a weight-sync ACK."""

        self._validate_wan_trainable_state(state_dict)

    def _validate_wan_trainable_state(
        self,
        state_dict: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        state = dict(state_dict)
        modules = self.trainable_modules
        accepted_prefixes = tuple(f"{name}." for name in modules)
        unexpected = sorted(key for key in state if not key.startswith(accepted_prefixes))
        if unexpected:
            raise ValueError(
                f"{type(self).__name__}: unexpected trainable state keys {unexpected[:5]}",
            )
        validated: dict[str, dict[str, Any]] = {}
        for name, module in modules.items():
            prefix = f"{name}."
            module_state = {key: value for key, value in state.items() if key.startswith(prefix)}
            validated[name] = require_weights_for(module, module_state, prefix=name)
        return validated

    def _set_wan_transformer(self, name: str, transformer: Any) -> None:
        if name == "transformer":
            self._set_transformer(transformer)
            return
        if name == "transformer_2":
            self._set_transformer_2(transformer)
            return
        raise ValueError(f"unknown Wan transformer name: {name!r}")

    def _transformer_for_step(
        self,
        state: WanT2VSamplingState | WanI2VSamplingState,
        step_idx: int,
    ) -> tuple[str, Any, float]:
        if state.boundary_ratio is None:
            return "transformer", self.transformer, state.guidance_scale
        if state.num_train_timesteps is None:
            raise ValueError(
                "Wan dual-stage routing requires scheduler.config.num_train_timesteps",
            )
        # Choosing the expert is a Python-level decision (which module to
        # call), so the boundary test needs the timestep value on the host.
        # Reading it from the device schedule would synchronise the stream on
        # every forward and stall the launch queue behind the previous step's
        # tail. The schedule is fixed for the life of a sampling state, so it
        # is copied to the host once and indexed from there; the reads below
        # are free.
        if state.host_timesteps is None:
            state.host_timesteps = state.timesteps.detach().to(device="cpu", dtype=torch.float32)
        boundary_timestep = float(state.boundary_ratio) * int(state.num_train_timesteps)
        low_noise = state.host_timesteps[step_idx] < boundary_timestep
        first = bool(low_noise.reshape(-1)[0].item())
        if bool((low_noise != first).any().item()):
            raise ValueError(
                "Wan dual-stage replay batch crosses the transformer boundary within "
                "one forward_step; batch samples must share the same denoise step",
            )
        if first:
            transformer_2 = self.transformer_2
            if transformer_2 is None:
                raise RuntimeError("Wan dual-stage sampling requires transformer_2")
            return "transformer_2", transformer_2, state.guidance_scale
        return "transformer", self.transformer, state.guidance_scale

    def forward_step(
        self,
        state: WanT2VSamplingState | WanI2VSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Wan transformer forward + optional batched CFG.

        Returns noise_pred plus the un/conditional branches; the caller owns
        scheduler.step / SDE. ``_backbone_extra`` adds the family's extra
        conditioning (I2V's image condition).

        Timestep shape convention (mirrors SD3 model):
        - rollouts: ``state.timesteps`` is 1-D ``[T]``; we expand a scalar to ``[B]``.
        - eval/training: collector packs per-sample timestep as ``[B]``; expand is a no-op.
        """

        bsz = state.latents.shape[0]
        timestep = state.timesteps[step_idx]
        expert_name, transformer, guidance_scale = self._transformer_for_step(
            state,
            step_idx,
        )
        td = getattr(transformer, "dtype", None)
        if td is None:
            try:
                td = next(transformer.parameters()).dtype
            except StopIteration as exc:
                raise RuntimeError(
                    f"{type(transformer).__name__} has no parameters to infer dtype",
                ) from exc
        latent_input = state.latents.to(td)
        timestep_batch = expand_batch_timestep(timestep, bsz).to(
            device=latent_input.device,
            dtype=td,
        )
        prompt_embeds = state.prompt_embeds.to(td)
        negative_prompt_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        with self._expert_lifecycle_trace(expert_name, timestep):
            output = DenoiseBackboneCaller(
                transformer,
                self,
            )(
                DenoiseBackboneInput(
                    hidden_states=latent_input,
                    timestep=timestep_batch,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=guidance_scale,
                    do_cfg=state.do_cfg,
                    output_dtype=td,
                    extra=self._backbone_extra(state, td),
                ),
            )
        return output.as_dict()

    @contextlib.contextmanager
    def _expert_lifecycle_trace(
        self,
        expert_name: str,
        timestep: torch.Tensor,
    ) -> Iterator[None]:
        if not self._expert_lifecycle_profiling or not torch.cuda.is_available():
            yield
            return

        device = torch.device(self.device)
        if device.type != "cuda":
            yield
            return
        torch.cuda.synchronize(device)
        switched = self._last_expert_name not in {None, expert_name}
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        before_free, total = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.cuda.nvtx.range(f"wan.expert.{expert_name}"):
            yield
        torch.cuda.synchronize(device)
        after_free, _ = torch.cuda.mem_get_info(device)
        modules = self._wan_transformers()
        inactive_name = "transformer_2" if expert_name == "transformer" else "transformer"
        record = {
            "event": "wan_expert_lifecycle",
            "expert": expert_name,
            "timestep": float(torch.as_tensor(timestep).reshape(-1)[0].item()),
            "switched": switched,
            "allocated_before_bytes": before_allocated,
            "allocated_after_bytes": torch.cuda.memory_allocated(device),
            "reserved_before_bytes": before_reserved,
            "reserved_after_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "physical_used_before_bytes": total - before_free,
            "physical_used_after_bytes": total - after_free,
            "active_cuda_parameter_bytes": _cuda_parameter_bytes(modules[expert_name]),
            "inactive_cuda_parameter_bytes": _cuda_parameter_bytes(modules[inactive_name]),
        }
        logger.info("WAN_EXPERT_LIFECYCLE %s", json.dumps(record, sort_keys=True))
        self._last_expert_name = expert_name

    # -- forward_step --------------------------------------------------

    def _backbone_extra(self, state: Any, dtype: torch.dtype) -> dict[str, Any]:
        """Extra transformer inputs beyond latents, timestep and prompts; none for T2V."""

        del state, dtype
        return {}

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> WanT2VSamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice."""
        ts = replay_tensors["timesteps"]
        # Pack as [1, B] so forward_step's state.timesteps[0] returns [B].
        timesteps = pack_eval_timestep(ts, step_idx)
        return WanT2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            boundary_ratio=batch_context.get("boundary_ratio"),
            num_train_timesteps=_scheduler_num_train_timesteps(self.scheduler),
        )


class WanT2VPipelineModel(DiffusersPipelineModelBase, WanT2VModel):
    """Wan T2V plus the diffusers pipeline that sampling needs.

    The pipeline brings prompt encoding, latent preparation, VAE encode and
    decode, and the Accelerate CPU-offload lifecycle; this class also
    exports the trajectory tensors replay reads back.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
        trainable_transformers: tuple[WanTransformerName, ...] | None = None,
        expert_lifecycle_profiling: bool = False,
    ) -> None:
        super().__init__(pipeline=pipeline, device=device)
        self._pipeline_offload: PipelineOffloadMode | _PipelineOffloadFailure = (
            PipelineOffloadMode.NONE
        )
        self.transformer_2 = getattr(pipeline, "transformer_2", None)
        self._init_wan_state(
            boundary_ratio=normalize_wan_boundary_ratio(
                _config_value(getattr(pipeline, "config", None), "boundary_ratio"),
                field_name="pipeline boundary_ratio",
            ),
            trainable_transformers=trainable_transformers,
            expert_lifecycle_profiling=expert_lifecycle_profiling,
        )

    def _set_transformer_2(self, transformer: Any) -> None:
        self.transformer_2 = transformer
        self.pipeline.transformer_2 = transformer

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_build(cls, build: ModelBuild) -> WanT2VPipelineModel:
        """Load the diffusers WanPipeline + freeze non-trainable modules."""
        _, trainable_transformers = wan_topology_from_build(build)
        from diffusers import WanPipeline

        eager_module_dtypes = {
            "vae": torch.float32,
            "text_encoder": build.parameter_dtype,
        }
        pipeline = WanPipeline.from_pretrained(
            build.model_name_or_path,
            torch_dtype={"default": build.parameter_dtype, **eager_module_dtypes},
            **build.pretrained_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        offload_mode = build.require_rollout().pipeline_offload_mode
        _stage_eager_wan_modules(
            pipeline,
            build,
            offload_mode=offload_mode,
            eager_module_dtypes=eager_module_dtypes,
        )
        return cls(
            pipeline=pipeline,
            device=build.device,
            trainable_transformers=trainable_transformers,
            expert_lifecycle_profiling=bool(
                (build.model_config or {}).get("expert_lifecycle_profiling", False),
            ),
        )

    def apply_full_finetune(self, build: ModelBuild) -> None:
        # Normalize the CPU storage now so trainer payloads and rollout
        # parameters share one exact dtype. The builder places the module
        # afterwards (or Accelerate's hooks own residency).
        super().apply_full_finetune(build)
        for module in self.trainable_modules.values():
            module.to(dtype=build.parameter_dtype)

    @property
    def uses_pipeline_cpu_offload(self) -> bool:
        """Whether this model is configured for Accelerate-owned residency."""

        return self._pipeline_offload is not PipelineOffloadMode.NONE

    @property
    def pipeline_cpu_offload_healthy(self) -> bool:
        """Whether configured pipeline hooks are installed and reusable."""

        return self._pipeline_offload is not _PipelineOffloadFailure.BROKEN

    def apply_generation_offload(self, build: ModelBuild) -> None:
        """Install pipeline offload hooks after LoRA has changed the module tree."""

        mode = build.rollout.pipeline_offload_mode
        state = self._pipeline_offload
        if state is _PipelineOffloadFailure.BROKEN:
            raise RuntimeError("Wan pipeline CPU offload is not reusable")
        loaded_mode = state.value
        if state is not PipelineOffloadMode.NONE and mode is not state:
            raise RuntimeError(
                "Wan pipeline offload mode changed during rollout construction: "
                f"loaded={loaded_mode!r}, final={mode!r}",
            )
        if mode is PipelineOffloadMode.NONE:
            return
        if state is mode:
            return
        # Initial installation is transactional too. A partial hook graph must
        # never be published as a reusable rollout model.
        self._pipeline_offload = _PipelineOffloadFailure.BROKEN
        self._enable_pipeline_offload(mode)
        self._pipeline_offload = mode

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        validated = self._validate_wan_trainable_state(state_dict)
        if not self.uses_pipeline_cpu_offload:
            return self._load_validated_wan_trainable_state(validated)

        return self._with_pipeline_cpu_offload_suspended(
            lambda: self._load_validated_wan_trainable_state(validated),
            operation="trainable weight sync",
        )

    def reset_pipeline_cpu_offload(self) -> None:
        """Return every component to CPU and reinstall fresh streaming hooks.

        Accelerate post-forward hooks do not run when a module forward raises,
        and model-level offload deliberately leaves the final component on the
        execution device. Cycling Diffusers' public hook API is the common
        phase-parking and error-recovery boundary for both cases.
        """

        if not self.uses_pipeline_cpu_offload:
            return
        self._with_pipeline_cpu_offload_suspended(
            lambda: None,
            operation="pipeline residency reset",
        )

    def _with_pipeline_cpu_offload_suspended(
        self,
        operation_fn: Callable[[], Any],
        *,
        operation: str,
    ) -> Any:
        """Run one CPU-resident operation and transactionally re-arm hooks."""

        if not self.pipeline_cpu_offload_healthy:
            raise RuntimeError(
                "Wan pipeline CPU offload is not reusable: its hook lifecycle "
                "is incomplete or previously failed",
            )

        if not self.uses_pipeline_cpu_offload:
            return operation_fn()
        state = self._pipeline_offload
        # Publish the transition before touching hooks. Partial removal, hook
        # reinstall, or weight mutation must leave one terminal state.
        self._pipeline_offload = _PipelineOffloadFailure.BROKEN

        try:
            self._disable_pipeline_offload(operation=operation)
        except BaseException as remove_error:
            raise RuntimeError(
                f"Wan pipeline CPU offload hook removal failed during {operation}; "
                "the rollout model is no longer reusable",
            ) from remove_error

        result: Any = None
        operation_error: BaseException | None = None
        try:
            result = operation_fn()
        except BaseException as error:
            operation_error = error

        try:
            self._enable_pipeline_offload(state)
        except BaseException as reinstall_error:
            detail = f"Wan pipeline CPU offload hook reinstall failed during {operation}"
            if operation_error is not None:
                detail += f" after operation failure {operation_error!r}"
            raise RuntimeError(
                f"{detail}; the rollout model is no longer reusable",
            ) from reinstall_error

        if operation_error is not None:
            # A state-dict copy can fail after mutating an arbitrary prefix. Fresh
            # hooks make teardown safe, but they cannot make partially updated
            # policy weights valid. Never publish this model as reusable.
            raise RuntimeError(
                f"Wan {operation} failed after pipeline hooks were removed; "
                "the rollout model may contain partial state and is no longer reusable",
            ) from operation_error

        self._pipeline_offload = state
        return result

    def _enable_pipeline_offload(self, mode: PipelineOffloadMode) -> None:
        """Install residency hooks after adapters and wrappers are final."""

        # Diffusers exposes two mutually exclusive accelerate hooks here:
        # sequential streams per layer and is the 32 GB Wan I2V escape hatch; model
        # offload stages full modules and only works when the transformer fits.
        gpu_id = getattr(self.device, "index", None) or 0
        if mode is PipelineOffloadMode.SEQUENTIAL:
            self.pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id)
            return
        if mode is PipelineOffloadMode.MODEL:
            self.pipeline.enable_model_cpu_offload(gpu_id=gpu_id)

    def _disable_pipeline_offload(self, *, operation: str) -> None:
        """Strip the residency hooks ``_enable_pipeline_offload`` installed.

        The model's own transformer roots may be PEFT wrapped; their Accelerate
        hooks are detached child-first before the pipeline's own cleanup runs.
        """

        remove_hooks = getattr(self.pipeline, "remove_all_hooks", None)
        if not callable(remove_hooks):
            raise RuntimeError(
                f"Wan pipeline CPU offload requires pipeline.remove_all_hooks() for {operation}",
            )
        from accelerate.hooks import remove_hook_from_module

        # PEFT delegates attributes to its child, so Accelerate's hasattr
        # traversal can detach a child's hook against the wrong owner. Remove
        # explicitly owned hooks child-first before pipeline cleanup.
        for owned in self._wan_transformers().values():
            if isinstance(owned, torch.nn.Module):
                for module in reversed(list(owned.modules())):
                    if "_hf_hook" in vars(module) or "_old_forward" in vars(module):
                        remove_hook_from_module(module, recurse=False)
        remove_hooks()

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Wan's T5 text encoder.

        Returns ``prompt_embeds`` and the matching ``negative_prompt_embeds``
        when CFG is active. Wan does not use a pooled CLIP embedding.
        """
        max_seq = kwargs.get("max_sequence_length", 512) or 512
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
        )

        td = self.pipeline.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        *,
        initial_latents: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> WanT2VSamplingState:
        """Build the per-request SamplingState for a Wan T2V denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        # Multistep history belongs to this batch, not the resident pipeline.
        scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # Wan prepare_latents signature:
        # (batch, channels, height, width, num_frames, dtype, device, generator, latents)
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            request.frame_count,
            torch.float32,
            device,
            generator,
            initial_latents,
        )

        return WanT2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            boundary_ratio=self._boundary_ratio,
            num_train_timesteps=_scheduler_num_train_timesteps(scheduler),
        )

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: WanT2VSamplingState) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.context (shared metadata)."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "boundary_ratio": state.boundary_ratio,
        }

    def export_replay_tensors(self, state: WanT2VSamplingState) -> dict[str, Any]:
        """Project SamplingState into trajectory replay tensors."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "latents_clean": state.latents.detach(),
        }

    # -- decode_latents ------------------------------------------------

    def encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode clean target video into Wan's normalized latent domain.

        ``video`` is ``[B,C,T,H,W]`` in ``[0,1]`` at the training geometry.
        This is the exact inverse of :meth:`decode_latents`'s per-channel
        denormalization and uses the deterministic posterior mode so an offline
        SFT shard is reproducible.
        """

        pipe = self.pipeline
        x = (video.to(device=self.device, dtype=pipe.vae.dtype) * 2.0) - 1.0
        with torch.no_grad():
            encoded = torch.cat(
                [pipe.vae.encode(sample.unsqueeze(0)).latent_dist.mode() for sample in x],
                dim=0,
            )
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(encoded.device, encoded.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1,
            pipe.vae.config.z_dim,
            1,
            1,
            1,
        ).to(encoded.device, encoded.dtype)
        return (encoded - latents_mean) * latents_std

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode 5D latents -> video [B, C, T, H, W] via Wan VAE.

        Applies Wan-specific per-channel denormalization using the VAE's
        ``latents_mean`` / ``latents_std`` over the ``z_dim`` channel axis.
        """
        pipe = self.pipeline

        def _transform(batch: torch.Tensor) -> torch.Tensor:
            x = batch.to(pipe.vae.dtype)
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(x.device, x.dtype)
            )
            latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                1, pipe.vae.config.z_dim, 1, 1, 1
            ).to(x.device, x.dtype)
            return x / latents_std + latents_mean

        decoder = ChunkedLatentDecoder(
            prepare_latents=_transform,
            vae_decode=lambda batch: pipe.vae.decode(
                batch,
                return_dict=False,
            )[0],
            postprocess=lambda video: pipe.video_processor.postprocess_video(
                video,
                output_type="pt",
            ),
            output_layout="video_btchw",
        )
        return decoder(latents)


class WanI2VModel(WanT2VModel):
    """Wan I2V per-step math: the channel-wise image conditioning and CLIP image embeds."""

    def build_branch(
        self,
        request: DenoiseBackboneInput,
        branch: str,
    ) -> DenoiseBranch:
        """I2V branch: channel-wise conditioning concat + CLIP image embeds.

        - Hidden states are ``cat([noise_latents, condition], dim=1)`` so the
          transformer sees both the noisy frames and the (mask + first-frame
          latent) conditioning channel-wise concatenated. The condition tensor
          is produced once per request by ``pipe.prepare_latents`` and rides
          ``request.extra``.
        - ``encoder_hidden_states_image`` is the CLIP visual feature from
          ``pipe.encode_image``; cond/uncond share the same tensor object so
          the batched-CFG packer concatenates it along dim 0 alongside
          ``prompt_embeds``.

        Wan 2.2 5B's expand_timesteps mode (mask-blended hidden states) remains
        a separate future upgrade path; the A14B dual-stage routing selects
        ``transformer`` / ``transformer_2`` outside this method.
        """
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = request.negative_prompt_embeds

        extra = request.extra
        condition = expand_tensor_to_batch(
            extra["condition"],
            request.hidden_states.shape[0],
        )
        # Channel-wise concat (mimics WanImageToVideoPipeline `__call__`:
        # `latent_model_input = torch.cat([latents, condition], dim=1)`).
        hidden_states = torch.cat(
            [request.hidden_states, condition.to(request.hidden_states.dtype)],
            dim=1,
        )

        extra_kwargs: dict[str, torch.Tensor] = {}
        image_embeds = extra.get("image_embeds")
        if image_embeds is not None:
            extra_kwargs["encoder_hidden_states_image"] = expand_tensor_to_batch(
                image_embeds,
                request.hidden_states.shape[0],
            )

        return DenoiseBranch(
            hidden_states=hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs=extra_kwargs,
        )

    def _backbone_extra(self, state: WanI2VSamplingState, dtype: torch.dtype) -> dict[str, Any]:
        """I2V conditions the transformer on the first-frame latent and image embeds."""

        return {
            "condition": state.condition.to(dtype),
            "image_embeds": None if state.image_embeds is None else state.image_embeds.to(dtype),
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> WanI2VSamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return WanI2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            image_embeds=replay_tensors.get("image_embeds"),
            condition=replay_tensors["condition"],
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            boundary_ratio=batch_context.get("boundary_ratio"),
            num_train_timesteps=_scheduler_num_train_timesteps(self.scheduler),
        )


class WanI2VPipelineModel(WanT2VPipelineModel, WanI2VModel):
    """Wan I2V plus the diffusers pipeline: image encoding, conditioning latents, export.

    The I2V math (``build_branch`` / ``_backbone_extra`` / ``restore_eval_state``)
    comes from ``WanI2VModel`` and the pipeline plumbing (constructor, offload
    lifecycle, second-expert sync) from ``WanT2VPipelineModel``. Neither base
    defines the other's members, so their order does not change what resolves.
    """

    @classmethod
    def from_build(cls, build: ModelBuild) -> WanI2VPipelineModel:
        """Load WanImageToVideoPipeline + freeze generation-only modules."""
        _, trainable_transformers = wan_topology_from_build(build)
        from diffusers import WanImageToVideoPipeline

        eager_module_dtypes = {
            "vae": torch.float32,
            "text_encoder": build.parameter_dtype,
            "image_encoder": build.parameter_dtype,
        }
        pipeline = WanImageToVideoPipeline.from_pretrained(
            build.model_name_or_path,
            torch_dtype={"default": build.parameter_dtype, **eager_module_dtypes},
            **build.pretrained_kwargs,
        )
        pipeline.set_progress_bar_config(disable=True)

        for module_name in ("vae", "text_encoder", "image_encoder"):
            module = getattr(pipeline, module_name, None)
            if module is not None:
                module.requires_grad_(False)

        offload_mode = build.require_rollout().pipeline_offload_mode
        _stage_eager_wan_modules(
            pipeline,
            build,
            offload_mode=offload_mode,
            eager_module_dtypes=eager_module_dtypes,
        )
        return cls(
            pipeline=pipeline,
            device=build.device,
            trainable_transformers=trainable_transformers,
            expert_lifecycle_profiling=bool(
                (build.model_config or {}).get("expert_lifecycle_profiling", False),
            ),
        )

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode Wan I2V text and optional CLIP image conditioning."""

        max_seq = kwargs.get("max_sequence_length", 512) or 512
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
            dtype=self.transformer.dtype,
        )

        td = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)

        reference_image = kwargs.get("reference_image")
        image_embeds = kwargs.get("image_embeds")
        if image_embeds is None and reference_image is not None:
            image_embeds = self._encode_image_embeds(reference_image)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "image_embeds": image_embeds,
            "reference_image": reference_image,
        }

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        *,
        initial_latents: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> WanI2VSamplingState:
        """Build the per-request SamplingState for a Wan I2V denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        reference_image = kwargs.get("reference_image", encoded.get("reference_image"))
        if reference_image is None:
            raise ValueError("Wan I2V sampling requires a reference_image")

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        # Multistep history belongs to this batch, not the resident pipeline.
        scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = scheduler.timesteps

        batch_size = prompt_embeds.shape[0]
        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        image = pipe.video_processor.preprocess(
            reference_image,
            height=request.height,
            width=request.width,
        ).to(device, dtype=torch.float32)

        num_channels_latents = int(pipe.vae.config.z_dim)
        with torch.no_grad():
            latents, condition = pipe.prepare_latents(
                image,
                batch_size,
                num_channels_latents,
                request.height,
                request.width,
                request.frame_count,
                torch.float32,
                device,
                generator,
                initial_latents,
                None,
            )

        image_embeds = encoded.get("image_embeds")
        if image_embeds is None:
            image_embeds = self._encode_image_embeds(reference_image)
        if image_embeds is not None and image_embeds.shape[:1] != (batch_size,):
            if image_embeds.shape[:1] != (1,):
                raise ValueError(
                    "Wan I2V image_embeds batch dimension is incompatible with rollout "
                    f"batch_size={batch_size}: {tuple(image_embeds.shape)}",
                )
            image_embeds = image_embeds.repeat(batch_size, *([1] * (image_embeds.ndim - 1)))

        return WanI2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            condition=condition,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            boundary_ratio=self._boundary_ratio,
            num_train_timesteps=_scheduler_num_train_timesteps(scheduler),
        )

    def export_batch_context(self, state: WanI2VSamplingState) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.context (shared metadata)."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "conditioning": "reference_image",
            "boundary_ratio": state.boundary_ratio,
        }

    def export_replay_tensors(self, state: WanI2VSamplingState) -> dict[str, Any]:
        """Project SamplingState into trajectory replay tensors."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "image_embeds": state.image_embeds,
            "condition": state.condition,
            "latents_clean": state.latents.detach(),
        }

    def _encode_image_embeds(self, reference_image: Any) -> torch.Tensor | None:
        # Only a transformer whose config carries ``image_dim`` consumes image
        # embeds; a PEFT wrapper keeps that config on its base model.
        transformer = self.transformer
        config = getattr(transformer, "config", None)
        if not (
            (isinstance(config, dict) and "image_dim" in config)
            or (config is not None and hasattr(config, "image_dim"))
        ):
            base_model = getattr(transformer, "base_model", None)
            inner_config = getattr(getattr(base_model, "model", base_model), "config", None)
            if inner_config is not None:
                config = inner_config
        if _config_value(config, "image_dim") is None:
            return None
        with torch.no_grad():
            image_embeds = self.pipeline.encode_image(reference_image, self.device)
        return image_embeds.to(self.transformer.dtype)


def _stage_eager_wan_modules(
    pipeline: Any,
    build: ModelBuild,
    *,
    offload_mode: str,
    eager_module_dtypes: Mapping[str, torch.dtype],
) -> None:
    """Apply frozen-module dtypes before optional pipeline offload hooks."""

    for module_name, dtype in eager_module_dtypes.items():
        module = getattr(pipeline, module_name, None)
        if module is not None:
            # Diffusers loads the whole pipeline at the actor dtype. Wan's VAE
            # must still decode in fp32, including when Accelerate will later
            # stream it from CPU. A dtype-only conversion preserves CPU
            # residency; the no-offload path additionally stages the module on
            # the execution device.
            if offload_mode == "none":
                module.to(build.device, dtype=dtype)
            else:
                module.to(dtype=dtype)


def _scheduler_num_train_timesteps(scheduler: Any) -> int | None:
    value = _config_value(getattr(scheduler, "config", None), "num_train_timesteps")
    return None if value is None else int(value)


def _cuda_parameter_bytes(module: Any) -> int:
    """Bytes of this expert's parameter shards currently resident on CUDA."""

    total = 0
    for parameter in module.parameters():
        local = getattr(parameter, "_local_tensor", parameter)
        if local.device.type == "cuda":
            total += local.numel() * local.element_size()
    return total


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


__all__ = [
    "WanI2VModel",
    "WanI2VPipelineModel",
    "WanI2VSamplingState",
    "WanT2VModel",
    "WanT2VPipelineModel",
    "WanT2VSamplingState",
]
