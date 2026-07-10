"""Runtime build spec and runtime bundle (CONTRACT.md, SPRINT_model_refactor.md §5.1, §5.3.E).

These two dataclasses are the only sanctioned interface between training scripts
and family-adjacent builders. Scripts must not import family model classes
directly; builders consume a ``RuntimeBuildSpec`` and return a ``RuntimeBundle``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.models.interfaces.replay import RuntimeModel

MEMORY_POLICY_METADATA_KEY = "memory_policy"

# Single source of truth for valid ``model.memory`` subsection names, shared by
# the config schema's unknown-key lint and the generation memory policy's typo
# check. Today only ``vae_decode`` (sliced/tiled VAE decode, applied on the
# rollout/generation side). The trainer needs no section here: each family
# builds a minimal ReplayModel that never loads the generation-only modules, so
# there is nothing to offload. Adding a section means editing this tuple once;
# both consumers derive from it.
MODEL_MEMORY_SECTIONS: tuple[str, ...] = ("vae_decode",)

# Single source of truth for the model_config compile block that the
# ``RuntimeBuildSpec.torch_compile`` property below consumes.
TORCH_COMPILE_MODEL_KEY = "torch_compile"


@dataclass
class RuntimeBuildSpec:
    """Runtime-only slice of the whole RL config.

    Builders take this, not the whole RL cfg. Reward / algorithm / trainer /
    dataset / logging cadence are explicitly out of scope.

    ``model_config`` / ``sampling_config`` carry the runtime-relevant config
    blocks (``cfg.model`` / ``cfg.sampling``) wholesale as deep-converted plain
    dicts. The read properties below expose the common curated views so
    consumers read ``spec.memory`` / ``spec.lora`` / ``spec.num_steps`` directly
    instead of re-deriving from the raw block. They centralize the lora /
    scheduler / memory / compile transforms in one place, so no read-time logic
    is duplicated per family. Family-specific fields (e.g. anima checkpoint
    paths) are read straight from ``model_config``. The universal typed fields
    are runtime-injected or needed by every family, so they stay typed.
    """

    model_name_or_path: str
    device: Any
    dtype: Any
    # Canonical rollout family name, set by the registry-descriptor extractor
    # (vrl.models.diffusion.build:extract_family_runtime_spec) so the generic
    # builders can look the family's build recipe up worker-side. None on the
    # legacy per-family builder path, which binds its model class in code.
    family: str | None = None
    # Diffusion t2v/i2v axis only. AR families must NOT reuse this field; they
    # carry their trajectory variant in ``ar_task`` so a single field never holds
    # two disjoint enums (t2v/i2v vs ar_t2i/ar_t2i_r1).
    task_variant: str | None = None
    # AR trajectory variant (``ar_t2i`` / ``ar_t2i_r1``). Selects the AR family
    # capability and replay shape; serialized across the Ray launch contract
    # alongside ``task_variant`` (both ride through ``asdict``).
    ar_task: str | None = None
    model_config: dict[str, Any] | None = None
    sampling_config: dict[str, Any] | None = None
    # ``frozen`` precision axis as a ``torch.dtype`` (encoders / VAE). Like
    # ``dtype``, it is a real dtype in memory and serialized to a name-string
    # across the Ray launch contract. None -> the family's historical derivation
    # (fp16 when the model runs fp32).
    frozen_dtype: Any = None
    # Rollout-only quantized GEMM token derived from ``precision.rollout``
    # ("fp8"/"fp4"), or None. The runtime builder swaps the transformer's big
    # linears to fp8 when this is "fp8" (storage stays the bf16 master ``dtype``).
    rollout_quantization: str | None = None
    # Kernel recipe for the quantized rollout (``precision.rollout_recipe``), or
    # None for the scheme default (fp8: "rowwise"). Consumed by
    # ``apply_rollout_quantization``; only ever set alongside a quantized rollout
    # (the precision resolver rejects it otherwise).
    rollout_quantization_recipe: str | None = None
    # Whether base weights will ever be synced INTO this rollout model. True
    # for trainer-driven rollouts (full-finetune sync loads base weights;
    # LoRA sync loads adapters — the quantizer already distinguishes via
    # use_lora). Sync-free contexts (generation probes, eval-only runs) set
    # False so fp8 can drop the bf16 masters BEFORE the device move — the
    # difference between a 17B rollout fitting a 32GB card or not.
    # Consumed by vrl.models.loader.apply_rollout_quantization.
    rollout_weight_sync: bool = True

    @property
    def use_lora(self) -> bool:
        """Whether the family should attach a LoRA adapter.

        ``False`` fallback when the block is absent (safe for fake test specs);
        every real experiment config sets ``model.use_lora`` explicitly.
        """
        return bool((self.model_config or {}).get("use_lora", False))

    @property
    def lora_path(self) -> str | None:
        """Resolved LoRA checkpoint path, or ``None`` when not loading one."""
        lora = (self.model_config or {}).get("lora") or {}
        return lora.get("path") or None

    @property
    def lora(self) -> dict[str, Any] | None:
        """Curated LoRA config (``rank``/``alpha``/``target_modules`` + extras).

        ``None`` when ``use_lora`` is off. Casts and the ``init_lora_weights`` /
        ``dropout`` / ``init`` extras are carried from the raw ``model.lora``
        block only when present, preserving per-family presence semantics. AR
        families layer their own defaults via ``_resolve_lora_block``.
        """
        if not self.use_lora:
            return None
        lora = (self.model_config or {}).get("lora") or {}
        config: dict[str, Any] = {
            "rank": int(lora["rank"]),
            "alpha": int(lora["alpha"]),
            "target_modules": list(lora["target_modules"]),
        }
        for key in ("init_lora_weights", "dropout", "init"):
            if key in lora:
                config[key] = lora[key]
        return config

    @property
    def num_steps(self) -> int | None:
        """Diffusion scheduler step count from ``sampling.num_steps``."""
        num_steps = (self.sampling_config or {}).get("num_steps")
        return None if num_steps is None else int(num_steps)

    @property
    def torch_compile(self) -> dict[str, Any] | None:
        """``model.torch_compile`` block only when ``enable`` is truthy."""
        block = (self.model_config or {}).get(TORCH_COMPILE_MODEL_KEY) or {}
        if not block.get("enable"):
            return None
        return {"enable": True, "mode": block.get("mode", "default")}

    @property
    def memory(self) -> dict[str, Any] | None:
        """The whole ``model.memory`` block (consumer extracts its sub-block)."""
        return (self.model_config or {}).get("memory")


@dataclass
class RuntimeBundle:
    """Sole output of a family builder. Consumed by scripts / collectors / trainers.

    ``raw_handle`` carries the raw family object (e.g. the diffusers pipeline
    or AR upstream wrapper) and must be treated as builder-internal — scripts
    and trainers must not reach into it.

    ``model`` is the family-provided general inference object. Shared trainer
    runtime code only relies on the narrow ``vrl.models.interfaces.RuntimeModel``
    contract: ``replay_forward``, ``disable_adapter``, and
    ``load_trainable_state``.
    Generation-only methods remain family/runtime implementation details.

    ``trainable_modules`` is the training-checkpoint contract. Every module
    registered here must expose PyTorch-compatible ``state_dict`` and
    ``load_state_dict`` methods. Generic trainer checkpointing saves and
    restores only these modules, so family builders must include every
    trainable adapter/backbone needed for exact resume.

    ``metadata`` may include generic replay/runtime flags used by shared
    trainer infrastructure. Build these through ``vrl.models.replay_loading`` so
    family runtimes share one contract:

    - ``loads_full_generation_modules``: true when the bundle owns
      generation-only modules such as prompt encoders, VAE/VQ decoders, or a
      full pipeline object. Consumed by the colocated-RAM guard
      (``validate_colocated_replay_memory``).
    """

    model: RuntimeModel
    trainable_modules: dict[str, Any]
    scheduler: Any
    raw_handle: Any
    metadata: dict[str, Any] = field(default_factory=dict)
