"""Runtime build spec and runtime bundle (CONTRACT.md, SPRINT_model_refactor.md §5.1, §5.3.E).

These two dataclasses are the only sanctioned interface between training scripts
and family-adjacent builders. Scripts must not import backend classes directly;
builders consume a ``RuntimeBuildSpec`` and return a ``RuntimeBundle``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.models.interfaces.replay import RuntimeModel

MEMORY_POLICY_METADATA_KEY = "memory_policy"


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
    backend_preference: tuple[str, ...] = ("diffusers",)
    task_variant: str | None = None
    model_config: dict[str, Any] | None = None
    sampling_config: dict[str, Any] | None = None

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
        block = (self.model_config or {}).get("torch_compile") or {}
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

    ``backend_handle`` carries the raw backend object (e.g. diffusers pipeline
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
    trainer infrastructure. Build these fields through
    ``vrl.models.replay_loading`` so family runtimes share one contract:

    - ``runtime_role``: e.g. ``"full_generation_model"`` or
      ``"minimal_replay_model"``.
    - ``loads_full_generation_modules``: true when the trainer bundle owns
      generation-only modules such as prompt encoders, VAE/VQ decoders, or a
      full pipeline object.
    - ``requires_minimal_replay_loader``: true when colocated Ray training is
      known to benefit from a family-specific replay-only loader.
    """

    model: RuntimeModel
    trainable_modules: dict[str, Any]
    scheduler: Any
    backend_kind: str
    backend_handle: Any
    ref_modules: dict[str, Any] | None = None
    runtime_caps: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
