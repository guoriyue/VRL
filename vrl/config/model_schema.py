"""Lightweight public schema for the YAML ``model`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery validate family-owned keys
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from pydantic import Field, model_validator

from vrl.config.base import ConfigBase
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class LoraSection(ConfigBase):
    """Shared adapter inputs consumed by ``ModelBuild.lora``."""

    rank: int | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", required=True),
    )
    alpha: int | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", required=True),
    )
    path: str | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    target_modules: list[str] | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "value",
            required=True,
            canonicalize="sorted_unique",
        ),
    )
    init_lora_weights: str | bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    dropout: float | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=0.0),
    )
    autocast_adapter_dtype: bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value"),
    )
    parameter_dtype: Literal["float32"] | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value"),
    )


class VaeDecodeMemorySection(ConfigBase):
    """VAE decode memory switches consumed by the decode-memory policy."""

    tiling: bool | None = None
    slicing: bool | None = None


class ModelMemorySection(ConfigBase):
    """Target-keyed generation memory configuration."""

    vae_decode: VaeDecodeMemorySection | None = None


# Runtime capabilities and generation-memory targets share this public section
# namespace. Derive it from the typed structure so adding a section cannot leave
# a stale hand-maintained allow-list behind.
MODEL_MEMORY_SECTIONS: tuple[str, ...] = tuple(ModelMemorySection.model_fields)


class TorchCompileSection(ConfigBase):
    """Transformer compile inputs consumed by ``ModelBuild.torch_compile``."""

    enable: bool | None = None
    mode: str | None = None
    # Which build roles compile: all | rollout | replay. Vocabulary owned by
    # ``vrl.models.interfaces.runtime.TorchCompileScope`` (validated
    # at config load by the compile matrix and per build by the property).
    scope: str | None = None
    # Compile each repeated transformer block in place instead of the root
    # module: one trace serves every block, so cold compile and recompiles cost
    # one block rather than the whole graph. Steady-state math is unchanged.
    regional: bool | None = None


class ModelExecutorSection(ConfigBase):
    """Shared ``GenericDiffusionBatchExecutor`` constructor inputs."""

    num_frames: int | None = None
    max_sequence_length: int | None = None
    fps: int | None = None
    batch_passthrough_keys: list[str] | None = None


class ModelSection(ConfigBase):
    """Keys shared by every registered model family."""

    # Defaults belong to the lightweight family schema, not the loaded model.
    # Target names/rank/alpha remain recipe inputs in the model presets.
    lora_defaults: ClassVar[LoraSection] = LoraSection(
        init_lora_weights="gaussian",
        autocast_adapter_dtype=True,
        dropout=0.0,
    )

    @classmethod
    def resolve_lora(cls, values: dict[str, Any] | LoraSection | None) -> LoraSection:
        """Resolve explicit settings over family defaults without mutating either."""

        requested = (
            values if isinstance(values, LoraSection) else LoraSection.model_validate(values or {})
        )
        return LoraSection.model_validate(
            {
                **ModelSection.lora_defaults.model_dump(exclude_none=True),
                **cls.lora_defaults.model_dump(exclude_none=True),
                **requested.model_dump(exclude_none=True),
            }
        )

    family: str = Field(
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    # Readers: ModelBuild plus family runtime LoRA projections.
    lora: LoraSection | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "lora",
            enabled_by="use_lora",
        ),
    )
    # Global section shape; the selected family validates supported targets.
    memory: ModelMemorySection | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    path: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="main",
            revision_field="revision",
        ),
    )
    # Immutable Hub snapshot used by full-pipeline rollout and component replay.
    revision: str | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source_revision",
            source="main",
        ),
    )
    # Reader: ModelBuild.pretrained_kwargs. Long-run configs can
    # fail closed on a missing cached artifact instead of consulting the Hub.
    # Load-source knob only (same weights either way), so identity-excluded.
    local_files_only: bool = Field(
        default=False,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    torch_compile: TorchCompileSection | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    # Run the policy's hand-written RMSNorm modules (diffusers ``RMSNorm``, the
    # Q/K norms of Cosmos, SD3.5 and Qwen-Image) as one fused kernel. Applied
    # to rollout AND replay so both roles share one rounding; kernel choice
    # only, same weights, so identity-excluded like torch_compile.
    fused_rms_norm: bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    # Run each feed-forward up-projection and its tanh-GELU (diffusers ``GELU``
    # in SD3.5, Wan, Flux, Qwen-Image) as one GEMM with the activation in the
    # epilogue. Rollout only: the fused op has no backward, so replay keeps the
    # reference kernel. Kernel choice, same weights, so identity-excluded.
    fused_gelu_projection: bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    # Add each LoRA site's delta into the base output in place instead of
    # materializing the fp32 sum and casting it back. Rollout only; bit-identical
    # to the peft forward, so replay needs no mirror. Kernel choice, same
    # weights, so identity-excluded.
    fused_lora_branch: bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )
    use_lora: bool | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=False),
    )
    # Shared GenericDiffusionBatchExecutor constructor values. The selected family
    # validates this block at typed parse and again at launch projection.
    executor: ModelExecutorSection | None = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("exclude"),
    )

    @model_validator(mode="after")
    def _validate_lora(self) -> Self:
        # Without adapters the pass cannot replace any LoRA branch.
        if self.fused_lora_branch and not self.use_lora:
            raise ValueError("model.fused_lora_branch requires model.use_lora=true")
        resolved = self.resolve_lora(self.lora)
        if resolved.parameter_dtype is not None and not self.use_lora:
            raise ValueError("model.lora.parameter_dtype requires model.use_lora=true")
        return self


__all__ = [
    "MODEL_MEMORY_SECTIONS",
    "LoraSection",
    "ModelExecutorSection",
    "ModelMemorySection",
    "ModelSection",
    "TorchCompileSection",
    "VaeDecodeMemorySection",
]
