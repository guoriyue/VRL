"""Fail-closed PEFT LoRA warm-start loading.

PEFT reconstructs an adapter from its saved ``adapter_config.json``.  Public
wm-infra LoRA settings must therefore be checked before PEFT mutates the base
model; otherwise the saved topology silently wins over the requested one.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_MISSING = object()


@dataclass(frozen=True, slots=True)
class _LoraAdapterTopology:
    """The LoRA shape and behavior that wm-infra can express publicly."""

    rank: int
    alpha: int
    dropout: float
    target_modules: frozenset[str]

    @classmethod
    def from_values(
        cls,
        *,
        rank: Any,
        alpha: Any,
        dropout: Any,
        target_modules: Any,
        owner: str,
    ) -> _LoraAdapterTopology:
        if isinstance(target_modules, str) or not isinstance(
            target_modules,
            (list, tuple, set, frozenset),
        ):
            raise ValueError(
                f"{owner} target_modules must be an explicit list or set of module names; "
                "regex and all-linear selectors are unsupported",
            )
        if not target_modules or any(
            not isinstance(target, str) or not target for target in target_modules
        ):
            raise ValueError(f"{owner} target_modules must contain non-empty strings")
        return cls(
            rank=int(rank),
            alpha=int(alpha),
            dropout=float(dropout),
            target_modules=frozenset(target_modules),
        )


def _validate_supported_lora_semantics(config: Any, *, adapter_path: str) -> None:
    from peft import LoraConfig

    defaults = LoraConfig()
    # This is a deliberate PEFT compatibility table, not a copy of LoraConfig:
    # these fields change adapter/base-model semantics but have no wm-infra
    # public configuration contract. Their dependency defaults are derived
    # directly so a PEFT default change cannot leave stale values here.
    unsupported_semantic_fields = (
        "exclude_modules",
        "fan_in_fan_out",
        "bias",
        "use_rslora",
        "modules_to_save",
        "layers_to_transform",
        "layers_pattern",
        "rank_pattern",
        "alpha_pattern",
        "megatron_config",
        "megatron_core",
        "trainable_token_indices",
        "loftq_config",
        "eva_config",
        "corda_config",
        "use_dora",
        "alora_invocation_tokens",
        "use_qalora",
        "qalora_group_size",
        "layer_replication",
        "runtime_config",
        "lora_bias",
        "target_parameters",
        "arrow_config",
        "ensure_weight_tying",
    )
    changed = []
    for field_name in unsupported_semantic_fields:
        actual = getattr(config, field_name, _MISSING)
        default = getattr(defaults, field_name, _MISSING)
        # PEFT is declared with a lower-bound rather than one exact version.
        # Fields absent from both installed config types carry no semantics to
        # validate; future saved fields still fail while parsing above.
        if actual is _MISSING and default is _MISSING:
            continue
        if actual != default:
            changed.append(field_name)

    init = config.init_lora_weights
    if isinstance(init, str) and init.lower() not in {"gaussian", "orthogonal"}:
        changed.append("init_lora_weights")

    if changed:
        raise ValueError(
            f"LoRA adapter {adapter_path!r} uses unsupported semantic field(s): "
            f"{', '.join(changed)}",
        )


def load_trainable_lora_adapter(
    base: Any,
    adapter_path: str,
    *,
    expected_rank: int,
    expected_alpha: int,
    expected_dropout: float,
    expected_target_modules: Iterable[str],
    expected_task_type: str | None = None,
    adapter_name: str = "default",
) -> Any:
    """Validate and load one trainable LoRA adapter before mutating ``base``."""

    from peft import LoraConfig, PeftConfig, PeftModel

    expected = _LoraAdapterTopology.from_values(
        rank=expected_rank,
        alpha=expected_alpha,
        dropout=expected_dropout,
        target_modules=expected_target_modules,
        owner="configured LoRA",
    )
    try:
        with warnings.catch_warnings():
            # PEFT normally drops future adapter_config keys after warning. A
            # dropped field could carry semantics this runtime cannot honor, so
            # compatibility uncertainty is a hard error at this trust boundary.
            warnings.filterwarnings(
                "error",
                message=r"Unexpected keyword arguments .* for class .*Config.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "error",
                message=r".*\bignored\b.*",
                category=UserWarning,
            )
            config = PeftConfig.from_pretrained(adapter_path)
    except UserWarning as error:
        raise ValueError(
            f"LoRA adapter {adapter_path!r} contains adapter_config fields "
            f"unsupported by the installed PEFT version: {error}",
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"failed to read PEFT adapter_config from {adapter_path!r}",
        ) from error

    if type(config) is not LoraConfig:
        raise ValueError(
            f"adapter {adapter_path!r} is {type(config).__name__}, not a PEFT LoRA adapter",
        )
    _validate_supported_lora_semantics(config, adapter_path=adapter_path)
    actual = _LoraAdapterTopology.from_values(
        rank=config.r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.target_modules,
        owner=f"LoRA adapter {adapter_path!r}",
    )

    mismatches = []
    if actual.rank != expected.rank:
        mismatches.append(f"rank saved={actual.rank} configured={expected.rank}")
    if actual.alpha != expected.alpha:
        mismatches.append(f"alpha saved={actual.alpha} configured={expected.alpha}")
    if actual.dropout != expected.dropout:
        mismatches.append(
            f"dropout saved={actual.dropout} configured={expected.dropout}",
        )
    if actual.target_modules != expected.target_modules:
        mismatches.append(
            "target_modules "
            f"saved={sorted(actual.target_modules)!r} "
            f"configured={sorted(expected.target_modules)!r}",
        )
    actual_task_type = getattr(config.task_type, "value", config.task_type)
    if actual_task_type != expected_task_type:
        mismatches.append(
            f"task_type saved={actual_task_type!r} expected={expected_task_type!r}",
        )
    if mismatches:
        raise ValueError(
            f"LoRA adapter topology mismatch for {adapter_path!r}: {'; '.join(mismatches)}",
        )

    return PeftModel.from_pretrained(
        base,
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=True,
        config=config,
    )


__all__ = ["load_trainable_lora_adapter"]
