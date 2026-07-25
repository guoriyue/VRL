"""PEFT/LoRA adapter lifecycle: warm-start loading, peeling, and on/off switching.

Loading is fail-closed. PEFT reconstructs an adapter from its saved
``adapter_config.json``.  Public wm-infra LoRA settings must therefore be checked
before PEFT mutates the base model; otherwise the saved topology silently wins
over the requested one.

The helpers below :func:`load_trainable_lora_adapter` act on an *already attached*
adapter: :func:`peel_peft` reaches the wrapped inner module, while
:func:`disable_adapter_on` / :func:`activate_adapter_on` switch the adapter for a
reference or previous-policy forward pass.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from vrl.models.weight_utils import unwrap_compile_and_ddp

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


def peel_peft(module: Any) -> Any:
    """Peel a PEFT wrapper (``base_model.model``) off ``module``, else return it.

    PEFT replaces the target ``nn.Linear`` modules in place, so the peeled inner
    module still routes through the LoRA layers. Cannot key off
    ``hasattr(module, "base_model")`` alone: HF ``PreTrainedModel`` exposes
    ``base_model`` as a property returning ``self`` even without a PEFT wrap, and
    that object has no ``.model`` attr. Only peel when the PEFT inner path exists.
    """

    inner = getattr(module, "base_model", None)
    if inner is not None and hasattr(inner, "model") and inner.model is not module:
        return inner.model
    return module


def disable_adapter_on(module: Any) -> contextlib.AbstractContextManager[None]:
    """Context manager disabling ``module``'s LoRA/PEFT adapter, or a no-op when absent.

    Used for the reference (adapter-off) forward pass. Switches on the module
    behind any DDP/``torch.compile`` wrapper (same reason as
    :func:`activate_adapter_on`) and covers both adapter-disable surfaces:

    - PEFT ``PeftModel.disable_adapter()`` — already a context manager;
    - diffusers ``PeftAdapterMixin`` ``disable_adapters()`` / ``enable_adapters()``.

    The plural surface matters: ``WanTransformer3DModel`` / cosmos-predict2 carry
    LoRA via ``PeftAdapterMixin.add_adapter`` and expose ONLY the plural pair, so
    checking the singular method alone silently failed to disable the adapter —
    the reference forward (e.g. the diffusion GRPO KL term in
    ``evaluators/diffusion/sde_logprob.py``) then ran with the policy adapter
    still on. A module exposing neither surface is genuinely adapter-less and
    returns a null context.
    """

    host = unwrap_compile_and_ddp(module)

    disable = getattr(host, "disable_adapter", None)
    if callable(disable):
        return disable()

    disable_adapters = getattr(host, "disable_adapters", None)
    enable_adapters = getattr(host, "enable_adapters", None)
    if callable(disable_adapters) and callable(enable_adapters):
        if not _has_plural_adapter_loaded(host):
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def _disabled() -> Iterator[None]:
            disabled_before = _plural_adapter_disable_flags(host)
            restore_enabled = (
                any(not disabled for disabled in disabled_before) if disabled_before else True
            )
            disable_adapters()
            try:
                yield
            finally:
                if restore_enabled:
                    enable_adapters()

        return _disabled()

    return contextlib.nullcontext()


def _has_plural_adapter_loaded(module: Any) -> bool:
    """Whether a diffusers-style plural adapter surface has real adapters loaded."""

    loaded = getattr(module, "_hf_peft_config_loaded", None)
    if loaded is False:
        return False
    peft_config = getattr(module, "peft_config", None)
    return not (isinstance(peft_config, Mapping) and not peft_config)


def _plural_adapter_disable_flags(module: Any) -> tuple[bool, ...]:
    """Return per-layer disable flags for PEFT tuner layers when exposed."""

    named_modules = getattr(module, "named_modules", None)
    if not callable(named_modules):
        return ()
    flags: list[bool] = []
    for _name, child in named_modules():
        disabled = getattr(child, "disable_adapters", None)
        if isinstance(disabled, bool):
            flags.append(disabled)
            continue
        disabled = getattr(child, "_disable_adapters", None)
        if isinstance(disabled, bool):
            flags.append(disabled)
    return tuple(flags)


def activate_adapter_on(module: Any, name: str) -> contextlib.AbstractContextManager[None]:
    """Context manager activating PEFT adapter ``name``, restoring ``"default"`` on exit.

    The named-adapter sibling of :func:`disable_adapter_on`, used for the
    previous-policy forward pass. The adapter is switched on the module *behind*
    any DDP / ``torch.compile`` wrapper: those wrappers do not proxy PEFT's
    ``set_adapter``, but they wrap the same underlying module, so the switch is
    visible to the wrapped grad forward too. ``set_adapter`` exists on both the
    PEFT ``PeftModel`` and the diffusers ``PeftAdapterMixin`` surfaces. Unlike
    *disabling* — a sensible no-op on an adapter-less module — activating a
    *named* adapter requires one, so a module without ``set_adapter`` raises
    rather than silently forwarding the wrong weights.
    """

    host = unwrap_compile_and_ddp(module)
    set_adapter = getattr(host, "set_adapter", None)
    if not callable(set_adapter):
        raise RuntimeError(
            f"cannot activate adapter {name!r}: {type(host).__name__} has no "
            "set_adapter(); attach a PEFT adapter before requesting it",
        )

    @contextlib.contextmanager
    def _activated() -> Iterator[None]:
        set_adapter(name)
        try:
            yield
        finally:
            set_adapter("default")

    return _activated()


__all__ = [
    "activate_adapter_on",
    "disable_adapter_on",
    "load_trainable_lora_adapter",
    "peel_peft",
]
