"""PEFT LoRA primitives shared by every denoise family.

``DiffusionModelBase.apply_lora`` is the single attach site: it walks the
model's ``trainable_modules`` and hands each root to ``attach_lora_adapter``
here (validated saved adapter for a warm start, ``LoraConfig`` +
``get_peft_model`` for a fresh one). Families declare data on the base class
(default init, adapter dtype policy, whether the frozen ``previous`` mirror is
mandatory) and never re-implement the PEFT call sequence.

The previous-policy adapter primitives (``build_lora_config`` /
``copy_adapter_weights`` / ``freeze_checkpoint_owned_adapter_params`` /
``attach_previous_policy_adapter``) are pure PEFT operations too. The frozen
``previous`` LoRA mirror they build is what DiffusionNFT's negative branch and
V-GRPO's importance ratio evaluate the behaviour policy through.
"""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces.runtime import ModelBuild, register_checkpoint_owned_state
from vrl.models.peft_adapter import load_trainable_lora_adapter


def require_lora_config(build: ModelBuild) -> dict[str, Any]:
    """Return ``build.lora`` or fail: attach without a LoRA block is a config bug."""

    lora_config = getattr(build, "lora", None)
    if lora_config is None:
        raise ValueError("LoRA runtime build requires model.lora configuration")
    return lora_config


def attach_lora_adapter(
    module: Any,
    lora_config: dict[str, Any],
    *,
    lora_path: str | None = None,
    init_weights_default: Any = "gaussian",
    autocast_adapter_dtype: bool = True,
    adapter_name: str = "default",
) -> Any:
    """Wrap one trainable root with a PEFT adapter per ``model.lora``.

    A warm start (``lora_path``) validates the saved adapter's topology against
    the configured one before mutating ``module``; a fresh adapter takes
    ``model.lora.init_lora_weights`` or the family's ``init_weights_default``.
    ``autocast_adapter_dtype`` is PEFT's fp32 adapter upcast; it is threaded
    through both branches so a warm-started adapter matches a fresh one.
    """

    if lora_path:
        wrapped = load_trainable_lora_adapter(
            module,
            lora_path,
            expected_rank=lora_config["rank"],
            expected_alpha=lora_config["alpha"],
            expected_dropout=lora_config.get("dropout", 0.0),
            expected_target_modules=lora_config["target_modules"],
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
        )
        wrapped.set_adapter(adapter_name)
        return wrapped

    from peft import get_peft_model

    return get_peft_model(
        module,
        build_lora_config(
            lora_config,
            init_lora_weights=lora_config.get("init_lora_weights", init_weights_default),
        ),
        adapter_name=adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
    )


def attach_previous_policy_adapter(transformer: Any, lora_config: dict[str, Any]) -> None:
    """Build the frozen ``previous`` adapter on ``transformer``, seeded from ``default``.

    Idempotent on the adapter slot: only adds it once, then (re)seeds it from
    the current ``default`` so ``previous == default`` at attach time (the
    lr=0 invariants of NFT and V-GRPO). Leaves ``default`` active.
    """

    if "previous" not in getattr(transformer, "peft_config", {}):
        transformer.add_adapter("previous", build_lora_config(lora_config))
    copy_adapter_weights(transformer, src="default", dst="previous")
    freeze_checkpoint_owned_adapter_params(transformer, "previous")
    transformer.set_adapter("default")


def previous_policy_adapter_requested(build: ModelBuild) -> bool:
    """Whether ``model.nft_previous_adapter`` asks for the frozen mirror."""

    # Bare test builds are namespaces without model_config; treat as "no".
    model_config = getattr(build, "model_config", None) or {}
    return bool(model_config.get("nft_previous_adapter", False))


def require_lora_for_previous_policy_adapter(build: ModelBuild) -> None:
    """Reject the previous-adapter switch without LoRA before paying a model load."""

    if previous_policy_adapter_requested(build) and not build.use_lora:
        raise RuntimeError(
            "model.nft_previous_adapter requires LoRA (the frozen previous "
            "adapter is a PEFT adapter); set model.use_lora=true.",
        )


def build_lora_config(lora_config: Any, *, init_lora_weights: Any = "gaussian") -> Any:
    """Build one PEFT ``LoraConfig`` from a ``model.lora`` block.

    The ``default`` adapter and its frozen ``previous`` mirror share this
    shape. Init only matters for a fresh ``default`` (``attach_lora_adapter``
    passes the resolved value); ``previous`` is overwritten by
    ``copy_adapter_weights`` right after creation, so the default suffices.
    """

    from peft import LoraConfig

    return LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config.get("dropout", 0.0),
        init_lora_weights=init_lora_weights,
        target_modules=lora_config["target_modules"],
    )


def copy_adapter_weights(
    module: Any,
    *,
    src: str,
    dst: str,
    decay: float = 0.0,
) -> None:
    """Copy (or EMA-blend) one PEFT adapter's params into another in place.

    ``decay=0`` is an exact copy; ``decay`` in (0, 1] is a soft update
    ``dst <- decay*dst + (1-decay)*src`` (NFT ``weight_copy_decay``). Matches
    params by the ``.{src}.`` / ``.{dst}.`` marker PEFT puts in every adapter path.
    """

    named = dict(module.named_parameters())
    copied = 0
    decay = float(decay)
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"adapter weight copy decay must be in [0, 1], got {decay}")
    for name, param in named.items():
        src_marker = f".{src}."
        if src_marker not in name:
            continue
        dst_name = name.replace(src_marker, f".{dst}.")
        dst_param = named.get(dst_name)
        if dst_param is None:
            continue
        if decay == 0.0:
            dst_param.data.copy_(param.data)
        else:
            dst_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        copied += 1
    if copied == 0:
        raise RuntimeError(
            f"failed to copy adapter weights from {src!r} to {dst!r}; "
            "no matching adapter parameters were found",
        )


def freeze_checkpoint_owned_adapter_params(module: Any, adapter: str) -> None:
    """Freeze a mutable PEFT adapter and register it for exact checkpoint resume.

    Used for NFT's ``previous`` adapter: it is only forward-evaluated under
    no_grad and refreshed by weight copy (``sync_previous_policy_adapter``),
    never optimized. PEFT creates adapter params with ``requires_grad=True``, so
    without this DDP's reducer expects a gradient for them that the no_grad
    replay never produces — failing the first backward unless the more expensive
    ``find_unused_parameters=true`` is forced. Freezing it keeps the cheaper
    ``find_unused_parameters=false`` correct.
    """

    marker = f".{adapter}."
    matched = [
        (name, parameter)
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if marker in name
    ]
    if not matched:
        raise RuntimeError(
            f"no parameters found for adapter {adapter!r} to freeze",
        )
    prior_requires_grad = [parameter.requires_grad for _, parameter in matched]
    try:
        for _, parameter in matched:
            parameter.requires_grad_(False)
        register_checkpoint_owned_state(module, (name for name, _ in matched))
    except BaseException:
        for (_, parameter), requires_grad in zip(
            matched,
            prior_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)
        raise


__all__ = [
    "attach_lora_adapter",
    "attach_previous_policy_adapter",
    "build_lora_config",
    "copy_adapter_weights",
    "freeze_checkpoint_owned_adapter_params",
    "previous_policy_adapter_requested",
    "require_lora_config",
    "require_lora_for_previous_policy_adapter",
]
