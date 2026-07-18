"""Shared PEFT LoRA attach logic for diffusion family models.

Five families carried near-identical copies of ``apply_lora``; the PEFT call
convention (PeftModel.from_pretrained for resume vs LoraConfig+get_peft_model
for fresh adapters) is family-agnostic, so it lives here once. Families only
override the small hooks that actually differ.

The DiffusionNFT previous-policy adapter primitives (``build_lora_config`` /
``copy_adapter_weights`` / ``freeze_adapter_params``) also live here: they are
pure PEFT operations with no family specifics, shared by both cosmos/predict2.5
(its custom ``apply_lora``) and flux (its ``attach_previous_policy_adapter`` /
``sync_previous_policy_adapter`` methods). Keeping one copy here is the same
de-duplication rationale as the attach logic above — a second copy would rot the
moment the PEFT param-naming convention shifts under one family but not the other.
"""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces.runtime import ModelBuild


class LoraModelMixin:
    """Attach a PEFT LoRA adapter to the family transformer per runtime build."""

    # Fresh-adapter weight init when the config does not set init_lora_weights.
    # Wan overrides to True: empty training adapters must initially preserve
    # base Wan output.
    _lora_default_init_weights: Any = "gaussian"

    def _lora_transformer(self) -> Any:
        """The trainable transformer to wrap.

        Reads ``self.transformer`` — kept in sync with ``pipeline.transformer`` by
        every family's ``_set_transformer`` — instead of ``pipeline.transformer``,
        so this one attach path also serves the pipeline-less replay models (which
        set ``self.transformer`` directly and raise on ``pipeline``).
        """
        return self.transformer

    def _lora_dtype(self, build: ModelBuild) -> Any | None:
        """Dtype for the pre-wrap device move; ``None`` skips the cast.

        Default: the build's parameter dtype — the near-universal diffusers-family
        behavior. Overrides: cosmos predict2/cosmos3 return ``None`` (their
        transformer is already cast at load); anima/echo return their stored
        ``self._dtype`` (single-file checkpoints with their own parameter storage).
        """

        return build.parameter_dtype

    def apply_lora(self, build: ModelBuild) -> None:
        """Wrap the family transformer with PEFT LoRA per ``build.lora_*``."""
        from peft import LoraConfig, PeftModel, get_peft_model

        transformer = self._lora_transformer()
        transformer.requires_grad_(False)
        dtype = self._lora_dtype(build)
        # Quantized LoRA rollouts attach PEFT while the checkpoint is still on
        # CPU. The shared builder then swaps the wrapped base linears, drops
        # their unsynced masters, and only moves the compact policy to the GPU.
        # Moving here would make a 17B bf16 checkpoint exceed a 32GB card before
        # quantization has a chance to compact it. Plain-precision LoRA keeps
        # the historical direct move; replay builders never request rollout
        # quantization.
        rollout = getattr(build, "rollout", None)
        defer_device_move = bool(
            rollout is not None
            and getattr(getattr(build, "precision", None), "quantization", None),
        )
        if not defer_device_move:
            if dtype is None:
                transformer.to(self.device)
            else:
                transformer.to(self.device, dtype=dtype)

        lora_path = build.lora_path
        if lora_path:
            wrapped = PeftModel.from_pretrained(
                transformer,
                lora_path,
                is_trainable=True,
            )
            wrapped.set_adapter("default")
            self._set_transformer(wrapped)
            return

        lora_config = build.lora
        assert lora_config is not None
        cfg = LoraConfig(
            r=lora_config["rank"],
            lora_alpha=lora_config["alpha"],
            init_lora_weights=lora_config.get(
                "init_lora_weights",
                self._lora_default_init_weights,
            ),
            target_modules=lora_config["target_modules"],
        )
        self._set_transformer(get_peft_model(transformer, cfg))


def build_lora_config(lora_config: Any) -> Any:
    """Build the LoRA config for a family's adapters from one ``model.lora`` block.

    The ``default`` and the frozen ``previous`` mirror use identical settings, so
    this names that single shape instead of repeating the literal. Init does not
    matter for ``previous`` (it is overwritten by ``copy_adapter_weights`` right
    after creation), so ``gaussian`` matches the fresh-default init.
    """

    from peft import LoraConfig

    return LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        init_lora_weights="gaussian",
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


def freeze_adapter_params(module: Any, adapter: str) -> None:
    """Set ``requires_grad=False`` on every parameter of the named PEFT adapter.

    Used for NFT's ``previous`` adapter: it is only forward-evaluated under
    no_grad and refreshed by weight copy (``sync_previous_policy_adapter``),
    never optimized. PEFT creates adapter params with ``requires_grad=True``, so
    without this DDP's reducer expects a gradient for them that the no_grad
    replay never produces — failing the first backward unless the more expensive
    ``find_unused_parameters=true`` is forced. Freezing it keeps the cheaper
    ``find_unused_parameters=false`` correct.
    """

    marker = f".{adapter}."
    frozen = 0
    for name, param in module.named_parameters():
        if marker in name:
            param.requires_grad_(False)
            frozen += 1
    if frozen == 0:
        raise RuntimeError(
            f"no parameters found for adapter {adapter!r} to freeze",
        )


__all__ = [
    "LoraModelMixin",
    "build_lora_config",
    "copy_adapter_weights",
    "freeze_adapter_params",
]
