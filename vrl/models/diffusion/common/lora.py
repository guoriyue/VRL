"""Shared PEFT LoRA attach logic for diffusion family models.

Five families carried near-identical copies of ``apply_lora``; the PEFT call
convention (PeftModel.from_pretrained for resume vs LoraConfig+get_peft_model
for fresh adapters) is family-agnostic, so it lives here once. Families only
override the small hooks that actually differ.

Cosmos Predict2.5 deliberately does NOT use this mixin: its ``apply_lora``
additionally manages a second "previous" adapter for NFT previous-policy
replay, which is a different shape, not a copy.
"""

from __future__ import annotations

from typing import Any


class LoraModelMixin:
    """Attach a PEFT LoRA adapter to the family transformer per runtime spec."""

    # Fresh-adapter weight init when the config does not set init_lora_weights.
    # Wan overrides to True: empty training adapters must initially preserve
    # base Wan output.
    _lora_default_init_weights: Any = "gaussian"

    def _lora_transformer(self) -> Any:
        """The transformer module to wrap (anima keeps it off-pipeline)."""
        return self.pipeline.transformer

    def _lora_dtype(self, spec: Any) -> Any | None:
        """Optional dtype for the pre-wrap device move (sd3_5/anima cast)."""
        del spec
        return None

    def apply_lora(self, spec: Any) -> None:
        """Wrap the family transformer with PEFT LoRA per ``spec.lora_*``."""
        from peft import LoraConfig, PeftModel, get_peft_model

        transformer = self._lora_transformer()
        transformer.requires_grad_(False)
        dtype = self._lora_dtype(spec)
        if dtype is None:
            transformer.to(self.device)
        else:
            transformer.to(self.device, dtype=dtype)

        lora_path = spec.lora_path
        if lora_path:
            wrapped = PeftModel.from_pretrained(
                transformer,
                lora_path,
                is_trainable=True,
            )
            wrapped.set_adapter("default")
            self._set_transformer(wrapped)
            return

        lora_config = spec.lora
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


def wrap_transformer_lora(
    transformer: Any,
    spec: Any,
    device: Any,
    *,
    default_init: Any = "gaussian",
) -> Any:
    """Wrap a single transformer with a fresh PEFT LoRA adapter and return it.

    Mirrors ``LoraModelMixin.apply_lora`` / ``apply_lora_to_transformer`` for one
    module, without the pipeline/self coupling — used to attach a second adapter
    to Wan 2.2's low-noise expert (``transformer_2``). Resume-from-checkpoint
    (``spec.lora_path``) is rejected here because a single adapter dir cannot
    carry two distinct experts; per-expert adapter dirs are a follow-up.
    """
    from peft import LoraConfig, get_peft_model

    if spec.lora_path:
        raise NotImplementedError(
            "Resuming a second LoRA expert (Wan 2.2 transformer_2) from a single "
            "lora_path is not supported; fresh adapters only for now.",
        )
    transformer.requires_grad_(False)
    transformer.to(device)
    lora_config = spec.lora
    assert lora_config is not None
    cfg = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        init_lora_weights=lora_config.get("init_lora_weights", default_init),
        target_modules=lora_config["target_modules"],
    )
    return get_peft_model(transformer, cfg)


__all__ = ["LoraModelMixin", "wrap_transformer_lora"]
