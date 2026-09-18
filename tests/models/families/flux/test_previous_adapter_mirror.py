"""FLUX's frozen ``previous`` LoRA mirror.

The forward-process objectives (DiffusionNFT / V-GRPO) reach FLUX only through
the shared replay contract, so the family-specific piece left to verify is the
adapter lifecycle: a genuine tiny PEFT transformer goes through
``attach_previous_policy_adapter`` (the real ``add_adapter`` / freeze / copy
path the runtime builder runs) and ``sync_previous_policy_adapter``, without
loading the real ~12B FLUX transformer.
"""

from __future__ import annotations

import torch

from tests.models.steps.denoise.fixtures import (
    _TINY_WAN_LORA_TARGETS,
    build_tiny_wan_transformer,
)
from vrl.models.families.flux.model import FluxReplayModel


def _peft_default_only_model() -> FluxReplayModel:
    """A FLUX replay model carrying a real PEFT ``default`` adapter (no previous yet).

    Built via ``get_peft_model`` exactly as the production LoRA path does, so the
    transformer is a PeftModel whose ``add_adapter(name, config)`` signature the
    ``previous`` attach relies on.
    """

    from peft import LoraConfig, get_peft_model

    base = build_tiny_wan_transformer()
    base.requires_grad_(False)
    peft_t = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=_TINY_WAN_LORA_TARGETS,
        ),
    )
    return FluxReplayModel(transformer=peft_t, scheduler=None, device="cpu")


def test_attach_previous_policy_adapter_builds_frozen_mirror() -> None:
    """attach builds a frozen ``previous`` adapter seeded from ``default``."""

    model = _peft_default_only_model()
    assert "previous" not in model.transformer.peft_config

    model.attach_previous_policy_adapter()

    assert "previous" in model.transformer.peft_config
    prev = {n: p for n, p in model.transformer.named_parameters() if ".previous." in n}
    assert prev, "no previous-adapter params were created"
    # forward-only mirror: every previous param is frozen (DDP reducer ignores it).
    assert all(not p.requires_grad for p in prev.values())
    # at least one default param stays trainable (the optimized adapter).
    trainable = [
        n for n, p in model.transformer.named_parameters() if ".default." in n and p.requires_grad
    ]
    assert trainable
    # seeded == default: a default lora_A and its previous twin match after attach.
    a_name = next(n for n in prev if "lora_A" in n)
    d_name = a_name.replace(".previous.", ".default.")
    named = dict(model.transformer.named_parameters())
    assert torch.allclose(named[a_name], named[d_name])


def test_sync_previous_policy_adapter_refreshes_from_default() -> None:
    """sync(decay=0) re-copies the trained ``default`` weights into ``previous``."""

    model = _peft_default_only_model()
    model.attach_previous_policy_adapter()
    named = dict(model.transformer.named_parameters())
    a_name = next(n for n in named if ".previous." in n and "lora_A" in n)
    d_name = a_name.replace(".previous.", ".default.")

    # Simulate an optimizer step moving the trainable default away from previous.
    with torch.no_grad():
        named[d_name].add_(1.0)
    assert not torch.allclose(named[a_name], named[d_name])

    model.sync_previous_policy_adapter(decay=0.0)
    assert torch.allclose(named[a_name], named[d_name])
