"""``algorithm.kind`` dispatch shared by the schema, the rules and the factory.

One table names, per kind, the module that owns the algorithm, its config
dataclass (what the YAML section becomes) and the algorithm class (whose
class attributes say what the run must provide). Imports are lazy so config
parsing does not load torch-backed algorithm modules it does not select.
"""

from __future__ import annotations

import importlib
from typing import Any

_KINDS: dict[str, tuple[str, str, str]] = {
    "grpo": ("vrl.algorithms.grpo.continuous", "GRPOConfig", "GRPO"),
    "dance_grpo": ("vrl.algorithms.grpo.continuous", "GRPOConfig", "GRPO"),
    "flash_grpo": ("vrl.algorithms.grpo.continuous", "FlashGRPOConfig", "FlashGRPO"),
    "flow_dppo": ("vrl.algorithms.grpo.continuous", "FlowDPPOConfig", "FlowDPPO"),
    "grpo_guard": ("vrl.algorithms.grpo.continuous", "GRPOGuardConfig", "GRPOGuard"),
    "diffusion_dpo": ("vrl.algorithms.dpo", "DiffusionDPOConfig", "DiffusionDPO"),
    "diffusion_nft": ("vrl.algorithms.diffusion_nft", "DiffusionNFTConfig", "DiffusionNFT"),
    "v_grpo": ("vrl.algorithms.v_grpo", "VGRPOConfig", "VGRPO"),
}


def _entry(kind: str) -> tuple[str, str, str]:
    try:
        return _KINDS[kind]
    except KeyError:
        raise ValueError(f"unsupported algorithm kind: {kind!r}") from None


def algorithm_config_class(kind: str) -> type[Any]:
    """Return the runtime config dataclass selected by ``algorithm.kind``."""

    module, config_name, _ = _entry(kind)
    return getattr(importlib.import_module(module), config_name)


def algorithm_class(kind: str) -> type[Any]:
    """Return the algorithm class selected by ``algorithm.kind``.

    Its class attributes (``needs_sde_rollout``, ``sft_source``, ...) are the
    facts config resolution checks a run against; the online recipe factory
    constructs it from the resolved config.
    """

    module, _, class_name = _entry(kind)
    return getattr(importlib.import_module(module), class_name)


__all__ = ["algorithm_class", "algorithm_config_class"]
