"""Algorithm-kind to typed-config dispatch shared by schema and builders."""

from __future__ import annotations

from typing import Any


def algorithm_config_class(kind: str) -> type[Any]:
    """Return the runtime config dataclass selected by ``algorithm.kind``."""

    if kind in ("grpo", "dance_grpo"):
        from vrl.algorithms.grpo.continuous import GRPOConfig

        return GRPOConfig
    if kind == "flow_dppo":
        from vrl.algorithms.grpo.continuous import FlowDPPOConfig

        return FlowDPPOConfig
    if kind == "grpo_guard":
        from vrl.algorithms.grpo.continuous import GRPOGuardConfig

        return GRPOGuardConfig
    if kind == "token_grpo":
        from vrl.algorithms.grpo.token import TokenGRPOConfig

        return TokenGRPOConfig
    if kind == "token_grpo_multisegment":
        from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig

        return MultiSegmentTokenGRPOConfig
    if kind == "diffusion_dpo":
        from vrl.algorithms.dpo import DiffusionDPOConfig

        return DiffusionDPOConfig
    if kind == "diffusion_nft":
        from vrl.algorithms.diffusion_nft import DiffusionNFTConfig

        return DiffusionNFTConfig
    raise ValueError(f"unsupported algorithm kind: {kind!r}")


__all__ = ["algorithm_config_class"]
