"""Typed generation-memory policy carried across model build boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class VaeDecodeMemory:
    """Resolved VAE decode memory behavior."""

    tiling: bool = False
    slicing: bool = False

    def __post_init__(self) -> None:
        for policy_field in fields(self):
            value = getattr(self, policy_field.name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"VaeDecodeMemory.{policy_field.name} must be a bool",
                )


@dataclass(frozen=True, slots=True)
class GenerationMemoryPolicy:
    """Resolved target-keyed policy carried by a rollout ``ModelBuild``."""

    # A primitive mapping is accepted only when reconstructing the nested
    # dataclass from a Ray ``asdict(ModelBuild)`` payload.
    vae_decode: VaeDecodeMemory | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.vae_decode, Mapping):
            object.__setattr__(
                self,
                "vae_decode",
                VaeDecodeMemory(**dict(self.vae_decode)),
            )
        elif self.vae_decode is not None and not isinstance(
            self.vae_decode,
            VaeDecodeMemory,
        ):
            raise TypeError(
                "GenerationMemoryPolicy.vae_decode must be VaeDecodeMemory, a mapping, or None",
            )


__all__ = [
    "GenerationMemoryPolicy",
    "VaeDecodeMemory",
]
