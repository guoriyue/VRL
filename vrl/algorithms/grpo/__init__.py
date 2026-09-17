"""GRPO algorithm family.

Deliberately exports nothing. ``vrl.config.algorithm`` imports each config by
submodule to answer ``algorithm.kind``, so an eager re-export here made every
config parse load every torch-backed objective implementation.
"""

from __future__ import annotations

__all__: list[str] = []
