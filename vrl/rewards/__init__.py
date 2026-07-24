"""Reward functions for RL training."""

from __future__ import annotations

# Deliberately exports nothing. ``vrl.rewards.functions.registry`` is the real
# entry point and imports each reward by submodule, so a package-root facade is
# a second surface nothing uses -- and an eager one would import every reward
# implementation just to read the registry.
__all__: list[str] = []
