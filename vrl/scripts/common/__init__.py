"""Shared online training recipe helpers."""

# Public recipe entrypoint only. Internal factory helpers live in
# vrl.scripts.common.factory and are imported directly from there by online.py.
from vrl.scripts.common.online import run_online_recipe
from vrl.scripts.common.types import OnlineRecipeDefinition

__all__ = [
    "OnlineRecipeDefinition",
    "run_online_recipe",
]
