"""Shared online training recipe helpers."""

# Public recipe entrypoint only. Internal factory helpers live in
# vrl.scripts.common.factory and are imported directly from there by online.py.
from vrl.scripts.common.online import run_online_recipe

__all__ = [
    "run_online_recipe",
]
