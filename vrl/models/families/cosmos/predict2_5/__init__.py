"""Cosmos Predict2.5 family."""

from __future__ import annotations

# Deliberately exports nothing. The family registry dispatches by dotted
# submodule path (vrl/models/families/registry.py), so a package-root re-export is a
# second surface nothing imports; keeping this module empty is also what stops
# config discovery from pulling the torch-backed model runtime.
__all__: list[str] = []
