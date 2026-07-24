"""Standalone reward service package."""

from __future__ import annotations

# Deliberately exports nothing. Beyond having no importers, an empty package
# is what keeps ``python -m vrl.rewards.service.server`` from reaching server.py
# through the package first, which made runpy warn and run endpoint setup twice.
__all__: list[str] = []
