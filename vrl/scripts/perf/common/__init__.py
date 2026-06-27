"""Shared helpers for performance probes.

The top-level files in ``vrl.scripts.perf`` are stable CLI entrypoints. Shared
builders, timing utilities, and numeric kernels live here so probes do not import
from other probe CLIs.
"""
