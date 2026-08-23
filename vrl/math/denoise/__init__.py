"""Diffusion math shared by engines, evaluators, and algorithms.

Deliberately exports nothing: no production or test code imported this
package root (all callers import the owning submodule directly), so the
re-export facade bought no ergonomics while taxing import time and rotting
independently of the submodules. Import from the owning module instead.
"""

from __future__ import annotations

__all__: list[str] = []
