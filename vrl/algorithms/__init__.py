"""RL algorithms: advantage estimation and policy gradient losses.

Deliberately exports nothing. Re-exporting the algorithm ABC and adapters here
made every ``vrl.algorithms.*`` submodule import load the torch-backed objective
implementations — including the two pure dataclasses config parsing needs
(``logprob_mismatch.PrecisionCorrectionConfig``, ``types.TrainStepMetrics``).
Nothing imported the package root, so the cost bought no ergonomics. Import from
the owning module instead.
"""

from __future__ import annotations

__all__: list[str] = []
