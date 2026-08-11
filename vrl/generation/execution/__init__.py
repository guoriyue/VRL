"""Family-neutral batch execution: planning, identity, workers, memory.

Consumers import the concrete modules directly (``execution.sample_batches``,
``execution.planner``, ``execution.types``, ...) — that is the established
import shape across the repo, so this facade re-exports nothing. See
``vrl/generation/protocols.py`` for the map of which boundary each module
serves.
"""
