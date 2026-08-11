"""Rollout orchestration: drive the generation and reward engines into trainer batches.

Each subpackage owns one seam of the collection flow:

* ``batch``: the engine-neutral ``RolloutBatch`` both denoise and token
  trajectories pack into, plus operations over it.
* ``collector``: request -> ``GenerationRuntime`` -> ``RewardRuntime`` ->
  ``RolloutBatch`` for one prompt group, including shared-GPU phase handoffs.
* ``orchestration``: schedules (strict on-policy, continuous) deciding when
  collection runs relative to training and weight sync.
* ``evaluators``: train-time replay signal extraction from collected batches.
* ``stats``: the typed per-request stats accumulator that travels with items.

Consumers import the concrete modules directly; this facade re-exports
nothing.
"""
