"""Shared Ray substrate for domain runtimes.

Engine-neutral Ray primitives: lazy import and actor metadata (dependencies),
actor-group launch/teardown (actor_group), per-fleet call dispatch
(actor_pool), driver-side deadlines (operation_deadline), best-effort cleanup
that reports failures to owners (resource_cleanup), role-level GPU resolution
(resources), and the run-level placement group (placement).

Consumed by the generation engine's Ray transport (``vrl/generation/ray/``)
and the online launcher (``vrl/scripts/common/online.py``); the rollout
collector reads only the resolved lifecycle plan. Nothing here imports
``vrl.generation``, ``vrl.rewards``, or ``vrl.rollouts`` — that leaf position
is what lets one substrate serve every domain runtime without picking an
engine.
"""
