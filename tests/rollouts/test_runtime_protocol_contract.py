"""GenerationRuntime and weight-sync version contract tests.

Orchestration reads the version properties declared by each concrete boundary
instead of probing nested runtime internals.
"""

from __future__ import annotations

from vrl.generation.protocols import GenerationRuntime
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession


def _runtime(
    *,
    deferred: bool = False,
) -> RayGenerationRuntime:
    session = RayGenerationSession(
        executor=object(),
        weight_sync=None,
        owned_engines=[],
    )

    if deferred:

        async def create_session() -> RayGenerationSession:
            return session

        return RayGenerationRuntime(
            session=None,
            session_factory=create_session,
        )
    return RayGenerationRuntime(session=session)


# Note: the release-before-reward decision is no longer a runtime method; it is
# derived from GPU topology into the RayLifecyclePlan and read by the collector.
# See tests/ray/test_resources.py (plan derivation) and
# tests/rollouts/collector/test_runtime.py (collector consumption).


def test_concrete_runtimes_satisfy_generation_runtime_structurally() -> None:
    persistent = _runtime()
    deferred = _runtime(deferred=True)
    assert isinstance(persistent, GenerationRuntime)
    assert isinstance(deferred, GenerationRuntime)
