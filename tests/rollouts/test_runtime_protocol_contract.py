"""``RayGenerationRuntime`` satisfies the ``GenerationRuntime`` protocol structurally.

The collector accepts any runtime that satisfies the protocol; this pins the
one production implementation to it so a renamed or dropped member fails here
rather than at the first real launch.
"""

from __future__ import annotations

from vrl.generation.protocols import GenerationRuntime
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession


def test_ray_runtime_satisfies_generation_runtime_structurally() -> None:
    async def never_launched() -> RayGenerationSession:
        raise AssertionError("the structural check never launches a session")

    runtime = RayGenerationRuntime(session=None, session_factory=never_launched)

    assert isinstance(runtime, GenerationRuntime)
