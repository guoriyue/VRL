"""One real Ray cluster for the whole ``tests/generation/ray`` package.

The package-scoped shell below overrides the function-scoped ``local_ray`` in
``tests/conftest.py``, so every ``slow_test`` here pays one cluster start/stop
between them instead of one each (the directory's slow lane went from ~42s to
~15s over 3 runs each, while growing from 8 tests to 12).

The cluster carries the launcher's ``worker_process_setup_hook`` because it has
to: file order puts ``test_rollout_launcher`` between two other ``local_ray``
consumers, so a launcher-owned cluster would tear the shared one down mid-package
and leave the later tests holding a dead handle. Hanging the hook on the shared
cluster is also the simpler contract — it is test scaffolding (it publishes a
tiny executor on an importable production module), not behaviour under test, and
it is inert for everything else here. The hook body runs only in worker
processes, and no other actor in this package reads ``FAMILY_REGISTRY``,
``build_rollout`` or the checkpoint identity resolver. (``test_ray_resident_session``
rewrites those same three symbols, which looks like a conflict and is not: it
starts no actors at all and does its rewriting driver-side through
``monkeypatch``, so the two never meet.) The alternative — teaching the fixture
to re-init after somebody else shuts its cluster down — buys nothing and adds an
implicit ordering dependency.

Read ``real_local_ray``'s docstring before adding a test here: a shared cluster
has no per-test actor namespace, so every test must kill the actors it creates.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import real_local_ray

_REPO_ROOT = str(Path(__file__).resolve().parents[3])


def _worker_setup_hook(repo_root: str) -> Any:
    """Return a by-value hook that installs a tiny canonical family in workers."""

    def install() -> None:
        import contextlib
        import sys
        from dataclasses import replace

        # Ray workers do not inherit pytest's cwd-based import path. Put this
        # checkout first so the actor cannot accidentally load another editable
        # VRL checkout from the developer environment.
        sys.path.insert(0, repo_root)

        import vrl.families.registry as registry
        import vrl.models.checkpoint_identity as checkpoint_identity
        from vrl.models.interfaces import RuntimeBundle

        class TinyRuntimeModel:
            device = "cpu"

            def to(self, device: Any) -> TinyRuntimeModel:
                self.device = str(device)
                return self

            def replay_forward(self, batch: Any, timestep_idx: int = 0, **kwargs: Any) -> Any:
                raise NotImplementedError("Ray launcher test never calls replay_forward")

            def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
                return contextlib.nullcontext()

            def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
                self.loaded_state = dict(state_dict)

        class TinyChunkExecutor:
            family = "janus_pro"
            task = "ar_t2i"

            def __init__(self, model: TinyRuntimeModel) -> None:
                self.model = model

            def forward_chunk_plan(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    "Ray launcher test only verifies worker construction",
                )

            def gather_chunks(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    "Ray launcher test only verifies worker construction",
                )

        def build_tiny_rollout(_entry: Any, build: Any) -> RuntimeBundle:
            assert str(build.device) == "cpu"
            return RuntimeBundle(
                model=TinyRuntimeModel(),
                trainable_modules={},
                scheduler=None,
                raw_handle=None,
                precision=build.precision,
                loads_full_generation_modules=True,
            )

        # The hook executes before worker imports the launch contract. Publishing
        # the test executor on an importable production module lets canonical
        # registry dispatch stay intact without any compatibility fields.
        registry._RayLauncherTestExecutor = TinyChunkExecutor
        entry = registry.FAMILY_REGISTRY["janus_pro"]
        registry.FAMILY_REGISTRY["janus_pro"] = replace(
            entry,
            executor_cls="vrl.families.registry:_RayLauncherTestExecutor",
        )
        registry.ModelFamilyEntry.build_rollout = build_tiny_rollout
        checkpoint_identity.resolve_checkpoint_model_identity = lambda _build: {"schema": "test"}

    return install


@pytest.fixture(scope="package")
def local_ray() -> Iterator[Any]:
    with real_local_ray(
        runtime_env={"worker_process_setup_hook": _worker_setup_hook(_REPO_ROOT)},
    ) as ray:
        yield ray
