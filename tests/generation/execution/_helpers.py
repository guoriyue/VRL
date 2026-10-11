"""The minimal launch config the worker-core tests share.

The contract is pure data and identical across these files; what differs per
test (which executor the core gets, versioned slots, sleep offload) stays at the
call site where the test's intent is visible.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.launch_config import GenerationLaunchConfig


def launch_config(family: str = "sd3_5", **overrides: Any) -> GenerationLaunchConfig:
    kwargs: dict[str, Any] = {
        "family": family,
        "model_build": {},
        "expected_model_identity": {"schema": "test"},
    }
    kwargs.update(overrides)
    return GenerationLaunchConfig(**kwargs)
