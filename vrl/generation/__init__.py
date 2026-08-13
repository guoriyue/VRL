"""Generation runtime public API.

This package owns generation execution contracts and runtime helpers. RL
rollout collection, rewards, and trainer-ready batches live outside this layer.
"""

from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import (
    BatchPayload,
    GenerationBatchExecutor,
    GenerationBatchGatherer,
    GenerationRuntime,
)
from vrl.generation.types import (
    GenerationInput,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)

__all__ = [
    "BatchPayload",
    "GenerationBatchExecutor",
    "GenerationBatchGatherer",
    "GenerationInput",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationRuntime",
    "GenerationRuntimeLaunchContract",
    "GenerationSampleRow",
]
