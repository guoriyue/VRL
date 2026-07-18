"""Cosmos Predict2.5 family."""

from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25Model
from vrl.models.families.cosmos.predict2_5.runtime import (
    CosmosPredict25ChunkExecutor,
)

__all__ = [
    "CosmosPredict25ChunkExecutor",
    "CosmosPredict25Model",
]
