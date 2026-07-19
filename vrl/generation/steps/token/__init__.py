"""Token-step protocol for token-autoregressive and future compositions."""

from vrl.generation.steps.token.protocol import (
    TokenLoopInit,
    TokenStepBatch,
    TokenStepOutput,
)

__all__ = ["TokenLoopInit", "TokenStepBatch", "TokenStepOutput"]
