"""Shared module-path policy for rollout linear quantization."""

from __future__ import annotations

from enum import StrEnum

# Small, numerically sensitive layers stay in the model's master dtype. These
# names describe model structure rather than any one quantization scheme, so FP8
# and FP4 must consume the same table.
DEFAULT_EXCLUDE: tuple[str, ...] = (
    "norm",
    "embed",
    "proj_out",
    "time_",
    "_mod",
    "txt_in",
)

# Language-model trunks additionally protect vocabulary heads because their
# logits are the values scored by the RL objective.
LM_EXCLUDE: tuple[str, ...] = (*DEFAULT_EXCLUDE, "head", "output")

# The conservative FP4 production profile keeps attention in the master dtype
# until full-attention FP4 passes a real rollout -> BF16 replay SDE/reward gate.
# MLP-only is a risk boundary, not evidence that the aligned attention kernels
# cannot execute. This is a deliberately isolated model-name taxonomy.
MLP_PATH_SEGMENTS = frozenset(
    {"ff", "ffn", "feed_forward", "mlp", "img_mlp", "txt_mlp"},
)


class LinearTargetProfile(StrEnum):
    """Named sets of large policy Linears eligible for quantization."""

    MLP_ONLY = "mlp_only"
    ATTENTION_MLP = "attention_mlp"


def is_mlp_linear_path(path: str) -> bool:
    """Whether a dotted module path belongs to a transformer feed-forward block."""

    return any(
        segment in MLP_PATH_SEGMENTS
        or segment.startswith(("ff_", "mlp_"))
        or segment.endswith("_mlp")
        for segment in path.split(".")
    )


def matches_linear_target(path: str, profile: LinearTargetProfile) -> bool:
    """Whether ``path`` belongs to a validated quantization target profile."""

    if profile is LinearTargetProfile.ATTENTION_MLP:
        return True
    return is_mlp_linear_path(path)


__all__ = [
    "DEFAULT_EXCLUDE",
    "LM_EXCLUDE",
    "MLP_PATH_SEGMENTS",
    "LinearTargetProfile",
    "is_mlp_linear_path",
    "matches_linear_target",
]
