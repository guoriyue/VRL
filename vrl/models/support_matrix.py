"""Physical-AI model-support matrix (P0 deliverable).

A single typed table recording, per candidate model, the facts that decide
whether it can enter this repo and through which seam: input/output modalities,
action payload shape, training-logprob availability, environment requirement,
and the priority tier from the sprint. The probe scripts read their target row
from here, and ``tests/models/test_support_matrix.py`` pins the structural
invariants, so this table is a consumed asset rather than a rotting doc.

Source of truth for the priorities and seam decisions:
``docs/sprints/planned/SPRINT_physical_ai_model_support.md``.

This table is reference/decision data only; it is intentionally NOT wired into
the rollout family registry. A model gets a registry entry + runtime builder in
a later MR after its probe confirms a trainable contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# A model's seam: which existing (or new) rollout path it would use.
Seam = Literal[
    "video_diffusion",  # Wan / Cosmos2.5 / Echo — existing diffusion seam
    "vla_diffusion",    # diffusion/flow action policy (PI0.5) — new env seam
    "vla_token",        # token-action policy (OpenVLA-OFT) — new env seam
    "omnimodal",        # unified MoT reasoner+generator (Cosmos 3)
]

# Whether a training-grade, replayable action/denoise log-prob is available.
LogProb = Literal[
    "replayable",  # exact old-vs-new log-prob can be recomputed -> RL-eligible
    "eval_only",   # output is server JSON / non-distributional -> eval/SFT only
    "unknown",     # not yet probed -> must be resolved before any RL wiring
]

# Sprint priority tier (§4).
Tier = Literal["tier0", "tier1", "tier2", "tier3", "tier4"]

# Current engagement level in this repo.
Status = Literal[
    "supported",  # trainable asset live in-repo today
    "probe",      # P0/P1 inference/contract probe only
    "track",      # watch only; not a current target
]


@dataclass(frozen=True, slots=True)
class ModelSupportEntry:
    """One row of the physical-AI model-support matrix."""

    key: str
    display_name: str
    params_b: float | None
    seam: Seam
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    action_payload: str | None  # None when the model emits no action
    logprob: LogProb
    env: str | None             # required closed-loop env, if any
    tier: Tier
    status: Status
    path: str | None            # HF id / local path / server ref
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("ModelSupportEntry.key must be non-empty")
        if not self.inputs or not self.outputs:
            raise ValueError(f"{self.key}: inputs and outputs must be non-empty")
        # An action-emitting model must name its closed-loop env; a generator
        # without an action payload must not claim one.
        if self.action_payload is None and self.env is not None:
            raise ValueError(f"{self.key}: env set but no action payload")


MODEL_SUPPORT_MATRIX: tuple[ModelSupportEntry, ...] = (
    # ---- Tier 0: existing trainable video/world diffusion (守住主线) ----
    ModelSupportEntry(
        key="cosmos_predict2_5",
        display_name="Cosmos Predict2.5 2B",
        params_b=2.0,
        seam="video_diffusion",
        inputs=("text", "image"),
        outputs=("video",),
        action_payload=None,
        logprob="replayable",
        env=None,
        tier="tier0",
        status="supported",
        path="nvidia/Cosmos-Predict2.5-2B",
        notes="Live DiffusionNFT recipe + Kling/physics video reward.",
    ),
    ModelSupportEntry(
        key="wan_2_1",
        display_name="Wan 2.1 T2V",
        params_b=1.3,
        seam="video_diffusion",
        inputs=("text",),
        outputs=("video",),
        action_payload=None,
        logprob="replayable",
        env=None,
        tier="tier0",
        status="supported",
        path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        notes="GRPO/flow-matching logprob via existing diffusion seam.",
    ),
    ModelSupportEntry(
        key="echo",
        display_name="JoyAI-Echo (LTX-derived)",
        params_b=None,
        seam="video_diffusion",
        inputs=("text", "image"),
        outputs=("video",),
        action_payload=None,
        logprob="replayable",
        env=None,
        tier="tier0",
        status="supported",
        path="third_party/JoyAI-Echo",
        notes="Video flow-matching family; existing diffusion seam.",
    ),
    # ---- Tier 1: Cosmos 3 Nano probes (判定能否进仓库, 不训练) ----
    ModelSupportEntry(
        key="cosmos3_nano",
        display_name="Cosmos3-Nano (omnimodal)",
        params_b=16.0,
        seam="omnimodal",
        inputs=("text", "image", "video", "audio", "action"),
        outputs=("text", "image", "video", "audio", "action"),
        action_payload=None,  # generator probe target; action path probed separately
        logprob="unknown",
        env=None,
        tier="tier1",
        status="probe",
        path="nvidia/Cosmos3-Nano",
        notes="P0/P1 generator probe: does the Diffusers path expose a "
        "trainable denoiser with a replayable logprob? See "
        "vrl/scripts/eval/cosmos3_nano_generator_probe.py.",
    ),
    ModelSupportEntry(
        key="cosmos3_nano_policy_droid",
        display_name="Cosmos3-Nano-Policy-DROID",
        params_b=16.0,
        seam="omnimodal",
        inputs=("text", "image", "state"),
        outputs=("action",),
        action_payload="action_chunk",
        logprob="unknown",
        env="droid",
        tier="tier1",
        status="probe",
        path="nvidia/Cosmos3-Nano-Policy-DROID",
        notes="Robotics priority: eval/policy-server adapter first. Probe "
        "confirms action payload + embodiment gap vs LIBERO. See "
        "vrl/scripts/eval/cosmos3_nano_policy_droid_probe.py.",
    ),
    # ---- Tier 2: first RL-eligible robotics policies (eval first) ----
    ModelSupportEntry(
        key="pi05",
        display_name="PI0.5 (diffusion/flow action policy)",
        params_b=None,
        seam="vla_diffusion",
        inputs=("image", "state", "instruction"),
        outputs=("action",),
        action_payload="action_chunk",
        logprob="unknown",
        env="libero",
        tier="tier2",
        status="probe",
        path="cosmos-rl:configs/pi05/pi05-b1k-grpo-colocate.toml",
        notes="Closest fit to diffusion logprob reuse. P3 probe: which of "
        "flow_sde/flow_cps/flow_noise yields a training logprob.",
    ),
    ModelSupportEntry(
        key="openvla_oft",
        display_name="OpenVLA-OFT (token-action policy)",
        params_b=7.0,
        seam="vla_token",
        inputs=("image", "instruction"),
        outputs=("action",),
        action_payload="action_chunk",
        logprob="eval_only",
        env="libero",
        tier="tier2",
        status="probe",
        path="moojink/openvla-7b-oft-finetuned-libero-10",
        notes="Eval-validated on LIBERO-10 via the P1 Env/ActionPolicy seam "
        "(6/6 small-sample, vrl/scripts/eval/openvla_oft_libero_eval.py). FINDING: "
        "the released OFT head is deterministic L1-regression, NOT token sampling "
        "-> no replayable logprob -> eval/SFT only, not on-policy RL. "
        "See docs/sprints/info/SPRINT_physical_ai_tier2_openvla_libero.md.",
    ),
    # ---- Tier 4: 64B Super series — track only, no full training ----
    ModelSupportEntry(
        key="cosmos3_super",
        display_name="Cosmos3-Super (omnimodal)",
        params_b=64.0,
        seam="omnimodal",
        inputs=("text", "image", "video", "audio", "action"),
        outputs=("text", "image", "video", "audio", "action"),
        action_payload=None,
        logprob="unknown",
        env=None,
        tier="tier4",
        status="track",
        path="nvidia/Cosmos3-Super",
        notes="Eval/reward reference only; do not bind system design to 64B.",
    ),
    ModelSupportEntry(
        key="cosmos3_super_t2i",
        display_name="Cosmos3-Super-Text2Image",
        params_b=64.0,
        seam="omnimodal",
        inputs=("text",),
        outputs=("image",),
        action_payload=None,
        logprob="unknown",
        env=None,
        tier="tier4",
        status="track",
        path="nvidia/Cosmos3-Super-Text2Image",
        notes="Track only; not a main line.",
    ),
    ModelSupportEntry(
        key="cosmos3_super_i2v",
        display_name="Cosmos3-Super-Image2Video",
        params_b=64.0,
        seam="omnimodal",
        inputs=("image", "text"),
        outputs=("video",),
        action_payload=None,
        logprob="unknown",
        env=None,
        tier="tier4",
        status="track",
        path="nvidia/Cosmos3-Super-Image2Video",
        notes="Evaluator/probe only when resources allow.",
    ),
)

_BY_KEY: dict[str, ModelSupportEntry] = {entry.key: entry for entry in MODEL_SUPPORT_MATRIX}


def get_support_entry(key: str) -> ModelSupportEntry:
    """Return the matrix row for ``key`` or fail with the available keys."""
    try:
        return _BY_KEY[key]
    except KeyError as err:
        raise KeyError(
            f"unknown model-support key {key!r}; got {sorted(_BY_KEY)}",
        ) from err


__all__ = [
    "MODEL_SUPPORT_MATRIX",
    "LogProb",
    "ModelSupportEntry",
    "Seam",
    "Status",
    "Tier",
    "get_support_entry",
]
