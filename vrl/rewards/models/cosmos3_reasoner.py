"""Cosmos3 reasoner reward model for the in-process reward runtime.

Uses the **reasoner / understanding tower** of NVIDIA Cosmos3 (``cosmos3_omni``)
— a Qwen3-VL VLM — as a generative judge over a generated robot/physical-AI
video. Cosmos3's generator tower is a separate diffusion model (not this file);
here the reasoner reads the video + task instruction and prints four physical-AI
axes (1-5 integers), which we parse:

* ``task_success``           — does the action complete the instructed goal.
* ``contact_realism``        — grasps/contacts/forces are physically believable.
* ``temporal_consistency``   — object identity and motion are stable over frames.
* ``physical_plausibility``  — no clipping, teleporting, or physics violations.

This is the "domain judge" the robotic-data-factory sprint asks for (Kling
``overall_reward`` is video-quality, not task/contact-aware). It mirrors
``videoscore2.py``: a generative Qwen-VL judge, greedily decoded once, parsed
with a fixed-format regex. Public score keys are exactly the four axes plus
``overall`` (their mean) — the free-text wording never leaks as a score key.

CHECKPOINT NOTE — the raw ``nvidia/Cosmos3-Nano`` checkpoint is a **flat-key
unified** omni checkpoint (reasoner LLM/visual keys + ``*_moe_gen`` generator-
tower keys in one flat namespace). ``Qwen3VLForConditionalGeneration`` expects
the **nested** Qwen3-VL key layout, so the unified checkpoint does NOT load
HF-direct. ``worker_config.checkpoint_layout`` selects how the reasoner weights
are obtained:

* ``remapped`` — ``model_path`` points at a pre-remapped, reasoner-only Qwen3-VL
  checkpoint (produced offline by the vLLM ``cosmos3.py`` WeightsMapper: flat->
  nested, drop the generator/audio/action towers). Loaded cleanly here.
* unset / anything else — ``__init__`` raises with the remap prerequisite rather
  than attempting a doomed raw ``from_pretrained`` that mismatches every key.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.rewards.assets.video_judge_prompts import (
    COSMOS3_SCORE_REGEX,
    COSMOS3_SYSTEM_PROMPT,
    COSMOS3_USER_TEMPLATE,
)
from vrl.rewards.models.qwen_vl_judge import QwenVLVideoJudge


class Cosmos3ReasonerRewardModel(QwenVLVideoJudge):
    """Load the Cosmos3 reasoner (Qwen3-VL) and judge one (prompt, video) pair."""

    family = "Cosmos3 reasoner judge"

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        # Generator tower is a separate diffusion model; only the reasoner judges
        # here, and the unified checkpoint will not load HF-direct (see module
        # docstring). Require an explicit, supported layout.
        self.checkpoint_layout = str(worker_config.get("checkpoint_layout", "")).strip().lower()
        if self.checkpoint_layout != "remapped":
            raise ValueError(
                "Cosmos3 reasoner judge requires worker_config.checkpoint_layout='remapped' "
                "with model_path pointing at a pre-remapped, reasoner-only Qwen3-VL checkpoint. "
                "The raw nvidia/Cosmos3-Nano unified checkpoint is flat-key (reasoner + "
                "*_moe_gen generator tower in one namespace) and does NOT load under "
                "Qwen3VLForConditionalGeneration. Produce the remapped checkpoint offline via "
                "the vLLM cosmos3.py WeightsMapper (flat->nested, drop generator/audio/action "
                f"towers). Got checkpoint_layout={self.checkpoint_layout!r}.",
            )
        # 'remapped' means an offline-produced LOCAL dir; reward_model_name's
        # snapshot_download fallback would silently pull the raw flat-key unified
        # repo (which does NOT load), failing late after a multi-GB download. Require
        # model_path so the unfilled-config case fails fast with an actionable error.
        if not str(worker_config.get("model_path", "")).strip():
            raise ValueError(
                "Cosmos3 reasoner checkpoint_layout='remapped' requires a non-empty "
                "worker_config.model_path pointing at the offline pre-remapped, reasoner-only "
                "Qwen3-VL checkpoint dir. Refusing to snapshot_download reward_model_name "
                "(the raw unified nvidia/Cosmos3-Nano checkpoint is flat-key and will not load).",
            )

        # The ``remapped`` layout's ``model_path`` (validated non-empty above) is
        # the offline-produced reasoner-only Qwen3-VL dir. Resolve it directly with
        # an existence check; there is no snapshot_download fallback (the raw
        # unified repo does not load — see the guard above).
        model_root = Path(str(worker_config.get("model_path", "")).strip()).expanduser().resolve()
        if not model_root.exists():
            raise FileNotFoundError(f"Cosmos3 reasoner model_path missing: {model_root}")
        super().__init__(worker_config, model_root=model_root)

    def _load_model(self, load_kwargs: dict[str, Any]) -> Any:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration.from_pretrained(str(self.model_root), **load_kwargs)

    def _messages(self, video_path: str, prompt: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": COSMOS3_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    self._video_content(video_path),
                    {
                        "type": "text",
                        "text": COSMOS3_USER_TEMPLATE.format(prompt=prompt),
                    },
                ],
            },
        ]

    def _parse(self, decoded: str, generated: Any, generated_ids: list[int]) -> dict[str, float]:
        del generated, generated_ids
        parsed = _parse_integer_scores(decoded)
        if parsed is None:
            raise ValueError(
                "Cosmos3 reasoner produced no parseable score line; "
                f"output head was: {decoded[:200]!r}",
            )
        return _normalize_scores(*parsed)


def _parse_integer_scores(text: str) -> tuple[int, int, int, int] | None:
    """Extract the four 1-5 integer axes from the judge's generated text."""

    match = COSMOS3_SCORE_REGEX.search(text)
    if match is None:
        return None
    scores = tuple(int(match.group(i)) for i in (1, 2, 3, 4))
    if any(not (1 <= value <= 5) for value in scores):
        return None
    return scores  # type: ignore[return-value]


def _normalize_scores(
    task_success: float,
    contact_realism: float,
    temporal_consistency: float,
    physical_plausibility: float,
) -> dict[str, float]:
    """Map the four axes to the public score keys plus their mean ``overall``.

    This dict is the public scoring contract: only the documented keys, so a
    config cannot select an undocumented upstream key as ``score_key``.
    """

    task_success = float(task_success)
    contact_realism = float(contact_realism)
    temporal_consistency = float(temporal_consistency)
    physical_plausibility = float(physical_plausibility)
    return {
        "task_success": task_success,
        "contact_realism": contact_realism,
        "temporal_consistency": temporal_consistency,
        "physical_plausibility": physical_plausibility,
        "overall": (task_success + contact_realism + temporal_consistency + physical_plausibility)
        / 4.0,
    }


__all__ = ["Cosmos3ReasonerRewardModel"]
