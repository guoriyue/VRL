"""VideoScore2 reward model (Ray-actor backend).

Wraps ``TIGER-Lab/VideoScore2`` — a Qwen2.5-VL-7B judge that VideoScore2 SFTs on
VideoFeedback2 and then aligns with GRPO — to score a generated video on the
three axes the model is trained to emit:

* ``visual_quality``         — clarity, smoothness, artifacts.
* ``text_alignment``         — fidelity of the video to its text prompt.
* ``physical_common_sense``  — naturalness and physics plausibility.

VideoScore2 is a *generative* judge: it reasons (chain-of-thought) and then
prints the three scores as text in a fixed format (each a 1-5 integer). We load
it via ``AutoModelForVision2Seq`` + ``AutoProcessor`` with ``trust_remote_code``
(the checkpoint ships its own model code), greedily generate one judgement, and
parse the three scores with the upstream regex.

Two scoring modes (``worker_config.soft_scores``):

* hard (``soft_scores: false``) — the parsed 1-5 integers, exactly matching
  upstream ``vs2_inference.py`` text parsing. Robust, but only five levels: a
  GRPO group whose rollouts all land on the same integer gets zero advantage.
* soft (``soft_scores: true``, default) — the expected value ``Σ p(d)·d`` over
  digit tokens ``1..5`` at each emitted score position (continuous in ``[1,5]``),
  giving GRPO a smooth signal. This depends on aligning the score digit to its
  generation step; if that alignment fails for any axis we fall back to the hard
  integer for that axis, so the reward never crashes on an odd generation.

The public score keys are exactly ``visual_quality`` / ``text_alignment`` /
``physical_common_sense`` / ``overall`` (see :func:`_normalize_scores`); the
upstream free-text wording never leaks into the training config as a score key.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from vrl.models.dtypes import resolve_torch_dtype
from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceRequest
from vrl.rewards.models.base import RewardModel
from vrl.rewards.models.hub import parse_hf_repo_revision
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

_DEFAULT_REWARD_MODEL = "TIGER-Lab/VideoScore2"
_DEFAULT_FPS = 2.0
_DEFAULT_MAX_NEW_TOKENS = 1024
_SCORE_SCALE = (1, 2, 3, 4, 5)

# The system prompt fixes the judge's task and, crucially, the output grammar we
# parse below. Kept here (not in a YAML rubric asset) because it is a short,
# model-specific decode contract — not a tunable business rubric. A free-text
# rubric layer, if VideoScore2 ever grows one, belongs in worker_config.rubric_path.
_SYSTEM_PROMPT = (
    "You are an expert for evaluating AI-generated videos from three dimensions: "
    "(1) visual quality - clarity, smoothness, artifacts; "
    "(2) text-to-video alignment - fidelity to the prompt; "
    "(3) physical/common-sense consistency - naturalness and physics plausibility. "
    "Reason briefly, then output the final line exactly as: "
    "visual quality: <v_score>; text-to-video alignment: <t_score>, "
    "physical/common-sense consistency: <p_score>, "
    "where each score is an integer from 1 to 5."
)
_USER_TEMPLATE = (
    "Video prompt: {prompt}\n"
    "Please output in this format:\n"
    "visual quality: <v_score>; text-to-video alignment: <t_score>, "
    "physical/common-sense consistency: <p_score>"
)

# Upstream vs2_inference.py parsing. Anchors are also reused (as marker phrases)
# to locate each score's generation step for the soft expected-value path.
_SCORE_REGEX = re.compile(
    r"visual quality:\s*(\d+).*?"
    r"text-to-video alignment:\s*(\d+).*?"
    r"physical/common-sense consistency:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_DIMENSION_MARKERS: tuple[tuple[str, str], ...] = (
    ("visual_quality", "quality"),
    ("text_alignment", "alignment"),
    ("physical_common_sense", "consistency"),
)
# Max tokens between a marker's end and its score digit ("<marker>: <digit>").
_DIGIT_PROXIMITY_WINDOW = 8


class VideoScore2Model(RewardModel):
    """Load VideoScore2 and score one (prompt, video) pair per call."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        self.reward_model_name = str(
            self.worker_config.get("reward_model_name", ""),
        ).strip()
        self.model_root = _resolve_model_root(self.worker_config)
        self.dtype = resolve_torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.fps = float(self.worker_config.get("fps", _DEFAULT_FPS))
        self.max_new_tokens = int(
            self.worker_config.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS),
        )
        # Continuous expected-value scoring (default) vs upstream integer parsing.
        self.soft_scores = bool(self.worker_config.get("soft_scores", True))
        # Optional per-frame pixel bound for qwen smart_resize, matching the
        # Kling reward's knob; None keeps the processor default.
        max_frame_pixels = self.worker_config.get("max_frame_pixels")
        self.max_frame_pixels = None if max_frame_pixels is None else int(max_frame_pixels)
        self.local_files_only = bool(self.worker_config.get("local_files_only", False))

        logger.info(
            "loading VideoScore2 %s",
            kv(
                root=self.model_root,
                device=self.device,
                dtype=self.dtype,
                fps=self.fps,
                soft_scores=self.soft_scores,
            ),
        )

        from transformers import AutoModelForVision2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(self.model_root),
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            str(self.model_root),
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        model.eval()
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None) or processor
        self.model = model.to(self.device)
        # Pre-resolve digit token ids once; the soft path reads their logits.
        self.digit_token_ids = _resolve_digit_token_ids(self.tokenizer)

    def __call__(
        self,
        *,
        artifact: RewardInferenceArtifact,
        request: RewardInferenceRequest,
    ) -> dict[str, float]:
        del request
        prompt = artifact.prompt or str(artifact.metadata.get("prompt", ""))
        if not prompt:
            raise ValueError(
                f"VideoScore2 requires a prompt for artifact {artifact.artifact_id!r}; "
                "found none on artifact.prompt or artifact.metadata['prompt']",
            )
        video_path = str(Path(artifact.as_path()).expanduser().resolve())
        return self._score_video(video_path, prompt)

    def _score_video(self, video_path: str, prompt: str) -> dict[str, float]:
        from qwen_vl_utils import process_vision_info

        video_content: dict[str, Any] = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": self.fps,
        }
        if self.max_frame_pixels is not None:
            video_content["max_pixels"] = self.max_frame_pixels
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": _USER_TEMPLATE.format(prompt=prompt)},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=self.soft_scores,
                return_dict_in_generate=True,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = generated.sequences[0][prompt_len:].tolist()
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        hard = _parse_integer_scores(decoded)
        if hard is None:
            raise ValueError(
                "VideoScore2 produced no parseable score line; "
                f"output head was: {decoded[:200]!r}",
            )
        if not self.soft_scores:
            return _normalize_scores(*hard)

        step_logits = [step[0] for step in generated.scores]
        soft = _soft_scores_from_generation(
            generated_ids,
            step_logits,
            self.digit_token_ids,
            tokenizer=self.tokenizer,
        )
        return _normalize_scores(*_merge_soft_with_hard(soft, hard))


def preflight_videoscore2_backend() -> None:
    """Validate backend imports without downloading weights."""

    import qwen_vl_utils  # noqa: F401
    import transformers  # noqa: F401


def _parse_integer_scores(text: str) -> tuple[int, int, int] | None:
    """Extract the three 1-5 integer scores from generated text (upstream regex)."""

    match = _SCORE_REGEX.search(text)
    if match is None:
        return None
    scores = tuple(int(match.group(i)) for i in (1, 2, 3))
    if any(not (1 <= value <= 5) for value in scores):
        return None
    return scores  # type: ignore[return-value]


# Soft may drift from its greedy digit at most this far before it is treated
# as a misanchored slot rather than genuine probability spread.
_SOFT_HARD_TOLERANCE = 1.0


def _merge_soft_with_hard(
    soft: Mapping[str, float | None],
    hard: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Soft may only refine its hard integer, never contradict it.

    The hard regex parse is the upstream-faithful ground truth for which digit
    the judge actually emitted; the soft slot search is our extension and can
    misanchor (measured live: a list numeral "1" read as the score where the
    emitted digit was 3). A correctly anchored expectation sits near the greedy
    digit it averages around, so a soft value farther than the tolerance is a
    wrong slot — drop it with a warning instead of shipping a confident wrong
    score into the advantage.
    """

    merged: list[float] = []
    for i, (name, _marker) in enumerate(_DIMENSION_MARKERS):
        value = soft.get(name)
        if value is not None and abs(value - hard[i]) > _SOFT_HARD_TOLERANCE:
            logger.warning(
                "VideoScore2 soft score rejected as misanchored %s",
                kv(axis=name, soft=value, hard=hard[i]),
            )
            value = None
        merged.append(float(hard[i]) if value is None else float(value))
    return (merged[0], merged[1], merged[2])


def _resolve_digit_token_ids(tokenizer: Any) -> dict[int, int]:
    """Map each 1-5 score digit to its single token id (required by the soft path)."""

    ids: dict[int, int] = {}
    for digit in _SCORE_SCALE:
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(encoded) == 1:
            ids[digit] = int(encoded[0])
    return ids


def _soft_scores_from_generation(
    generated_ids: Sequence[int],
    step_logits: Sequence[Any],
    digit_token_ids: Mapping[int, int],
    *,
    tokenizer: Any,
) -> dict[str, float | None]:
    """Continuous per-axis scores: expected value over digits 1-5 at each slot.

    For each dimension we anchor on its marker word ("quality"/"alignment"/
    "consistency"), advance to the next generated digit token, and take the
    softmax-weighted mean of ``1..5`` from that step's logits. Any axis we cannot
    locate maps to ``None`` so the caller can fall back to the hard integer.
    """

    if not digit_token_ids:
        return {name: None for name, _ in _DIMENSION_MARKERS}
    digit_id_set = set(digit_token_ids.values())
    out: dict[str, float | None] = {}
    cursor = 0
    for name, marker_word in _DIMENSION_MARKERS:
        marker_ids = _marker_token_ids(tokenizer, marker_word)
        slot = _next_digit_step(generated_ids, marker_ids, digit_id_set, start=cursor)
        if slot is None or slot >= len(step_logits):
            out[name] = None
            continue
        out[name] = _expected_digit_value(step_logits[slot], digit_token_ids)
        cursor = slot + 1
    return out


def _marker_token_ids(tokenizer: Any, marker_word: str) -> list[list[int]]:
    """Tokenizations of a marker word to search for (with and without a space)."""

    variants = [marker_word, f" {marker_word}"]
    seen: list[list[int]] = []
    for variant in variants:
        ids = [int(i) for i in tokenizer.encode(variant, add_special_tokens=False)]
        if ids and ids not in seen:
            seen.append(ids)
    return seen


def _next_digit_step(
    generated_ids: Sequence[int],
    marker_variants: Sequence[Sequence[int]],
    digit_id_set: set[int],
    *,
    start: int,
) -> int | None:
    """Digit-token index for the marker's score slot, at/after ``start``.

    The judge repeats the marker words while reasoning ("Visual Quality
    Analysis: ...") long before the final score line, and the numbered answer
    list ("(1) visual quality: 3") plants a stray list numeral between a CoT
    mention and the real score — so binding to the FIRST marker match can lock
    onto the wrong digit (measured live: soft 1.0 vs hard 3, silently). Anchor
    on the LAST match instead, and only accept a digit within a few tokens of
    the marker: the score line always reads "<marker>: <digit>", while a digit
    trailing a CoT mention sits far away (out of window -> hard fallback).
    """

    best: int | None = None
    for marker_ids in marker_variants:
        pos = start
        while True:
            end = _find_subsequence(generated_ids, marker_ids, start=pos)
            if end is None:
                break
            stop = min(len(generated_ids), end + _DIGIT_PROXIMITY_WINDOW)
            for j in range(end, stop):
                if generated_ids[j] in digit_id_set:
                    if best is None or j > best:
                        best = j
                    break
            pos = end
    return best


def _find_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
    *,
    start: int = 0,
) -> int | None:
    """Return the index just past the first match of ``needle`` at/after ``start``."""

    if not needle:
        return None
    last = len(haystack) - len(needle)
    for i in range(start, last + 1):
        if all(haystack[i + k] == needle[k] for k in range(len(needle))):
            return i + len(needle)
    return None


def _expected_digit_value(logits: Any, digit_token_ids: Mapping[int, int]) -> float:
    """Softmax over the 1-5 digit logits, returned as the expected score."""

    digits = sorted(digit_token_ids)
    selected = torch.tensor(
        [float(logits[digit_token_ids[d]]) for d in digits],
        dtype=torch.float32,
    )
    probs = torch.softmax(selected, dim=0)
    values = torch.tensor([float(d) for d in digits], dtype=torch.float32)
    return float((probs * values).sum().item())


def _normalize_scores(
    visual_quality: float,
    text_alignment: float,
    physical_common_sense: float,
) -> dict[str, float]:
    """Map the three axes to the public score keys plus their mean ``overall``.

    This dict is the public scoring contract: it carries only the documented
    keys, so a config cannot select an undocumented upstream key as ``score_key``.
    """

    visual_quality = float(visual_quality)
    text_alignment = float(text_alignment)
    physical_common_sense = float(physical_common_sense)
    return {
        "visual_quality": visual_quality,
        "text_alignment": text_alignment,
        "physical_common_sense": physical_common_sense,
        "overall": (visual_quality + text_alignment + physical_common_sense) / 3.0,
    }


def _resolve_model_root(worker_config: Mapping[str, Any]) -> Path:
    """Return a local checkpoint dir: ``model_path`` if set, else snapshot_download."""

    model_path = str(worker_config.get("model_path", "")).strip()
    if model_path:
        root = Path(model_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"VideoScore2 model_path missing: {root}")
        return root

    reward_model_name = str(worker_config.get("reward_model_name", "")).strip()
    if not reward_model_name:
        reward_model_name = _DEFAULT_REWARD_MODEL
    model_ref = parse_hf_repo_revision(reward_model_name)

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_ref.repo_id,
            revision=model_ref.revision,
            local_files_only=bool(worker_config.get("local_files_only", False)),
        ),
    ).resolve()


__all__ = [
    "VideoScore2Model",
    "preflight_videoscore2_backend",
]
