"""OCR text-matching reward as a model-backed RewardModel.

The compatibility policy mirrors Flow-GRPO: detect text in sampled frames with
``paddleocr`` and score the concatenated result by normalized Levenshtein
distance to rollout metadata. Exact-text curricula may instead score the best
complete detected line, preserving line boundaries that the compatibility path
intentionally discards.

The PaddleOCR engine is lazy-loaded and injectable via ``worker_config["engine"]``
(or by assigning ``model._engine`` directly) so tests can supply a fake engine.
Returns the optimization score under ``ocr`` plus raw/duplicate audit values;
drive it with ``score_key="ocr"``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.rewards.ocr_text import (
    OcrEngineProfile,
    OcrScoringPolicy,
    OcrTextSelection,
    normalize_ocr_text,
)
from vrl.utils.media import to_uint8

logger = logging.getLogger(__name__)

# Persisted sidecar protocol used to audit the exact OCR decision behind a reward.
OCR_DEBUG_SCHEMA = "vrl.ocr-debug/v5"


@dataclass(frozen=True, slots=True)
class _OcrLine:
    """One public PaddleOCR recognition result after dependency adaptation."""

    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class _OcrCandidate:
    """One target-comparison candidate and the detected lines that formed it."""

    line_indices: tuple[int, ...] | None
    text: str


@dataclass(frozen=True, slots=True)
class _OcrDecision:
    """Resolved score and the line-level evidence that produced it."""

    raw_reward: float
    reward: float
    selected_line_indices: tuple[int, ...] | None
    selected_text: str
    rejected_extra_line_indices: tuple[int, ...]
    near_duplicate_line_indices: tuple[int, ...]


def _safe_filename_fragment(text: str, max_len: int = 24) -> str:
    """Sanitize arbitrary text for use inside a filename."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text)[:max_len].strip("_") or "empty"


class OCRRewardModel:
    """RewardModel returning one OCR reward and its audit observations.

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text in
    sampled frames and computes reward = mean over frames with reward > 0, per
    flow_grpo ``OcrScorer_video_or_image``. When ``debug_dir`` is set, dumps the
    best-scoring frame plus the exact line-level scoring decision for audit.
    """

    frame_interval: int = 4  # matches flow_grpo OcrScorer_video_or_image

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self._engine: Any = cfg.get("engine")
        self.engine_profile = OcrEngineProfile.parse(
            cfg.get("engine_profile", OcrEngineProfile.FLOW_GRPO_COMPAT.value),
            what="OCR reward engine_profile",
        )
        self.scoring_policy = OcrScoringPolicy.from_mapping(
            {
                "text_selection": cfg.get(
                    "text_selection",
                    OcrTextSelection.ALL_TEXT.value,
                ),
                "substring_full_credit": cfg.get("substring_full_credit", True),
                "exclusive_alphanumeric_lines": cfg.get(
                    "exclusive_alphanumeric_lines",
                    False,
                ),
                "extra_line_min_confidence": cfg.get("extra_line_min_confidence", 0.5),
                "near_duplicate_min_similarity": cfg.get("near_duplicate_min_similarity"),
            },
            what="OCR reward configuration",
        )
        debug_dir = cfg.get("debug_dir")
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_counter = 0
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            self._debug_counter = self._next_debug_index(self._debug_dir)

    @staticmethod
    def _next_debug_index(debug_dir: Path) -> int:
        """Continue the on-disk debug protocol without overwriting prior samples."""
        next_index = 0
        for path in debug_dir.iterdir():
            match = re.fullmatch(
                r"(?P<index>\d{6,})_.+_score-?\d+\.\d{3}\.(?:png|txt|json)",
                path.name,
            )
            if match is not None:
                next_index = max(next_index, int(match.group("index")) + 1)
        return next_index

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        self._engine = _build_paddle_ocr(self.engine_profile)

    def __call__(self, artifact: Any) -> dict[str, float]:
        import numpy as np
        import torch

        target_text_raw = str(artifact.metadata.get("target_text", ""))
        if not target_text_raw:
            return {"ocr": 0.0, "ocr_raw": 0.0, "ocr_near_duplicate_count": 0.0}

        target_text = normalize_ocr_text(target_text_raw)
        if not target_text:
            return {"ocr": 0.0, "ocr_raw": 0.0, "ocr_near_duplicate_count": 0.0}

        self._ensure_loaded()
        output = artifact.as_media()

        # ---- extract frames as list of numpy uint8 [H, W, C] ----
        # SD3 image OCR in flow_grpo uses OcrScorer, while video OCR uses
        # OcrScorer_video_or_image. The substring full-credit shortcut is
        # image-only, so keep track of whether this output is a single image.
        frames: list[np.ndarray] = []
        single_image = False

        if isinstance(output, torch.Tensor):
            raw = to_uint8(output)

            if raw.ndim == 4 and raw.shape[0] <= 4:
                # [C, T, H, W] video → [T, H, W, C]
                video = raw.permute(1, 2, 3, 0).cpu().numpy()
                frames = list(video[:: self.frame_interval])
            elif raw.ndim == 4 and raw.shape[0] > 4:
                # [T, C, H, W] or [B, C, H, W] — treat as T-first
                video = raw.permute(0, 2, 3, 1).cpu().numpy()
                frames = list(video[:: self.frame_interval])
            elif raw.ndim == 3:
                # [C, H, W] single image
                frames = [raw.permute(1, 2, 0).cpu().numpy()]
                single_image = True
            else:
                raise ValueError(
                    f"OCRReward expected image/video tensor with 3 or 4 dims, got {raw.ndim}",
                )
        else:
            # Assume PIL or numpy already; single image path
            array = np.asarray(output)
            frames = [array]
            single_image = array.ndim == 3

        # ---- per-frame OCR + Levenshtein, matches flow_grpo ----
        from Levenshtein import distance

        target_len = len(target_text)
        frame_rewards: list[float] = []
        frame_raw_rewards: list[float] = []
        # Start below the valid reward range so an all-zero sample still keeps
        # its first frame for reward-hacking audits.
        best_reward: float = -1.0
        best_frame: np.ndarray | None = None
        best_lines: tuple[_OcrLine, ...] = ()
        best_selected_line_indices: tuple[int, ...] | None = None
        best_selected_text = ""
        best_rejected_extra_line_indices: tuple[int, ...] = ()
        best_near_duplicate_line_indices: tuple[int, ...] = ()
        best_raw_reward = 0.0

        for frame in frames:
            lines = _extract_ocr_lines(_run_paddle_ocr(self._engine, frame))
            candidates = _scoring_candidates(lines, self.scoring_policy.text_selection)
            decision = _best_candidate_score(
                candidates,
                lines=lines,
                target_text=target_text,
                target_len=target_len,
                single_image=single_image,
                policy=self.scoring_policy,
                distance=distance,
            )
            if decision.reward > 0:
                frame_rewards.append(decision.reward)
            if decision.raw_reward > 0:
                frame_raw_rewards.append(decision.raw_reward)
            if decision.reward > best_reward:
                best_reward = decision.reward
                best_frame = frame
                best_lines = lines
                best_selected_line_indices = decision.selected_line_indices
                best_selected_text = decision.selected_text
                best_rejected_extra_line_indices = decision.rejected_extra_line_indices
                best_near_duplicate_line_indices = decision.near_duplicate_line_indices
                best_raw_reward = decision.raw_reward

        score_value = sum(frame_rewards) / len(frame_rewards) if frame_rewards else 0.0
        raw_score_value = (
            sum(frame_raw_rewards) / len(frame_raw_rewards) if frame_raw_rewards else 0.0
        )

        if self._debug_dir is not None and best_frame is not None:
            self._dump_debug_frame(
                best_frame,
                sample_id=artifact.sample_id,
                target=target_text_raw,
                recognized_lines=best_lines,
                selected_line_indices=best_selected_line_indices,
                selected_text=best_selected_text,
                rejected_extra_line_indices=best_rejected_extra_line_indices,
                near_duplicate_line_indices=best_near_duplicate_line_indices,
                best_frame_raw_score=best_raw_reward,
                best_frame_score=max(best_reward, 0.0),
                aggregate_raw_score=raw_score_value,
                aggregate_score=score_value,
            )

        return {
            "ocr": float(score_value),
            "ocr_raw": float(raw_score_value),
            "ocr_near_duplicate_count": float(len(best_near_duplicate_line_indices)),
        }

    def _dump_debug_frame(
        self,
        frame: Any,
        *,
        sample_id: str,
        target: str,
        recognized_lines: tuple[_OcrLine, ...],
        selected_line_indices: tuple[int, ...] | None,
        selected_text: str,
        rejected_extra_line_indices: tuple[int, ...],
        near_duplicate_line_indices: tuple[int, ...],
        best_frame_raw_score: float,
        best_frame_score: float,
        aggregate_raw_score: float,
        aggregate_score: float,
    ) -> None:
        """Save best frame + metadata to debug_dir. Failure is non-fatal."""
        idx = self._debug_counter
        self._debug_counter += 1
        try:
            from PIL import Image

            sample_tag = _safe_filename_fragment(sample_id)
            target_tag = _safe_filename_fragment(target)
            basename = (
                f"{idx:06d}_sample-{sample_tag}_target-{target_tag}_score{aggregate_score:.3f}"
            )
            img_path = self._debug_dir / f"{basename}.png"
            meta_path = self._debug_dir / f"{basename}.json"

            Image.fromarray(frame).save(img_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "schema": OCR_DEBUG_SCHEMA,
                        "sample_id": sample_id,
                        "target_text": target,
                        "normalized_target_text": normalize_ocr_text(target),
                        "engine_profile": self.engine_profile.value,
                        "scoring_policy": self.scoring_policy.to_record(),
                        "recognized_lines": [
                            {"text": line.text, "confidence": line.confidence}
                            for line in recognized_lines
                        ],
                        "all_recognized_text": "".join(line.text for line in recognized_lines),
                        "selected_line_indices": (
                            list(selected_line_indices)
                            if selected_line_indices is not None
                            else None
                        ),
                        "selected_recognized_text": selected_text,
                        "rejected_extra_line_indices": list(rejected_extra_line_indices),
                        "near_duplicate_line_indices": list(near_duplicate_line_indices),
                        "best_frame_raw_score": best_frame_raw_score,
                        "best_frame_score": best_frame_score,
                        "aggregate_raw_score": aggregate_raw_score,
                        "aggregate_score": aggregate_score,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.warning(
                "Failed to write OCR debug sample %06d to %s",
                idx,
                self._debug_dir,
                exc_info=True,
            )


def _build_paddle_ocr(
    profile: OcrEngineProfile = OcrEngineProfile.FLOW_GRPO_COMPAT,
) -> Any:
    """Build one pinned PaddleOCR profile across supported public APIs."""

    import inspect

    from paddleocr import PaddleOCR

    params = inspect.signature(PaddleOCR).parameters
    if "use_textline_orientation" in params:
        # Paddle 3.3.1's oneDNN executor cannot load the PP-OCRv6 static graph
        # ArrayAttribute. The OCR extra pins the locally qualified 3.2.1 stack,
        # where oneDNN is both stable and materially faster. Keep compatibility
        # mode on its pre-existing plain CPU path.
        common = {
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if profile is OcrEngineProfile.PP_OCRV6_MEDIUM:
            return PaddleOCR(
                enable_mkldnn=True,
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                **common,
            )
        return PaddleOCR(
            enable_mkldnn=False,
            lang="en",
            ocr_version="PP-OCRv4",
            **common,
        )
    if profile is not OcrEngineProfile.FLOW_GRPO_COMPAT:
        raise RuntimeError(
            "ppocrv6_medium requires the pinned OCR extra: "
            "paddleocr==3.7.0 and paddlepaddle==3.2.1",
        )
    return PaddleOCR(
        use_angle_cls=False,
        lang="en",
        use_gpu=False,
        show_log=False,
    )


def _run_paddle_ocr(engine: Any, frame: Any) -> Any:
    if hasattr(engine, "predict"):
        return engine.predict(frame)
    try:
        return engine.ocr(frame, cls=False)
    except (TypeError, ValueError) as exc:
        if "cls" not in str(exc) and "Unknown argument" not in str(exc):
            raise
        return engine.ocr(frame)


def _extract_ocr_lines(result: Any) -> tuple[_OcrLine, ...]:
    """Adapt public PaddleOCR 2.x/3.x results without losing line boundaries."""

    if result is None:
        return ()
    if isinstance(result, Mapping):
        if "rec_texts" in result:
            return _lines_from_columns(result.get("rec_texts"), result.get("rec_scores"))
        return ()
    if _is_legacy_ocr_row(result):
        text_score = result[1]
        text = text_score[0]
        confidence = float(text_score[1]) if len(text_score) > 1 else 1.0
        if not text or confidence <= 0.0:
            return ()
        return (_OcrLine(text=text, confidence=confidence),)
    if isinstance(result, (list, tuple)):
        lines: list[_OcrLine] = []
        for item in result:
            lines.extend(_extract_ocr_lines(item))
        return tuple(lines)
    return ()


def _is_legacy_ocr_row(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    text_score = value[1]
    return (
        isinstance(text_score, (list, tuple))
        and bool(text_score)
        and isinstance(text_score[0], str)
    )


def _lines_from_columns(texts: Any, scores: Any) -> tuple[_OcrLine, ...]:
    if texts is None:
        return ()
    if not isinstance(texts, (list, tuple)):
        raise TypeError("PaddleOCR rec_texts must be a list or tuple")
    if scores is None:
        scores = [1.0] * len(texts)
    if not isinstance(scores, (list, tuple)):
        raise TypeError("PaddleOCR rec_scores must be a list or tuple")
    if len(texts) != len(scores):
        raise ValueError(
            f"PaddleOCR rec_texts and rec_scores lengths differ: {len(texts)} != {len(scores)}",
        )
    lines: list[_OcrLine] = []
    for text, raw_confidence in zip(texts, scores, strict=True):
        confidence = float(raw_confidence)
        if isinstance(text, str) and text and confidence > 0.0:
            lines.append(_OcrLine(text=text, confidence=confidence))
    return tuple(lines)


def _scoring_candidates(
    lines: tuple[_OcrLine, ...],
    selection: OcrTextSelection,
) -> tuple[_OcrCandidate, ...]:
    if selection is OcrTextSelection.ALL_TEXT:
        return (_OcrCandidate(None, "".join(line.text for line in lines)),)
    if selection is OcrTextSelection.BEST_COMPLETE_LINE:
        return tuple(_OcrCandidate((index,), line.text) for index, line in enumerate(lines))
    return tuple(
        _OcrCandidate(
            tuple(range(start, start + width)),
            "".join(line.text for line in lines[start : start + width]),
        )
        for width in range(1, len(lines) + 1)
        for start in range(0, len(lines) - width + 1)
    )


def _best_candidate_score(
    candidates: tuple[_OcrCandidate, ...],
    *,
    lines: tuple[_OcrLine, ...],
    target_text: str,
    target_len: int,
    single_image: bool,
    policy: OcrScoringPolicy,
    distance: Callable[[str, str], int],
) -> _OcrDecision:
    best_reward = -1.0
    best_indices: tuple[int, ...] | None = None
    best_text = ""
    for candidate in candidates:
        normalized_candidate = normalize_ocr_text(candidate.text)
        dist = (
            0
            if single_image
            and policy.substring_full_credit
            and target_text in normalized_candidate
            else distance(normalized_candidate, target_text)
        )
        reward = 1.0 - min(dist, target_len) / target_len
        if reward > best_reward:
            best_reward = reward
            best_indices = candidate.line_indices
            best_text = candidate.text
    rejected_extra_line_indices = _rejected_extra_line_indices(
        lines,
        selected_line_indices=best_indices,
        policy=policy,
    )
    near_duplicate_line_indices = _near_duplicate_line_indices(
        lines,
        selected_line_indices=best_indices,
        target_text=target_text,
        target_len=target_len,
        policy=policy,
        distance=distance,
    )
    raw_reward = max(best_reward, 0.0)
    if rejected_extra_line_indices:
        reward = 0.0
    else:
        reward = raw_reward / (1 + len(near_duplicate_line_indices))
    return _OcrDecision(
        raw_reward=raw_reward,
        reward=reward,
        selected_line_indices=best_indices,
        selected_text=best_text,
        rejected_extra_line_indices=rejected_extra_line_indices,
        near_duplicate_line_indices=near_duplicate_line_indices,
    )


def _rejected_extra_line_indices(
    lines: tuple[_OcrLine, ...],
    *,
    selected_line_indices: tuple[int, ...] | None,
    policy: OcrScoringPolicy,
) -> tuple[int, ...]:
    if not policy.exclusive_alphanumeric_lines:
        return ()
    selected = frozenset(selected_line_indices or ())
    return tuple(
        index
        for index, line in enumerate(lines)
        if index not in selected
        and line.confidence >= policy.extra_line_min_confidence
        and any(character.isalnum() for character in line.text)
    )


def _near_duplicate_line_indices(
    lines: tuple[_OcrLine, ...],
    *,
    selected_line_indices: tuple[int, ...] | None,
    target_text: str,
    target_len: int,
    policy: OcrScoringPolicy,
    distance: Callable[[str, str], int],
) -> tuple[int, ...]:
    """Find confident unselected lines that look like another target attempt."""

    threshold = policy.near_duplicate_min_similarity
    if threshold is None:
        return ()
    selected = frozenset(selected_line_indices or ())
    duplicates: list[int] = []
    for index, line in enumerate(lines):
        if (
            index in selected
            or line.confidence < policy.extra_line_min_confidence
            or not any(character.isalnum() for character in line.text)
        ):
            continue
        normalized_line = normalize_ocr_text(line.text)
        similarity = 1.0 - min(distance(normalized_line, target_text), target_len) / target_len
        if similarity >= threshold:
            duplicates.append(index)
    return tuple(duplicates)


__all__ = ["OCRRewardModel"]
