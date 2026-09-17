"""OCR text-matching reward as a model-backed RewardModel.

Mirrors Flow-GRPO's ``OcrScorer_video_or_image``: detect text in sampled
frames with ``paddleocr``, concatenate the recognized lines, and score by
normalized Levenshtein distance to the ``target_text`` rollout metadata (a
single image whose text contains the target gets full credit, as upstream).

The PaddleOCR engine is lazy-loaded and injectable via ``worker_config["engine"]``
(or by assigning ``model._engine`` directly) so tests can supply a fake engine.
Returns edit similarity under ``ocr`` and the fraction of sampled frames whose
whole recognized text equals the target under ``ocr_match``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrl.utils.media import to_uint8

logger = logging.getLogger(__name__)

# Persisted sidecar protocol used to audit the exact OCR decision behind a reward.
OCR_DEBUG_SCHEMA = "vrl.ocr-debug/v6"


def normalize_ocr_text(text: str) -> str:
    """Match Flow-GRPO OCR targets by lowercasing and removing ASCII spaces.

    Punctuation remains significant; this is the exact comparison key the
    reward scores against, so dataset targets must be derived the same way.
    """

    return text.replace(" ", "").lower()


@dataclass(frozen=True, slots=True)
class _OcrLine:
    """One public PaddleOCR recognition result after dependency adaptation."""

    text: str
    confidence: float


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
        self._engine = _build_paddle_ocr()

    def __call__(self, artifact: Any) -> dict[str, float]:
        import numpy as np
        import torch

        target_text_raw = str(artifact.metadata.get("target_text", ""))
        if not target_text_raw:
            return {"ocr": 0.0, "ocr_match": 0.0}

        target_text = normalize_ocr_text(target_text_raw)
        if not target_text:
            return {"ocr": 0.0, "ocr_match": 0.0}

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
        match_frame_count = 0
        # Start below the valid reward range so an all-zero sample still keeps
        # its first frame for reward-hacking audits.
        best_reward: float = -1.0
        best_frame: np.ndarray | None = None
        best_lines: tuple[_OcrLine, ...] = ()

        for frame in frames:
            lines = _extract_ocr_lines(_run_paddle_ocr(self._engine, frame))
            recognized = normalize_ocr_text("".join(line.text for line in lines))
            # flow_grpo: the substring shortcut is image-only; video frames are
            # scored by edit distance alone.
            dist = (
                0
                if single_image and target_text in recognized
                else distance(recognized, target_text)
            )
            reward = 1.0 - min(dist, target_len) / target_len
            if reward > 0:
                frame_rewards.append(reward)
            # Exact-match success includes failed frames in its denominator: one
            # readable frame must not give an otherwise incorrect video full credit.
            match_frame_count += recognized == target_text
            if reward > best_reward:
                best_reward = reward
                best_frame = frame
                best_lines = lines

        score_value = sum(frame_rewards) / len(frame_rewards) if frame_rewards else 0.0
        match_score_value = match_frame_count / len(frames) if frames else 0.0

        if self._debug_dir is not None and best_frame is not None:
            self._dump_debug_frame(
                best_frame,
                sample_id=artifact.sample_id,
                target=target_text_raw,
                recognized_lines=best_lines,
                best_frame_score=max(best_reward, 0.0),
                aggregate_score=score_value,
                aggregate_match_score=match_score_value,
            )

        return {"ocr": float(score_value), "ocr_match": float(match_score_value)}

    def _dump_debug_frame(
        self,
        frame: Any,
        *,
        sample_id: str,
        target: str,
        recognized_lines: tuple[_OcrLine, ...],
        best_frame_score: float,
        aggregate_score: float,
        aggregate_match_score: float,
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
                        "recognized_lines": [
                            {"text": line.text, "confidence": line.confidence}
                            for line in recognized_lines
                        ],
                        "all_recognized_text": "".join(line.text for line in recognized_lines),
                        "best_frame_score": best_frame_score,
                        "aggregate_score": aggregate_score,
                        "aggregate_match_score": aggregate_match_score,
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


def _build_paddle_ocr() -> Any:
    """Build the flow_grpo-compatible PaddleOCR engine across supported public APIs."""

    import inspect

    from paddleocr import PaddleOCR

    params = inspect.signature(PaddleOCR).parameters
    if "use_textline_orientation" in params:
        # Paddle 3.3.1's oneDNN executor cannot load the PP-OCRv6 static graph
        # ArrayAttribute; compatibility mode stays on its pre-existing plain
        # CPU path.
        return PaddleOCR(
            enable_mkldnn=False,
            lang="en",
            ocr_version="PP-OCRv4",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
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


__all__ = ["OCRRewardModel", "normalize_ocr_text"]
