"""OCR reward function, behavior mirrors flow_grpo OCR scorers.

Scores generated image/video outputs by how well OCR-detected text matches a
target string provided in rollout metadata. Single-image SD3 rewards mirror
``OcrScorer``; video/multi-frame rewards mirror ``OcrScorer_video_or_image``.

flow_grpo references:
- ``flow_grpo/ocr.py::OcrScorer``
- ``flow_grpo/ocr.py::OcrScorer_video_or_image``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.types import RewardRollout


def _safe_filename_fragment(text: str, max_len: int = 24) -> str:
    """Sanitize arbitrary text for use inside a filename."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text)[:max_len].strip("_") or "empty"


def _normalize_ocr_text(text: str) -> str:
    """flow_grpo-compatible normalization: lowercase, strip spaces."""
    return text.replace(" ", "").lower()


def _normalize_text(text: str) -> str:
    """Normalize text for helper-level OCR edit-distance tests."""
    normalized = re.sub(r"[^a-z0-9\s]+", "", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _normalized_edit_distance(a: str, b: str) -> float:
    """Return Levenshtein distance normalized by the longer input length."""
    if not a and not b:
        return 0.0
    distance = _edit_distance(a, b)
    return distance / max(len(a), len(b), 1)


def _edit_distance(a: str, b: str) -> int:
    """Small dependency-free Levenshtein implementation for tests."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = curr
    return prev[-1]


class OCRReward(RewardFunction):
    """OCR-based text matching reward (flow_grpo-compatible).

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text
    in sampled frames and computes reward = mean over frames with reward>0,
    per the flow_grpo ``OcrScorer_video_or_image`` implementation.

    When ``debug_dir`` is set, dumps the best-scoring frame along with the
    OCR-detected text and target to disk for reward-hacking audit.
    """

    frame_interval: int = 4  # matches flow_grpo OcrScorer_video_or_image

    def __init__(self, device: str = "cuda", debug_dir: str | None = None) -> None:
        self._device = device
        self._engine: Any = None
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_counter = 0
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        self._engine = _build_paddle_ocr()

    async def score(self, rollout: RewardRollout) -> float:
        import numpy as np
        import torch

        target_text_raw = rollout.metadata.get("target_text", "")
        if not target_text_raw:
            return 0.0

        target_text = _normalize_ocr_text(target_text_raw)
        if not target_text:
            return 0.0

        self._ensure_loaded()
        output = rollout.trajectory.output

        # ---- extract frames as list of numpy uint8 [H, W, C] ----
        # SD3 image OCR in flow_grpo uses OcrScorer, while video OCR uses
        # OcrScorer_video_or_image. The substring full-credit shortcut is
        # image-only, so keep track of whether this output is a single image.
        frames: list[np.ndarray] = []
        single_image = False

        if isinstance(output, torch.Tensor):
            raw = (output * 255).round().clamp(0, 255).to(torch.uint8)

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
        best_reward: float = 0.0
        best_frame: np.ndarray | None = None
        best_ocr_text: str = ""

        for frame in frames:
            text_raw = _extract_ocr_text(_run_paddle_ocr(self._engine, frame))
            text = _normalize_ocr_text(text_raw)
            dist = 0 if single_image and target_text in text else distance(text, target_text)
            dist = min(dist, target_len)

            reward = 1.0 - dist / target_len
            if reward > 0:
                frame_rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_frame = frame
                best_ocr_text = text_raw

        score_value = sum(frame_rewards) / len(frame_rewards) if frame_rewards else 0.0

        if self._debug_dir is not None and best_frame is not None:
            self._dump_debug_frame(best_frame, target_text_raw, best_ocr_text, score_value)

        return score_value

    def _dump_debug_frame(
        self,
        frame: Any,
        target: str,
        ocr_text: str,
        score_value: float,
    ) -> None:
        """Save best frame + metadata to debug_dir. Failure is non-fatal."""
        try:
            from PIL import Image

            idx = self._debug_counter
            self._debug_counter += 1
            tag = _safe_filename_fragment(target)
            img_path = self._debug_dir / f"{idx:06d}_{tag}_score{score_value:.3f}.png"
            meta_path = self._debug_dir / f"{idx:06d}_{tag}_score{score_value:.3f}.txt"

            Image.fromarray(frame).save(img_path)
            meta_path.write_text(
                f"target: {target}\nocr:    {ocr_text}\nscore:  {score_value:.4f}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        return [await self.score(r) for r in rollouts]


def _build_paddle_ocr() -> Any:
    """Build PaddleOCR across both legacy 2.x and current 3.x constructor APIs."""

    import inspect

    from paddleocr import PaddleOCR

    params = inspect.signature(PaddleOCR).parameters
    if "use_textline_orientation" in params:
        return PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,
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


def _extract_ocr_text(result: Any) -> str:
    if not result:
        return ""
    if isinstance(result, dict):
        return _join_ocr_texts(result.get("rec_texts"), result.get("rec_scores"))
    if isinstance(result, list):
        if result and all(isinstance(item, dict) for item in result):
            return "".join(_extract_ocr_text(item) for item in result)

        rows = result[0] if len(result) == 1 and isinstance(result[0], list) else result
        texts: list[str] = []
        for row in rows:
            if isinstance(row, dict):
                texts.append(_extract_ocr_text(row))
                continue
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            text_score = row[1]
            if not isinstance(text_score, (list, tuple)) or not text_score:
                continue
            text = text_score[0]
            score = text_score[1] if len(text_score) > 1 else 1.0
            if isinstance(text, str) and float(score) > 0.0:
                texts.append(text)
        return "".join(texts)
    return ""


def _join_ocr_texts(texts: Any, scores: Any) -> str:
    if not texts:
        return ""
    if scores is None:
        scores = [1.0] * len(texts)
    return "".join(
        str(text) for text, score in zip(texts, scores, strict=False) if float(score) > 0.0
    )
