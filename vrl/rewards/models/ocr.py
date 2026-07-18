"""OCR text-matching reward as a model-backed RewardModel.

Scoring logic ported verbatim from the in-process ``OCRReward``: detect text in
sampled frames with ``paddleocr`` and score by normalized Levenshtein distance to
a target string carried in the rollout metadata. Single-image SD3 rewards mirror
flow_grpo ``OcrScorer`` (substring full-credit shortcut); video/multi-frame
rewards mirror ``OcrScorer_video_or_image``.

The PaddleOCR engine is lazy-loaded and injectable via ``worker_config["engine"]``
(or by assigning ``model._engine`` directly) so tests can supply a fake engine.
Returns ``{"ocr": value}``; drive it with ``score_key="ocr"``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vrl.utils.media import to_uint8


def _safe_filename_fragment(text: str, max_len: int = 24) -> str:
    """Sanitize arbitrary text for use inside a filename."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text)[:max_len].strip("_") or "empty"


def _normalize_ocr_text(text: str) -> str:
    """flow_grpo-compatible normalization: lowercase, strip spaces."""
    return text.replace(" ", "").lower()


class OCRRewardModel:
    """RewardModel returning ``{"ocr": reward}`` per artifact.

    Uses ``paddleocr`` (matches flow_grpo's engine choice) to detect text in
    sampled frames and computes reward = mean over frames with reward > 0, per
    flow_grpo ``OcrScorer_video_or_image``. When ``debug_dir`` is set, dumps the
    best-scoring frame plus OCR/target text for reward-hacking audit.
    """

    frame_interval: int = 4  # matches flow_grpo OcrScorer_video_or_image

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self._device = str(cfg.get("device", "cuda"))
        self._engine: Any = cfg.get("engine")
        debug_dir = cfg.get("debug_dir")
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_counter = 0
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        self._engine = _build_paddle_ocr()

    def __call__(self, *, artifact: Any, request: Any) -> dict[str, float]:
        del request
        import numpy as np
        import torch

        target_text_raw = str(artifact.metadata.get("target_text", ""))
        if not target_text_raw:
            return {"ocr": 0.0}

        target_text = _normalize_ocr_text(target_text_raw)
        if not target_text:
            return {"ocr": 0.0}

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

        return {"ocr": float(score_value)}

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


__all__ = ["OCRRewardModel"]
