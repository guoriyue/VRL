"""Paddle-free OCR text contract shared by data and reward code."""

from __future__ import annotations


def normalize_ocr_text(text: str) -> str:
    """Match Flow-GRPO OCR targets by lowercasing and removing ASCII spaces.

    Punctuation remains significant. Keeping this narrow contract independent
    from the OCR model lets dataset derivation use the exact runtime comparison
    key without importing PaddleOCR or media dependencies.
    """

    return text.replace(" ", "").lower()


__all__ = ["normalize_ocr_text"]
