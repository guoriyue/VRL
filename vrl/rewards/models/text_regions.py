"""Region-bound text diagnostics for image editing and dialogue layouts.

Regions are explicit task specifications, not detected speech bubbles. Scores
measure OCR agreement inside those regions, not narrative or visual quality.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vrl.rewards.inference import RewardInferenceArtifact, RewardInferenceResult
from vrl.rewards.models.ocr import _build_paddle_ocr, _extract_ocr_lines, _run_paddle_ocr
from vrl.rewards.text_spec import TextLayout

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def _has_word(text: str, word: str) -> bool:
    if not word:
        return False
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


class TextRegionsRewardModel:
    """CPU OCR with equal region weighting and complete per-region evidence.

    NFC Unicode normalization and collapsed whitespace tolerate OCR line wraps;
    case, punctuation, extra words and repeated words remain significant.
    Optional English word scores ignore case and separators, while retaining
    token order, spelling, numeric tokens, omissions and repetitions.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        # Tests inject an engine; otherwise the OCR reward's PaddleOCR backend.
        self._engine = worker_config.get("engine")
        # "ignore" drops ASCII punctuation before comparing: PaddleOCR adds and
        # drops periods at line ends, which would otherwise dominate the score
        # of a correctly typeset balloon. Case and words stay significant.
        self._punctuation = worker_config.get("punctuation", "keep")
        if self._punctuation not in ("keep", "ignore"):
            raise ValueError("text_regions punctuation must be keep or ignore")
        # Opt-in English word agreement tolerates OCR punctuation/line joins,
        # while charging a complete region for misspelled or missing words.
        self._word_matching = worker_config.get("word_matching")
        if self._word_matching not in (None, "english"):
            raise ValueError("text_regions word_matching must be english or omitted")

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        if self._punctuation == "ignore":
            text = text.translate(_PUNCTUATION)
        return " ".join(text.split())

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        return self.score_results((artifact,))[0].scores

    def score_results(
        self, artifacts: Sequence[RewardInferenceArtifact]
    ) -> list[RewardInferenceResult]:
        import torch
        from Levenshtein import distance
        from PIL import Image

        from vrl.utils.media import to_pil_image

        results = []
        for artifact in artifacts:
            layout = TextLayout.model_validate(artifact.metadata.get("text_layout"))
            if self._word_matching is not None and any(
                character.isalnum() and not character.isascii()
                for region in layout.regions
                for character in region.text
            ):
                raise ValueError(
                    "English word matching requires ASCII alphabetic or numeric targets"
                )
            if artifact.path and not artifact.path.endswith(".pt"):
                with Image.open(artifact.path) as source:
                    if getattr(source, "n_frames", 1) != 1:
                        raise ValueError("text_regions requires a single image")
                    media = source.copy()
            else:
                media = artifact.as_media()
            # Visual judges use the collector's C,T,H,W contract for one image.
            if isinstance(media, torch.Tensor) and media.ndim == 4 and media.shape[1] == 1:
                media = media[:, 0]
            if isinstance(media, Image.Image):
                if getattr(media, "n_frames", 1) != 1:
                    raise ValueError("text_regions requires a single image")
            elif not isinstance(media, (np.ndarray, torch.Tensor)) or media.ndim != 3:
                raise ValueError("text_regions requires a single image")
            image = to_pil_image(media)
            if image.size != (layout.width, layout.height):
                raise ValueError("text_regions requires the declared canvas without resizing")
            if self._engine is None:
                self._engine = _build_paddle_ocr()
            evidence = []
            for region in layout.regions:
                frame = np.asarray(image.crop(region.box))
                lines = _extract_ocr_lines(_run_paddle_ocr(self._engine, frame))
                recognized = " ".join(line.text for line in lines)
                target = self._normalize(region.text)
                observed = self._normalize(recognized)
                edits = distance(target, observed)
                evidence.append(
                    {
                        **region.model_dump(mode="json"),
                        "recognized": recognized,
                        "normalized_target": target,
                        "normalized_recognized": observed,
                        "edit_distance": edits,
                        "similarity": 1.0 - edits / max(len(target), len(observed), 1),
                        "exact": observed == target,
                        "lines": [
                            {"text": line.text, "confidence": line.confidence} for line in lines
                        ],
                    }
                )
                if self._word_matching is not None:
                    target_words = re.findall(r"[a-z]+|[0-9]+", region.text.casefold())
                    if not target_words:
                        raise ValueError("English word matching requires words or numbers")
                    observed_words = re.findall(r"[a-z]+|[0-9]+", recognized.casefold())
                    evidence[-1].update(
                        word_target=target_words,
                        word_recognized=observed_words,
                        word_exact=observed_words == target_words
                        and not any(
                            character.isalnum() and not character.isascii()
                            for character in recognized
                        ),
                    )
            similarities = [row["similarity"] for row in evidence]
            focus = artifact.metadata.get("text_focus")
            focus_scores: dict[str, float] = {}
            if focus is not None:
                # A single edit keeps its existing mapping. A sequence can pass
                # all still-required edits, so undoing an earlier changed word
                # cannot hide in the character similarity of a long transcript.
                focuses = [focus] if isinstance(focus, Mapping) else focus
                if not isinstance(focuses, (list, tuple)) or not focuses:
                    raise ValueError("text_focus must be a mapping or a non-empty list")
                completed, focus_similarities, focus_word_exact = [], [], []
                for item in focuses:
                    if not isinstance(item, Mapping) or not item.get("region_id"):
                        raise ValueError(
                            "text_focus needs region_id, must_contain and must_not_contain"
                        )
                    row = next((r for r in evidence if r["region_id"] == item["region_id"]), None)
                    if row is None:
                        raise ValueError(
                            f"text_focus names an unknown region: {item['region_id']}"
                        )
                    read = (
                        row["recognized"]
                        if self._word_matching is not None
                        else row["normalized_recognized"]
                    )
                    present = _has_word(read, str(item.get("must_contain", "")))
                    gone = not _has_word(read, str(item.get("must_not_contain", "")))
                    completed.append(float(present and gone))
                    focus_similarities.append(float(row["similarity"]))
                    if self._word_matching is not None:
                        focus_word_exact.append(float(row["word_exact"]))
                focus_scores = {
                    "text_focus_done": float(np.mean(completed)),
                    "text_focus_similarity": float(np.mean(focus_similarities)),
                }
                if self._word_matching is not None:
                    focus_scores["text_focus_word_exact_fraction"] = float(
                        np.mean(focus_word_exact)
                    )
            results.append(
                RewardInferenceResult(
                    artifact_id=artifact.artifact_id,
                    scores={
                        "text_similarity": float(np.mean(similarities)),
                        "text_worst_region": float(min(similarities)),
                        "text_exact_fraction": sum(row["exact"] for row in evidence)
                        / len(evidence),
                        "text_empty_fraction": sum(
                            not row["normalized_recognized"] for row in evidence
                        )
                        / len(evidence),
                        **{
                            f"region/{row['region_id']}/{axis}": float(row[key])
                            for row in evidence
                            for axis, key in (("similarity", "similarity"), ("exact", "exact"))
                        },
                        **focus_scores,
                        **(
                            {
                                "text_word_exact_fraction": float(
                                    np.mean([row["word_exact"] for row in evidence])
                                ),
                                **{
                                    f"region/{row['region_id']}/word_exact": float(
                                        row["word_exact"]
                                    )
                                    for row in evidence
                                },
                            }
                            if self._word_matching is not None
                            else {}
                        ),
                    },
                    diagnostics={
                        "schema": "vrl.text-regions.v1",
                        "normalization": "unicode-nfc-collapse-whitespace-case-sensitive"
                        + ("-ignore-punctuation" if self._punctuation == "ignore" else ""),
                        "ocr": "paddle_v4_en",
                        "regions": evidence,
                        **(
                            {
                                "word_matching": self._word_matching,
                                "word_normalization": "ascii-alphanumeric-casefold-ignore-separators-v1",
                            }
                            if self._word_matching
                            else {}
                        ),
                        "scope": "Specified regions only; no bubble detection, outside-text audit, "
                        "speaker attribution, global reading-order or aesthetic guarantee.",
                    },
                )
            )
        return results
