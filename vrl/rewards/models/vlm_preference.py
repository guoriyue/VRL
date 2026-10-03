"""Generative VLM preference reward: per-criterion answer probabilities from a large vision-language judge.

The judge (Qwen3-VL-32B by default) is asked one question per criterion about
the generated image -- aesthetics, anatomy/structure, artifacts, prompt
adherence -- and the reward is the weighted mean of the per-criterion scores.
Each score is read from the judge's own next-token distribution, not from a
regression head, so it inherits the judge's knowledge and scales with the
judge (Seedream 3.0 §2.3.3; RewardDance §3.2.3). Two answer formats:

* ``yesno``: ``P(yes) / (P(yes) + P(no))``. Sharp, good for guards (structure,
  adherence) where the judge should be near-certain.
* ``rating``: the judge answers excellent / good / fair / poor / bad and the
  score is the probability-weighted mean of the levels (1.0 .. 0.0), as in
  Q-Align. A single yes/no token saturates at 0 or 1 for a whole prompt group
  (measured: within-group std 0.000 for aesthetics), which gives GRPO nothing
  to rank; the graded expectation spreads the mass over neighbouring levels.

Why a large generative judge instead of CLIP-style preference models
(PickScore, HPSv3, SigLIP aesthetic): those see a 224-384 px thumbnail and a
global score, so six fingers, tangled cables or a cluttered background do not
move them, while a warm cinematic tint does -- which is exactly the drift the
earlier runs showed. The judge here sees the image at up to ``judge_max_side``
pixels and is asked about those failure modes directly.

All criteria questions for one image are batched into a single forward pass
(left-padded), so the cost is one 32B forward per image, not one per
criterion.

Scores: ``vlm_preference`` (weighted mean in [0, 1]) plus
``vlm_preference/<criterion>`` for every criterion. Group-normalized GRPO
advantages make the absolute scale irrelevant; the per-criterion keys are for
diagnostics and for the reward card's blind-spot bookkeeping.

``{prompt}`` in a question is replaced by the artifact's prompt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.base import LazyTorchModule
from vrl.rewards.models.media import artifact_middle_frame_image

RATING_LEVELS: tuple[tuple[str, float], ...] = (
    ("excellent", 1.0),
    ("good", 0.75),
    ("fair", 0.5),
    ("poor", 0.25),
    ("bad", 0.0),
)
ANSWER_SUFFIX = {
    "yesno": "\nAnswer with a single word: yes or no.",
    "rating": "\nAnswer with a single word: excellent, good, fair, poor, or bad.",
}

# Pairwise mode: the candidate is "Image 1" and the reference "Image 2" (and the
# reverse order is also asked, so position bias cancels). Every question is a
# yes/no "is Image 1 better" judgment.
PAIRWISE_PREAMBLE = (
    'Image 1 and Image 2 were both generated for the description: "{prompt}". '
    "Compare them carefully. "
)
DEFAULT_PAIRWISE_CRITERIA: tuple[dict[str, Any], ...] = (
    {
        "name": "overall",
        "weight": 2.0,
        "answer": "yesno",
        "question": (
            "Is Image 1 better than Image 2 as a finished picture: more pleasing "
            "composition, more natural and well-controlled lighting, more harmonious colors "
            "without an exaggerated color cast, and more refined detail?"
        ),
    },
    {
        "name": "structure",
        "weight": 1.0,
        "answer": "yesno",
        "question": (
            "Is Image 1 better than Image 2 in structural correctness: hands with five "
            "fingers, faces and bodies with normal proportions, and no merged, missing, "
            "extra or twisted limbs or parts?"
        ),
    },
    {
        "name": "clean",
        "weight": 1.0,
        "answer": "yesno",
        "question": (
            "Is Image 1 freer than Image 2 of rendering defects such as noise or grain, "
            "smudges, garbled text, floating fragments, purposeless tangled wires or cables, "
            "and meaningless clutter that the description did not call for?"
        ),
    },
)

DEFAULT_CRITERIA: tuple[dict[str, Any], ...] = (
    {
        "name": "aesthetic",
        "weight": 1.0,
        "answer": "rating",
        "question": (
            "As a demanding professional art director, how would you rate the overall "
            "visual quality of this image: composition, natural and well-controlled "
            "lighting, harmonious colors without an exaggerated color cast, and refined "
            "detail throughout?"
        ),
    },
    {
        "name": "structure",
        "weight": 1.0,
        "answer": "yesno",
        "question": (
            "Are all people, animals and objects in this image structurally correct: "
            "hands with five fingers, faces and bodies with normal proportions, and no "
            "merged, missing, extra or twisted limbs or parts?"
        ),
    },
    {
        "name": "clean",
        "weight": 1.0,
        "answer": "rating",
        "question": (
            'The description was: "{prompt}". Ignoring anything it explicitly asks for, '
            "how would you rate this image's freedom from rendering defects such as noise "
            "or grain, smudges, garbled text, floating fragments, purposeless tangled wires "
            "or cables, and meaningless clutter that it did not call for?"
        ),
    },
    {
        "name": "alignment",
        "weight": 1.0,
        "answer": "yesno",
        "question": 'Does this image faithfully depict the following description: "{prompt}"?',
    },
)


def _grid_crops(image: Any, grid: int) -> list[Any]:
    """The ``grid`` x ``grid`` tiles of an image, row-major, at native resolution."""

    w, h = image.size
    tw, th = w // grid, h // grid
    return [
        image.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
        for r in range(grid)
        for c in range(grid)
    ]


class VLMPreferenceRewardModel(LazyTorchModule):
    """Score one image with per-criterion questions to a VLM judge."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(worker_config)
        self.device = str(cfg.get("device", "cuda"))
        self._judge_model = str(cfg.get("judge_model", "Qwen/Qwen3-VL-32B-Instruct-FP8"))
        self._judge_revision = cfg.get("judge_revision")
        self._judge_max_side = int(cfg.get("judge_max_side", 1024))
        self._local_files_only = bool(cfg.get("local_files_only", False))
        # Pairwise mode: compare the candidate against a reference image whose
        # path the artifact carries under ``metadata[<reference_key>]``
        # (resolved under ``data_root``); ``both_orders`` asks each question in
        # both image orders and averages, cancelling position bias.
        self._reference_key = str(cfg.get("reference_key") or "").strip() or None
        self._both_orders = bool(cfg.get("both_orders", True))
        self.data_root = str(cfg.get("data_root") or "")
        raw = cfg.get("criteria") or (
            DEFAULT_PAIRWISE_CRITERIA if self._reference_key else DEFAULT_CRITERIA
        )
        self.criteria = [self._validate_criterion(c) for c in raw]
        if not self.criteria:
            raise ValueError("vlm_preference needs at least one criterion")
        names = [c["name"] for c in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError(f"vlm_preference criteria names must be unique: {names}")
        total = sum(c["weight"] for c in self.criteria)
        if total <= 0:
            raise ValueError("vlm_preference criteria weights must sum to a positive number")
        self._weights = [c["weight"] / total for c in self.criteria]

    @staticmethod
    def _validate_criterion(raw: Mapping[str, Any]) -> dict[str, Any]:
        name = str(raw.get("name") or "").strip()
        question = str(raw.get("question") or "").strip()
        weight = float(raw.get("weight", 1.0))
        answer = str(raw.get("answer", "yesno"))
        if not name or not question:
            raise ValueError("every vlm_preference criterion needs a name and a question")
        if weight < 0:
            raise ValueError(f"vlm_preference criterion {name!r} has a negative weight")
        if answer not in ANSWER_SUFFIX:
            raise ValueError(
                f"vlm_preference criterion {name!r} answer must be one of {sorted(ANSWER_SUFFIX)}"
            )
        crops = int(raw.get("crops", 0) or 0)
        if crops < 0 or crops == 1:
            raise ValueError(f"vlm_preference criterion {name!r} crops must be 0 or >= 2")
        return {
            "name": name,
            "question": question,
            "weight": weight,
            "answer": answer,
            "crops": crops,
        }

    def _load_module(self) -> Any:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self._judge_model,
            revision=self._judge_revision,
            local_files_only=self._local_files_only,
        )
        # Batched questions share one image; left padding keeps every row's
        # last position at the answer slot.
        processor.tokenizer.padding_side = "left"
        judge = AutoModelForImageTextToText.from_pretrained(
            self._judge_model,
            revision=self._judge_revision,
            local_files_only=self._local_files_only,
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        tokenizer = processor.tokenizer

        def first_tokens(word: str) -> list[int]:
            variants = (word, word.capitalize(), " " + word, " " + word.capitalize())
            return sorted({tokenizer.encode(v, add_special_tokens=False)[0] for v in variants})

        # answer format -> list of (token ids, value) options
        answers = {
            "yesno": [(first_tokens("yes"), 1.0), (first_tokens("no"), 0.0)],
            "rating": [(first_tokens(word), value) for word, value in RATING_LEVELS],
        }
        return {"judge": judge, "processor": processor, "answers": answers}

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        image = artifact_middle_frame_image(artifact)
        reference = None
        if self._reference_key:
            from PIL import Image

            from vrl.utils.artifacts import default_data_root, resolve_artifact_path

            raw = artifact.metadata.get(self._reference_key)
            if not raw:
                raise ValueError(
                    f"vlm_preference pairwise mode needs metadata.{self._reference_key}"
                )
            path = resolve_artifact_path(
                str(raw), data_root=self.data_root or default_data_root(), allow_absolute=True
            )
            with Image.open(path) as opened:
                reference = opened.convert("RGB")
        return self.score(image, prompt=artifact.prompt, reference=reference)

    def score(self, image: Any, *, prompt: str, reference: Any = None) -> dict[str, float]:
        """Weighted mean of the per-criterion scores, plus every criterion.

        With ``reference`` (pairwise mode) each score is P(candidate better than
        reference), averaged over both image orders when ``both_orders`` is set.
        """

        questions = [c["question"].replace("{prompt}", prompt.strip()) for c in self.criteria]
        answers = [c["answer"] for c in self.criteria]
        if reference is None:
            values = self.answer_scores(image, questions, answers)
        else:
            preamble = PAIRWISE_PREAMBLE.replace("{prompt}", prompt.strip())
            questions = [preamble + q for q in questions]
            values = [0.0] * len(self.criteria)
            # Criteria with ``crops: n`` are judged on an n x n grid of native-resolution
            # crops (k-th crop of the candidate vs k-th crop of the reference) and
            # averaged, so fine texture reaches the judge at full scale; the rest see
            # the whole images.
            by_grid: dict[int, list[int]] = {}
            for i, c in enumerate(self.criteria):
                by_grid.setdefault(int(c.get("crops", 0)), []).append(i)
            for grid, idx in by_grid.items():
                qs = [questions[i] for i in idx]
                ans = [answers[i] for i in idx]
                views = (
                    [(image, reference)]
                    if grid <= 1
                    else list(
                        zip(_grid_crops(image, grid), _grid_crops(reference, grid), strict=True)
                    )
                )
                acc = [0.0] * len(idx)
                for cand, ref in views:
                    forward = self.answer_scores(cand, qs, ans, reference=ref)
                    if self._both_orders:
                        backward = self.answer_scores(ref, qs, ans, reference=cand)
                        vals = [
                            (f + (1.0 - b)) / 2.0 for f, b in zip(forward, backward, strict=True)
                        ]
                    else:
                        vals = forward
                    acc = [a + v for a, v in zip(acc, vals, strict=True)]
                for j, i in enumerate(idx):
                    values[i] = acc[j] / len(views)
        out = {
            "vlm_preference": float(sum(w * v for w, v in zip(self._weights, values, strict=True)))
        }
        for c, v in zip(self.criteria, values, strict=True):
            out[f"vlm_preference/{c['name']}"] = float(v)
        return out

    def answer_scores(
        self,
        image: Any,
        questions: Sequence[str],
        answers: Sequence[str],
        reference: Any = None,
    ) -> list[float]:
        """Expected answer value in [0, 1] per question, read off one batched forward.

        For ``yesno`` this is P(yes) renormalized over {yes, no}; for ``rating`` the
        probability-weighted mean of the level values, renormalized over the levels.
        """

        import torch
        from PIL import Image

        module = self._module_for_inference()

        def fit(im: Any) -> Any:
            scale = self._judge_max_side / max(im.size)
            if scale < 1:
                im = im.resize(
                    (round(im.width * scale), round(im.height * scale)), Image.Resampling.BICUBIC
                )
            return im

        images = [fit(image)] + ([fit(reference)] if reference is not None else [])
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": im} for im in images]
                    + [{"type": "text", "text": question + ANSWER_SUFFIX[answer]}],
                }
            ]
            for question, answer in zip(questions, answers, strict=True)
        ]
        inputs = (
            module["processor"]
            .apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            .to(self.device)
        )
        with torch.inference_mode():
            logits = module["judge"](**inputs).logits[:, -1].float()
        probs = torch.softmax(logits, dim=-1)
        values: list[float] = []
        for row, answer in zip(probs, answers, strict=True):
            mass = [(float(row[ids].sum()), value) for ids, value in module["answers"][answer]]
            total = sum(p for p, _ in mass)
            values.append(sum(p * v for p, v in mass) / max(total, 1e-9))
        return values


__all__ = [
    "DEFAULT_CRITERIA",
    "DEFAULT_PAIRWISE_CRITERIA",
    "RATING_LEVELS",
    "VLMPreferenceRewardModel",
]
