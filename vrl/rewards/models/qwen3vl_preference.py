"""Qwen3-VL preference reward: per-criterion answer probabilities from a Qwen3-VL judge.

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

Scores: ``qwen3vl_preference`` (weighted mean in [0, 1]) plus
``qwen3vl_preference/<criterion>`` for every criterion. Group-normalized GRPO
advantages make the absolute scale irrelevant; the per-criterion keys are for
diagnostics and for the reward card's blind-spot bookkeeping.

``{prompt}`` in a question is replaced by the artifact's prompt. ``{intent}`` is
replaced by ``metadata[<intent_key>]`` when ``intent_key`` is configured and the
artifact carries it, otherwise by the prompt: an expanded reading of a terse or
idiomatic prompt ("gold jaguar print hoodie" -> a hoodie with a gold leopard-spot
animal-print pattern) that fidelity questions can judge against while the policy
still sees the raw prompt.
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
# Graded pairwise answer: how Image 1 compares with Image 2. Against a much
# stronger reference a yes/no "is Image 1 better" sits at P(yes) ~ 0 for every
# sibling of a group and gives GRPO nothing to rank; the expectation over these
# levels still separates "much worse" from "slightly worse".
COMPARE_LEVELS: tuple[tuple[str, float], ...] = (
    ("better", 1.0),
    ("slightly", 0.75),
    ("similar", 0.5),
    ("worse", 0.25),
    ("much", 0.0),
)
# Symmetric pairwise choice: P(Image 1) read off the two single-token letters,
# so neither end of the scale is privileged and the two image orders cancel.
AB_LEVELS: tuple[tuple[str, float], ...] = (("A", 1.0), ("B", 0.0))
# Absolute 1-5 score read as the expectation over the digit tokens.
DIGIT_LEVELS: tuple[tuple[str, float], ...] = tuple((str(i), (i - 1) / 4) for i in range(1, 6))
ANSWER_SUFFIX = {
    "yesno": "\nAnswer with a single word: yes or no.",
    "rating": "\nAnswer with a single word: excellent, good, fair, poor, or bad.",
    "compare": (
        "\nAnswer with one word for how Image 1 compares with Image 2 on this point: "
        "better (clearly better), slightly (slightly better), similar, worse (slightly worse), "
        "or much (much worse)."
    ),
    "ab": "\nImage 1 is A and Image 2 is B. Answer with a single letter: A or B.",
    "digit": "\nAnswer with a single digit from 1 (worst) to 5 (best).",
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


class Qwen3VLPreferenceRewardModel(LazyTorchModule):
    """Score one image with per-criterion questions to a Qwen3-VL judge."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(worker_config)
        self.device = str(cfg.get("device", "cuda"))
        self._judge_model = str(cfg.get("judge_model", "Qwen/Qwen3-VL-32B-Instruct-FP8"))
        self._judge_revision = cfg.get("judge_revision")
        self._judge_max_side = int(cfg.get("judge_max_side", 1024))
        # Upper bound on criterion questions batched into one forward. All
        # questions of an image share one forward by default; on a 44 GiB card
        # five pairwise questions at 896 px overflow, so the service caps it.
        self._questions_per_forward = int(cfg.get("questions_per_forward", 0))
        if self._questions_per_forward < 0:
            raise ValueError(
                "qwen3vl_preference questions_per_forward must be >= 0 (0 = unbounded)"
            )
        self._local_files_only = bool(cfg.get("local_files_only", False))
        # Pairwise mode: compare the candidate against a reference image whose
        # path the artifact carries under ``metadata[<reference_key>]``
        # (resolved under ``data_root``); ``both_orders`` asks each question in
        # both image orders and averages, cancelling position bias.
        self._reference_key = str(cfg.get("reference_key") or "").strip() or None
        self._intent_key = str(cfg.get("intent_key") or "").strip() or None
        # Checklist criteria read their items (short phrases naming what the
        # prompt requires: objects, actions, attributes) from this metadata key.
        self._checklist_key = str(cfg.get("checklist_key") or "").strip() or None
        self._both_orders = bool(cfg.get("both_orders", True))
        self.data_root = str(cfg.get("data_root") or "")
        raw = cfg.get("criteria") or (
            DEFAULT_PAIRWISE_CRITERIA if self._reference_key else DEFAULT_CRITERIA
        )
        self.criteria = [self._validate_criterion(c) for c in raw]
        if not self.criteria:
            raise ValueError("qwen3vl_preference needs at least one criterion")
        names = [c["name"] for c in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError(f"qwen3vl_preference criteria names must be unique: {names}")
        total = sum(c["weight"] for c in self.criteria)
        if total <= 0:
            raise ValueError("qwen3vl_preference criteria weights must sum to a positive number")
        self._weights = [c["weight"] / total for c in self.criteria]

    @staticmethod
    def _validate_criterion(raw: Mapping[str, Any]) -> dict[str, Any]:
        name = str(raw.get("name") or "").strip()
        question = str(raw.get("question") or "").strip()
        weight = float(raw.get("weight", 1.0))
        answer = str(raw.get("answer", "yesno"))
        if not name or not question:
            raise ValueError("every qwen3vl_preference criterion needs a name and a question")
        if weight < 0:
            raise ValueError(f"qwen3vl_preference criterion {name!r} has a negative weight")
        if answer not in ANSWER_SUFFIX:
            raise ValueError(
                f"qwen3vl_preference criterion {name!r} answer must be one of {sorted(ANSWER_SUFFIX)}"
            )
        crops = int(raw.get("crops", 0) or 0)
        if crops < 0 or crops == 1:
            raise ValueError(f"qwen3vl_preference criterion {name!r} crops must be 0 or >= 2")
        checklist = bool(raw.get("checklist", False))
        if checklist and ("{item}" not in question or answer != "yesno" or crops):
            raise ValueError(
                f"qwen3vl_preference checklist criterion {name!r} needs an {{item}} placeholder, "
                "answer: yesno and no crops"
            )
        # ``absolute``: asked about the candidate alone even in pairwise mode;
        # ``invert``: the question names a defect, so the score is 1 - P(yes).
        absolute = bool(raw.get("absolute", False)) or checklist
        invert = bool(raw.get("invert", False))
        if invert and (answer != "yesno" or not absolute):
            raise ValueError(
                f"qwen3vl_preference criterion {name!r}: invert needs answer: yesno and absolute: true"
            )
        if absolute and crops:
            raise ValueError(f"qwen3vl_preference criterion {name!r}: absolute takes no crops")
        # ``group``: scored by comparing siblings of one prompt group against each
        # other (Image 1 = A, Image 2 = B, Image 3 = the reference); only
        # ``score_batch`` can fill it, a lone ``__call__`` scores it 0.5.
        group = bool(raw.get("group", False))
        if group and (answer != "ab" or absolute or crops):
            raise ValueError(
                f"qwen3vl_preference group criterion {name!r} needs answer: ab, no absolute, no crops"
            )
        return {
            "name": name,
            "question": question,
            "weight": weight,
            "answer": answer,
            "crops": crops,
            "checklist": checklist,
            "absolute": absolute,
            "invert": invert,
            "group": group,
        }

    def _load_module(self) -> Any:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self._judge_model,
            revision=self._judge_revision,
            local_files_only=self._local_files_only,
        )
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
            "compare": [(first_tokens(word), value) for word, value in COMPARE_LEVELS],
            "ab": [(first_tokens(word), value) for word, value in AB_LEVELS],
            "digit": [(first_tokens(word), value) for word, value in DIGIT_LEVELS],
        }
        return {"judge": judge, "processor": processor, "answers": answers}

    def _resolve(self, artifact: RewardInferenceArtifact) -> tuple[Any, Any, str, dict[str, Any]]:
        """Candidate image, reference image (or None), reference path, score kwargs."""

        image = artifact_middle_frame_image(artifact)
        reference, ref_path = None, ""
        if self._reference_key:
            from PIL import Image

            from vrl.utils.paths import RootedPaths

            raw = artifact.metadata.get(self._reference_key)
            if not raw:
                raise ValueError(
                    f"qwen3vl_preference pairwise mode needs metadata.{self._reference_key}"
                )
            path = RootedPaths.data_root(self.data_root or None, allow_absolute=True).resolve(
                str(raw)
            )
            ref_path = str(path)
            with Image.open(path) as opened:
                reference = opened.convert("RGB")
        intent = artifact.metadata.get(self._intent_key) if self._intent_key else None
        checklist = artifact.metadata.get(self._checklist_key) if self._checklist_key else None
        kwargs = {
            "prompt": artifact.prompt,
            "reference": reference,
            "intent": str(intent) if intent else None,
            "checklist": [str(item) for item in checklist or () if str(item).strip()],
        }
        return image, reference, ref_path, kwargs

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        image, _, _, kwargs = self._resolve(artifact)
        return self.score(image, **kwargs)

    def score_batch(self, artifacts: Sequence[RewardInferenceArtifact]) -> list[dict[str, float]]:
        """Score a batch; ``group`` criteria compare siblings sharing prompt and reference.

        Siblings are compared on a ring schedule (each sample against the next and,
        for groups of more than three, the third-next), both image orders, with the
        reference shown as Image 3. A sample's score is its mean win probability
        over its comparisons; a sample with no sibling scores 0.5.
        """

        resolved = [self._resolve(a) for a in artifacts]
        values = [self._criterion_values(image, **kwargs) for image, _, _, kwargs in resolved]
        group_idx = [i for i, c in enumerate(self.criteria) if c["group"]]
        if group_idx:
            clusters: dict[tuple[str, str], list[int]] = {}
            for k, (artifact, (_, _, ref_path, _)) in enumerate(
                zip(artifacts, resolved, strict=True)
            ):
                clusters.setdefault((artifact.prompt.strip(), ref_path), []).append(k)
            for (prompt, _), members in clusters.items():
                for ci in group_idx:
                    wins = self._group_wins(
                        ci, prompt, [resolved[k][0] for k in members], resolved[members[0]][1]
                    )
                    for k, w in zip(members, wins, strict=True):
                        values[k][ci] = w
        return [self._compose(v) for v in values]

    def _group_wins(
        self, criterion: int, prompt: str, images: list[Any], reference: Any
    ) -> list[float]:
        n = len(images)
        if n < 2:
            return [0.5] * n
        offsets = [1, 3] if n > 3 else [1]
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for o in offsets:
            for i in range(n):
                j = (i + o) % n
                key = (min(i, j), max(i, j))
                if i != j and key not in seen:
                    seen.add(key)
                    pairs.append((i, j))
        question = self.criteria[criterion]["question"].replace("{prompt}", prompt)
        totals = [0.0] * n
        counts = [0] * n
        for i, j in pairs:
            refs = [images[j]] + ([reference] if reference is not None else [])
            forward = self.answer_scores(images[i], [question], ["ab"], reference=refs)[0]
            if self._both_orders:
                refs_b = [images[i]] + ([reference] if reference is not None else [])
                backward = self.answer_scores(images[j], [question], ["ab"], reference=refs_b)[0]
                p_i = (forward + (1.0 - backward)) / 2.0
            else:
                p_i = forward
            totals[i] += p_i
            totals[j] += 1.0 - p_i
            counts[i] += 1
            counts[j] += 1
        return [t / c if c else 0.5 for t, c in zip(totals, counts, strict=True)]

    def _compose(self, values: Sequence[float]) -> dict[str, float]:
        out = {
            "qwen3vl_preference": float(
                sum(w * v for w, v in zip(self._weights, values, strict=True))
            )
        }
        for c, v in zip(self.criteria, values, strict=True):
            out[f"qwen3vl_preference/{c['name']}"] = float(v)
        return out

    def score(
        self,
        image: Any,
        *,
        prompt: str,
        reference: Any = None,
        intent: str | None = None,
        checklist: Sequence[str] = (),
    ) -> dict[str, float]:
        """Weighted mean of the per-criterion scores, plus every criterion.

        With ``reference`` (pairwise mode) each score is P(candidate better than
        reference), averaged over both image orders when ``both_orders`` is set.
        ``intent`` fills ``{intent}`` in the questions; it defaults to the prompt.

        A ``checklist`` criterion is absolute, never pairwise: its question is asked
        once per ``checklist`` item about the candidate image alone and the score is
        the mean P(yes). Without items it scores 0.5 (a constant within a prompt
        group, so it adds no advantage). ``group`` criteria score 0.5 here; only
        :meth:`score_batch` can compare siblings.
        """

        return self._compose(
            self._criterion_values(
                image, prompt=prompt, reference=reference, intent=intent, checklist=checklist
            )
        )

    def _criterion_values(
        self,
        image: Any,
        *,
        prompt: str,
        reference: Any = None,
        intent: str | None = None,
        checklist: Sequence[str] = (),
    ) -> list[float]:
        prompt = prompt.strip()
        intent = (intent or prompt).strip()
        questions = [
            c["question"].replace("{prompt}", prompt).replace("{intent}", intent)
            for c in self.criteria
        ]
        answers = [c["answer"] for c in self.criteria]
        values = [0.5 if c["group"] else 0.0 for c in self.criteria]
        checks = [i for i, c in enumerate(self.criteria) if c["checklist"]]
        for i in checks:
            items = [item.strip() for item in checklist if item.strip()]
            if not items:
                values[i] = 0.5
                continue
            per_item = self.answer_scores(
                image,
                [questions[i].replace("{item}", item) for item in items],
                ["yesno"] * len(items),
            )
            values[i] = float(sum(per_item) / len(per_item))
        absolutes = [
            i for i, c in enumerate(self.criteria) if c["absolute"] and not c["checklist"]
        ]
        if absolutes:
            # Defect questions about the candidate alone ("does it have extra
            # limbs / unrequested text?"); one forward for all of them.
            for i, v in zip(
                absolutes,
                self.answer_scores(
                    image, [questions[i] for i in absolutes], [answers[i] for i in absolutes]
                ),
                strict=True,
            ):
                values[i] = 1.0 - v if self.criteria[i]["invert"] else v
        plain = [
            i
            for i, c in enumerate(self.criteria)
            if i not in checks and i not in absolutes and not c["group"]
        ]
        if reference is None:
            if plain:
                for i, v in zip(
                    plain,
                    self.answer_scores(
                        image, [questions[i] for i in plain], [answers[i] for i in plain]
                    ),
                    strict=True,
                ):
                    values[i] = v
        else:
            preamble = PAIRWISE_PREAMBLE.replace("{prompt}", prompt)
            questions = [preamble + q for q in questions]
            # Criteria with ``crops: n`` are judged on an n x n grid of native-resolution
            # crops (k-th crop of the candidate vs k-th crop of the reference) and
            # averaged, so fine texture reaches the judge at full scale; the rest see
            # the whole images.
            by_grid: dict[int, list[int]] = {}
            for i in plain:
                by_grid.setdefault(int(self.criteria[i].get("crops", 0)), []).append(i)
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
        return values

    def answer_scores(
        self,
        image: Any,
        questions: Sequence[str],
        answers: Sequence[str],
        reference: Any = None,
    ) -> list[float]:
        """Expected answer value in [0, 1] per question, read off batched forwards.

        For ``yesno`` this is P(yes) renormalized over {yes, no}; for ``rating`` the
        probability-weighted mean of the level values, renormalized over the levels.
        Questions share one forward, in chunks of ``questions_per_forward`` when set.

        Every question row starts with the same images, so they are encoded once
        per forward chunk: one forward over the shared image prefix fills a KV
        cache, and one batched forward over the question suffixes attends to it. A row therefore sees the
        same tokens at the same positions as a full forward of its own, without
        the vision tower and the image-prefix tokens being recomputed per question.
        """

        limit = self._questions_per_forward
        if limit and len(questions) > limit:
            values: list[float] = []
            for start in range(0, len(questions), limit):
                values.extend(
                    self.answer_scores(
                        image,
                        questions[start : start + limit],
                        answers[start : start + limit],
                        reference,
                    )
                )
            return values

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

        # ``reference`` may be one image (Image 2) or a sequence (Image 2, 3, ...).
        refs = (
            []
            if reference is None
            else list(reference)
            if isinstance(reference, (list, tuple))
            else [reference]
        )
        images = [fit(image)] + [fit(r) for r in refs]
        processor, judge = module["processor"], module["judge"]
        texts = [
            processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": im} for im in images]
                        + [{"type": "text", "text": question + ANSWER_SUFFIX[answer]}],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question, answer in zip(questions, answers, strict=True)
        ]
        # The images precede the question text, so every row shares the text up
        # to the last vision-end marker; a special token, so splitting there
        # tokenizes both halves exactly as the whole row would be.
        split = texts[0].rindex("<|vision_end|>") + len("<|vision_end|>")
        prefix = processor(text=[texts[0][:split]], images=images, return_tensors="pt").to(
            self.device
        )
        suffix = processor.tokenizer(
            [text[split:] for text in texts],
            add_special_tokens=False,
            padding=True,
            padding_side="right",
            return_tensors="pt",
        ).to(self.device)
        rows, suffix_len = suffix["input_ids"].shape
        prefix_len = prefix["input_ids"].shape[1]
        with torch.inference_mode():
            cache = judge(**prefix, use_cache=True).past_key_values
            cache.batch_repeat_interleave(rows)
            # Text after the images continues M-RoPE as one shared position per
            # token: the row's index plus the prefix's rope delta, on all axes.
            positions = torch.arange(prefix_len, prefix_len + suffix_len, device=self.device)
            positions = positions.view(1, 1, -1).expand(3, rows, -1) + judge.model.rope_deltas
            logits = judge(
                input_ids=suffix["input_ids"],
                attention_mask=torch.cat(
                    [
                        suffix["attention_mask"].new_ones(rows, prefix_len),
                        suffix["attention_mask"],
                    ],
                    dim=1,
                ),
                past_key_values=cache,
                position_ids=positions,
            ).logits
            last = suffix["attention_mask"].sum(dim=1) - 1
            logits = logits[torch.arange(rows, device=self.device), last].float()
        probs = torch.softmax(logits, dim=-1)
        values: list[float] = []
        for row, answer in zip(probs, answers, strict=True):
            mass = [(float(row[ids].sum()), value) for ids, value in module["answers"][answer]]
            total = sum(p for p, _ in mass)
            values.append(sum(p * v for p, v in mass) / max(total, 1e-9))
        return values


__all__ = [
    "AB_LEVELS",
    "COMPARE_LEVELS",
    "DEFAULT_CRITERIA",
    "DEFAULT_PAIRWISE_CRITERIA",
    "DIGIT_LEVELS",
    "RATING_LEVELS",
    "Qwen3VLPreferenceRewardModel",
]
