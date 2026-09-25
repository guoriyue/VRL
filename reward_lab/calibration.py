"""Preference pairs, frozen-combination fitting, and blinded review packets.

Pairwise logistic fitting uses calibration-split labels only; holdout
evaluation never refits scales or weights. These preferences supervise a
combination of pointwise scores, not a new image-to-image pairwise judge. The
frozen combination itself lives in ``vrl.rewards.calibration`` so training
applies exactly the arithmetic fitted here.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from vrl.config.base import ConfigBase
from vrl.rewards.calibration import COMBINATION_SCHEMA, FrozenRewardCombination
from vrl.rewards.evaluation import Evaluation
from vrl.utils.json_files import canonical_json_sha256, write_json
from vrl.utils.score_statistics import bootstrap_mean_interval

REVIEW_SCHEMA = "vrl.preference-review.v1"
REVIEW_MEDIA = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".mp4": "video",
    ".webm": "video",
}
FIT_ITERATIONS = 100


class PreferencePair(ConfigBase):
    """One annotation; repeated annotations retain distinct pair IDs."""

    pair_id: str = Field(min_length=1)
    left: str = Field(min_length=1)
    right: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    split: Literal["calibration", "holdout"]
    preference: Literal["left", "right", "tie", "unsure"]
    dimension: str = Field(default="overall", min_length=1)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def load_jsonl(cls, path: Path) -> list[PreferencePair]:
        pairs = [
            cls.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if not pairs or len({pair.pair_id for pair in pairs}) != len(pairs):
            raise ValueError("preference pair IDs must be non-empty and unique")
        return pairs

    @classmethod
    def from_review(cls, review: dict[str, Any], answers: dict[str, Any]) -> list[PreferencePair]:
        """Map the explicitly answered A/B choices of a review packet back to sample IDs."""

        if review.get("schema") != REVIEW_SCHEMA:
            raise ValueError("unsupported review schema")
        if answers.get("review_id") != review["review_id"]:
            raise ValueError("answers belong to a different review")
        given = answers.get("answers")
        if not isinstance(given, dict) or not given:
            raise ValueError("review has no explicit answers")
        pairs = []
        for key, answer in given.items():
            if key not in review["mapping"] or answer not in ("a", "b", "tie", "unsure"):
                raise ValueError("unknown review pair or answer")
            entry = review["mapping"][key]
            preference = answer
            if answer in ("a", "b"):
                preference = "left" if (answer == "a") != entry["reversed"] else "right"
            pairs.append(cls(**entry["pair"], preference=preference))
        return pairs


class Calibration:
    """Fit, apply and evaluate a frozen combination over one scoring run."""

    def __init__(self, evaluation: Evaluation) -> None:
        self.evaluation = evaluation

    def validate(self, pairs: list[PreferencePair]) -> None:
        """Reject source, prompt, sample and media leakage across declared splits."""

        assignments: dict[tuple[str, str], str] = {}
        sources: dict[str, str] = {}
        records = self.evaluation.records
        if len({pair.pair_id for pair in pairs}) != len(pairs):
            raise ValueError("duplicate preference pair IDs")
        for pair in pairs:
            if pair.left == pair.right:
                raise ValueError("preference cannot compare a sample with itself")
            prompt_ids = set()
            identities = [("source", pair.source_group)]
            for sample_id in (pair.left, pair.right):
                if sample_id not in records or records[sample_id]["status"] != "success":
                    raise ValueError(f"preference references an unscored sample: {sample_id}")
                row = records[sample_id]["input"]
                prompt_ids.add(row["prompt_id"])
                identities.extend((("sample", sample_id), ("prompt", row["prompt_id"])))
                if row.get("sha256"):
                    identities.append(("media", row["sha256"]))
                identities.extend(("asset", v) for v in row.get("asset_sha256", {}).values())
                if sources.setdefault(sample_id, pair.source_group) != pair.source_group:
                    raise ValueError("one sample cannot belong to multiple source groups")
            if len(prompt_ids) != 1:
                raise ValueError("preference pairs must compare samples from the same prompt")
            for identity in identities:
                if assignments.setdefault(identity, pair.split) != pair.split:
                    raise ValueError(
                        f"calibration/holdout leakage through {identity[0]}: {identity[1]}"
                    )

    def fit(
        self,
        pairs: list[PreferencePair],
        *,
        axes: list[str],
        dimension: str = "overall",
        l2: float = 0.1,
        tie_margin: float = 0.1,
    ) -> dict[str, Any]:
        """Fit a convex L2 logistic preference loss with equal source-group weight.

        Axes are standardized on unique calibration samples only. A constant
        axis gets unit scale and zero information. Ties target 0.5; unsure
        labels are excluded. A positive regularizer keeps the Hessian
        invertible with redundant axes. Non-convergence raises.
        """

        self.validate(pairs)
        if not axes or any(not axis for axis in axes) or len(set(axes)) != len(axes):
            raise ValueError("axes must be non-empty and unique")
        if not math.isfinite(l2) or l2 <= 0 or not math.isfinite(tie_margin) or tie_margin < 0:
            raise ValueError("l2 must be positive and tie_margin non-negative, both finite")
        selected = [
            p
            for p in pairs
            if p.split == "calibration" and p.dimension == dimension and p.preference != "unsure"
        ]
        if len({p.source_group for p in selected}) < 2 or not any(
            p.preference in {"left", "right"} for p in selected
        ):
            raise ValueError(
                "fitting needs two source groups and at least one decisive calibration label"
            )
        records = self.evaluation.records
        ids = sorted({sample_id for p in selected for sample_id in (p.left, p.right)})
        try:
            raw = np.asarray(
                [
                    [records[sample_id]["result"]["scores"][axis] for axis in axes]
                    for sample_id in ids
                ],
                dtype=np.float64,
            )
        except KeyError as error:
            raise ValueError("calibration score axis missing") from error
        if not np.isfinite(raw).all():
            raise ValueError("calibration scores must be finite")
        means, scales = raw.mean(axis=0), raw.std(axis=0)
        scales = np.where(scales > 0, scales, 1.0)
        features = dict(zip(ids, (raw - means) / scales, strict=True))
        x = np.asarray([features[p.left] - features[p.right] for p in selected])
        y = np.asarray(
            [
                1.0 if p.preference == "left" else 0.0 if p.preference == "right" else 0.5
                for p in selected
            ]
        )
        counts = Counter(p.source_group for p in selected)
        sample_weights = np.asarray(
            [1.0 / (len(counts) * counts[p.source_group]) for p in selected]
        )
        weights = np.zeros(len(axes), dtype=np.float64)

        def objective(candidate: np.ndarray) -> float:
            logits = x @ candidate
            return float(
                sample_weights @ (np.logaddexp(0, logits) - y * logits)
                + l2 * (candidate @ candidate) / 2
            )

        for _iteration in range(FIT_ITERATIONS):
            logits = x @ weights
            probability = np.exp(-np.logaddexp(0, -logits))
            gradient = x.T @ (sample_weights * (probability - y)) + l2 * weights
            if np.linalg.norm(gradient, ord=np.inf) < 1e-8:
                break
            hessian = x.T @ (
                (sample_weights * probability * (1 - probability))[:, None] * x
            ) + l2 * np.eye(len(axes))
            step = np.linalg.solve(hessian, gradient)
            rate = 1.0
            current = objective(weights)
            while rate > 1e-10 and objective(
                weights - rate * step
            ) > current - 1e-4 * rate * float(gradient @ step):
                rate *= 0.5
            if rate <= 1e-10:
                raise RuntimeError("calibration line search failed")
            weights -= rate * step
        else:
            raise RuntimeError("calibration did not converge")
        payload = {
            "schema": COMBINATION_SCHEMA,
            "scoring_config_hash": canonical_json_sha256(self.evaluation.config, allow_nan=False),
            "fit_run_id": self.evaluation.run_id,
            "dimension": dimension,
            "axes": axes,
            "means": means.tolist(),
            "scales": scales.tolist(),
            "weights": weights.tolist(),
            "raw_weights": (weights / scales).tolist(),
            "raw_bias": float(-(means * weights / scales).sum()),
            "tie_margin": tie_margin,
            "l2": l2,
            "calibration_source_groups": sorted(counts),
            "calibration_prompt_ids": sorted({records[key]["input"]["prompt_id"] for key in ids}),
            "fit_pairs": len(selected),
            "fit_samples": len(ids),
            "iterations": _iteration,
            "objective": objective(weights),
        }
        return {"combination_id": canonical_json_sha256(payload, allow_nan=False), **payload}

    def apply(self, combination: dict[str, Any]) -> dict[str, Any]:
        """Apply frozen coefficients to this run's raw scores, without labels or refitting.

        Failed or absent source measurements remain unscored. This is a derived
        report, not a replacement raw evaluation or proof of preference accuracy.
        """

        frozen = FrozenRewardCombination(combination, scoring_config=self.evaluation.config)
        records = {}
        for sample_id, row in self.evaluation.records.items():
            result = {"input": row["input"], "status": row["status"]}
            if row["status"] == "success":
                score, contributions = frozen.apply(row["result"]["scores"], sample_id=sample_id)
                result.update(
                    score=score, contributions=contributions, source_result=row["result"]
                )
            elif row["status"] == "error":
                result["error"] = row["error"]
            elif row["status"] != "missing":
                raise ValueError(f"invalid source status: {sample_id}")
            records[sample_id] = result
        return {
            "schema": "vrl.reward-combination-application.v1",
            "combination_id": combination["combination_id"],
            "evaluation_run_id": self.evaluation.run_id,
            "dimension": frozen.dimension,
            "status_counts": dict(Counter(row["status"] for row in records.values())),
            "records": records,
        }

    def evaluate(self, pairs: list[PreferencePair], combination: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the frozen combination on holdout labels with explicit three-way ties."""

        frozen = FrozenRewardCombination(combination, scoring_config=self.evaluation.config)
        self.validate(pairs)
        selected = [p for p in pairs if p.split == "holdout" and p.dimension == frozen.dimension]
        if not selected:
            raise ValueError("holdout has no labels for this dimension")
        records = self.evaluation.records
        outcomes, grouped = [], defaultdict(list)
        tags: dict[str, list[float]] = defaultdict(list)
        for pair in selected:
            if pair.source_group in combination["calibration_source_groups"]:
                raise ValueError("holdout source group was used for calibration")
            scores = []
            for sample_id in (pair.left, pair.right):
                row = records[sample_id]
                if row["input"]["prompt_id"] in combination["calibration_prompt_ids"]:
                    raise ValueError("holdout prompt was used for calibration")
                score, _ = frozen.apply(row["result"]["scores"], sample_id=sample_id)
                scores.append(score)
            delta = scores[0] - scores[1]
            if not math.isfinite(delta):
                raise ValueError("combination difference is nonfinite")
            prediction = (
                "tie"
                if abs(delta) <= combination["tie_margin"]
                else "left"
                if delta > 0
                else "right"
            )
            correct = None if pair.preference == "unsure" else float(prediction == pair.preference)
            outcomes.append(
                {
                    **pair.model_dump(mode="json"),
                    "delta": delta,
                    "prediction": prediction,
                    "correct": correct,
                }
            )
            if correct is not None:
                grouped[pair.source_group].append(correct)
                for tag in set(pair.tags):
                    tags[tag].append(correct)
        group_means = [statistics.fmean(values) for values in grouped.values()]
        return {
            "combination_id": combination["combination_id"],
            "evaluation_run_id": self.evaluation.run_id,
            "source_balanced_agreement": statistics.fmean(group_means) if group_means else None,
            "bootstrap_95ci": list(
                bootstrap_mean_interval(
                    group_means,
                    schema="vrl.reward-preferences.v1",
                    label=combination["combination_id"],
                    score_key=frozen.dimension,
                )
            )
            if len(group_means) >= 2
            else None,
            "judged_source_groups": len(group_means),
            "unsure_count": sum(p.preference == "unsure" for p in selected),
            "by_tag": {
                tag: {"annotations": len(values), "agreement": statistics.fmean(values)}
                for tag, values in tags.items()
            },
            "outcomes": outcomes,
            "tie_policy": "Three-way exact agreement; unsure excluded; tie margin frozen at fit.",
        }

    def review_packet(
        self, pairs: list[dict[str, Any]], output: Path, *, seed: int
    ) -> dict[str, Any]:
        """Write a portable local HTML review with randomized order and A/B sides.

        Pair selection and split assignment are operator inputs; the packet
        copies the original media and never derives labels from scores. The
        audit manifest binds display order to sample identities. This is
        presentation blinding, not access control against inspecting files.
        """

        if type(seed) is not int or not pairs:
            raise ValueError("review needs an integer seed and non-empty pair specifications")
        if any("preference" in pair for pair in pairs):
            raise ValueError("review pair specifications must not contain preference labels")
        # Reuse the leakage gate without emitting placeholder labels.
        validated = [PreferencePair(**pair, preference="unsure") for pair in pairs]
        self.validate(validated)
        if output.exists():
            raise FileExistsError(output)
        rng = random.Random(seed)
        rng.shuffle(validated)
        records = self.evaluation.records
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        (stage / "media").mkdir()
        copied: dict[tuple[str, str], dict[str, str]] = {}

        def copy_media(path: str, digest: str) -> dict[str, str]:
            source = Path(path)
            extension = source.suffix.lower()
            if extension not in REVIEW_MEDIA:
                raise ValueError(f"unsupported browser review media extension: {extension}")
            key = (digest, extension)
            if key not in copied:
                relative = f"media/{len(copied):06d}{extension}"
                shutil.copyfile(source, stage / relative)
                copied[key] = {"path": relative, "kind": REVIEW_MEDIA[extension]}
            return copied[key]

        try:
            display, mapping = [], {}
            for index, pair in enumerate(validated):
                left = records[pair.left]["input"]
                right = records[pair.right]["input"]
                if left["prompt"] != right["prompt"]:
                    raise ValueError("review pairs must share exact prompt text")
                if left.get("assets", {}) != right.get("assets", {}) or left.get(
                    "asset_sha256", {}
                ) != right.get("asset_sha256", {}):
                    raise ValueError("review pairs must share auxiliary inputs")
                first, second = (
                    copy_media(item["path"], item["sha256"]) for item in (left, right)
                )
                reverse = bool(rng.getrandbits(1))
                if reverse:
                    first, second = second, first
                review_key = f"pair-{index:06d}"
                reference = left.get("assets", {}).get("reference_image")
                display.append(
                    {
                        "key": review_key,
                        "prompt": left["prompt"],
                        "dimension": pair.dimension,
                        "a": first,
                        "b": second,
                        "reference": copy_media(reference, left["asset_sha256"]["reference_image"])
                        if reference
                        else None,
                    }
                )
                mapping[review_key] = {
                    "pair": pair.model_dump(mode="json", exclude={"preference"}),
                    "reversed": reverse,
                    "left_sha256": left["sha256"],
                    "right_sha256": right["sha256"],
                    "asset_sha256": left.get("asset_sha256", {}),
                }
            manifest = {
                "schema": REVIEW_SCHEMA,
                "run_id": self.evaluation.run_id,
                "seed": seed,
                "mapping": mapping,
                "display": display,
            }
            manifest = {"review_id": canonical_json_sha256(manifest, allow_nan=False), **manifest}
            write_json(stage / "audit.json", manifest)
            public = json.dumps(
                {"review_id": manifest["review_id"], "pairs": display}, ensure_ascii=True
            ).replace("<", "\\u003c")
            template = files("reward_lab").joinpath("preference_review.html").read_text()
            (stage / "index.html").write_text(template.replace("__REVIEW_DATA__", public))
            stage.rename(output)
            return {
                "review_id": manifest["review_id"],
                "pairs": len(display),
                "output": str(output),
            }
        finally:
            if stage.exists():
                shutil.rmtree(stage)


__all__ = ["REVIEW_SCHEMA", "Calibration", "PreferencePair"]
