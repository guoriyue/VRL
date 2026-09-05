"""Codex-CLI image-QA reward as a batch-capable RewardModel.

A rubric-driven LLM judge: write rollout images to temporary PNGs, render the
command/prompt placeholders, run the ``codex exec`` subprocess (timeout + error
handling), then parse the judge output. Absolute mode returns clamped ``[0, 1]``
scores. Reference-listwise mode compares one complete rollout group with its
frozen base anchor and returns categorical rewards in ``[-2, 2]``.
Binary-guard and exact-count modes require two order-mirrored passes before
returning a reward of one. Exact-count keeps the judge's observed integer
separate from the metadata-owned target so prose cannot satisfy the reward.

Restored 2026-08-22 (removed in 51c78968) and adapted to the current
score_batch/InProcessRewardScorer interface. Judge calls fan out over a thread
pool bounded by ``max_concurrency``.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from vrl.rewards.types import REWARD_GROUP_ID_METADATA_KEY
from vrl.utils.artifacts import resolve_artifact_path
from vrl.utils.media import write_png

DEFAULT_PROMPT_TEMPLATE = """You are a strict image-text alignment judge.
Evaluate whether the attached image matches the text prompt.
Return exactly one JSON object and no extra text:
{{"score": 0.37}}

Use a dense continuous score in [0, 1]. Do not collapse most samples to 0.
Reserve 0.0 only for blank, broken, or completely unrelated images.
Assign fine-grained decimals based on visible evidence; avoid repeated generic
scores such as 0.10, 0.12, or 0.50 when images differ.

Scoring rubric:
- 0.85-1.00: the image clearly matches the prompt.
- 0.60-0.84: the image mostly matches with minor missing details.
- 0.35-0.59: the image partially matches but misses important details.
- 0.10-0.34: the image is coherent but weakly related to the prompt.
- 0.00-0.09: the image is blank, broken, or completely unrelated.

Text prompt: {prompt}
"""

DEFAULT_GRID_PROMPT_TEMPLATE = """You are a strict anime image judge.
The attached image is a montage of {count} separate generations arranged in a
grid, each cell labeled with a number (1..{count}) in its top-left corner,
ordered left-to-right, top-to-bottom. Every cell was generated from the SAME
text prompt below.

Score EACH cell independently in [0, 1]. Use dense continuous decimals based on
visible evidence; do not collapse cells to the same value when they differ.

Return exactly one JSON object and no extra text, with {count} scores in cell
order:
{{"scores": [0.37, 0.81, ...]}}

Scoring rubric:
- 0.85-1.00: clear, high-quality anime that matches the prompt.
- 0.60-0.84: mostly good with minor issues.
- 0.35-0.59: partial match or noticeable quality problems.
- 0.10-0.34: coherent but weak.
- 0.00-0.09: blank, broken, or unrelated.

Text prompt: {prompt}
"""


class _ReferenceIntegrity(StrEnum):
    PASS = "pass"
    REGRESS = "regress"


class _ReferencePreference(StrEnum):
    STRONG_LOSS = "strong_loss"
    LOSS = "loss"
    TIE = "tie"
    WIN = "win"
    STRONG_WIN = "strong_win"

    @property
    def reward(self) -> float:
        match self:
            case _ReferencePreference.STRONG_LOSS:
                return -2.0
            case _ReferencePreference.LOSS:
                return -1.0
            case _ReferencePreference.TIE:
                return 0.0
            case _ReferencePreference.WIN:
                return 1.0
            case _ReferencePreference.STRONG_WIN:
                return 2.0

    @property
    def is_win(self) -> bool:
        return self in {
            _ReferencePreference.WIN,
            _ReferencePreference.STRONG_WIN,
        }


@dataclass(frozen=True, slots=True)
class _ReferenceVerdict:
    candidate_id: str
    integrity: _ReferenceIntegrity
    preference: _ReferencePreference


class _BinaryGuardDecision(StrEnum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class _BinaryGuardVerdict:
    candidate_id: str
    carrier_grounding: _BinaryGuardDecision
    image_integrity: _BinaryGuardDecision

    @property
    def passes(self) -> bool:
        return (
            self.carrier_grounding is _BinaryGuardDecision.PASS
            and self.image_integrity is _BinaryGuardDecision.PASS
        )


@dataclass(frozen=True, slots=True)
class _ExactCountVerdict:
    candidate_id: str
    observed_count: int
    unambiguous: bool


@dataclass(frozen=True, slots=True)
class _CandidateGroup:
    group_id: str
    prompt: str
    candidate_indices: tuple[int, ...]
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ReferenceGroup:
    group_id: str
    prompt: str
    reference_path: Path
    candidate_indices: tuple[int, ...]
    candidate_ids: tuple[str, ...]


class CodexImageQARewardModel:
    """RewardModel returning one ``codex_image_qa`` score per rollout artifact.

    The command may contain ``{image_path}``, ``{output_path}``, ``{prompt}``,
    and ``{output_schema_path}`` placeholders. The rendered prompt is sent to
    stdin by default. Native ``codex exec`` commands automatically receive the
    generated ``--output-schema`` argument; compatible commands can opt in with
    the schema-path placeholder. ``comparison_mode=reference_listwise`` adds a
    frozen base image to the judge montage but never to the returned rollout
    rows; its categorical rewards are therefore in ``[-2, 2]`` rather than the
    legacy absolute mode's ``[0, 1]``. ``comparison_mode=binary_guard`` scores
    complete rollout groups twice and accepts only order-invariant passes.
    ``comparison_mode=exact_count`` additionally requires the judge to report
    the observed integer; code, rather than the judge, compares it with the
    metadata-derived target.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        command = cfg.get("command")
        if command is None:
            raise ValueError("CodexImageQAReward requires reward.kwargs.codex_image_qa.command")
        self.command = _normalize_command(command)
        self.timeout_s = float(cfg.get("timeout_s", 300.0))
        self.prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        raw_prompt_metadata_key = cfg.get("prompt_metadata_key", "")
        if not isinstance(raw_prompt_metadata_key, str):
            raise TypeError("Codex image-QA prompt_metadata_key must be a string")
        self.prompt_metadata_key = raw_prompt_metadata_key.strip()
        self.pass_prompt_stdin = bool(cfg.get("pass_prompt_stdin", True))
        self.max_concurrency = max(1, int(cfg.get("max_concurrency", 1)))
        # Grid batching: pack up to N same-prompt images into one downscaled,
        # cell-numbered montage and score them in a SINGLE CLI call, cutting
        # calls and image tokens ~N x. 1 keeps the one-image-per-call behavior.
        self.images_per_call = max(1, int(cfg.get("images_per_call", 1)))
        self.tile_size = max(64, int(cfg.get("tile_size", 256)))
        self.grid_prompt_template = cfg.get("grid_prompt_template", DEFAULT_GRID_PROMPT_TEMPLATE)
        self.comparison_mode = str(cfg.get("comparison_mode", "absolute")).strip()
        if self.comparison_mode not in {
            "absolute",
            "reference_listwise",
            "binary_guard",
            "exact_count",
        }:
            raise ValueError(
                "Codex image-QA comparison_mode must be 'absolute', "
                "'reference_listwise', 'binary_guard', or 'exact_count', "
                f"got {self.comparison_mode!r}",
            )
        self.reference_data_root = str(cfg.get("reference_data_root", "")).strip()
        self.expected_group_size = int(cfg.get("expected_group_size", 0))
        self.reference_prompt_template = str(cfg.get("reference_prompt_template", ""))
        self.binary_guard_prompt_template = str(cfg.get("binary_guard_prompt_template", ""))
        self.exact_count_prompt_template = str(cfg.get("exact_count_prompt_template", ""))
        if self.comparison_mode == "reference_listwise":
            if not self.reference_data_root:
                raise ValueError(
                    "reference_listwise Codex image-QA requires reference_data_root",
                )
            if self.expected_group_size < 2:
                raise ValueError(
                    "reference_listwise Codex image-QA requires expected_group_size >= 2",
                )
            if self.images_per_call != self.expected_group_size:
                raise ValueError(
                    "reference_listwise Codex image-QA requires images_per_call "
                    "to equal expected_group_size so a rollout group is never chunked",
                )
            if not self.reference_prompt_template.strip():
                raise ValueError(
                    "reference_listwise Codex image-QA requires reference_prompt_template",
                )
        if self.comparison_mode == "binary_guard":
            if self.expected_group_size < 2:
                raise ValueError(
                    "binary_guard Codex image-QA requires expected_group_size >= 2",
                )
            if self.images_per_call != self.expected_group_size:
                raise ValueError(
                    "binary_guard Codex image-QA requires images_per_call "
                    "to equal expected_group_size so a rollout group is never chunked",
                )
            if not self.binary_guard_prompt_template.strip():
                raise ValueError(
                    "binary_guard Codex image-QA requires binary_guard_prompt_template",
                )
        if self.comparison_mode == "exact_count":
            if self.expected_group_size < 2:
                raise ValueError(
                    "exact_count Codex image-QA requires expected_group_size >= 2",
                )
            if self.images_per_call != self.expected_group_size:
                raise ValueError(
                    "exact_count Codex image-QA requires images_per_call to equal "
                    "expected_group_size so a rollout group is never chunked",
                )
            if not self.prompt_metadata_key:
                raise ValueError(
                    "exact_count Codex image-QA requires prompt_metadata_key",
                )
            if not self.exact_count_prompt_template.strip():
                raise ValueError(
                    "exact_count Codex image-QA requires exact_count_prompt_template",
                )
        scored_rollout_dir = str(cfg.get("scored_rollout_dir", "")).strip()
        self.scored_rollout_dir = Path(scored_rollout_dir) if scored_rollout_dir else None
        self._saved_batch_index = _next_saved_batch_index(self.scored_rollout_dir)

    def score_batch(self, artifacts: Sequence[Any]) -> list[dict[str, float]]:
        artifacts = list(artifacts)
        if not artifacts:
            return []
        if self.comparison_mode == "reference_listwise":
            scores = self._score_batch_reference_listwise(artifacts)
        elif self.comparison_mode == "binary_guard":
            scores = self._score_batch_binary_guard(artifacts)
        elif self.comparison_mode == "exact_count":
            scores = self._score_batch_exact_count(artifacts)
        elif self.images_per_call > 1:
            scores = self._score_batch_grid(artifacts)
        else:
            workers = min(self.max_concurrency, len(artifacts))
            if workers <= 1:
                scores = [self(artifact) for artifact in artifacts]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    scores = list(pool.map(self.__call__, artifacts))
        self._save_scored_rollouts(artifacts, scores)
        return scores

    def _save_scored_rollouts(
        self,
        artifacts: list[Any],
        scores: list[dict[str, float]],
    ) -> None:
        """Persist underlying rollout pixels and scores for visual audit."""

        root = self.scored_rollout_dir
        if root is None:
            return
        if len(artifacts) != len(scores):
            raise ValueError(
                "cannot save scored rollouts with mismatched artifacts and scores: "
                f"{len(artifacts)} != {len(scores)}",
            )

        root.mkdir(parents=True, exist_ok=True)
        batch_index = self._saved_batch_index
        final_dir = root / f"batch-{batch_index:06d}"
        if final_dir.exists():
            raise FileExistsError(f"scored rollout batch already exists: {final_dir}")

        with tempfile.TemporaryDirectory(prefix=".batch-", dir=root) as tmp:
            staging_dir = Path(tmp)
            medias = [artifact.as_media() for artifact in artifacts]
            policy_versions = {
                int(artifact.metadata["rollout_policy_version"])
                for artifact in artifacts
                if "rollout_policy_version" in artifact.metadata
            }
            if len(policy_versions) > 1:
                raise ValueError(
                    "one scored rollout batch cannot mix policy versions: "
                    f"{sorted(policy_versions)}",
                )
            items: list[dict[str, Any]] = []
            for sample_index, (artifact, media, score_map) in enumerate(
                zip(artifacts, medias, scores, strict=True),
            ):
                image_name = f"sample-{sample_index:02d}.png"
                write_png(media, staging_dir / image_name)
                items.append(
                    {
                        "artifact_id": str(artifact.artifact_id),
                        "image": image_name,
                        "prompt": str(getattr(artifact, "prompt", "")),
                        "judge_prompt": self._prompt_for_artifact(artifact),
                        "scores": {str(name): float(value) for name, value in score_map.items()},
                    },
                )

            montage_name = "montage.png"
            _compose_grid(medias, self.tile_size, staging_dir / montage_name)
            manifest = {
                "schema_version": 1,
                "batch_index": batch_index,
                "count": len(items),
                "montage": montage_name,
                "rollout_policy_version": next(iter(policy_versions), None),
                "items": items,
            }
            (staging_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging_dir.replace(final_dir)
        self._saved_batch_index += 1

    def _score_batch_binary_guard(
        self,
        artifacts: list[Any],
    ) -> list[dict[str, float]]:
        """Accept only candidates that pass both checks in both cell orders."""

        groups = self._candidate_groups(artifacts)
        jobs = [(group, reverse) for group in groups for reverse in (False, True)]

        def run_job(
            job: tuple[_CandidateGroup, bool],
        ) -> tuple[str, bool, dict[str, _BinaryGuardVerdict]]:
            group, reverse = job
            return (
                group.group_id,
                reverse,
                self._score_binary_guard_pass(artifacts, group, reverse=reverse),
            )

        workers = min(self.max_concurrency, len(jobs))
        if workers <= 1:
            pass_results = [run_job(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pass_results = list(pool.map(run_job, jobs))

        verdicts_by_pass = {
            (group_id, reverse): verdicts for group_id, reverse, verdicts in pass_results
        }
        scores: list[dict[str, float] | None] = [None] * len(artifacts)
        for group in groups:
            forward = verdicts_by_pass[(group.group_id, False)]
            reverse = verdicts_by_pass[(group.group_id, True)]
            for candidate_id, artifact_index in zip(
                group.candidate_ids,
                group.candidate_indices,
                strict=True,
            ):
                forward_verdict = forward[candidate_id]
                reverse_verdict = reverse[candidate_id]
                forward_pass = forward_verdict.passes
                reverse_pass = reverse_verdict.passes
                consensus = forward_pass and reverse_pass
                scores[artifact_index] = {
                    "codex_image_qa": float(consensus),
                    "codex_image_qa_forward": float(forward_pass),
                    "codex_image_qa_reverse": float(reverse_pass),
                    "codex_image_qa_consensus": float(consensus),
                    "codex_image_qa_mirror_agreement": float(
                        forward_verdict == reverse_verdict,
                    ),
                }

        ordered_scores: list[dict[str, float]] = []
        for artifact_index, score_map in enumerate(scores):
            if score_map is None:
                raise RuntimeError(
                    "binary-guard scoring did not produce a result for "
                    f"artifact index {artifact_index}",
                )
            ordered_scores.append(score_map)
        return ordered_scores

    def _score_binary_guard_pass(
        self,
        artifacts: list[Any],
        group: _CandidateGroup,
        *,
        reverse: bool,
    ) -> dict[str, _BinaryGuardVerdict]:
        """Run one guard pass while keeping ids attached to candidate media."""

        labeled_media = list(
            zip(
                group.candidate_ids,
                (artifacts[index].as_media() for index in group.candidate_indices),
                strict=True,
            ),
        )
        if reverse:
            # Reverse complete (id, media) pairs so the second montage is a
            # genuine spatial order reversal, not a relabeling of the first.
            labeled_media.reverse()

        response_contract = json.dumps(
            {
                "candidates": [
                    {
                        "id": candidate_id,
                        "carrier_grounding": _BinaryGuardDecision.PASS.value,
                        "image_integrity": _BinaryGuardDecision.PASS.value,
                    }
                    for candidate_id in group.candidate_ids
                ],
            },
            separators=(",", ":"),
        )
        prompt_text = _render_prompt_template(
            self.binary_guard_prompt_template,
            prompt=group.prompt,
            count=len(group.candidate_ids),
            response_contract=response_contract,
        )
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-binary-guard-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "grid.png"
            output_path = tmp_path / "judge_output.txt"
            output_schema_path = tmp_path / "output_schema.json"
            _compose_grid(
                [media for _, media in labeled_media],
                self.tile_size,
                image_path,
                labels=[label for label, _ in labeled_media],
            )
            _write_binary_guard_output_schema(output_schema_path, group.candidate_ids)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                output_schema_path=output_schema_path,
                prompt=group.prompt,
            )
            output_text = self._run_command(
                command,
                stdin_text=prompt_text if self.pass_prompt_stdin else "",
                output_path=output_path,
                workdir=tmp_path,
            )
        return _extract_binary_guard_verdicts(output_text, group.candidate_ids)

    def _score_batch_exact_count(
        self,
        artifacts: list[Any],
    ) -> list[dict[str, float]]:
        """Reward only mirrored, unambiguous agreement with the typed target."""

        groups = self._candidate_groups(artifacts)
        targets: dict[str, int] = {}
        for group in groups:
            try:
                target = int(group.prompt)
            except ValueError as exc:
                raise ValueError(
                    f"exact-count group {group.group_id!r} target must be an integer, "
                    f"got {group.prompt!r}",
                ) from exc
            if str(target) != group.prompt or target < 1:
                raise ValueError(
                    f"exact-count group {group.group_id!r} target must be a canonical "
                    f"positive integer, got {group.prompt!r}",
                )
            targets[group.group_id] = target

        jobs = [(group, reverse) for group in groups for reverse in (False, True)]

        def run_job(
            job: tuple[_CandidateGroup, bool],
        ) -> tuple[str, bool, dict[str, _ExactCountVerdict]]:
            group, reverse = job
            return (
                group.group_id,
                reverse,
                self._score_exact_count_pass(artifacts, group, reverse=reverse),
            )

        workers = min(self.max_concurrency, len(jobs))
        if workers <= 1:
            pass_results = [run_job(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pass_results = list(pool.map(run_job, jobs))

        verdicts_by_pass = {
            (group_id, reverse): verdicts for group_id, reverse, verdicts in pass_results
        }
        scores: list[dict[str, float] | None] = [None] * len(artifacts)
        for group in groups:
            target = targets[group.group_id]
            forward = verdicts_by_pass[(group.group_id, False)]
            reverse = verdicts_by_pass[(group.group_id, True)]
            for candidate_id, artifact_index in zip(
                group.candidate_ids,
                group.candidate_indices,
                strict=True,
            ):
                forward_verdict = forward[candidate_id]
                reverse_verdict = reverse[candidate_id]
                forward_pass = (
                    forward_verdict.unambiguous and forward_verdict.observed_count == target
                )
                reverse_pass = (
                    reverse_verdict.unambiguous and reverse_verdict.observed_count == target
                )
                consensus = forward_pass and reverse_pass
                scores[artifact_index] = {
                    "codex_image_qa": float(consensus),
                    "codex_image_qa_forward": float(forward_pass),
                    "codex_image_qa_reverse": float(reverse_pass),
                    "codex_image_qa_consensus": float(consensus),
                    "codex_image_qa_mirror_agreement": float(
                        forward_verdict == reverse_verdict,
                    ),
                    "codex_image_qa_observed_forward": float(
                        forward_verdict.observed_count,
                    ),
                    "codex_image_qa_observed_reverse": float(
                        reverse_verdict.observed_count,
                    ),
                    "codex_image_qa_target": float(target),
                    "codex_image_qa_unambiguous_forward": float(
                        forward_verdict.unambiguous,
                    ),
                    "codex_image_qa_unambiguous_reverse": float(
                        reverse_verdict.unambiguous,
                    ),
                }

        ordered_scores: list[dict[str, float]] = []
        for artifact_index, score_map in enumerate(scores):
            if score_map is None:
                raise RuntimeError(
                    "exact-count scoring did not produce a result for "
                    f"artifact index {artifact_index}",
                )
            ordered_scores.append(score_map)
        return ordered_scores

    def _score_exact_count_pass(
        self,
        artifacts: list[Any],
        group: _CandidateGroup,
        *,
        reverse: bool,
    ) -> dict[str, _ExactCountVerdict]:
        """Collect observed counts while stable ids preserve candidate identity."""

        labeled_media = list(
            zip(
                group.candidate_ids,
                (artifacts[index].as_media() for index in group.candidate_indices),
                strict=True,
            ),
        )
        if reverse:
            labeled_media.reverse()

        response_contract = json.dumps(
            {
                "candidates": [
                    {
                        "id": candidate_id,
                        "observed_count": 0,
                        "unambiguous": False,
                    }
                    for candidate_id in group.candidate_ids
                ],
            },
            separators=(",", ":"),
        )
        prompt_text = _render_prompt_template(
            self.exact_count_prompt_template,
            prompt=group.prompt,
            count=len(group.candidate_ids),
            response_contract=response_contract,
        )
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-exact-count-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "grid.png"
            output_path = tmp_path / "judge_output.txt"
            output_schema_path = tmp_path / "output_schema.json"
            _compose_grid(
                [media for _, media in labeled_media],
                self.tile_size,
                image_path,
                labels=[label for label, _ in labeled_media],
            )
            _write_exact_count_output_schema(output_schema_path, group.candidate_ids)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                output_schema_path=output_schema_path,
                prompt=group.prompt,
            )
            output_text = self._run_command(
                command,
                stdin_text=prompt_text if self.pass_prompt_stdin else "",
                output_path=output_path,
                workdir=tmp_path,
            )
        return _extract_exact_count_verdicts(output_text, group.candidate_ids)

    def _score_batch_reference_listwise(
        self,
        artifacts: list[Any],
    ) -> list[dict[str, float]]:
        """Score complete rollout groups against one frozen base reference.

        Every group is judged twice with the 3x3 cell order mirrored. Stable
        cell ids let the parser restore candidate identity; a directional
        disagreement becomes a tie. If neither pass finds a consensus candidate
        that beats the reference without an integrity regression, the whole
        group's optimization reward is exactly zero.
        """

        groups = self._reference_groups(artifacts)
        jobs = [(group, reverse) for group in groups for reverse in (False, True)]

        def run_job(
            job: tuple[_ReferenceGroup, bool],
        ) -> tuple[str, bool, dict[str, _ReferenceVerdict]]:
            group, reverse = job
            return (
                group.group_id,
                reverse,
                self._score_reference_pass(artifacts, group, reverse=reverse),
            )

        workers = min(self.max_concurrency, len(jobs))
        if workers <= 1:
            pass_results = [run_job(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pass_results = list(pool.map(run_job, jobs))

        verdicts_by_pass = {
            (group_id, reverse): verdicts for group_id, reverse, verdicts in pass_results
        }
        scores: list[dict[str, float] | None] = [None] * len(artifacts)
        for group in groups:
            forward = verdicts_by_pass[(group.group_id, False)]
            mirrored = verdicts_by_pass[(group.group_id, True)]
            consensus = {
                candidate_id: _consensus_reference_verdict(
                    forward[candidate_id],
                    mirrored[candidate_id],
                )
                for candidate_id in group.candidate_ids
            }
            improvements = {
                candidate_id: (
                    verdict.preference.reward
                    if verdict.integrity is _ReferenceIntegrity.PASS and verdict.preference.is_win
                    else 0.0
                )
                for candidate_id, verdict in consensus.items()
            }
            group_active = any(reward > 0.0 for reward in improvements.values())
            for candidate_id, artifact_index in zip(
                group.candidate_ids,
                group.candidate_indices,
                strict=True,
            ):
                verdict = consensus[candidate_id]
                relative_reward = (
                    -2.0
                    if verdict.integrity is _ReferenceIntegrity.REGRESS
                    else verdict.preference.reward
                )
                scores[artifact_index] = {
                    # GRPO mean-centers this value. Keeping non-winners at zero
                    # ensures a tied candidate cannot become positive merely
                    # because several worse candidates pulled the group mean down.
                    "codex_image_qa": improvements[candidate_id] if group_active else 0.0,
                    "codex_image_qa_group_active": float(group_active),
                    "codex_image_qa_integrity_pass": float(
                        verdict.integrity is _ReferenceIntegrity.PASS,
                    ),
                    "codex_image_qa_mirror_agreement": float(
                        forward[candidate_id] == mirrored[candidate_id],
                    ),
                    "codex_image_qa_relative": relative_reward,
                }

        ordered_scores: list[dict[str, float]] = []
        for artifact_index, score_map in enumerate(scores):
            if score_map is None:
                raise RuntimeError(
                    "reference-listwise scoring did not produce a result for "
                    f"artifact index {artifact_index}",
                )
            ordered_scores.append(score_map)
        return ordered_scores

    def _candidate_groups(self, artifacts: list[Any]) -> list[_CandidateGroup]:
        grouped_indices: dict[str, list[int]] = {}
        for artifact_index, artifact in enumerate(artifacts):
            raw_group_id = artifact.metadata.get(REWARD_GROUP_ID_METADATA_KEY)
            if not isinstance(raw_group_id, str) or not raw_group_id.strip():
                raise ValueError(
                    f"{self.comparison_mode.replace('_', '-')} Codex image-QA artifact "
                    f"{artifact.artifact_id!r} is missing non-empty "
                    f"metadata[{REWARD_GROUP_ID_METADATA_KEY!r}]",
                )
            grouped_indices.setdefault(raw_group_id, []).append(artifact_index)

        groups: list[_CandidateGroup] = []
        for group_id, candidate_indices_list in grouped_indices.items():
            if len(candidate_indices_list) != self.expected_group_size:
                raise ValueError(
                    f"{self.comparison_mode.replace('_', '-')} group {group_id!r} has "
                    f"{len(candidate_indices_list)} candidates; expected "
                    f"{self.expected_group_size}",
                )
            group_artifacts = [artifacts[index] for index in candidate_indices_list]
            generation_prompts = {
                str(getattr(artifact, "prompt", "")) for artifact in group_artifacts
            }
            if len(generation_prompts) != 1:
                raise ValueError(
                    f"{self.comparison_mode.replace('_', '-')} group {group_id!r} "
                    "mixes generation prompts",
                )
            judge_targets = {self._prompt_for_artifact(artifact) for artifact in group_artifacts}
            if len(judge_targets) != 1:
                raise ValueError(
                    f"{self.comparison_mode.replace('_', '-')} group {group_id!r} "
                    "mixes metadata-derived judge targets",
                )
            candidate_indices = tuple(candidate_indices_list)
            groups.append(
                _CandidateGroup(
                    group_id=group_id,
                    prompt=next(iter(judge_targets)),
                    candidate_indices=candidate_indices,
                    candidate_ids=tuple(
                        f"C{position + 1}" for position in range(len(candidate_indices))
                    ),
                ),
            )
        return groups

    def _reference_groups(self, artifacts: list[Any]) -> list[_ReferenceGroup]:
        groups: list[_ReferenceGroup] = []
        for candidate_group in self._candidate_groups(artifacts):
            group_artifacts = [artifacts[index] for index in candidate_group.candidate_indices]
            target_images = {
                str(artifact.metadata.get("target_image", "") or "").strip()
                for artifact in group_artifacts
            }
            if "" in target_images:
                raise ValueError(
                    f"reference-listwise group {candidate_group.group_id!r} has no target_image",
                )
            if len(target_images) != 1:
                raise ValueError(
                    "reference-listwise group "
                    f"{candidate_group.group_id!r} mixes target_image values",
                )
            target_image = next(iter(target_images))
            reference_path = resolve_artifact_path(
                target_image,
                data_root=self.reference_data_root,
                allow_absolute=False,
            )
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"reference-listwise target image does not exist: {reference_path}",
                )
            groups.append(
                _ReferenceGroup(
                    group_id=candidate_group.group_id,
                    prompt=candidate_group.prompt,
                    reference_path=reference_path,
                    candidate_indices=candidate_group.candidate_indices,
                    candidate_ids=candidate_group.candidate_ids,
                ),
            )
        return groups

    def _score_reference_pass(
        self,
        artifacts: list[Any],
        group: _ReferenceGroup,
        *,
        reverse: bool,
    ) -> dict[str, _ReferenceVerdict]:
        """Run one stable-id ordering of a reference-listwise comparison."""

        from PIL import Image

        with Image.open(group.reference_path) as source:
            reference = source.convert("RGB")
        labeled_media = [("R", reference)]
        labeled_media.extend(
            (candidate_id, artifacts[index].as_media())
            for candidate_id, index in zip(
                group.candidate_ids,
                group.candidate_indices,
                strict=True,
            )
        )
        if reverse:
            labeled_media.reverse()

        response_contract = json.dumps(
            {
                "candidates": [
                    {
                        "id": candidate_id,
                        "integrity": _ReferenceIntegrity.PASS.value,
                        "preference": _ReferencePreference.TIE.value,
                    }
                    for candidate_id in group.candidate_ids
                ],
            },
            separators=(",", ":"),
        )
        prompt_text = _render_prompt_template(
            self.reference_prompt_template,
            prompt=group.prompt,
            count=len(group.candidate_ids),
            response_contract=response_contract,
        )
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-reference-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "grid.png"
            output_path = tmp_path / "judge_output.txt"
            output_schema_path = tmp_path / "output_schema.json"
            _compose_grid(
                [media for _, media in labeled_media],
                self.tile_size,
                image_path,
                labels=[label for label, _ in labeled_media],
            )
            _write_reference_output_schema(output_schema_path, group.candidate_ids)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                output_schema_path=output_schema_path,
                prompt=group.prompt,
            )
            output_text = self._run_command(
                command,
                stdin_text=prompt_text if self.pass_prompt_stdin else "",
                output_path=output_path,
                workdir=tmp_path,
            )
        return _extract_reference_verdicts(output_text, group.candidate_ids)

    def _score_batch_grid(self, artifacts: list[Any]) -> list[dict[str, float]]:
        """Group by prompt, tile each group into montages, one CLI call per tile.

        Grouping by identical prompt keeps every cell in a montage sharing one
        text prompt, so the rubric names the prompt once and asks for a score per
        numbered cell. Scores are mapped back to each artifact's original index.
        """

        groups: dict[str, list[int]] = {}
        for idx, artifact in enumerate(artifacts):
            groups.setdefault(self._prompt_for_artifact(artifact), []).append(idx)

        chunks: list[tuple[str, list[int]]] = []
        for prompt, indices in groups.items():
            for start in range(0, len(indices), self.images_per_call):
                chunks.append((prompt, indices[start : start + self.images_per_call]))

        scores: list[float | None] = [None] * len(artifacts)

        def run_chunk(chunk: tuple[str, list[int]]) -> None:
            prompt, indices = chunk
            cell_scores = self._score_grid([artifacts[i] for i in indices], prompt)
            for i, score in zip(indices, cell_scores, strict=True):
                scores[i] = score

        workers = min(self.max_concurrency, len(chunks))
        if workers <= 1:
            for chunk in chunks:
                run_chunk(chunk)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(run_chunk, chunks))

        return [{"codex_image_qa": (0.0 if s is None else s)} for s in scores]

    def _score_grid(self, group: list[Any], prompt: str) -> list[float]:
        """Compose one numbered montage of ``group`` and parse a score per cell."""

        n = len(group)
        if n == 1:
            return [self(group[0])["codex_image_qa"]]
        prompt_text = _render_prompt_template(
            self.grid_prompt_template,
            prompt=prompt,
            count=n,
        )
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-grid-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "grid.png"
            output_path = tmp_path / "judge_output.txt"
            output_schema_path = tmp_path / "output_schema.json"
            _compose_grid([a.as_media() for a in group], self.tile_size, image_path)
            _write_output_schema(output_schema_path, count=n)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                output_schema_path=output_schema_path,
                prompt=prompt,
            )
            output_text = self._run_command(
                command,
                stdin_text=prompt_text if self.pass_prompt_stdin else "",
                output_path=output_path,
                workdir=tmp_path,
            )
        return _extract_grid_scores(output_text, n)

    def __call__(self, artifact: Any) -> dict[str, float]:
        prompt = self._prompt_for_artifact(artifact)
        prompt_text = _render_prompt_template(self.prompt_template, prompt=prompt)
        with tempfile.TemporaryDirectory(prefix="vrl-codex-image-qa-") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "image.png"
            output_path = tmp_path / "judge_output.txt"
            output_schema_path = tmp_path / "output_schema.json"
            write_png(artifact.as_media(), image_path)
            _write_output_schema(output_schema_path)
            command = _render_command(
                self.command,
                image_path=image_path,
                output_path=output_path,
                output_schema_path=output_schema_path,
                prompt=prompt,
            )
            output_text = self._run_command(
                command,
                stdin_text=prompt_text if self.pass_prompt_stdin else "",
                output_path=output_path,
                workdir=tmp_path,
            )
        return {"codex_image_qa": _extract_score_from_text(output_text)}

    def _prompt_for_artifact(self, artifact: Any) -> str:
        """Resolve the rubric target without reparsing generation prose."""

        if not self.prompt_metadata_key:
            return str(getattr(artifact, "prompt", ""))
        metadata = getattr(artifact, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"Codex image-QA artifact {artifact.artifact_id!r} metadata must be a mapping",
            )
        if self.prompt_metadata_key not in metadata:
            raise ValueError(
                f"Codex image-QA artifact {artifact.artifact_id!r} is missing "
                f"metadata[{self.prompt_metadata_key!r}]",
            )
        value = metadata[self.prompt_metadata_key]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise TypeError(
                f"Codex image-QA artifact {artifact.artifact_id!r} "
                f"metadata[{self.prompt_metadata_key!r}] must be a string or number",
            )
        prompt = str(value).strip()
        if not prompt:
            raise ValueError(
                f"Codex image-QA artifact {artifact.artifact_id!r} "
                f"metadata[{self.prompt_metadata_key!r}] must be non-empty",
            )
        return prompt

    def _run_command(
        self,
        command: list[str],
        *,
        stdin_text: str,
        output_path: Path,
        workdir: Path,
    ) -> str:
        try:
            completed = subprocess.run(
                command,
                input=stdin_text.encode("utf-8"),
                capture_output=True,
                cwd=str(workdir),
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Codex image-QA timed out after {self.timeout_s:.1f}s: {command!r}",
            ) from exc

        stdout_text = completed.stdout.decode("utf-8", errors="replace")
        stderr_text = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(
                "Codex image-QA failed "
                f"(exit={completed.returncode}): {command!r}\n"
                f"STDERR:\n{stderr_text}\nSTDOUT:\n{stdout_text}",
            )

        if output_path.exists():
            file_text = output_path.read_text(encoding="utf-8", errors="replace").strip()
            if file_text:
                return file_text
        return stdout_text


def _normalize_command(command: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def _render_command(
    command: list[str],
    *,
    image_path: Path,
    output_path: Path,
    output_schema_path: Path,
    prompt: str,
) -> list[str]:
    values = {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "output_schema_path": str(output_schema_path),
        "prompt": prompt,
    }
    rendered = [part.format(**values) for part in command]
    is_codex_exec = (
        len(rendered) >= 2 and Path(rendered[0]).stem == "codex" and rendered[1] in {"exec", "e"}
    )
    has_output_schema = any(
        part == "--output-schema" or part.startswith("--output-schema=") for part in rendered
    )
    if is_codex_exec and not has_output_schema:
        # Insert next to the subcommand instead of appending after the positional
        # prompt, whose placement is intentionally owned by the configured command.
        rendered[2:2] = ["--output-schema", str(output_schema_path)]
    return rendered


def _write_output_schema(path: Path, *, count: int | None = None) -> None:
    """Write the per-call response contract beside the disposable image.

    Montage chunks can have different sizes (especially the final chunk), so
    their exact cardinality belongs to the invocation rather than static config.
    """

    score = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    if count is None:
        properties: dict[str, Any] = {"score": score}
        required = ["score"]
    else:
        if count < 1:
            raise ValueError(f"Codex image-QA output schema count must be positive, got {count}")
        properties = {
            "scores": {
                "type": "array",
                "items": score,
                "minItems": count,
                "maxItems": count,
            },
        }
        required = ["scores"]
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema, separators=(",", ":")), encoding="utf-8")


def _write_reference_output_schema(
    path: Path,
    candidate_ids: Sequence[str],
) -> None:
    """Write the exact structured contract for one reference comparison."""

    candidate_ids = tuple(candidate_ids)
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("reference candidate ids must be non-empty and unique")
    verdict = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": list(candidate_ids)},
            "integrity": {
                "type": "string",
                "enum": [member.value for member in _ReferenceIntegrity],
            },
            "preference": {
                "type": "string",
                "enum": [member.value for member in _ReferencePreference],
            },
        },
        "required": ["id", "integrity", "preference"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": verdict,
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
            },
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema, separators=(",", ":")), encoding="utf-8")


def _write_binary_guard_output_schema(
    path: Path,
    candidate_ids: Sequence[str],
) -> None:
    """Write the exact structured contract for one binary-guard pass."""

    candidate_ids = tuple(candidate_ids)
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("binary-guard candidate ids must be non-empty and unique")
    decision = {
        "type": "string",
        "enum": [member.value for member in _BinaryGuardDecision],
    }
    verdict = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": list(candidate_ids)},
            "carrier_grounding": decision,
            "image_integrity": decision,
        },
        "required": ["id", "carrier_grounding", "image_integrity"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": verdict,
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
            },
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema, separators=(",", ":")), encoding="utf-8")


def _write_exact_count_output_schema(
    path: Path,
    candidate_ids: Sequence[str],
) -> None:
    """Write the exact structured contract for one observed-count pass."""

    candidate_ids = tuple(candidate_ids)
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("exact-count candidate ids must be non-empty and unique")
    verdict = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": list(candidate_ids)},
            "observed_count": {"type": "integer", "minimum": 0},
            "unambiguous": {"type": "boolean"},
        },
        "required": ["id", "observed_count", "unambiguous"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": verdict,
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
            },
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema, separators=(",", ":")), encoding="utf-8")


def _next_saved_batch_index(root: Path | None) -> int:
    """Continue after complete batches when a supervised run resumes."""

    if root is None or not root.exists():
        return 0
    indices = []
    for candidate in root.glob("batch-*"):
        if not (candidate / "manifest.json").is_file():
            continue
        try:
            indices.append(int(candidate.name.removeprefix("batch-")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def _compose_grid(
    medias: Sequence[Any],
    tile: int,
    out_path: Path,
    *,
    labels: Sequence[str] | None = None,
) -> None:
    """Downscale media and tile it into a montage with stable visible labels."""

    import math

    from PIL import Image, ImageDraw

    from vrl.utils.media import to_pil_image

    imgs = [to_pil_image(m).convert("RGB").resize((tile, tile), Image.LANCZOS) for m in medias]
    n = len(imgs)
    resolved_labels = tuple(str(index + 1) for index in range(n))
    if labels is not None:
        resolved_labels = tuple(str(label) for label in labels)
        if len(resolved_labels) != n:
            raise ValueError(
                f"Codex image-QA montage label/media mismatch: {len(resolved_labels)} != {n}",
            )
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    pad = 2
    canvas = Image.new(
        "RGB", (cols * tile + (cols + 1) * pad, rows * tile + (rows + 1) * pad), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(imgs, resolved_labels, strict=True)):
        r, c = divmod(i, cols)
        x = pad + c * (tile + pad)
        y = pad + r * (tile + pad)
        canvas.paste(img, (x, y))
        # Stable ids survive mirrored cell order and restore candidate identity.
        draw.rectangle([x, y, x + 9 * len(label) + 6, y + 18], fill="black")
        draw.text((x + 3, y + 3), label, fill="white")
    canvas.save(out_path, format="PNG")


def _extract_grid_scores(text: str, count: int) -> list[float]:
    """Parse ``count`` per-cell scores from a grid judge response."""

    value = _find_first_json_value(text.strip())
    scores: list[Any] | None = None
    if isinstance(value, dict):
        for key in ("scores", "cells", "results"):
            if isinstance(value.get(key), list):
                scores = value[key]
                break
    elif isinstance(value, list):
        scores = value
    if scores is None:
        raise ValueError(f"Cannot parse {count} grid scores from output: {text!r}")
    if len(scores) != count:
        raise ValueError(
            f"Expected {count} grid scores, got {len(scores)} in output: {text!r}",
        )
    return [_score_from_value(score) for score in scores]


def _extract_reference_verdicts(
    text: str,
    candidate_ids: Sequence[str],
) -> dict[str, _ReferenceVerdict]:
    """Parse a complete, uniquely identified reference-comparison response."""

    candidate_ids = tuple(candidate_ids)
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot parse reference-listwise Codex image-QA output: {text!r}",
        ) from exc
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise ValueError("reference-listwise output must contain only 'candidates'")
    rows = value["candidates"]
    if not isinstance(rows, list) or len(rows) != len(candidate_ids):
        observed = len(rows) if isinstance(rows, list) else type(rows).__name__
        raise ValueError(
            "reference-listwise output candidate count mismatch: "
            f"expected {len(candidate_ids)}, got {observed}",
        )

    parsed: dict[str, _ReferenceVerdict] = {}
    expected_ids = set(candidate_ids)
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "integrity", "preference"}:
            raise ValueError(
                "each reference-listwise candidate must contain exactly "
                "'id', 'integrity', and 'preference'",
            )
        candidate_id = row["id"]
        if not isinstance(candidate_id, str) or candidate_id not in expected_ids:
            raise ValueError(f"unexpected reference-listwise candidate id: {candidate_id!r}")
        if candidate_id in parsed:
            raise ValueError(f"duplicate reference-listwise candidate id: {candidate_id!r}")
        try:
            parsed[candidate_id] = _ReferenceVerdict(
                candidate_id=candidate_id,
                integrity=_ReferenceIntegrity(row["integrity"]),
                preference=_ReferencePreference(row["preference"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid reference-listwise verdict for {candidate_id!r}: {row!r}",
            ) from exc

    missing_ids = expected_ids.difference(parsed)
    if missing_ids:
        raise ValueError(
            f"reference-listwise output is missing candidate ids: {sorted(missing_ids)}",
        )
    return {candidate_id: parsed[candidate_id] for candidate_id in candidate_ids}


def _extract_binary_guard_verdicts(
    text: str,
    candidate_ids: Sequence[str],
) -> dict[str, _BinaryGuardVerdict]:
    """Parse one complete binary-guard response without permissive fallback."""

    candidate_ids = tuple(candidate_ids)
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot parse binary-guard Codex image-QA output: {text!r}",
        ) from exc
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise ValueError("binary-guard output must contain only 'candidates'")
    rows = value["candidates"]
    if not isinstance(rows, list) or len(rows) != len(candidate_ids):
        observed = len(rows) if isinstance(rows, list) else type(rows).__name__
        raise ValueError(
            "binary-guard output candidate count mismatch: "
            f"expected {len(candidate_ids)}, got {observed}",
        )

    parsed: dict[str, _BinaryGuardVerdict] = {}
    expected_ids = set(candidate_ids)
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "carrier_grounding",
            "image_integrity",
        }:
            raise ValueError(
                "each binary-guard candidate must contain exactly 'id', "
                "'carrier_grounding', and 'image_integrity'",
            )
        candidate_id = row["id"]
        if not isinstance(candidate_id, str) or candidate_id not in expected_ids:
            raise ValueError(f"unexpected binary-guard candidate id: {candidate_id!r}")
        if candidate_id in parsed:
            raise ValueError(f"duplicate binary-guard candidate id: {candidate_id!r}")
        try:
            parsed[candidate_id] = _BinaryGuardVerdict(
                candidate_id=candidate_id,
                carrier_grounding=_BinaryGuardDecision(row["carrier_grounding"]),
                image_integrity=_BinaryGuardDecision(row["image_integrity"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid binary-guard verdict for {candidate_id!r}: {row!r}",
            ) from exc

    missing_ids = expected_ids.difference(parsed)
    if missing_ids:
        raise ValueError(
            f"binary-guard output is missing candidate ids: {sorted(missing_ids)}",
        )
    return {candidate_id: parsed[candidate_id] for candidate_id in candidate_ids}


def _extract_exact_count_verdicts(
    text: str,
    candidate_ids: Sequence[str],
) -> dict[str, _ExactCountVerdict]:
    """Parse complete observed counts without interpreting generation prose."""

    candidate_ids = tuple(candidate_ids)
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot parse exact-count Codex image-QA output: {text!r}",
        ) from exc
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise ValueError("exact-count output must contain only 'candidates'")
    rows = value["candidates"]
    if not isinstance(rows, list) or len(rows) != len(candidate_ids):
        observed = len(rows) if isinstance(rows, list) else type(rows).__name__
        raise ValueError(
            "exact-count output candidate count mismatch: "
            f"expected {len(candidate_ids)}, got {observed}",
        )

    parsed: dict[str, _ExactCountVerdict] = {}
    expected_ids = set(candidate_ids)
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "observed_count",
            "unambiguous",
        }:
            raise ValueError(
                "each exact-count candidate must contain exactly 'id', "
                "'observed_count', and 'unambiguous'",
            )
        candidate_id = row["id"]
        if not isinstance(candidate_id, str) or candidate_id not in expected_ids:
            raise ValueError(f"unexpected exact-count candidate id: {candidate_id!r}")
        if candidate_id in parsed:
            raise ValueError(f"duplicate exact-count candidate id: {candidate_id!r}")
        observed_count = row["observed_count"]
        unambiguous = row["unambiguous"]
        if type(observed_count) is not int or observed_count < 0:
            raise ValueError(
                f"invalid exact-count observed_count for {candidate_id!r}: {observed_count!r}",
            )
        if type(unambiguous) is not bool:
            raise ValueError(
                f"invalid exact-count unambiguous value for {candidate_id!r}: {unambiguous!r}",
            )
        parsed[candidate_id] = _ExactCountVerdict(
            candidate_id=candidate_id,
            observed_count=observed_count,
            unambiguous=unambiguous,
        )

    missing_ids = expected_ids.difference(parsed)
    if missing_ids:
        raise ValueError(
            f"exact-count output is missing candidate ids: {sorted(missing_ids)}",
        )
    return {candidate_id: parsed[candidate_id] for candidate_id in candidate_ids}


def _consensus_reference_verdict(
    forward: _ReferenceVerdict,
    mirrored: _ReferenceVerdict,
) -> _ReferenceVerdict:
    """Keep exact mirror agreement; convert every order-sensitive result to a tie."""

    if forward.candidate_id != mirrored.candidate_id:
        raise ValueError(
            "cannot combine reference verdicts for different candidates: "
            f"{forward.candidate_id!r} != {mirrored.candidate_id!r}",
        )
    if forward == mirrored:
        return forward
    return _ReferenceVerdict(
        candidate_id=forward.candidate_id,
        integrity=_ReferenceIntegrity.PASS,
        preference=_ReferencePreference.TIE,
    )


def _render_prompt_template(
    template: str,
    *,
    prompt: str,
    count: int | None = None,
    response_contract: str | None = None,
) -> str:
    """Render invocation-owned fields while preserving literal JSON braces."""

    if response_contract is None:
        response_contract = '{"score": 0.37}' if count is None else '{"scores": [0.37, 0.81, ...]}'
    values = {
        "count": str(1 if count is None else count),
        "prompt": prompt,
        "response_contract": response_contract,
    }

    rendered = template
    placeholders: dict[str, str] = {}
    for name, value in values.items():
        placeholder = f"__VRL_IMAGE_QA_{name.upper()}_PLACEHOLDER__"
        while placeholder in template:
            placeholder = f"_{placeholder}"
        rendered = rendered.replace(f"{{{name}}}", placeholder)
        placeholders[placeholder] = value
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    for placeholder, value in placeholders.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _extract_score_from_text(text: str) -> float:
    """Parse a judge response into a clamped score in ``[0, 1]``."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("Codex image-QA returned empty output")

    json_value = _find_first_json_value(stripped)
    if json_value is not None:
        return _score_from_value(json_value)

    lowered = stripped.lower()
    if lowered.startswith("yes"):
        return 1.0
    if lowered.startswith("no"):
        return 0.0

    score_match = re.search(r'"?score"?\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))', stripped)
    if score_match:
        return _clamp_score(float(score_match.group(1)))

    try:
        return _clamp_score(float(stripped))
    except ValueError as exc:
        raise ValueError(f"Cannot parse Codex image-QA score from output: {text!r}") from exc


def _find_first_json_value(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def _score_from_value(value: Any) -> float:
    if isinstance(value, dict):
        if "score" in value:
            return _clamp_score(float(value["score"]))
        if "answer" in value:
            return _score_answer(value["answer"])
        if "scores" in value:
            return _score_from_value(value["scores"])
        if "resultMap" in value:
            return _score_from_value(value["resultMap"])
    if isinstance(value, list):
        if not value:
            raise ValueError("Codex image-QA returned an empty score list")
        return _score_from_value(value[0])
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered.startswith(("yes", "no")):
            return _score_answer(value)
        return _clamp_score(float(value))
    return _clamp_score(float(value))


def _score_answer(answer: Any) -> float:
    text = str(answer).strip().lower()
    if text.startswith("yes"):
        return 1.0
    if text.startswith("no"):
        return 0.0
    raise ValueError(f"Cannot convert Codex image-QA answer to reward score: {answer!r}")


def _clamp_score(value: float) -> float:
    return min(max(value, 0.0), 1.0)


__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "CodexImageQARewardModel",
    "_extract_score_from_text",
    "_render_prompt_template",
]
