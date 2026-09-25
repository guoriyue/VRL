"""Audit requirement preservation across an ordered sequence of scored states."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, FiniteFloat, StrictInt, field_validator, model_validator

from vrl.config.base import ConfigBase
from vrl.rewards.evaluation import Evaluation


class SequenceRequirement(ConfigBase):
    """A frozen criterion; ``active_from`` says from which state it is due."""

    name: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    direction: Literal[-1, 1] = 1
    threshold: FiniteFloat = Field(strict=True)
    active_from: StrictInt = Field(default=0, ge=0)

    @field_validator("direction", mode="before")
    @classmethod
    def validate_direction(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("requirement direction must be -1 or 1, not a boolean")
        return value


class EditSequenceSpec(ConfigBase):
    """Ordered sample identities and the requirements to check across them."""

    sequence_id: str = Field(min_length=1)
    samples: list[str] = Field(min_length=2)
    requirements: list[SequenceRequirement] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sequence(self) -> EditSequenceSpec:
        if any(not sample for sample in self.samples) or len(set(self.samples)) != len(
            self.samples
        ):
            raise ValueError("sequence sample IDs must be nonempty and unique")
        if len({rule.name for rule in self.requirements}) != len(self.requirements):
            raise ValueError("sequence requirement names must be unique")
        if any(rule.active_from >= len(self.samples) for rule in self.requirements):
            raise ValueError("requirement activation is outside the sequence")
        return self

    def report(self, evaluation: Evaluation) -> dict[str, Any]:
        """Per state and per requirement: pass, fail, unknown, or not yet active.

        A failed or missing score is ``unknown``, never zero, and no regression
        or improvement is inferred across an unknown interval. All states must
        describe the same task (prompt and reward inputs).
        """

        records = evaluation.records
        missing = set(self.samples) - set(records)
        if missing:
            raise ValueError(f"sequence references unknown sample IDs: {sorted(missing)}")
        task = None
        states = []
        previous: dict[str, str] = {}
        observations: dict[str, list[dict[str, Any]]] = {r.name: [] for r in self.requirements}
        for step, sample_id in enumerate(self.samples):
            row = records[sample_id]
            item = row["input"]
            identity = {k: v for k, v in item.items() if k not in {"sample_id", "path", "sha256"}}
            if task is not None and identity != task:
                raise ValueError("sequence task inputs or verifier specification changed")
            task = identity
            measurements = {}
            for rule in self.requirements:
                value = reason = None
                if step < rule.active_from:
                    status = "not_active"
                elif row["status"] != "success":
                    status, reason = "unknown", row["status"]
                elif rule.axis not in row["result"]["scores"]:
                    status, reason = "unknown", "missing_axis"
                else:
                    value = row["result"]["scores"][rule.axis]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError("sequence scores must be finite numbers")
                    if not math.isfinite(value):
                        raise ValueError("sequence scores must be finite numbers")
                    passed = (
                        value >= rule.threshold if rule.direction == 1 else value <= rule.threshold
                    )
                    status = "pass" if passed else "fail"
                before = previous.get(rule.name)
                transition = None
                if before == "pass" and status == "fail":
                    transition = "regressed"
                elif before == "fail" and status == "pass":
                    transition = "improved"
                elif before == "pass" and status == "pass":
                    transition = "preserved"
                measurements[rule.name] = {
                    "status": status,
                    "value": value,
                    "reason": reason,
                    "transition": transition,
                }
                observations[rule.name].append(measurements[rule.name])
                previous[rule.name] = status
            states.append(
                {
                    "step": step,
                    "sample_id": sample_id,
                    "media_sha256": item.get("sha256"),
                    "source_status": row["status"],
                    "requirements": measurements,
                    **({"error": row["error"]} if row["status"] == "error" else {}),
                }
            )
        summaries = {}
        for rule in self.requirements:
            values = observations[rule.name]
            summaries[rule.name] = {
                "first_satisfied_step": next(
                    (i for i, v in enumerate(values) if v["status"] == "pass"), None
                ),
                "regression_steps": [
                    i for i, v in enumerate(values) if v["transition"] == "regressed"
                ],
                "improvement_steps": [
                    i for i, v in enumerate(values) if v["transition"] == "improved"
                ],
                "unknown_steps": [i for i, v in enumerate(values) if v["status"] == "unknown"],
                "final_status": values[-1]["status"],
            }
        final = {row["final_status"] for row in summaries.values()}
        return {
            "schema": "vrl.edit-sequence-audit.v1",
            "evaluation_run_id": evaluation.run_id,
            "spec": self.model_dump(mode="json"),
            "coverage_complete": not any(row["unknown_steps"] for row in summaries.values()),
            "final_requirements_met": False
            if "fail" in final
            else None
            if "unknown" in final
            else True,
            "requirements": summaries,
            "states": states,
        }


__all__ = ["EditSequenceSpec", "SequenceRequirement"]
