"""Audit requirement preservation across declared, independently scored edit states."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, FiniteFloat, StrictInt, field_validator, model_validator

from vrl.config.base import ConfigBase
from vrl.utils.json_files import canonical_json_sha256


class SequenceRequirement(ConfigBase):
    """A frozen criterion; activation declares when a requirement becomes due."""

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
    """Ordered sample identities and explicit measurement criteria, without inference."""

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


def sequence_report(evaluation: dict[str, Any], spec: EditSequenceSpec) -> dict[str, Any]:
    """Report regressions without turning failures or missing axes into zero scores.

    All states must be scored against exactly the same task/reference/specification.
    Activation can stage its requirements, but changing the expected transcript
    midway cannot masquerade as preserving the old text. Declared sequence order
    is not proof that a generator actually consumed the previous image.
    """
    records = evaluation["records"]
    missing = set(spec.samples) - set(records)
    if missing:
        raise ValueError(f"sequence references unknown sample IDs: {sorted(missing)}")
    task_identity = None
    states = []
    previous = {}
    observations = {rule.name: [] for rule in spec.requirements}
    for step, sample_id in enumerate(spec.samples):
        row = records[sample_id]
        item = row["input"]
        task = {
            key: value for key, value in item.items() if key not in {"sample_id", "path", "sha256"}
        }
        identity = canonical_json_sha256(task, allow_nan=False)
        if task_identity is not None and identity != task_identity:
            raise ValueError("sequence task inputs or verifier specification changed")
        task_identity = identity
        if row["status"] not in {"success", "error", "missing"}:
            raise ValueError(f"invalid sequence source status: {row['status']}")
        measurements = {}
        for rule in spec.requirements:
            value = None
            reason = None
            if step < rule.active_from:
                status = "not_active"
            elif row["status"] != "success":
                status, reason = "unknown", row["status"]
            elif rule.axis not in row["result"]["scores"]:
                status, reason = "unknown", "missing_axis"
            else:
                value = row["result"]["scores"][rule.axis]
                # Validate public in-memory callers as well as persisted records.
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
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
                "media_sha256": item["sha256"],
                "source_status": row["status"],
                "requirements": measurements,
                **({"error": row["error"]} if row["status"] == "error" else {}),
            }
        )
    summaries = {}
    for rule in spec.requirements:
        values = observations[rule.name]
        summaries[rule.name] = {
            "first_satisfied_step": next(
                (i for i, value in enumerate(values) if value["status"] == "pass"), None
            ),
            "regression_steps": [
                i for i, value in enumerate(values) if value["transition"] == "regressed"
            ],
            "improvement_steps": [
                i for i, value in enumerate(values) if value["transition"] == "improved"
            ],
            "unknown_steps": [i for i, value in enumerate(values) if value["status"] == "unknown"],
            "final_status": values[-1]["status"],
        }
    final = {row["final_status"] for row in summaries.values()}
    report = {
        "schema": "vrl.edit-sequence-audit.v1",
        "evaluation_run_id": evaluation["run_id"],
        "scoring_config_hash": canonical_json_sha256(evaluation["config"], allow_nan=False),
        "spec": spec.model_dump(mode="json"),
        "task_identity": task_identity,
        "coverage_complete": not any(row["unknown_steps"] for row in summaries.values()),
        "final_requirements_met": False
        if "fail" in final
        else None
        if "unknown" in final
        else True,
        "requirements": summaries,
        "states": states,
        "limitations": [
            "Thresholds are operator-specified measurement criteria, not calibrated quality guarantees.",
            "Unknown intervals are not assigned regression or improvement events.",
            "Declared state order does not attest generation lineage or prove learned planning.",
        ],
    }
    return {"report_id": canonical_json_sha256(report, allow_nan=False), **report}
