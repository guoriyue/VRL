"""Test-owned profile resolution and inference-evidence evaluation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from tests.quality.contracts import (
    InferencePhase,
    ModelIdentity,
    QualityProfile,
    QualityProtocol,
)
from tests.quality.evidence import evaluate_artifact_evidence
from tests.quality.io import (
    canonical_sha256,
    inference_environment,
    read_json_object,
    sha256_file,
    source_tree_sha256,
)
from vrl.config.validation import require
from vrl.rollouts.family_names import normalize_rollout_family

ASSET_ROOT = Path(__file__).with_name("protocols")


def family_profile_path(family: str) -> Path:
    """Map a canonical registry key directly to its test-owned approval fixture."""

    return ASSET_ROOT / "families" / f"{family}.yaml"


def load_quality_profile(cfg: DictConfig) -> tuple[QualityProfile, QualityProtocol]:
    family = normalize_rollout_family(str(require(cfg, "model.family")))
    profile_path = family_profile_path(family)
    profile_raw = _read_yaml_mapping(profile_path)
    profile = QualityProfile.from_mapping(profile_raw)
    protocol_raw = _read_yaml_mapping(ASSET_ROOT / profile.extends)
    protocol = QualityProtocol.from_mapping(
        {**dict(protocol_raw), **dict(profile.protocol_overrides)},
    )
    identity = resolve_model_identity(cfg)
    if identity not in profile.model_identities:
        allowed = [asdict(candidate) for candidate in profile.model_identities]
        raise ValueError(
            "resolved primary model identity is not approved by the test fixture: "
            f"family={family!r}, identity={asdict(identity)!r}, allowed={allowed!r}",
        )
    return profile, protocol


def load_quality_protocol(cfg: DictConfig) -> QualityProtocol:
    """Public test facade returning only the resolved behavioral protocol."""

    return load_quality_profile(cfg)[1]


def resolve_model_identity(cfg: DictConfig) -> ModelIdentity:
    path = str(OmegaConf.select(cfg, "model.path", default="") or "").strip()
    if not path:
        raise ValueError("inference preflight requires model.path")
    revision_raw = OmegaConf.select(cfg, "model.revision", default=None)
    revision = None if revision_raw is None else str(revision_raw).strip() or None
    return ModelIdentity(path=path, revision=revision)


def quality_provenance(cfg: DictConfig) -> dict[str, str]:
    profile, protocol = load_quality_profile(cfg)
    identity = resolve_model_identity(cfg)
    return {
        "source_tree_sha256": source_tree_sha256(),
        "environment_sha256": canonical_sha256(inference_environment()),
        "config_sha256": canonical_sha256(_quality_config_payload(cfg)),
        "protocol_sha256": canonical_sha256(
            {"profile": asdict(profile), "protocol": asdict(protocol)},
        ),
        "model_identity": (
            identity.path if identity.revision is None else f"{identity.path}@{identity.revision}"
        ),
    }


def evaluate_inference_preflight(
    cfg: DictConfig,
    evidence_path: str | Path,
    *,
    phase: InferencePhase = InferencePhase.BASE,
    checkpoint_file: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one evidence manifest in-process and return a pytest-friendly report."""

    family = normalize_rollout_family(str(require(cfg, "model.family")))
    profile, protocol = load_quality_profile(cfg)
    manifest_path = Path(evidence_path).expanduser().resolve()
    manifest = read_json_object(manifest_path)
    checkpoint_sha256 = (
        sha256_file(Path(checkpoint_file).expanduser().resolve())
        if checkpoint_file is not None
        else None
    )
    assertions, artifacts, identities = evaluate_artifact_evidence(
        manifest,
        manifest_path=manifest_path,
        protocol=protocol,
        family=family,
        phase=phase,
        checkpoint_sha256=checkpoint_sha256,
        expected_provenance=quality_provenance(cfg),
    )
    return {
        "verdict": "pass" if assertions and all(item.passed for item in assertions) else "fail",
        "phase": phase.value,
        "family": family,
        "comparison_mode": (
            "strict" if protocol.min_native_similarity > 0.0 else "guarded_divergence"
        ),
        "protocol": {
            "profile": str(family_profile_path(family)),
            "sha256": canonical_sha256(
                {"profile": asdict(profile), "protocol": asdict(protocol)},
            ),
        },
        "evidence": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "checkpoint_sha256": checkpoint_sha256,
        "identities": identities,
        "artifacts": artifacts,
        "assertions": [item.to_dict() for item in assertions],
    }


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing inference-quality fixture: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"inference-quality fixture must contain a mapping: {path}")
    return value


def _quality_config_payload(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("resolved training config must be a mapping")
    payload = copy.deepcopy(raw)
    trainer = payload.get("trainer")
    if isinstance(trainer, dict):
        trainer.pop("resume_from", None)
        trainer.pop("output_dir", None)
    model = payload.get("model")
    if isinstance(model, dict):
        lora = model.get("lora")
        if isinstance(lora, dict):
            lora.pop("path", None)
    return payload
