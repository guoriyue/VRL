"""Pydantic typed boundary for merged training configs.

OmegaConf handles YAML defaults, interpolation, and CLI overrides.
Pydantic validates the fully-resolved, merged container after OmegaConf finishes.
The schema intentionally uses extra="ignore" during migration so that YAML fields
not yet represented here are silently accepted rather than rejected.
"""

from __future__ import annotations

from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# ── Kling VideoReward kwargs model ────────────────────────────────────────────


class KlingVideoRewardKwargs(BaseModel):
    """Validates non-production Kling VideoReward kwargs.

    Two scopes handled here:
      - removed top-level fields raise immediately with clear migration messages
      - inference_runtime and scheduling are checked unconditionally

    Production-only checks live in RootConfig._validate_production_kling_video_reward,
    gated on production.kling_video_reward.enabled.
    """

    model_config = ConfigDict(extra="ignore")

    inference_runtime: str
    reward_name: str
    score_key: str
    scheduling: str = "sync"
    worker_config: dict[str, Any]
    # captured for production cross-field check in RootConfig
    media_type: str | None = None
    artifact_format: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_top_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # backend has a specific migration message pointing to inference_runtime
        if "backend" in data:
            raise ValueError(
                "reward.kwargs.kling_video_reward.backend is no longer supported; "
                "use reward.kwargs.kling_video_reward.inference_runtime=ray",
            )
        if not isinstance(data.get("worker_config"), dict):
            raise ValueError(
                "reward.kwargs.kling_video_reward.worker_config must be a mapping"
            )
        removed = sorted(
            k for k in ("enqueue_url", "fetch_url", "token", "poll_interval_s",
                        "max_wait_s", "stub_scale", "device")
            if k in data
        )
        if removed:
            raise ValueError(
                "reward.kwargs.kling_video_reward no longer supports external "
                "reward endpoint fields: "
                + ", ".join(removed),
            )
        return data

    @model_validator(mode="after")
    def _validate_runtime_constraints(self) -> KlingVideoRewardKwargs:
        if self.inference_runtime != "ray":
            raise ValueError(
                "reward.kwargs.kling_video_reward.inference_runtime must be 'ray'"
            )
        if self.scheduling != "sync":
            raise ValueError(
                "reward.kwargs.kling_video_reward.scheduling currently supports only 'sync'"
            )
        return self


VideoRewardKwargs = KlingVideoRewardKwargs


# ── Reward section ────────────────────────────────────────────────────────────


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    components: dict[str, Any]
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reward(self) -> RewardConfig:
        # All kwargs entries must be mappings (or null)
        for name, sub in self.kwargs.items():
            if sub is not None and not isinstance(sub, dict):
                raise ValueError(
                    f"reward.kwargs.{name} must be a mapping, got {type(sub).__name__}",
                )

        # Per-component: weight must be numeric and >= 0
        for name, weight_raw in self.components.items():
            try:
                weight = float(weight_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"reward.components.{name} must be numeric, got {weight_raw!r}",
                ) from exc
            if weight < 0:
                raise ValueError(f"reward.components.{name} must be >= 0, got {weight}")
            if weight == 0:
                continue
            # Kling VideoReward has constraints that scripts cannot self-heal.
            if name in {"kling_video_reward", "video_reward"}:
                sub = self.kwargs.get(name)
                if not isinstance(sub, dict):
                    raise ValueError(
                        f"config missing required field: reward.kwargs.{name} "
                        f"(component {name!r} has non-zero weight)",
                    )
                try:
                    KlingVideoRewardKwargs.model_validate(sub)
                except ValidationError as exc:
                    first = exc.errors(include_url=False)[0]
                    error_type = first["type"]
                    msg = first["msg"]
                    loc = ".".join(str(p) for p in first["loc"])
                    if error_type == "missing":
                        raise ValueError(
                            f"config missing required field: reward.kwargs.{name}.{loc}"
                        ) from exc
                    if msg.startswith("Value error, "):
                        msg = msg[len("Value error, "):]
                    raise ValueError(msg) from exc

        return self


# ── Algorithm section ─────────────────────────────────────────────────────────


class AlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal[
        "grpo", "token_grpo", "token_grpo_multisegment", "diffusion_dpo", "diffusion_nft"
    ]
    # Deprecated field; captured explicitly so the validator can emit a clear message
    # rather than silently ignoring it under extra="ignore".
    adv_estimator: str | None = None

    @model_validator(mode="after")
    def _reject_adv_estimator(self) -> AlgorithmConfig:
        if self.adv_estimator is not None:
            raise ValueError(
                "algorithm.adv_estimator is no longer supported; use algorithm.kind"
            )
        return self


# ── Data section ──────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    loader: Literal["pickapic_preference", "prompt_manifest", "prompt_image_manifest"]
    manifest: str | None = None
    eval_manifest: str | None = None
    preprocessing: dict[str, Any] | None = None
    sampler: dict[str, Any] | None = None
    dataset_name: str | None = None
    split: str | None = None
    cache_dir: str | None = None
    max_train_samples: Any = None
    task_type: str | None = None

    @model_validator(mode="after")
    def _validate_data(self) -> DataConfig:
        # loader validity already enforced by the Literal field type
        if self.loader == "prompt_manifest":
            if not self.manifest:
                raise ValueError("config missing required field: data.manifest")
            if self.preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            sampler = self.sampler or {}
            sampler_type = str(sampler.get("type", "")) if "type" in sampler else ""
            if not sampler_type:
                raise ValueError("config missing required field: data.sampler.type")
            valid_samplers = {"random_without_replacement", "sequential_window"}
            if sampler_type not in valid_samplers:
                expected = " / ".join(sorted(valid_samplers))
                raise ValueError(
                    f"unknown data.sampler.type={sampler_type!r}; expected {expected}"
                )

        if self.loader == "prompt_image_manifest":
            if not self.manifest:
                raise ValueError("config missing required field: data.manifest")
            if not self.eval_manifest:
                raise ValueError("config missing required field: data.eval_manifest")
            if self.preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            for field in ("format", "image_field", "caption_field", "media_type", "conditioning"):
                if field not in self.preprocessing:
                    raise ValueError(
                        f"config missing required field: data.preprocessing.{field}"
                    )
            sampler = self.sampler or {}
            sampler_type = str(sampler.get("type", "")) if "type" in sampler else ""
            if not sampler_type:
                raise ValueError("config missing required field: data.sampler.type")
            valid_samplers = {"random_without_replacement", "sequential_window"}
            if sampler_type not in valid_samplers:
                expected = " / ".join(sorted(valid_samplers))
                raise ValueError(
                    f"unknown data.sampler.type={sampler_type!r}; expected {expected}"
                )

        if self.loader == "pickapic_preference":
            for field in ("dataset_name", "split", "cache_dir"):
                # Allow empty strings (matches require()'s semantics — only None/absent is invalid)
                if getattr(self, field) is None:
                    raise ValueError(f"config missing required field: data.{field}")
            if self.preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            for field in ("resolution", "random_crop", "horizontal_flip"):
                if field not in self.preprocessing:
                    raise ValueError(
                        f"config missing required field: data.preprocessing.{field}"
                    )
            sampler = self.sampler or {}
            for field in ("shuffle", "drop_last", "dataloader_num_workers"):
                if field not in sampler:
                    raise ValueError(
                        f"config missing required field: data.sampler.{field}"
                    )

        return self


# ── Supporting sections for cross-field validation ────────────────────────────


class RolloutConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sde: dict[str, Any] | None = None
    noise_level: float | None = None
    final_image_policy: str | None = None
    n: int | None = None
    n_samples_per_prompt: int | None = None
    rollout_batch_size: int | None = None


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # keep as raw dict to avoid nested model complexity during migration
    r1: dict[str, Any] | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    family: str | None = None


class ProductionKlingVideoRewardConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    report_path: str | None = None


ProductionVideoRewardConfig = ProductionKlingVideoRewardConfig


class ProductionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kling_video_reward: ProductionKlingVideoRewardConfig | None = None
    video_reward: ProductionKlingVideoRewardConfig | None = None


# ── Root config ───────────────────────────────────────────────────────────────


class RootConfig(BaseModel):
    """Top-level typed boundary for all training config sections.

    Only sections relevant to validation are modelled; the rest are silently
    accepted via extra="ignore" so that YAML fields not yet in the schema never
    break existing runs during the migration period.
    """

    model_config = ConfigDict(extra="ignore")

    algorithm: AlgorithmConfig | None = None
    data: DataConfig | None = None
    reward: RewardConfig | None = None
    rollout: RolloutConfig | None = None
    model: ModelConfig | None = None
    sampling: SamplingConfig | None = None
    production: ProductionConfig | None = None

    @model_validator(mode="after")
    def _cross_field_validate(self) -> RootConfig:
        algo = self.algorithm
        if algo is None:
            return self

        kind = algo.kind
        rollout = self.rollout

        # grpo / diffusion_nft: SDE type must be sde or cps
        if kind in {"grpo", "diffusion_nft"}:
            sde = rollout.sde if rollout else None
            sde_type = str(sde.get("type", "")) if isinstance(sde, dict) else ""
            if sde_type not in {"sde", "cps"}:
                raise ValueError("rollout.sde.type must be 'sde' or 'cps'")

        # token_grpo: nextstep_1 family requires rollout.noise_level
        if kind == "token_grpo":
            model_family = self.model.family if self.model else None
            if model_family == "nextstep_1" and (
                rollout is None or rollout.noise_level is None
            ):
                raise ValueError("config missing required field: rollout.noise_level")

        # token_grpo_multisegment: janus_pro only, final_image_policy must match sampling
        if kind == "token_grpo_multisegment":
            model_family = (self.model.family or "") if self.model else ""
            if model_family != "janus_pro":
                raise ValueError(
                    "token_grpo_multisegment currently requires model.family=janus_pro"
                )
            final_image_policy = (rollout.final_image_policy or "") if rollout else ""
            if final_image_policy not in {"always_generate", "use_selfcheck"}:
                raise ValueError(
                    "rollout.final_image_policy must be 'always_generate' or 'use_selfcheck'"
                )
            sampling_r1 = (self.sampling.r1 or {}) if self.sampling else {}
            sampling_final_policy = str(sampling_r1.get("final_image_policy", ""))
            if sampling_final_policy != final_image_policy:
                raise ValueError(
                    "sampling.r1.final_image_policy must match rollout.final_image_policy"
                )

        # production Kling VideoReward structural rules (path existence stays separate)
        prod = self.production
        production_enabled = bool(
            prod
            and (
                (prod.kling_video_reward and prod.kling_video_reward.enabled)
                or (prod.video_reward and prod.video_reward.enabled)
            )
        )
        if production_enabled:
            self._validate_production_kling_video_reward()

        return self

    def _validate_production_kling_video_reward(self) -> None:
        vr_kwargs: dict[str, Any] = {}
        if self.reward and self.reward.kwargs:
            vr_kwargs = (
                self.reward.kwargs.get("kling_video_reward")
                or self.reward.kwargs.get("video_reward")
                or {}
            )

        media_type = str(vr_kwargs.get("media_type", ""))
        if media_type != "video":
            raise ValueError(
                "production.kling_video_reward requires "
                "reward.kwargs.kling_video_reward.media_type=video"
            )
        artifact_format = str(vr_kwargs.get("artifact_format", ""))
        if artifact_format != "mp4":
            raise ValueError("production.kling_video_reward requires artifact_format=mp4")
        reward_name = str(vr_kwargs.get("reward_name", "")).strip()
        if not reward_name:
            raise ValueError(
                "production.kling_video_reward requires "
                "reward.kwargs.kling_video_reward.reward_name"
            )

        worker_config = vr_kwargs.get("worker_config") or {}
        forbidden = sorted(
            k for k in ("backend", "backend_import_path", "backend_code_dir",
                        "import_path", "model_subdir", "score_key_map", "model_factory")
            if k in worker_config
        )
        if forbidden:
            raise ValueError(
                "production.kling_video_reward worker_config should name the reward "
                "model directly; "
                f"remove extra loader fields: {', '.join(forbidden)}",
            )

        task_type = str((self.data.task_type or "") if self.data else "")
        if task_type not in {"text_to_video", "image_to_video", "video2world"}:
            raise ValueError(
                "production.kling_video_reward requires "
                "data.task_type=text_to_video, image_to_video, or video2world"
            )


# ── Parse boundary ────────────────────────────────────────────────────────────


def _extract_error_message(exc: ValidationError) -> str:
    """Extract a clean ValueError message from a Pydantic ValidationError."""
    first = exc.errors(include_url=False)[0]
    error_type = first["type"]
    msg = first["msg"]
    loc = ".".join(str(p) for p in first["loc"])
    # Missing required field — remap to repo-standard message format
    if error_type == "missing":
        return f"config missing required field: {loc}"
    # Literal enum mismatch — reformat to "unknown {loc}={input!r}; expected ..."
    if error_type == "literal_error":
        input_val = first.get("input", "")
        expected = msg.replace("Input should be", "expected")
        return f"unknown {loc}={input_val!r}; {expected}"
    # ValueError raised inside a validator — strip Pydantic's "Value error, " prefix
    if msg.startswith("Value error, "):
        return msg[len("Value error, "):]
    return msg


def parse_config(cfg: DictConfig) -> RootConfig:
    """Validate a fully-merged, resolved DictConfig through the typed schema.

    OmegaConf resolves interpolations and enforces ??? missing-value semantics;
    Pydantic validates structure, enum discriminators, and cross-field rules.
    """
    try:
        raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue as exc:
        missing_path = getattr(exc, "full_key", None) or str(exc)
        raise ValueError(f"config missing required field: {missing_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("config must be a top-level mapping")

    try:
        return RootConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(_extract_error_message(exc)) from exc


__all__ = [
    "AlgorithmConfig",
    "DataConfig",
    "KlingVideoRewardKwargs",
    "ModelConfig",
    "ProductionConfig",
    "ProductionKlingVideoRewardConfig",
    "ProductionVideoRewardConfig",
    "RewardConfig",
    "RolloutConfig",
    "RootConfig",
    "SamplingConfig",
    "VideoRewardKwargs",
    "parse_config",
]
