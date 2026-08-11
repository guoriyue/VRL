"""Build typed runtime config objects from merged YAML."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from omegaconf import DictConfig, OmegaConf

from vrl.config.algorithm import algorithm_config_class
from vrl.config.precision import (
    PrecisionPolicy,
    resolve_precision_policy,
)
from vrl.config.reward_inference import (
    RewardInferenceConfig,
    parse_reward_inference_config,
)
from vrl.config.schema import RewardConfig, RootConfig
from vrl.config.validation import (
    path_exists,
    require,
    validate_training_config,
)

if TYPE_CHECKING:
    from vrl.algorithms.dpo import DiffusionDPOConfig
    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.checkpointing import TrainingResumeConfig
    from vrl.trainers.core.types import PrecisionDriftGuardConfig
    from vrl.trainers.offline import OfflineDPOTrainerConfig
    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig


@dataclass(frozen=True, slots=True)
class RewardRuntimeConfig:
    """Resolved reward weights and per-component runtime kwargs."""

    weights: dict[str, float]
    kwargs: dict[str, dict[str, Any]]
    # Per-component inference deployment, resolved once here (compute-once) so GPU
    # placement reads it off this bundle instead of re-walking the raw reward cfg.
    inference_configs: dict[str, RewardInferenceConfig]

    def __post_init__(self) -> None:
        component_names = set(self.weights)
        for field_name, component_map in (
            ("kwargs", self.kwargs),
            ("inference_configs", self.inference_configs),
        ):
            names = set(component_map)
            if names != component_names:
                missing = sorted(component_names - names)
                unknown = sorted(names - component_names)
                raise ValueError(
                    f"reward runtime {field_name} keys must match component keys; "
                    f"missing={missing}, unknown={unknown}",
                )
        for name, component_kwargs in self.kwargs.items():
            for key in ("sleep_offload", "memory_parking_residual_bytes_limit"):
                if key in component_kwargs:
                    raise ValueError(
                        f"reward.kwargs.{name}.{key} is topology-derived and cannot "
                        "be set in YAML; remove it and select shared or dedicated "
                        "reward GPU ownership under distributed.resources.reward",
                    )

    @property
    def all_external_inference(self) -> bool:
        """Whether every configured component executes through an HTTP service."""

        return bool(self.inference_configs) and all(
            inference.kind == "http" for inference in self.inference_configs.values()
        )


@dataclass(frozen=True, slots=True)
class BuiltConfigs:
    """Named outputs derived from one validated public config."""

    root: RootConfig
    algorithm: Any
    precision: PrecisionPolicy
    trainer: TrainerConfig | None
    reward: RewardRuntimeConfig | None
    resume: TrainingResumeConfig


def build_precision_split_safety_configs() -> tuple[
    PrecisionCorrectionConfig,
    PrecisionDriftGuardConfig,
]:
    """Build the production correction and guard policy for a precision split.

    Hardware validation probes consume this same typed source so a measured gate
    cannot silently validate thresholds different from live training.
    """

    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.core.types import PrecisionDriftGuardConfig

    return (
        PrecisionCorrectionConfig(
            tis_mode="truncate",
            rs_mode="seq_mean_k1",
        ),
        PrecisionDriftGuardConfig(
            mode="fail",
            max_abs_log_ratio=math.log(10.0),
            max_ratio_abs_dev=9.0,
            fail_on_nonfinite=True,
        ),
    )


def _dataclass_field_names(cls: type[Any]) -> set[str]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} must be a dataclass type")
    return {field.name for field in fields(cls) if field.init}


def _section_payload_and_missing(
    cls: type[Any],
    cfg: DictConfig,
    path: str,
) -> tuple[dict[str, Any], list[str]]:
    """Select ``cls`` fields from the section; report missing required paths.

    An explicitly null section (``actor.ema: null``) raises instead of
    silently replacing the section with all-defaults — only true absence
    means "use the dataclass defaults". Unknown keys raise: for a typed
    section the dataclass is the complete vocabulary, so a typo'd
    hyperparameter must refuse to start rather than silently train with the
    default behind a lint warning.
    """

    node = OmegaConf.select(cfg, path)
    if node is None:
        if path_exists(cfg, path):
            raise ValueError(
                f"config section {path} is null; delete the key or fill the section",
            )
        raw: dict[str, Any] = {}
    else:
        raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
        if not isinstance(raw, dict):
            raise ValueError(f"config section {path} must be a mapping")

    allowed = _dataclass_field_names(cls)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        keys = ", ".join(f"{path}.{key}" for key in unknown)
        raise ValueError(f"unknown {cls.__name__} key(s): {keys}")
    payload = {key: value for key, value in raw.items() if key in allowed}
    # Required = no default, torch signature semantics.
    required = {
        field.name
        for field in fields(cls)
        if field.init and field.default is MISSING and field.default_factory is MISSING
    }
    missing = sorted(f"{path}.{name}" for name in required - set(payload))
    return payload, missing


def _dataclass_payload(cls: type[Any], node: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} config must be a mapping")
    allowed = _dataclass_field_names(cls)
    ignored_keys = {"kind", "kl_reward_coef"}
    unknown = sorted(set(raw) - allowed - ignored_keys)
    if unknown:
        fields_text = ", ".join(f"algorithm.{key}" for key in unknown)
        raise ValueError(f"unknown {cls.__name__} config field(s): {fields_text}")
    return {key: value for key, value in raw.items() if key in allowed}


def _validate_yaml_home(
    field_name: str,
    home: str,
    *,
    owner: str = "TrainerConfig",
) -> None:
    """Reject metadata addresses whose top-level section is not a known one.

    Guards the silent failure mode of a typo'd address on an OPTIONAL field
    (it would fall back to the default instead of reading the user's value).
    The valid-section vocabulary is derived from the schema's RootConfig —
    the existing source of truth for top-level sections — so this check can
    never drift into rejecting a legitimately added section.
    """

    from vrl.config.schema import RootConfig

    top = home.split(".", 1)[0]
    if top not in RootConfig.model_fields:
        expected = ", ".join(sorted(RootConfig.model_fields))
        raise AssertionError(
            f"{owner}.{field_name} declares unknown yaml home {home!r}; "
            f"the top-level section must be one of: {expected}",
        )


def _optional_non_negative_int(cfg: DictConfig, path: str) -> int | None:
    if not path_exists(cfg, path):
        return None
    value = require(cfg, path)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer (got {value!r})")
    return value


def build_online_batch_plan(cfg: DictConfig) -> OnlineBatchPlan:
    """Resolve public size/count inputs into one canonical optimizer batch plan."""

    from vrl.trainers.online.config import OnlineBatchPlan

    base = OnlineBatchPlan(
        prompts_per_batch=require(cfg, "rollout.prompts_per_batch"),
        n_samples_per_prompt=require(cfg, "rollout.n_samples_per_prompt"),
    )
    prompts = base.prompts_per_batch
    accumulation_steps = _optional_non_negative_int(
        cfg,
        "actor.gradient_accumulation_steps",
    )
    microbatch_size = _optional_non_negative_int(cfg, "actor.microbatch_size")

    active_accumulation = int(accumulation_steps or 0)
    active_microbatch = int(microbatch_size or 0)
    if active_accumulation > 0 and active_microbatch > 0:
        if active_accumulation * active_microbatch != prompts:
            raise ValueError(
                "actor.microbatch_size * actor.gradient_accumulation_steps "
                f"must equal rollout.prompts_per_batch "
                f"({active_microbatch} * {active_accumulation} != {prompts}); "
                "set only one of them.",
            )
    elif active_microbatch > 0:
        if prompts % active_microbatch != 0:
            raise ValueError(
                "actor.microbatch_size must evenly divide "
                f"rollout.prompts_per_batch ({prompts} % {active_microbatch} != 0)",
            )
        active_accumulation = prompts // active_microbatch

    payload: dict[str, Any] = {
        "prompts_per_batch": prompts,
        "n_samples_per_prompt": base.n_samples_per_prompt,
    }
    if active_accumulation > 0:
        payload["gradient_accumulation_steps"] = active_accumulation
    if path_exists(cfg, "actor.replay_samples_per_chunk"):
        raise ValueError(
            "actor.replay_samples_per_chunk was renamed to "
            "actor.replay_samples_per_batch; update the config key",
        )
    for field_name in ("replay_samples_per_batch", "host_memory_budget_fraction"):
        path = f"actor.{field_name}"
        if path_exists(cfg, path):
            payload[field_name] = require(cfg, path)
    return OnlineBatchPlan(**payload)


def build_trainer_config(
    cfg: DictConfig,
    *,
    precision: PrecisionPolicy | None = None,
):
    """Slice merged YAML into ``TrainerConfig``.

    The layout is derived from each field's ``metadata={"yaml": ...}`` on the
    dataclass (section name for scalars, dotted path for nested sections,
    "bridged" for builder-computed values), and requiredness from the field
    defaults — the dataclass is the single declaration of both. Missing
    required keys across sections and scalars are
    collected and reported together with full YAML paths.
    """

    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

    trainer_block = getattr(cfg, "trainer", None)
    if trainer_block is not None and "eval" in trainer_block:
        raise ValueError(
            "trainer.eval was removed: online training no longer runs fixed eval. "
            "Evaluate saved checkpoints with a script under vrl/scripts/eval; "
            "for Cosmos Predict2.5 + Kling use "
            "`python -m vrl.scripts.eval.cosmos_predict25_kling_eval`.",
        )
    orchestration_block = (
        getattr(trainer_block, "rollout_orchestration", None)
        if trainer_block is not None
        else None
    )
    if orchestration_block is not None and "require_separate_gpus" in orchestration_block:
        raise ValueError(
            "trainer.rollout_orchestration.require_separate_gpus was removed: "
            "rollout topology is derived from resolved GPU ownership. Shared GPUs "
            "use strict_on_policy with distributed.resources.rollout.gpu_pool=trainer; "
            "continuous requires disjoint trainer and rollout GPUs.",
        )

    hints = get_type_hints(TrainerConfig)
    payload: dict[str, Any] = {}
    missing: list[str] = []

    # ``batch_plan`` is bridged because its public inputs span rollout and actor.
    # Derive its required public paths from the plan fields so missing-key
    # reporting cannot drift when the typed plan changes.
    for plan_field in fields(OnlineBatchPlan):
        home = plan_field.metadata.get("yaml")
        if home is None:
            raise AssertionError(
                f"OnlineBatchPlan.{plan_field.name} does not declare its YAML home "
                "(field metadata {'yaml': ...})",
            )
        _validate_yaml_home(plan_field.name, home, owner="OnlineBatchPlan")
        path = f"{home}.{plan_field.name}"
        if (
            plan_field.default is MISSING
            and plan_field.default_factory is MISSING
            and not path_exists(cfg, path)
        ):
            missing.append(path)

    for f in fields(TrainerConfig):
        if not f.init:
            continue
        home = f.metadata.get("yaml")
        if home is None:
            raise AssertionError(
                f"TrainerConfig.{f.name} does not declare its YAML home "
                "(field metadata {'yaml': ...})",
            )
        if home == "bridged":
            continue
        _validate_yaml_home(f.name, home)
        field_cls = hints[f.name]
        if is_dataclass(field_cls):
            section_payload, section_missing = _section_payload_and_missing(
                field_cls,
                cfg,
                home,
            )
            if section_missing:
                missing.extend(section_missing)
            else:
                payload[f.name] = field_cls(**section_payload)
        else:
            path = f"{home}.{f.name}"
            if path_exists(cfg, path):
                payload[f.name] = require(cfg, path)
            elif f.default is MISSING and f.default_factory is MISSING:
                missing.append(path)

    if missing:
        raise ValueError("config missing required key(s): " + ", ".join(sorted(missing)))

    # Resolve the public policy once; trainer fields are its runtime projection.
    precision = precision or resolve_precision_policy(cfg)
    payload.update(
        batch_plan=build_online_batch_plan(cfg),
        train_precision=precision.training.label,
        rollout_precision=precision.rollout.label,
    )
    # On a rollout/train precision split, the correction mechanism is an
    # implementation detail the user should not have to spell out: default to
    # TIS/RS correction plus a catastrophic-drift guard. Explicit expert
    # trainer.precision_* blocks are still respected.
    if not precision.stages_match:
        correction, guard = build_precision_split_safety_configs()
        if not path_exists(cfg, "trainer.precision_correction"):
            payload["precision_correction"] = correction
        if not path_exists(cfg, "trainer.precision_drift_guard"):
            payload["precision_drift_guard"] = guard

    return TrainerConfig(**payload)


def build_offline_dpo_trainer_config(
    cfg: DictConfig,
    dpo_config: DiffusionDPOConfig,
    *,
    train_batch_size: int,
    gradient_accumulation_steps: int,
) -> OfflineDPOTrainerConfig:
    """Slice merged YAML into ``OfflineDPOTrainerConfig``.

    The offline twin of ``build_trainer_config``: same public ``actor.*``
    optimizer section, projected into the offline trainer instead. It lives here
    rather than on the config dataclass because ``vrl.trainers.offline`` holds no
    YAML knowledge, and rather than in the recipe script because a public config
    projection is not a script-private detail.
    """

    from vrl.trainers.core.types import OptimConfig
    from vrl.trainers.offline import OfflineDPOTrainerConfig

    raw_optim = OmegaConf.to_container(
        cfg.actor.optim,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(raw_optim, dict):
        raise ValueError("actor.optim must be a mapping")
    optim = OptimConfig(**raw_optim)
    if optim.optim_8bit:
        raise ValueError(
            "actor.optim.optim_8bit=true is not supported by OfflineDPOTrainer; "
            "use AdamW/Adafactor without 8-bit optimizer state",
        )
    use_adafactor = bool(require(cfg, "actor.use_adafactor"))
    if use_adafactor:
        adam_only_keys = sorted({"adam_beta1", "adam_beta2", "eps"} & raw_optim.keys())
        if adam_only_keys:
            paths = ", ".join(f"actor.optim.{key}" for key in adam_only_keys)
            raise ValueError(
                f"actor.use_adafactor=true does not consume AdamW-only key(s): {paths}",
            )

    scale_lr = bool(require(cfg, "actor.scale_lr"))
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    lr = float(optim.lr) * effective_batch_size if scale_lr else float(optim.lr)
    max_grad_norm = OmegaConf.select(cfg, "actor.max_norm")
    if max_grad_norm is None:
        max_grad_norm = OfflineDPOTrainerConfig().max_grad_norm
    return OfflineDPOTrainerConfig(
        beta=float(dpo_config.beta),
        sft_weight=float(dpo_config.sft_weight),
        lr=lr,
        adam_beta1=float(optim.adam_beta1),
        adam_beta2=float(optim.adam_beta2),
        adam_weight_decay=float(optim.weight_decay),
        adam_epsilon=float(optim.eps),
        max_grad_norm=float(max_grad_norm),
        gradient_accumulation_steps=gradient_accumulation_steps,
        prediction_type=str(require(cfg, "actor.prediction_type")),
        use_adafactor=use_adafactor,
    )


def build_algorithm_config(cfg: DictConfig):
    """Dispatch on ``algorithm.kind`` and return the typed algorithm config."""

    if "algorithm" not in cfg:
        raise ValueError("config missing `algorithm` section")
    kind = str(require(cfg, "algorithm.kind"))
    cls = algorithm_config_class(kind)
    return cls(**_dataclass_payload(cls, cfg.algorithm))


def build_reward_config(cfg: DictConfig | RewardConfig) -> RewardRuntimeConfig:
    """Resolve one public reward section into its runtime config.

    Zero-weight components remain present so they can be scored and logged as
    observation-only safeguards without changing the optimization reward.
    """

    from vrl.config.validation import validate_reward_config

    reward = cfg if isinstance(cfg, RewardConfig) else validate_reward_config(cfg)
    weights = {name: float(weight) for name, weight in reward.components.items()}
    unknown_kwargs = sorted(set(reward.kwargs) - set(weights))
    if unknown_kwargs:
        keys = ", ".join(f"reward.kwargs.{name}" for name in unknown_kwargs)
        raise ValueError(f"reward kwargs configured for unknown component(s): {keys}")
    kwargs = {name: dict(reward.kwargs.get(name) or {}) for name in weights}
    inference_configs = {
        name: parse_reward_inference_config(
            (kwargs.get(name) or {}).get("inference"),
            context=f"reward.kwargs.{name}.inference",
        )
        for name in weights
    }
    return RewardRuntimeConfig(
        weights=weights,
        kwargs=kwargs,
        inference_configs=inference_configs,
    )


def build_configs(cfg: DictConfig) -> BuiltConfigs:
    """Bundle typed configs for downstream training scripts."""

    from vrl.trainers.checkpointing import (
        prepare_model_config_for_training_resume,
        resolve_training_resume_config,
    )

    resume = resolve_training_resume_config(cfg)
    # A full checkpoint, not model.lora.path, owns trainable state on resume.
    # Normalize the raw source before typed parsing so persisted config and all
    # runtime consumers receive one truthful model tree.
    prepare_model_config_for_training_resume(cfg, resume)
    root, precision = validate_training_config(cfg)
    algorithm = build_algorithm_config(cfg)
    is_offline_dpo = root.algorithm is not None and root.algorithm.kind == "diffusion_dpo"
    trainer = None if is_offline_dpo else build_trainer_config(cfg, precision=precision)
    reward = build_reward_config(root.reward) if root.reward is not None else None
    if not is_offline_dpo:
        if reward is None:
            raise ValueError("online recipe requires a reward section")
        if not any(weight > 0 for weight in reward.weights.values()):
            raise ValueError("At least one reward component must have weight > 0.")
    return BuiltConfigs(
        root=root,
        algorithm=algorithm,
        precision=precision,
        trainer=trainer,
        reward=reward,
        resume=resume,
    )


__all__ = [
    "BuiltConfigs",
    "RewardRuntimeConfig",
    "build_algorithm_config",
    "build_configs",
    "build_offline_dpo_trainer_config",
    "build_online_batch_plan",
    "build_reward_config",
    "build_trainer_config",
]
