"""Build typed runtime config objects from merged YAML."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from vrl.config.algorithm import algorithm_config_class
from vrl.config.precision import (
    PrecisionPolicy,
)
from vrl.config.reward_inference import (
    RewardInferenceConfig,
    parse_reward_inference_config,
)
from vrl.config.schema import RewardConfig, RootConfig
from vrl.config.validation import (
    dataclass_field_names,
    require,
    validate_training_config,
)

if TYPE_CHECKING:
    from vrl.algorithms.dpo import DiffusionDPOConfig
    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.checkpointing import TrainingResumeConfig
    from vrl.trainers.core.types import PrecisionDriftGuardConfig
    from vrl.trainers.offline import OfflineDPOTrainerConfig
    from vrl.trainers.online.config import TrainerConfig


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

    @classmethod
    def from_cfg(cls, cfg: DictConfig | RewardConfig) -> RewardRuntimeConfig:
        """Resolve one public reward section into its runtime config.

        Zero-weight components remain present so they can be scored and logged
        as observation-only safeguards without changing the optimization
        reward.
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
        return cls(
            weights=weights,
            kwargs=kwargs,
            inference_configs=inference_configs,
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


def _dataclass_payload(cls: type[Any], node: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(node, resolve=True, throw_on_missing=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} config must be a mapping")
    allowed = dataclass_field_names(cls)
    ignored_keys = {"kind", "kl_reward_coef"}
    unknown = sorted(set(raw) - allowed - ignored_keys)
    if unknown:
        fields_text = ", ".join(f"algorithm.{key}" for key in unknown)
        raise ValueError(f"unknown {cls.__name__} config field(s): {fields_text}")
    return {key: value for key, value in raw.items() if key in allowed}


def build_offline_dpo_trainer_config(
    cfg: DictConfig,
    dpo_config: DiffusionDPOConfig,
) -> OfflineDPOTrainerConfig:
    """Slice merged YAML into ``OfflineDPOTrainerConfig``.

    The offline twin of ``TrainerConfig.from_cfg``: same public ``actor.*``
    optimizer section, projected into the offline trainer instead. It stays a
    free builder because ``vrl.trainers.offline`` deliberately holds no YAML
    knowledge (unlike the online config, whose fields declare their own YAML
    homes), and it takes two sources — the raw cfg plus the already-built
    algorithm config.
    """

    from vrl.trainers.core.types import OptimConfig
    from vrl.trainers.offline import OfflineDPOTrainerConfig

    train_batch_size = int(require(cfg, "actor.train_batch_size"))
    gradient_accumulation_steps = int(require(cfg, "actor.gradient_accumulation_steps"))

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


def build_configs(cfg: DictConfig) -> BuiltConfigs:
    """Bundle typed configs for downstream training scripts."""

    from vrl.trainers.checkpointing import (
        prepare_model_config_for_training_resume,
        resolve_training_resume_config,
    )
    from vrl.trainers.online.config import TrainerConfig

    resume = resolve_training_resume_config(cfg)
    # A full checkpoint, not model.lora.path, owns trainable state on resume.
    # Normalize the raw source before typed parsing so persisted config and all
    # runtime consumers receive one truthful model tree.
    prepare_model_config_for_training_resume(cfg, resume)
    root, precision = validate_training_config(cfg)
    algorithm = build_algorithm_config(cfg)
    is_offline_dpo = root.algorithm is not None and root.algorithm.kind == "diffusion_dpo"
    trainer = None if is_offline_dpo else TrainerConfig.from_cfg(cfg, precision=precision)
    reward = RewardRuntimeConfig.from_cfg(root.reward) if root.reward is not None else None
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
]
