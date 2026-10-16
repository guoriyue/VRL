"""Qwen-Image-Layered supervised flow matching on cached layered scenes (offline).

The unified ``vrl.scripts.train`` entry point dispatches
``experiment/qwen_image_layered/offline_sft_*`` configs here. The policy is the
family's replay model (transformer and scheduler only: the scenes are already
encoded, so no VAE or text encoder loads), wrapped with the configured LoRA and
started from ``model.lora.path``. Each step draws one cached scene, noises its
target frames at a schedule position inside ``algorithm.timestep_index_range``
and penalizes the velocity error under the scene's token weights
(``data.token_weights``). ``actor.stream_frozen_blocks`` keeps the frozen
blocks on the host for a 32 GB card.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

if TYPE_CHECKING:
    # Config dispatch imports this module to find the entrypoint; keep torch
    # out of that import path.
    import torch

logger = logging.getLogger(__name__)

FAMILY = "qwen_image_layered"


def layered_forward(
    model: Any,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    conditioning: dict[str, Any],
) -> torch.Tensor:
    """The ``ForwardFn`` of ``OfflineSFTTrainer`` for cached layered scenes."""

    device = noisy_latents.device
    mask = conditioning["prompt_embeds_mask"]
    return model.predict_velocity(
        noisy_latents,
        timesteps,
        condition_latents=conditioning["condition_latents"].to(device),
        prompt_embeds=conditioning["prompt_embeds"].to(device),
        prompt_embeds_mask=None if mask is None else mask.to(device),
        layers=int(conditioning["layers"]),
        height=int(conditioning["height"]),
        width=int(conditioning["width"]),
    )


def place_trainable_roots(bundle: Any, actor: Any, *, device: Any) -> None:
    """Put the replay build's host-resident roots on ``device``, whole or streamed.

    ``actor.stream_frozen_blocks`` keeps each root's frozen blocks on the host.
    """

    from vrl.models.host_streaming import stream_frozen_blocks
    from vrl.models.parking import module_on_host

    stream = bool(actor.stream_frozen_blocks)
    for name, module in bundle.trainable_modules.items():
        if not module_on_host(module):
            continue
        if stream:
            count = stream_frozen_blocks(module, device=device)
            logger.info("%s: %d frozen blocks streamed from the host", name, count)
        else:
            module.to(device)


def _required(value: Any, path: str) -> Any:
    if value is None:
        raise ValueError(f"config missing required field: {path}")
    return value


def train_qwen_image_layered_sft(cfg: DictConfig) -> None:
    """Run layered supervised flow matching driven by a merged YAML config."""

    import torch
    from torch.utils.data import DataLoader

    from vrl.algorithms.sft import DiffusionSFT, DiffusionSFTConfig
    from vrl.config.builders import BuiltConfigs
    from vrl.config.data import manifest_sources
    from vrl.ray.resources import ResourcePlan
    from vrl.trainers.activation_checkpointing import enable_transformer_gradient_checkpointing
    from vrl.trainers.checkpointing import (
        TrainingCheckpoint,
        build_adapter_exports,
        capture_rng_state,
        restore_rng_state,
        save_resolved_config,
        save_training_checkpoint,
    )
    from vrl.trainers.data.layered_scenes import LayeredSceneDataset, collate_layered_scenes
    from vrl.trainers.metrics_io import MetricsCSV
    from vrl.trainers.offline import OfflineSFTTrainer, OfflineTrainerConfig
    from vrl.trainers.offline.sft import SFTStepMetrics

    built = BuiltConfigs.from_cfg(cfg)
    family_entry = built.family
    if family_entry.family != FAMILY:
        raise ValueError(
            f"layered SFT requires model.family={FAMILY!r}; got {family_entry.family!r}",
        )
    root = built.root
    resources = ResourcePlan.from_root(root)
    device = torch.device(resources.trainer_torch_device)
    trainer_section = _required(root.trainer, "trainer")
    data_cfg = _required(root.data, "data")
    sampler = _required(data_cfg.sampler, "data.sampler")
    if data_cfg.loader != "layered_scenes":
        raise ValueError(
            f"layered SFT requires data.loader='layered_scenes'; got {data_cfg.loader!r}",
        )
    sft_config = built.algorithm
    if not isinstance(sft_config, DiffusionSFTConfig):
        raise TypeError(
            f"layered SFT expects algorithm.kind=diffusion_sft, got {type(sft_config).__name__}",
        )
    actor = _required(root.actor, "actor")
    seed = int(_required(trainer_section.seed, "trainer.seed"))
    trainer_cfg = OfflineTrainerConfig.from_root(root)
    train_batch_size = int(actor.train_batch_size)
    resume_checkpoint = (
        TrainingCheckpoint.load(built.resume_from) if built.resume_from is not None else None
    )
    logger.info("%s", resources)

    # The replay build: transformer and scheduler, the trainer's own policy
    # shape. The scheduler grid comes from sampling.{num_steps,height,width}.
    model_build = family_entry.resolve_model_build(
        root, device, precision=built.precision, for_rollout=False
    )
    model_identity = model_build.checkpoint_identity()
    if resume_checkpoint is not None:
        resume_checkpoint.validate_compatibility(
            family=FAMILY, expected_model_identity=model_identity
        )
    bundle = family_entry.build_replay(model_build, expected_identity=model_identity)
    model = bundle.model
    enable_transformer_gradient_checkpointing(bundle, root)
    place_trainable_roots(bundle, actor, device=device)

    start_step = resume_checkpoint.next_step if resume_checkpoint is not None else 0
    dataset = LayeredSceneDataset(
        manifest_sources(data_cfg.manifest),
        split=str(data_cfg.split),
        token_weights=str(data_cfg.token_weights),
        spare_layers_max=int(data_cfg.spare_layers_max),
        family=FAMILY,
        model_revision=None if root.model.revision is None else str(root.model.revision),
    )
    # A resumed segment draws scenes from a seed offset by its start step, so
    # the scene order does not repeat the first segment's.
    torch.manual_seed(seed + start_step)
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=bool(sampler.shuffle),
        num_workers=int(sampler.dataloader_num_workers),
        collate_fn=collate_layered_scenes,
        drop_last=bool(sampler.drop_last),
    )
    logger.info("Loaded %d cached scenes", len(dataset))

    trainer = OfflineSFTTrainer(
        model=model,
        algorithm=DiffusionSFT(sft_config),
        forward_fn=layered_forward,
        noise_scheduler=model.scheduler,
        config=trainer_cfg,
        device=device,
    )
    if resume_checkpoint is not None:
        resume_checkpoint.restore_training(
            trainer=trainer,
            bundle=bundle,
            family=FAMILY,
            expected_model_identity=model_identity,
        )
        logger.info(
            "Resuming from %s, start_step=%d", resume_checkpoint.checkpoint_dir, start_step
        )
        restore_rng_state(resume_checkpoint.rng_state)

    out_dir = Path(str(trainer_section.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out_dir, resumed=resume_checkpoint is not None)
    from dataclasses import fields as _dc_fields

    metric_fields = tuple(f.name for f in _dc_fields(SFTStepMetrics))
    metrics_csv = MetricsCSV(
        out_dir / "metrics.csv",
        ("step", *metric_fields),
        resume_at=("step", start_step) if resume_checkpoint is not None else None,
    )

    max_train_steps = int(_required(trainer_section.max_train_steps, "trainer.max_train_steps"))
    checkpointing_steps = int(
        _required(trainer_section.checkpointing_steps, "trainer.checkpointing_steps"),
    )
    log_interval = int(_required(trainer_section.log_interval, "trainer.log_interval"))
    adapter_exports = build_adapter_exports(bundle, use_lora=bool(root.model.use_lora))
    logger.info(
        "Starting layered SFT — %d steps, lr=%g, timestep range %s, token weights %s, "
        "spare layers up to %d",
        max_train_steps,
        trainer_cfg.lr,
        sft_config.timestep_index_range,
        data_cfg.token_weights,
        int(data_cfg.spare_layers_max),
    )

    def save(step: int, name: str) -> None:
        save_training_checkpoint(
            out_dir / name,
            trainer=trainer,
            bundle=bundle,
            family=FAMILY,
            progress={
                "completed_step": step,
                "next_step": step,
                "global_step": trainer.global_step,
            },
            rng_state=capture_rng_state(),
            adapter_exports=adapter_exports,
            model_identity=model_identity,
        )
        logger.info("Saved checkpoint -> %s", out_dir / name)

    step = start_step
    if step > max_train_steps:
        raise ValueError(
            "resume checkpoint starts after configured max_train_steps: "
            f"start_step={step}, max_train_steps={max_train_steps}",
        )
    data_iter = iter(dataloader)
    while step < max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        m = trainer.step(batch)

        if step % log_interval == 0:
            logger.info(
                "step %d | loss=%.4f mse=%.4f gn=%.3f layers=%d scene=%s",
                step,
                m.loss,
                m.mse,
                m.grad_norm,
                int(batch.conditioning["layers"]),
                batch.conditioning["scene"],
            )
            row = ",".join(f"{getattr(m, name):.6f}" for name in metric_fields)
            metrics_csv.append(f"{step},{row}\n")

        if checkpointing_steps > 0 and (step + 1) % checkpointing_steps == 0:
            save(step + 1, f"checkpoint-{step + 1}")
        step += 1

    save(max_train_steps, "checkpoint-final")
    logger.info("Layered SFT complete.")


__all__ = ["layered_forward", "place_trainable_roots", "train_qwen_image_layered_sft"]
