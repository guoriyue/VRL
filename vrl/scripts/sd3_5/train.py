"""SD 3.5 GRPO training recipe.

The unified ``vrl.scripts.train`` entry point dispatches SD3.5 configs here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from vrl.trainers.checkpointing import (
    LORA_WEIGHTS_NAME,
    capture_rng_state,
    load_training_checkpoint_from_config,
    prepare_metrics_csv,
    prepare_model_config_for_training_resume,
    restore_rng_state,
    restore_training_checkpoint,
    sample_prompt_indices,
    save_resolved_config,
    save_training_checkpoint,
)

logger = logging.getLogger(__name__)


async def train_sd3_5_grpo(cfg: DictConfig) -> None:
    """Run SD 3.5 GRPO training driven by a merged YAML config."""
    import os

    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.algorithms.grpo_token import TokenGRPOConfig
    from vrl.algorithms.stat_tracking import PerPromptStatTracker
    from vrl.config.loader import build_configs, require
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collector import (
        SD3_5CollectorConfig,
        build_rollout_collector,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.rollouts.runtime.backend import build_rollout_backend_from_cfg
    from vrl.rollouts.runtime.launch_inputs import build_rollout_runtime_inputs
    from vrl.trainers.data import PromptExample, load_prompt_manifest
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.weight_sync import build_runtime_weight_syncer

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    grpo_config = built["algorithm"]
    if not isinstance(grpo_config, GRPOConfig) or isinstance(grpo_config, TokenGRPOConfig):
        raise TypeError(
            f"SD3.5 expects algorithm.kind=grpo, got {type(grpo_config).__name__}",
        )

    if trainer_config.profile:
        os.environ["VRL_PROFILE_COLLECT"] = "1"

    resume_checkpoint = load_training_checkpoint_from_config(cfg)
    prepare_model_config_for_training_resume(
        cfg,
        resume_checkpoint,
        strict=trainer_config.resume_strict,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if trainer_config.bf16 else torch.float16
    torch.manual_seed(int(trainer_config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(trainer_config.seed))

    # 1. Build policy bundle (backend construction lives in family builder)
    from vrl.models.families.sd3_5.builder import build_sd3_5_runtime_bundle_from_cfg

    bundle = build_sd3_5_runtime_bundle_from_cfg(cfg, device, weight_dtype)
    sd3_5_model = bundle.policy
    transformer = sd3_5_model.transformer

    if trainer_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    _offload_driver_frozen_modules(sd3_5_model)

    # 2. Reward
    reward_weights, reward_kwargs = built["reward"]
    if not reward_weights:
        raise ValueError("At least one reward component must have weight > 0.")
    reward_fn = MultiReward.from_dict(
        reward_weights,
        device=str(device),
        reward_kwargs=reward_kwargs,
    )
    logger.info("Reward mix: %s", reward_weights)

    # 3. Collector + algorithm
    noise_level = float(require(cfg, "rollout.noise_level"))
    collector_config = SD3_5CollectorConfig(
        num_steps=cfg.sampling.num_steps,
        guidance_scale=cfg.sampling.guidance_scale,
        height=cfg.sampling.height,
        width=cfg.sampling.width,
        max_sequence_length=int(getattr(cfg.sampling, "max_sequence_length", 128)),
        noise_level=noise_level,
        cfg=cfg.sampling.cfg,
        sample_batch_size=int(require(cfg, "rollout.sample_batch_size")),
        kl_reward=cfg.algorithm.kl_reward,
        sde_type=str(require(cfg, "rollout.sde.type")),
        sde_window_size=cfg.rollout.sde.window_size,
        sde_window_range=tuple(cfg.rollout.sde.window_range),
        same_latent=cfg.rollout.same_latent,
    )
    collector = build_rollout_collector(
        "sd3_5",
        model=sd3_5_model,
        reward_fn=reward_fn,
        config=collector_config,
    )
    rollout_runtime_inputs = build_rollout_runtime_inputs(
        cfg,
        "sd3_5",
        weight_dtype=weight_dtype,
        executor_kwargs={"sample_batch_size": collector_config.sample_batch_size},
    )
    collector.set_runtime(
        build_rollout_backend_from_cfg(
            cfg,
            driver_bundle=bundle,
            runtime_spec=rollout_runtime_inputs.runtime_spec,
            gatherer=rollout_runtime_inputs.gatherer,
        ),
    )

    evaluator = FlowMatchingEvaluator(
        bundle.scheduler,
        noise_level=noise_level,
        sde_type=collector_config.sde_type,
    )
    algorithm = GRPO(grpo_config)

    use_lora_kl = cfg.model.use_lora and grpo_config.init_kl_coef > 0
    ref_model = sd3_5_model if use_lora_kl else None
    stat_tracker = (
        PerPromptStatTracker(global_std=grpo_config.global_std)
        if cfg.algorithm.per_prompt_stat_tracking
        else None
    )

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=sd3_5_model,
        ref_model=ref_model,
        weight_syncer=build_runtime_weight_syncer(
            collector.runtime,
            initial_policy_version=resume_checkpoint.next_step
            if resume_checkpoint is not None
            else None,
        ),
        config=trainer_config,
        device=device,
        stat_tracker=stat_tracker,
    )
    if resume_checkpoint is not None:
        restore_training_checkpoint(
            resume_checkpoint,
            trainer=trainer,
            bundle=bundle,
            strict=trainer_config.resume_strict,
        )
        logger.info(
            "Resuming from %s, start_epoch=%d",
            resume_checkpoint.checkpoint_dir,
            resume_checkpoint.next_epoch,
        )

    # 4. Prompts + loop
    manifest_path = Path(cfg.data.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    examples: list[PromptExample] = load_prompt_manifest(manifest_path)

    eval_examples: list[PromptExample] = []
    eval_enabled = bool(require(cfg, "eval.enable"))
    if eval_enabled:
        eval_manifest = Path(str(require(cfg, "eval.manifest")))
        if not eval_manifest.exists():
            raise FileNotFoundError(f"Eval manifest not found: {eval_manifest}")
        max_eval_prompts = int(require(cfg, "eval.max_prompts"))
        eval_examples = load_prompt_manifest(eval_manifest)[:max_eval_prompts]
        if not eval_examples:
            raise ValueError(f"Eval manifest contains no prompts: {eval_manifest}")

    logger.info(
        "Starting SD3 GRPO — %d epochs, %d examples, n=%d",
        trainer_config.total_epochs,
        len(examples),
        trainer_config.n,
    )

    start_epoch = resume_checkpoint.next_epoch if resume_checkpoint is not None else 0
    if start_epoch > trainer_config.total_epochs:
        raise ValueError(
            "resume checkpoint starts after configured total_epochs: "
            f"start_epoch={start_epoch}, total_epochs={trainer_config.total_epochs}",
        )

    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output_dir, resumed=resume_checkpoint is not None)

    csv_path = output_dir / "metrics.csv"
    component_names = list(reward_weights.keys())
    component_cols = ",".join(f"r_{n}" for n in component_names)
    metrics_header = (
        "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
        "clip_fraction,approx_kl,advantage_mean,grad_norm,adv_saturation,"
        "adv_zero_rate,group_size,trained_prompt_num," + component_cols + "\n"
    )
    prepare_metrics_csv(csv_path, metrics_header, resume=resume_checkpoint is not None)

    eval_csv_path = output_dir / "eval_metrics.csv"
    if eval_enabled:
        prepare_metrics_csv(
            eval_csv_path,
            "epoch,reward_mean,reward_std," + component_cols + "\n",
            resume=resume_checkpoint is not None,
        )

    rng = torch.Generator().manual_seed(trainer_config.seed)
    if resume_checkpoint is not None:
        restore_rng_state(resume_checkpoint.rng_state, prompt_generator=rng)
    for epoch in range(start_epoch, trainer_config.total_epochs):
        if eval_enabled and epoch % int(require(cfg, "eval.freq")) == 0:
            eval_metrics = await _run_fixed_eval(
                trainer=trainer,
                collector=collector,
                reward_fn=reward_fn,
                examples=eval_examples,
                cfg=cfg,
                output_dir=output_dir,
                epoch=epoch,
                component_names=component_names,
                eval_csv_path=eval_csv_path,
                torch=torch,
            )
            logger.info(
                "Eval epoch %d | reward=%.4f+/-%.4f",
                epoch,
                eval_metrics["reward_mean"],
                eval_metrics["reward_std"],
            )

        idx = sample_prompt_indices(
            rng,
            num_examples=len(examples),
            rollout_batch_size=trainer_config.rollout_batch_size,
        )
        example_batch = [examples[i] for i in idx]
        reward_fn.reset_components()
        metrics = await trainer.step(example_batch)

        if epoch % trainer_config.log_freq == 0:
            last = getattr(reward_fn, "last_components", {}) or {}
            component_means = {
                n: (sum(last.get(n, [])) / len(last.get(n, []))) if last.get(n) else float("nan")
                for n in component_names
            }
            component_str = " ".join(f"{n}={component_means[n]:.3f}" for n in component_names)
            logger.info(
                "Epoch %d | loss=%.4f kl=%.4f reward=%.4f+/-%.4f "
                "grad_norm=%.4f adv_sat=%.3f adv_zero=%.3f | %s",
                epoch,
                metrics.loss,
                metrics.kl_penalty,
                metrics.reward_mean,
                metrics.reward_std,
                metrics.grad_norm,
                metrics.adv_saturation,
                metrics.adv_zero_rate,
                component_str,
            )
            with open(csv_path, "a") as f:
                vals = ",".join(f"{component_means[n]:.4f}" for n in component_names)
                f.write(
                    f"{epoch},{metrics.loss:.6f},{metrics.policy_loss:.6f},"
                    f"{metrics.kl_penalty:.6f},{metrics.reward_mean:.4f},"
                    f"{metrics.reward_std:.4f},{metrics.clip_fraction:.4f},"
                    f"{metrics.approx_kl:.6f},{metrics.advantage_mean:.6f},"
                    f"{metrics.grad_norm:.6f},{metrics.adv_saturation:.4f},"
                    f"{metrics.adv_zero_rate:.4f},{metrics.group_size:.2f},"
                    f"{metrics.trained_prompt_num}," + vals + "\n"
                )

        if trainer_config.save_freq > 0 and (epoch + 1) % trainer_config.save_freq == 0:
            ckpt_path = output_dir / f"checkpoint-{epoch + 1}"
            save_training_checkpoint(
                ckpt_path,
                trainer=trainer,
                bundle=bundle,
                family="sd3_5",
                progress={
                    "completed_epoch": epoch + 1,
                    "next_epoch": epoch + 1,
                    "global_step": trainer.state.global_step,
                },
                rng_state=capture_rng_state(prompt_generator=rng),
                export_modules={LORA_WEIGHTS_NAME: transformer}
                if bool(cfg.model.use_lora) and hasattr(transformer, "save_pretrained")
                else None,
                export_ema=trainer._ema,
            )
            logger.info("Saved checkpoint to %s", ckpt_path)

    final_path = output_dir / "checkpoint-final"
    save_training_checkpoint(
        final_path,
        trainer=trainer,
        bundle=bundle,
        family="sd3_5",
        progress={
            "completed_epoch": trainer_config.total_epochs,
            "next_epoch": trainer_config.total_epochs,
            "global_step": trainer.state.global_step,
        },
        rng_state=capture_rng_state(prompt_generator=rng),
        export_modules={LORA_WEIGHTS_NAME: transformer}
        if bool(cfg.model.use_lora) and hasattr(transformer, "save_pretrained")
        else None,
        export_ema=trainer._ema,
    )
    logger.info("Training complete. Final checkpoint: %s", final_path)


def _offload_driver_frozen_modules(policy: object) -> None:
    """Move frozen driver-only modules off CUDA before Ray workers load.

    The driver replay path needs the trainable transformer and scheduler only:
    rollout workers own prompt encoding and VAE decoding during generation, while
    trainer replay consumes prompt embeddings and latents already packed in the
    ``RolloutBatch``. Keeping frozen encoders/VAE on the driver GPU duplicates
    the rollout worker footprint and makes single-GPU Ray smoke runs OOM.
    """

    import torch

    pipeline = getattr(policy, "pipeline", None)
    if pipeline is None:
        return

    for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
        module = getattr(pipeline, name, None)
        if module is not None:
            module.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def _run_fixed_eval(
    *,
    trainer: Any,
    collector: Any,
    reward_fn: Any,
    examples: list[Any],
    cfg: Any,
    output_dir: Path,
    epoch: int,
    component_names: list[str],
    eval_csv_path: Path,
    torch: Any,
) -> dict[str, float]:
    """Run Flow-GRPO-style fixed SD3 eval with EMA weights when configured."""

    from omegaconf import OmegaConf

    from vrl.config.loader import require

    eval_seed = int(
        OmegaConf.select(
            cfg,
            "eval.seed",
            default=OmegaConf.select(cfg, "trainer.seed", default=0),
        ),
    )
    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    ema = trainer._ensure_ema() if bool(require(cfg, "eval.use_ema")) else None
    if ema is not None:
        ema.copy_ema_to(trainable, store_temp=True)
        if trainer.weight_syncer is not None:
            await trainer.weight_syncer.push(trainer.model.state_dict())
    else:
        await trainer._ensure_rollout_weights_initialized()

    batches = []
    reward_fn.reset_components()
    eval_dir = output_dir / f"eval_epoch_{epoch:04d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, example in enumerate(examples):
            batch = await collector.collect(
                [example.prompt],
                group_size=1,
                target_text=example.target_text,
                references=example.references,
                task_type=example.task_type,
                sample_metadata=example.metadata,
                request_overrides={
                    "num_steps": int(require(cfg, "eval.num_steps")),
                    "sample_batch_size": int(require(cfg, "eval.sample_batch_size")),
                    "noise_level": float(require(cfg, "eval.noise_level")),
                    "seed": eval_seed + idx,
                },
            )
            batches.append(batch)
            if batch.videos is not None:
                _save_eval_image(
                    batch.videos[0],
                    eval_dir / f"{idx:03d}.png",
                )
    finally:
        release = getattr(collector, "release_runtime_memory", None)
        if callable(release):
            await release()
        if ema is not None:
            ema.copy_temp_to(trainable)
            if trainer.weight_syncer is not None:
                await trainer.weight_syncer.push(trainer.model.state_dict())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rewards = torch.cat([batch.rewards.detach().float().cpu() for batch in batches])
    component_means = _component_means(reward_fn, component_names)
    with eval_csv_path.open("a", encoding="utf-8") as f:
        vals = ",".join(f"{component_means[name]:.4f}" for name in component_names)
        f.write(
            f"{epoch},{rewards.mean().item():.6f},"
            f"{rewards.std().item() if rewards.numel() > 1 else 0.0:.6f},"
            f"{vals}\n",
        )

    _make_contact_sheet(
        sorted(eval_dir.glob("*.png")),
        eval_dir / "contact_sheet.png",
    )

    return {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std().item() if rewards.numel() > 1 else 0.0),
    }


def _component_means(reward_fn: Any, component_names: list[str]) -> dict[str, float]:
    last = getattr(reward_fn, "last_components", {}) or {}
    return {
        name: (sum(last.get(name, [])) / len(last.get(name, [])))
        if last.get(name)
        else float("nan")
        for name in component_names
    }


def _save_eval_image(tensor: Any, path: Path) -> None:
    from PIL import Image

    array = tensor.detach().float().cpu().clamp(0, 1)
    if array.ndim == 4:
        array = array[:, 0]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0)
    pixels = (array.numpy() * 255).astype("uint8")
    Image.fromarray(pixels.squeeze()).save(path)


def _make_contact_sheet(image_paths: list[Path], out_path: Path) -> None:
    if not image_paths:
        return
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_w, thumb_h = 256, 256
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    for idx, image in enumerate(images):
        image.thumbnail((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(image, (x, y))
    sheet.save(out_path)
