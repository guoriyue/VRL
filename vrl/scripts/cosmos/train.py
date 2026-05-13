"""Cosmos Predict2 GRPO training recipe.

The unified ``vrl.scripts.train`` entry point dispatches Cosmos configs here.
Pipeline construction, LoRA, collector wiring, training loop, checkpointing,
paired LoRA-vs-base eval, and middle-frame PNG dumps all live in this module.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

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


async def train_cosmos_predict2_grpo(cfg: DictConfig) -> None:
    """Run Cosmos Predict2 GRPO training driven by a merged YAML config."""
    import os

    import torch

    from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
    from vrl.algorithms.grpo.token import TokenGRPOConfig
    from vrl.algorithms.stat_tracking import PerPromptStatTracker
    from vrl.config.loader import build_configs, require
    from vrl.distributed.resources import (
        format_distributed_resource_plan,
        resolve_distributed_resources,
        trainer_torch_device,
    )
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collector import (
        CosmosPredict2CollectorConfig,
        build_rollout_collector,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.rollouts.runtime.backend import build_rollout_backend_from_cfg
    from vrl.rollouts.runtime.launch_inputs import build_rollout_runtime_inputs
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.weight_sync import build_runtime_weight_syncer

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    grpo_config = built["algorithm"]
    model_family = str(require(cfg, "model.family"))
    if not isinstance(grpo_config, GRPOConfig) or isinstance(grpo_config, TokenGRPOConfig):
        raise TypeError(
            f"Cosmos expects algorithm.kind=grpo, got {type(grpo_config).__name__}",
        )

    if trainer_config.profile:
        os.environ["VRL_PROFILE_COLLECT"] = "1"

    resume_checkpoint = load_training_checkpoint_from_config(cfg)
    prepare_model_config_for_training_resume(
        cfg,
        resume_checkpoint,
        strict=trainer_config.resume_strict,
    )

    distributed_resources = resolve_distributed_resources(cfg)
    logger.info(format_distributed_resource_plan(distributed_resources))
    device = torch.device(trainer_torch_device(distributed_resources))
    weight_dtype = torch.bfloat16 if trainer_config.bf16 else torch.float16

    reference_mode = _cosmos_reference_mode(cfg)
    manifest_path = Path(cfg.data.manifest)
    examples = _load_cosmos_prompt_examples(
        manifest_path,
        reference_mode=reference_mode,
        rollout_batch_size=trainer_config.rollout_batch_size,
    )

    # Video2World conditioning is required for meaningful Cosmos training.
    global_reference_image_path = str(require(cfg, "model.reference_image") or "")
    if reference_mode == "global":
        reference_image = _load_required_reference_image(global_reference_image_path)
        active_reference_image_path = str(Path(global_reference_image_path).expanduser())
        logger.info("Loaded global reference image from %s", cfg.model.reference_image)
    else:
        first_reference = examples[0].reference_image
        reference_image = _load_required_reference_image(first_reference)
        active_reference_image_path = first_reference
        logger.info(
            "Using per-sample reference images from %s; default eval reference=%s",
            manifest_path,
            active_reference_image_path,
        )

    # 1. Build policy bundle (backend construction lives in family runtime)
    from vrl.models.families.cosmos.predict2.runtime import (
        build_cosmos_predict2_runtime_bundle_from_cfg,
    )

    bundle = build_cosmos_predict2_runtime_bundle_from_cfg(
        cfg,
        device,
        weight_dtype,
    )
    cosmos_model = bundle.policy
    transformer = cosmos_model.transformer

    if trainer_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

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

    # 3. Collector + evaluator + algorithm
    noise_level = float(require(cfg, "rollout.noise_level"))
    collector_config = CosmosPredict2CollectorConfig(
        num_steps=cfg.sampling.num_steps,
        guidance_scale=cfg.sampling.guidance_scale,
        height=cfg.sampling.height,
        width=cfg.sampling.width,
        num_frames=cfg.sampling.num_frames,
        fps=cfg.sampling.fps,
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
        model_family,
        model=cosmos_model,
        reward_fn=reward_fn,
        config=collector_config,
        reference_image=reference_image,
    )
    rollout_runtime_inputs = build_rollout_runtime_inputs(
        cfg,
        model_family,
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
    ref_model = cosmos_model if use_lora_kl else None
    stat_tracker = (
        PerPromptStatTracker(global_std=grpo_config.global_std)
        if cfg.algorithm.per_prompt_stat_tracking
        else None
    )

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=cosmos_model,
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

    # Eval prompts: held-out file or first 4 training prompts
    eval_prompts: list[str] = []
    eval_prompts_file = str(require(cfg, "eval.prompts_file") or "")
    if eval_prompts_file and Path(eval_prompts_file).exists():
        eval_prompts = [
            line.strip()
            for line in Path(eval_prompts_file).read_text().strip().splitlines()
            if line.strip()
        ]
    if not eval_prompts:
        eval_prompts = [example.prompt for example in examples[:4]]

    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output_dir, resumed=resume_checkpoint is not None)
    _write_reference_artifacts(
        output_dir=output_dir,
        reference_mode=reference_mode,
        examples=examples,
        global_reference_image_path=active_reference_image_path,
    )

    # --- eval-only mode: paired base-vs-LoRA comparison, no training ---
    if bool(require(cfg, "eval.eval_only")):
        if not str(require(cfg, "model.lora.path") or ""):
            raise ValueError(
                "eval.eval_only=true requires model.lora.path pointing to a "
                "trained LoRA checkpoint. Without it, 'LoRA' scores would "
                "come from a random adapter."
            )
        eval_seeds = int(require(cfg, "eval.seeds"))
        await _run_eval_only(
            transformer=transformer,
            collector=collector,
            eval_prompts=eval_prompts,
            eval_seeds=eval_seeds,
            reference_image=reference_image,
            output_dir=output_dir,
            torch=torch,
        )
        return

    # --- training loop ---
    logger.info(
        "Starting Cosmos Predict2 GRPO — %d epochs, %d prompts, n=%d",
        trainer_config.total_epochs,
        len(examples),
        trainer_config.n,
    )

    if grpo_config.global_std and trainer_config.rollout_batch_size == 1:
        logger.warning(
            "global_std collapses to per-group std with rollout_batch_size=1; "
            "consider rollout.rollout_batch_size>=2",
        )

    csv_path = output_dir / "metrics.csv"
    component_names = list(reward_weights.keys())
    component_cols = ",".join(f"r_{n}" for n in component_names)
    prepare_metrics_csv(
        csv_path,
        "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
        "clip_fraction,approx_kl,advantage_mean,grad_norm,adv_saturation,"
        "adv_zero_rate,group_size,trained_prompt_num,reference_image_present,"
        "reference_image_path," + component_cols + "\n",
        resume=resume_checkpoint is not None,
    )

    gate_enable = bool(require(cfg, "eval.sanity_gates.enable"))
    clip_thresh = float(require(cfg, "eval.sanity_gates.clip_fraction_threshold"))
    kl_thresh = float(require(cfg, "eval.sanity_gates.approx_kl_threshold"))

    rng = torch.Generator().manual_seed(trainer_config.seed)
    start_epoch = resume_checkpoint.next_epoch if resume_checkpoint is not None else 0
    if start_epoch > trainer_config.total_epochs:
        raise ValueError(
            "resume checkpoint starts after configured total_epochs: "
            f"start_epoch={start_epoch}, total_epochs={trainer_config.total_epochs}",
        )
    if resume_checkpoint is not None:
        restore_rng_state(resume_checkpoint.rng_state, prompt_generator=rng)
    for epoch in range(start_epoch, trainer_config.total_epochs):
        idx = sample_prompt_indices(
            rng,
            num_examples=len(examples),
            rollout_batch_size=trainer_config.rollout_batch_size,
        )
        example_batch = [examples[i] for i in idx]
        active_reference_image_path = _active_reference_image_path(
            example_batch,
            reference_mode=reference_mode,
            global_reference_image_path=global_reference_image_path,
        )
        prompt_batch = (
            [example.prompt for example in example_batch]
            if reference_mode == "global"
            else example_batch
        )

        reward_fn.reset_components()
        metrics = await trainer.step(prompt_batch)

        # Epoch-0 on-policy sanity check.
        if epoch == 0 and gate_enable:
            logger.info(
                "Epoch 0 sanity check: approx_kl=%.6f clip_fraction=%.4f",
                metrics.approx_kl,
                metrics.clip_fraction,
            )
            if metrics.clip_fraction > clip_thresh:
                logger.warning(
                    "HIGH clip_fraction at epoch 0 (%.3f) — likely ratio "
                    "mismatch between collect and forward_step. Training may "
                    "not converge.",
                    metrics.clip_fraction,
                )
            if metrics.approx_kl > kl_thresh:
                logger.warning(
                    "HIGH approx_kl at epoch 0 (%.6f) — log-probs from "
                    "collect and forward_step may not match. Check "
                    "_predict_noise_impl consistency.",
                    metrics.approx_kl,
                )

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
                    f"{metrics.trained_prompt_num},1,{active_reference_image_path},"
                    + vals + "\n"
                )

        if trainer_config.save_freq > 0 and (epoch + 1) % trainer_config.save_freq == 0:
            ckpt_path = output_dir / f"checkpoint-{epoch + 1}"
            save_training_checkpoint(
                ckpt_path,
                trainer=trainer,
                bundle=bundle,
                family=model_family,
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

            # Generate eval samples for visual comparison (middle-frame PNGs)
            logger.info("Generating eval samples at epoch %d...", epoch + 1)
            transformer.eval()
            eval_dir = ckpt_path / "eval_samples"
            eval_dir.mkdir(exist_ok=True)
            reward_debug_dir = output_dir / "reward_debug" / f"checkpoint-{epoch + 1}"
            reward_debug_dir.mkdir(parents=True, exist_ok=True)
            reward_debug_rows: list[dict[str, Any]] = []
            eval_scores: list[float] = []
            offload_driver_model_for_eval = (
                device.type == "cuda"
                and _collector_runtime_requires_driver_model_offload(collector)
            )
            if offload_driver_model_for_eval:
                _move_model_to_device(cosmos_model, "cpu")
            try:
                for i, ep in enumerate(eval_prompts):
                    with torch.no_grad():
                        eval_batch = await collector.collect(
                            [ep],
                            reference_image=reference_image,
                            seed=i,
                        )
                    score = eval_batch.rewards[0].item()
                    eval_scores.append(score)
                    if eval_batch.videos is not None:
                        filename = f"prompt_{i}_score_{score:.2f}.png"
                        _save_middle_frame(
                            eval_batch.videos[0],
                            eval_dir,
                            filename,
                            torch,
                        )
                        _save_middle_frame(
                            eval_batch.videos[0],
                            reward_debug_dir,
                            filename,
                            torch,
                        )
                    reward_debug_rows.append(
                        {
                            "prompt": ep,
                            "score": score,
                            "reference_image": active_reference_image_path,
                        },
                    )
            finally:
                if offload_driver_model_for_eval:
                    _move_model_to_device(cosmos_model, device)
            _write_jsonl(reward_debug_dir / "reward_debug.jsonl", reward_debug_rows)
            transformer.train()

            avg_eval = sum(eval_scores) / len(eval_scores) if eval_scores else 0.0
            logger.info(
                "Checkpoint %d | eval_reward=%.4f (%s) | saved to %s",
                epoch + 1,
                avg_eval,
                ", ".join(f"{s:.2f}" for s in eval_scores),
                ckpt_path,
            )

    final_path = output_dir / "checkpoint-final"
    save_training_checkpoint(
        final_path,
        trainer=trainer,
        bundle=bundle,
        family=model_family,
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


async def train_cosmos_predict25_diffusion_nft(cfg: DictConfig) -> None:
    """Run Cosmos Predict2.5 DiffusionNFT training driven by YAML config."""
    import os

    import torch

    from vrl.algorithms.diffusion_nft import DiffusionNFT, DiffusionNFTConfig
    from vrl.algorithms.grpo.token import TokenGRPOConfig
    from vrl.algorithms.stat_tracking import PerPromptStatTracker
    from vrl.config.loader import build_configs, require
    from vrl.distributed.resources import (
        format_distributed_resource_plan,
        resolve_distributed_resources,
        trainer_torch_device,
    )
    from vrl.models.families.cosmos.predict2_5.runtime import (
        build_cosmos_predict25_runtime_bundle_from_cfg,
    )
    from vrl.rewards.multi import MultiReward
    from vrl.rollouts.collector import CosmosPredict2CollectorConfig, build_rollout_collector
    from vrl.rollouts.runtime.backend import build_rollout_backend_from_cfg
    from vrl.rollouts.runtime.launch_inputs import build_rollout_runtime_inputs
    from vrl.trainers.data import load_prompt_manifest
    from vrl.trainers.diagnostics import parameter_state_summary, trainable_state_digest
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.weight_sync import build_runtime_weight_syncer

    built = build_configs(cfg)
    trainer_config = built["trainer"]
    nft_config = built["algorithm"]
    model_family = str(require(cfg, "model.family"))
    if not isinstance(nft_config, DiffusionNFTConfig) or isinstance(nft_config, TokenGRPOConfig):
        raise TypeError(
            "Cosmos Predict2.5 DiffusionNFT expects algorithm.kind=diffusion_nft, "
            f"got {type(nft_config).__name__}",
        )
    if not bool(require(cfg, "model.use_lora")):
        raise ValueError("Cosmos Predict2.5 DiffusionNFT requires model.use_lora=true")

    if trainer_config.profile:
        os.environ["VRL_PROFILE_COLLECT"] = "1"

    resume_checkpoint = load_training_checkpoint_from_config(cfg)
    prepare_model_config_for_training_resume(
        cfg,
        resume_checkpoint,
        strict=trainer_config.resume_strict,
    )

    distributed_resources = resolve_distributed_resources(cfg)
    logger.info(format_distributed_resource_plan(distributed_resources))
    device = torch.device(trainer_torch_device(distributed_resources))
    weight_dtype = torch.bfloat16 if trainer_config.bf16 else torch.float16

    manifest_path = Path(cfg.data.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    examples = load_prompt_manifest(manifest_path)
    prompts = [example.prompt for example in examples if example.prompt]
    if not prompts:
        raise ValueError(f"Manifest contains no prompts: {manifest_path}")

    bundle = build_cosmos_predict25_runtime_bundle_from_cfg(cfg, device, weight_dtype)
    model = bundle.policy
    transformer = model.transformer
    if trainer_config.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    reward_weights, reward_kwargs = built["reward"]
    if not reward_weights:
        raise ValueError("At least one reward component must have weight > 0.")
    reward_fn = MultiReward.from_dict(
        reward_weights,
        device=str(device),
        reward_kwargs=reward_kwargs,
    )
    logger.info("Reward mix: %s", reward_weights)

    collector_config = CosmosPredict2CollectorConfig(
        num_steps=int(require(cfg, "sampling.num_steps")),
        guidance_scale=float(require(cfg, "sampling.guidance_scale")),
        height=int(require(cfg, "sampling.height")),
        width=int(require(cfg, "sampling.width")),
        num_frames=int(require(cfg, "sampling.num_frames")),
        fps=int(require(cfg, "sampling.fps")),
        noise_level=float(require(cfg, "rollout.noise_level")),
        cfg=bool(require(cfg, "sampling.cfg")),
        sample_batch_size=int(require(cfg, "rollout.sample_batch_size")),
        kl_reward=float(getattr(cfg.algorithm, "kl_reward", 0.0)),
        sde_type=str(require(cfg, "rollout.sde.type")),
        sde_window_size=int(require(cfg, "rollout.sde.window_size")),
        sde_window_range=tuple(require(cfg, "rollout.sde.window_range")),
        same_latent=bool(require(cfg, "rollout.same_latent")),
    )
    collector = build_rollout_collector(
        model_family,
        model=model,
        reward_fn=reward_fn,
        config=collector_config,
    )
    rollout_runtime_inputs = build_rollout_runtime_inputs(
        cfg,
        model_family,
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

    algorithm = DiffusionNFT(nft_config)
    stat_tracker = (
        PerPromptStatTracker(global_std=nft_config.global_std)
        if nft_config.per_prompt_stat_tracking
        else None
    )
    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=None,
        model=model,
        ref_model=None,
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

    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output_dir, resumed=resume_checkpoint is not None)

    csv_path = output_dir / "metrics.csv"
    component_names = list(reward_weights.keys())
    component_cols = ",".join(f"r_{name}" for name in component_names)
    prepare_metrics_csv(
        csv_path,
        "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
        "clip_fraction,approx_kl,advantage_mean,grad_norm,adv_saturation,"
        "adv_zero_rate,group_size,trained_prompt_num,global_step,"
        + component_cols
        + "\n",
        resume=resume_checkpoint is not None,
    )

    rng = torch.Generator().manual_seed(trainer_config.seed)
    start_epoch = resume_checkpoint.next_epoch if resume_checkpoint is not None else 0
    if resume_checkpoint is not None:
        restore_rng_state(resume_checkpoint.rng_state, prompt_generator=rng)

    logger.info(
        "Starting Cosmos Predict2.5 DiffusionNFT — %d epochs, %d prompts, n=%d",
        trainer_config.total_epochs,
        len(prompts),
        trainer_config.n,
    )
    optimization_probe_before = trainable_state_digest(model)
    for epoch in range(start_epoch, trainer_config.total_epochs):
        idx = sample_prompt_indices(
            rng,
            num_examples=len(prompts),
            rollout_batch_size=trainer_config.rollout_batch_size,
        )
        prompt_batch = [prompts[i] for i in idx]
        reward_fn.reset_components()
        metrics = await trainer.step(prompt_batch)
        if epoch == start_epoch:
            optimization_probe_after = trainable_state_digest(model)
            optimization_check = {
                "epoch": epoch,
                "global_step": int(trainer.state.global_step),
                "loss": float(metrics.loss),
                "policy_loss": float(metrics.policy_loss),
                "kl_penalty": float(metrics.kl_penalty),
                "reward_mean": float(metrics.reward_mean),
                "reward_std": float(metrics.reward_std),
                "advantage_mean": float(metrics.advantage_mean),
                "adv_zero_rate": float(metrics.adv_zero_rate),
                "grad_norm": float(metrics.grad_norm),
                "trainable_sha256_changed": (
                    optimization_probe_before.get("sha256")
                    != optimization_probe_after.get("sha256")
                ),
                "trainable_before": optimization_probe_before,
                "trainable_after": optimization_probe_after,
                "parameter_state_after": parameter_state_summary(model),
            }
            with (output_dir / "optimization_check.json").open("w", encoding="utf-8") as f:
                json.dump(optimization_check, f, indent=2, sort_keys=True)
            if not optimization_check["trainable_sha256_changed"]:
                logger.warning(
                    "First DiffusionNFT step did not change trainable parameter digest; "
                    "check gradients and advantage filtering."
                )

        if epoch % trainer_config.log_freq == 0:
            last = getattr(reward_fn, "last_components", {}) or {}
            component_means = {
                name: (sum(last.get(name, [])) / len(last.get(name, [])))
                if last.get(name)
                else float("nan")
                for name in component_names
            }
            logger.info(
                "Epoch %d | loss=%.4f policy=%.4f kl=%.4f reward=%.4f+/-%.4f "
                "grad_norm=%.4f adv_zero=%.3f global_step=%d",
                epoch,
                metrics.loss,
                metrics.policy_loss,
                metrics.kl_penalty,
                metrics.reward_mean,
                metrics.reward_std,
                metrics.grad_norm,
                metrics.adv_zero_rate,
                trainer.state.global_step,
            )
            with open(csv_path, "a") as f:
                vals = ",".join(f"{component_means[name]:.4f}" for name in component_names)
                f.write(
                    f"{epoch},{metrics.loss:.6f},{metrics.policy_loss:.6f},"
                    f"{metrics.kl_penalty:.6f},{metrics.reward_mean:.4f},"
                    f"{metrics.reward_std:.4f},{metrics.clip_fraction:.4f},"
                    f"{metrics.approx_kl:.6f},{metrics.advantage_mean:.6f},"
                    f"{metrics.grad_norm:.6f},{metrics.adv_saturation:.4f},"
                    f"{metrics.adv_zero_rate:.4f},{metrics.group_size:.2f},"
                    f"{metrics.trained_prompt_num},{trainer.state.global_step},"
                    + vals
                    + "\n"
                )

        if trainer_config.save_freq > 0 and (epoch + 1) % trainer_config.save_freq == 0:
            ckpt_path = output_dir / f"checkpoint-{epoch + 1}"
            save_training_checkpoint(
                ckpt_path,
                trainer=trainer,
                bundle=bundle,
                family=model_family,
                progress={
                    "completed_epoch": epoch + 1,
                    "next_epoch": epoch + 1,
                    "global_step": trainer.state.global_step,
                },
                rng_state=capture_rng_state(prompt_generator=rng),
                export_modules={LORA_WEIGHTS_NAME: transformer}
                if hasattr(transformer, "save_pretrained")
                else None,
                export_ema=trainer._ema,
            )
            logger.info("Saved checkpoint to %s", ckpt_path)

    final_path = output_dir / "checkpoint-final"
    save_training_checkpoint(
        final_path,
        trainer=trainer,
        bundle=bundle,
        family=model_family,
        progress={
            "completed_epoch": trainer_config.total_epochs,
            "next_epoch": trainer_config.total_epochs,
            "global_step": trainer.state.global_step,
        },
        rng_state=capture_rng_state(prompt_generator=rng),
        export_modules={LORA_WEIGHTS_NAME: transformer}
        if hasattr(transformer, "save_pretrained")
        else None,
        export_ema=trainer._ema,
    )
    logger.info("Training complete. Final checkpoint: %s", final_path)


def _save_middle_frame(
    video: Any,
    out_dir: Path,
    filename: str,
    torch: Any,
) -> None:
    """Extract middle frame from video tensor and save as PNG."""
    try:
        from PIL import Image

        vid = video
        if vid.ndim == 4:
            mid = vid.shape[1] // 2
            frame = vid[:, mid, :, :]  # [C, H, W]
        else:
            frame = vid
        frame = (frame * 255).clamp(0, 255).to(torch.uint8)
        frame = frame.cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray(frame)
        img.save(out_dir / filename)
    except Exception:
        logger.debug("Failed to save frame %s", filename)


async def _run_eval_only(
    *,
    transformer: Any,
    collector: Any,
    eval_prompts: list[str],
    eval_seeds: int,
    reference_image: Any,
    output_dir: Path,
    torch: Any,
) -> None:
    """Eval-only mode: paired base-vs-LoRA comparison.

    Same seed for both LoRA and base runs on each (prompt, seed) pair so
    the comparison is paired / apples-to-apples.
    """
    import csv

    eval_dir = output_dir / "eval_only"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "eval_results.csv"

    rows: list[dict[str, Any]] = []

    # Evaluate LoRA model
    logger.info(
        "Evaluating LoRA model on %d prompts x %d seeds...",
        len(eval_prompts),
        eval_seeds,
    )
    transformer.eval()
    lora_scores: list[float] = []
    for i, prompt in enumerate(eval_prompts):
        for seed in range(eval_seeds):
            with torch.no_grad():
                batch = await collector.collect(
                    [prompt],
                    reference_image=reference_image,
                    seed=seed,
                )
            score = batch.rewards[0].item()
            lora_scores.append(score)
            if batch.videos is not None:
                _save_middle_frame(
                    batch.videos[0],
                    eval_dir,
                    f"lora_prompt_{i}_seed_{seed}_score_{score:.2f}.png",
                    torch,
                )
            rows.append(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "lora_score": f"{score:.4f}",
                    "base_score": "",
                    "delta": "",
                }
            )

    # Evaluate base model (disable LoRA adapter) — same seeds for paired comparison
    base_scores: list[float] = []
    if hasattr(transformer, "disable_adapter"):
        logger.info(
            "Evaluating base model (LoRA disabled) on %d prompts x %d seeds...",
            len(eval_prompts),
            eval_seeds,
        )
        with transformer.disable_adapter():
            row_idx = 0
            for i, prompt in enumerate(eval_prompts):
                for seed in range(eval_seeds):
                    with torch.no_grad():
                        batch = await collector.collect(
                            [prompt],
                            reference_image=reference_image,
                            seed=seed,
                        )
                    score = batch.rewards[0].item()
                    base_scores.append(score)
                    if batch.videos is not None:
                        _save_middle_frame(
                            batch.videos[0],
                            eval_dir,
                            f"base_prompt_{i}_seed_{seed}_score_{score:.2f}.png",
                            torch,
                        )
                    rows[row_idx]["base_score"] = f"{score:.4f}"
                    rows[row_idx]["delta"] = f"{lora_scores[row_idx] - score:.4f}"
                    row_idx += 1
    else:
        logger.warning(
            "Model does not support disable_adapter — skipping base comparison.",
        )

    transformer.train()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt", "seed", "lora_score", "base_score", "delta"],
        )
        writer.writeheader()
        writer.writerows(rows)

    avg_lora = sum(lora_scores) / len(lora_scores) if lora_scores else 0.0
    avg_base = sum(base_scores) / len(base_scores) if base_scores else 0.0
    delta = avg_lora - avg_base if base_scores else float("nan")
    logger.info(
        "Eval-only results: lora=%.4f base=%.4f delta=%.4f | %s",
        avg_lora,
        avg_base,
        delta,
        csv_path,
    )


def _load_required_reference_image(reference_image_path: str) -> Any:
    """Load the Video2World conditioning image and reject degenerate training."""

    path_text = str(reference_image_path or "").strip()
    if not path_text:
        raise ValueError(
            "Cosmos Predict2 Video2World GRPO requires model.reference_image. "
            "The empty-string default would use zero conditioning, which is only "
            "valid for smoke/debug runs and is not a meaningful Video2World "
            "training objective.",
        )

    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"model.reference_image does not exist: {path}")

    from PIL import Image

    return Image.open(path).convert("RGB")


def _collector_runtime_requires_driver_model_offload(collector: Any) -> bool:
    try:
        runtime = collector.runtime
    except Exception:
        runtime = getattr(collector, "_runtime", None)
    return bool(getattr(runtime, "requires_driver_model_offload", False))


def _move_model_to_device(model: Any, device: Any) -> None:
    import torch

    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cosmos_reference_mode(cfg: DictConfig) -> str:
    mode = str(OmegaConf.select(cfg, "cosmos.reference_mode", default="global"))
    if mode not in {"global", "per_sample"}:
        raise ValueError("cosmos.reference_mode must be 'global' or 'per_sample'")
    return mode


def _load_cosmos_prompt_examples(
    manifest_path: Path,
    *,
    reference_mode: str,
    rollout_batch_size: int,
) -> list[Any]:
    from vrl.trainers.data import load_prompt_manifest

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    examples = load_prompt_manifest(manifest_path)
    if not examples:
        raise ValueError(f"Manifest contains no prompts: {manifest_path}")

    if reference_mode == "per_sample":
        if int(rollout_batch_size) != 1:
            raise ValueError(
                "cosmos.reference_mode=per_sample currently requires "
                "rollout.rollout_batch_size=1 so each rollout request has one "
                "reference image.",
            )
        base_dir = manifest_path.parent
        for idx, example in enumerate(examples):
            raw_path = str(example.reference_image or "").strip()
            if not raw_path:
                raise ValueError(
                    f"{manifest_path}: row {idx} is missing required field "
                    "reference_image for cosmos.reference_mode=per_sample",
                )
            ref_path = Path(raw_path).expanduser()
            if not ref_path.is_absolute():
                ref_path = (base_dir / ref_path).resolve()
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"{manifest_path}: row {idx} reference_image does not exist: {ref_path}",
                )
            example.reference_image = str(ref_path)
            metadata = dict(example.metadata or {})
            metadata["reference_image"] = str(ref_path)
            example.metadata = metadata

    return examples


def _active_reference_image_path(
    examples: list[Any],
    *,
    reference_mode: str,
    global_reference_image_path: str,
) -> str:
    if reference_mode == "global":
        return str(Path(global_reference_image_path).expanduser())
    if len(examples) != 1:
        raise ValueError("per-sample Cosmos reference mode expects one example per step")
    return examples[0].reference_image


def _write_reference_artifacts(
    *,
    output_dir: Path,
    reference_mode: str,
    examples: list[Any],
    global_reference_image_path: str,
) -> None:
    reference_dir = output_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if reference_mode == "global":
        source = Path(global_reference_image_path).expanduser()
        if source.exists():
            target = reference_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        for example in examples:
            rows.append(
                {
                    "prompt": example.prompt,
                    "reference_image": str(source),
                    "reference_mode": reference_mode,
                },
            )
    else:
        for example in examples:
            rows.append(
                {
                    "prompt": example.prompt,
                    "reference_image": example.reference_image,
                    "reference_mode": reference_mode,
                },
            )

    _write_jsonl(output_dir / "reference_manifest.jsonl", rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
