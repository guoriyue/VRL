"""Compare serial, two-stream and delayed-dW updates on one recipe's trajectory.

This is an isolated, single-device scheduling experiment, not a training mode.
It uses the real denoise loop, trajectory, evaluator, objective and trainer.
Rewards are fixed sample-index scores: results measure update cost/correctness,
not learned quality or end-to-end rollout throughput. No weights are published.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from vrl.scripts.generation.full_sequence_probe import build_parser, prepare_inputs
from vrl.scripts.perf.replay_overlap import UpdateComparisonError, compare_updates


def backend_preflight() -> dict:
    """Report local prerequisites; installed packages do not imply qualification."""

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    capability = torch.cuda.get_device_capability() if count else None
    other_processes = []
    if count:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=GPU-" + str(torch.cuda.get_device_properties().uuid),
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            pid, memory = line.split(",", 1)
            if int(pid.strip()) != os.getpid():
                other_processes.append({"pid": int(pid.strip()), "used_mib": memory.strip()})
    return {
        "other_compute_processes": other_processes,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "device_count": count,
        "device": torch.cuda.get_device_name() if count else "cpu",
        "capability": capability,
        "dualpipe": {
            "installed": importlib.util.find_spec("dualpipe") is not None,
            "qualified": False,
            "reason": "Requires model stage partitioning and custom overlapped_forward_backward; "
            + (
                "at least two GPUs are absent."
                if count < 2
                else "multi-GPU count alone is insufficient."
            ),
            "source": "https://github.com/deepseek-ai/DualPipe",
        },
        "deep_ep": {
            "installed": importlib.util.find_spec("deep_ep") is not None,
            "qualified": False,
            "reason": "No token-routed expert-parallel policy in this probe; EP needs multiple "
            "GPUs and supported interconnect. Experimental bucket collectives are not FSDP integration.",
            "source": "https://github.com/deepseek-ai/DeepEP",
        },
        "deep_gemm": {
            "installed": importlib.util.find_spec("deep_gemm") is not None,
            "qualified": False,
            "advertised_architecture_match": capability is not None
            and capability in {(9, 0), (10, 0)},
            "reason": "Published NVIDIA path targets SM90/SM100; validate shape/layout/scale and "
            "gradient contracts in a separate backend trial on matching hardware.",
            "source": "https://github.com/deepseek-ai/DeepGEMM",
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="check updates without timing; small numerical tests may share a GPU",
    )
    parser.add_argument("--modes", default="serial,streams,deferred_wgrad")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=None, help="override trainer.seed for this probe"
    )
    args = parser.parse_args(argv)
    if args.artifact_dir is None:
        parser.error("--artifact-dir is required; reports never overwrite an existing directory")
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=False)
    preflight = backend_preflight()
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    from dataclasses import replace

    from omegaconf import OmegaConf

    from vrl.config.builders import BuiltConfigs
    from vrl.config.loading import load_config
    from vrl.generation.steps.denoise.loop import run_denoise_loop
    from vrl.rollouts.batch import RolloutBatch
    from vrl.scripts.common.factory import AlgorithmEvaluatorPair
    from vrl.trainers.activation_checkpointing import enable_transformer_gradient_checkpointing
    from vrl.trainers.distributed import DistributedTrainingContext
    from vrl.trainers.online.trainer import OnlineTrainer, TrainingBatch
    from vrl.trainers.strategy import SingleProcessStrategy
    from vrl.trajectory.builders import build_diffusion_trajectory

    started = time.perf_counter()
    cfg = None
    modes = tuple(args.modes.split(","))
    report = {
        "passed": False,
        "config": args.config,
        "overrides": args.overrides,
        "preflight": preflight,
        "reward_source": "synthetic sample-index scores (not quality evaluation)",
        "seed_override": args.seed,
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_CUDA_ALLOC_CONF",
                "HF_HUB_OFFLINE",
            )
        },
        "atol": args.atol,
        "rtol": args.rtol,
    }
    try:
        cfg = load_config(args.config, overrides=args.overrides)
        built = BuiltConfigs.from_cfg(cfg)
        if preflight["other_compute_processes"] and not args.correctness_only:
            raise RuntimeError(
                "GPU is shared with another compute process; performance measurement refused"
            )
        training = built.root.distributed.training
        if training is not None and training.strategy != "single_process":
            raise ValueError(
                "this probe does not replace a distributed recipe with single_process"
            )
        if built.root.actor.offload is not None or built.resume_from is not None:
            raise ValueError(
                "offload/resume recipes need their own lifecycle; unsupported in this pilot"
            )
        if built.precision.training.dtype == "fp16":
            raise ValueError("FP16 needs GradScaler; start this isolated pilot with BF16 or FP32")
        if built.trainer is None or built.root.algorithm.kind != "grpo":
            raise ValueError("this pilot supports evaluator-backed GRPO only")
        if built.algorithm.kl_coef:
            raise ValueError(
                "disable KL explicitly; reference-policy weight swaps are outside this pilot"
            )
        if (
            built.trainer.ppo_epochs != 1
            or built.trainer.batch_config.optimizer_steps_per_batch != 1
        ):
            raise ValueError(
                "pilot requires one optimizer update: actor.ppo_epochs=1 and optimizer_steps_per_batch=1"
            )
        if built.trainer.rollout_orchestration.schedule_mode != "strict_on_policy":
            raise ValueError("the fixed-trajectory probe requires strict_on_policy scheduling")
        from vrl.models.precision import apply_float32_precision

        apply_float32_precision(built.precision.rollout.float32_precision)
        from vrl.run import OnlineRunConfig

        run_config = OnlineRunConfig.from_root(built.root)
        if args.seed is not None:
            run_config = replace(run_config, seed=args.seed)
        run_config.initialize_process_rng()
        report["initialization_seed"] = run_config.seed
        report["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        inputs = prepare_inputs(args, built.root)
        if inputs.quantization is not None or inputs.compiled:
            raise ValueError(
                "start with an unquantized eager baseline; compiled/cache interaction is not qualified"
            )
        if inputs.loop_config.sample_count != inputs.request.samples_per_prompt:
            raise ValueError("probe needs a complete prompt group in one generation batch")
        if inputs.loop_config.sample_count < 2:
            raise ValueError("need at least two samples for a nonzero group-relative advantage")
        with torch.no_grad():
            result = run_denoise_loop(
                model=inputs.model, state=inputs.state, config=inputs.loop_config
            )
        trajectory = build_diffusion_trajectory(
            request=inputs.request,
            sample_rows=inputs.request.sample_rows(),
            observations=result.observations,
            actions=result.actions,
            old_log_prob=result.log_probs,
            timesteps=result.timesteps,
            replay_tensors=inputs.model.export_replay_tensors(result.state),
            context=inputs.model.export_batch_context(result.state),
        )
        device = result.observations.device
        rewards = torch.linspace(-1, 1, inputs.loop_config.sample_count, device=device)
        group_ids = torch.zeros(len(rewards), device=device, dtype=torch.long)
        rollout_batch = RolloutBatch(
            rewards=rewards,
            group_ids=group_ids,
            trajectory=trajectory,
            context=dict(trajectory.context),
        )
        source_state = {
            k: v.detach().cpu().clone()
            for k, v in inputs.model.transformer.state_dict().items()
            if "lora_" in k or not built.root.model.use_lora
        }
        replay_build = inputs.entry.resolve_model_build(
            built.root,
            device,
            precision=built.precision,
            for_rollout=False,
        )
        model_identity = replay_build.checkpoint_identity()
        entry = inputs.entry
        del result, trajectory, inputs
        bundle = entry.build_replay(replay_build, expected_identity=model_identity)
        loaded = bundle.model.transformer.load_state_dict(
            source_state, strict=not built.root.model.use_lora
        )
        if loaded.unexpected_keys or any("lora_" in name for name in loaded.missing_keys):
            raise RuntimeError("rollout/replay trainable state schemas differ")
        del source_state
        enable_transformer_gradient_checkpointing(bundle, built.root)
        pair = AlgorithmEvaluatorPair.from_built(built, scheduler=bundle.scheduler)
        strategy = SingleProcessStrategy(
            DistributedTrainingContext("single_process", 0, 1, device)
        )
        trainer = OnlineTrainer(
            pair.algorithm,
            None,
            pair.evaluator,
            bundle.model,
            replace(built.trainer, output_dir=str(output / "trainer")),
            strategy=strategy,
        )
        advantages = pair.algorithm.compute_advantages_from_tensors(rewards, group_ids)
        batch = TrainingBatch(
            batches=[rollout_batch],
            advantages=[advantages],
            group_size=len(rewards),
            trained_prompt_num=1,
            sample_count=len(rewards),
            adv_zero_rate=0.0,
            adv_saturation=0.0,
            pre_filter_reward_mean=0.0,
            pre_filter_reward_std=float(rewards.std(correction=0)),
            pre_filter_adv_mean=float(advantages.mean()),
            reward_components={},
        )
        report.update(
            compare_updates(
                trainer,
                batch,
                modes=modes,
                warmup=args.warmup,
                repeats=args.repeats,
                atol=args.atol,
                rtol=args.rtol,
                trace_dir=output if args.trace else None,
                measure=not args.correctness_only,
            )
        )
        report["preflight_after"] = backend_preflight()
        if report["preflight_after"]["other_compute_processes"] and not args.correctness_only:
            raise RuntimeError(
                "another GPU compute process appeared during measurement; result rejected"
            )
        report["update_device"] = str(device)
        report["family"] = entry.family
        report["trainable_parameters"] = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
    except Exception as error:
        report["passed"] = False
        if isinstance(error, UpdateComparisonError):
            report.update(error.report)
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["seconds_including_load"] = time.perf_counter() - started
        repo_root = Path(__file__).resolve().parents[3]
        report["probe_source_sha256"] = {
            str(p.relative_to(repo_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                Path(__file__).with_name("replay_overlap.py").resolve(),
                Path(__file__).parents[1] / "generation" / "full_sequence_probe.py",
            )
        }
        report["revision"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
        report["worktree_status"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
        )
        if cfg is not None:
            (output / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
