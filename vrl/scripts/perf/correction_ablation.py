"""Paired short SANA training runs to measure existing correction modes.

This is a diagnostic ablation, not a held-out reward-improvement benchmark.
All arms use one optimizer update per rollout so recompute is admissible.
Use a fresh output directory; existing run directories are never overwritten.
Run with the Python environment that supplies Ray and vLLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def probe_loss(output: Path) -> None:
    """Controlled log-prob perturbations through the production GRPO loss.

    These are synthetic scalar signals, not model forwards or reward evidence.
    Separate +/- advantages expose which side PPO clipping already suppresses.
    """
    import torch

    from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.algorithms.trajectory import AlgorithmInput
    from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

    records = []
    for delta in [0.0, 0.001, -0.001, 0.01, -0.01, 0.8, -0.8]:
        for mode in ["off", "tis", "rs", "recompute"]:
            fresh = torch.full((2,), delta, requires_grad=True)
            signals = TrajectorySignalBatch(
                segments={
                    "denoise": SegmentSignal(
                        name="denoise",
                        distribution="flow_matching",
                        log_prob=fresh,
                        old_log_prob=torch.zeros(2),
                        mask=torch.ones(2),
                    )
                },
                group_ids=torch.arange(2),
                primary_segment="denoise",
            )
            algorithm = GRPO(GRPOConfig(clip_ratio=1e-4, kl_coef=0.0))
            algorithm.precision_correction = PrecisionCorrectionConfig(
                tis_mode="truncate" if mode == "tis" else "off",
                rs_mode="seq_mean_k1" if mode == "rs" else "off",
                recompute_old_logprob="on" if mode == "recompute" else "off",
            )
            loss, metrics = algorithm.compute_loss(
                AlgorithmInput(
                    signals=signals,
                    advantages=torch.tensor([1.0, -1.0]),
                )
            )
            loss.backward()
            records.append(
                {
                    "synthetic_logprob_delta": delta,
                    "mode": mode,
                    "loss": loss.item(),
                    "d_loss_d_logprob": fresh.grad.tolist(),
                    "tis_clip_fraction": metrics.update.tis_clip_fraction,
                    "rs_masked_fraction": metrics.update.rs_seq_masked_fraction,
                }
            )
    (output / "synthetic_loss_probe.json").write_text(json.dumps(records, indent=2))


def compare_runs(output: Path, baseline: Path) -> None:
    """Compare recorded metrics and actual tensors, not serialized-file hashes."""
    import torch

    reference = torch.load(
        baseline / "checkpoint-final/checkpoint.pt", map_location="cpu", weights_only=False
    )["model"]["owned_state"]
    comparisons = []
    for path in sorted(output.rglob("metrics.csv")):
        run = path.parent
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        parity = []
        debug = run / "training_debug.jsonl"
        if debug.exists():
            parity = [
                r
                for line in debug.read_text().splitlines()
                if (r := json.loads(line)).get("event") == "replay_parity_gate"
            ]
        checkpoint = run / "checkpoint-final/checkpoint.pt"
        tensor_count = different = 0
        max_diff = None
        if checkpoint.exists():
            max_diff = 0.0
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"][
                "owned_state"
            ]
            if state.keys() != reference.keys():
                raise ValueError(f"Checkpoint module keys differ: {run}")
            for module, weights in reference.items():
                if weights.keys() != state[module].keys():
                    raise ValueError(f"Checkpoint tensor keys differ: {run}/{module}")
                for name, weight in weights.items():
                    other = state[module][name]
                    if weight.shape != other.shape or not (
                        torch.isfinite(weight).all() and torch.isfinite(other).all()
                    ):
                        raise ValueError(f"Invalid tensor comparison: {run}/{module}/{name}")
                    tensor_count += 1
                    different += not torch.equal(weight, other)
                    max_diff = max(max_diff, (weight.float() - other.float()).abs().max().item())
        comparisons.append(
            {
                "run": str(run.relative_to(output)),
                "metrics": rows,
                "parity": parity,
                "checkpoint_exists": checkpoint.exists(),
                "compared_tensors": tensor_count,
                "different_tensors": different,
                "max_weight_abs_diff": max_diff,
            }
        )
    (output / "comparison.json").write_text(json.dumps(comparisons, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["off", "tis", "rs", "recompute"],
        default=["off", "tis", "rs", "recompute"],
    )
    parser.add_argument("--replay-batch", type=int, default=4)
    parser.add_argument(
        "--model-path", type=Path, help="Local snapshot of the preset's exact revision"
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--compare-to", type=Path, help="Summarize existing runs against this baseline"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.replay_batch < 1:
        parser.error("epochs and replay-batch must be positive")
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.compare_to is not None:
        compare_runs(output, args.compare_to.resolve())
        return
    probe_loss(output)
    if args.probe_only:
        return
    env = dict(os.environ, PYTHONPATH=str(root))
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            run = output / f"seed{seed}_{mode}"
            if run.exists():
                raise FileExistsError(f"Refusing to overwrite experiment {run}")
            overrides = [
                f"trainer.output_dir={run}",
                f"trainer.total_epochs={args.epochs}",
                "trainer.save_freq=0",
                f"trainer.seed={seed}",
                "actor.ppo_epochs=1",
                f"actor.training_microbatch_size={args.replay_batch}",
                "actor.timestep_fraction=0.5",
                "actor.ema.enable=false",
                "rollout.prompts_per_batch=2",
                "rollout.n_samples_per_prompt=4",
                "rollout.samples_per_generation_batch=4",
                "trainer.replay_parity.every_update=true",
                "trainer.precision_drift_guard.mode=warn",
                "reward.components={aesthetic: 1.0}",
                "reward.kwargs={aesthetic: {device: null, model_name: google/siglip-so400m-patch14-384}}",
            ]
            if args.model_path is not None:
                overrides += [f"model.path={args.model_path.resolve()}"]
            if mode == "tis":
                overrides += ["trainer.precision_correction.tis_mode=truncate"]
            elif mode == "rs":
                overrides += ["trainer.precision_correction.rs_mode=seq_mean_k1"]
            elif mode == "recompute":
                # YAML treats an unquoted `on` as boolean True.
                overrides += ['trainer.precision_correction.recompute_old_logprob="on"']
            from vrl.config.loading import load_config
            from vrl.config.schema import parse_config

            parse_config(load_config("experiment/sana/online_grpo_aesthetic", overrides=overrides))
            command = [
                sys.executable,
                "-m",
                "vrl.scripts.train",
                "--config",
                "experiment/sana/online_grpo_aesthetic",
                *overrides,
            ]
            manifest = {
                "mode": mode,
                "seed": seed,
                "command": command,
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
            }
            (output / f"{run.name}.command.json").write_text(json.dumps(manifest, indent=2))
            started = time.monotonic()
            print(f"Starting {run.name}", flush=True)
            with (output / f"{run.name}.log").open("w") as log:
                process = subprocess.run(
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            rows = []
            if (run / "metrics.csv").exists():
                with (run / "metrics.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
            results.append(
                {
                    **manifest,
                    "returncode": process.returncode,
                    "elapsed_s": time.monotonic() - started,
                    "metrics": rows,
                }
            )
            (output / "summary.json").write_text(json.dumps(results, indent=2))
            print(f"Finished {run.name}: exit={process.returncode}, rows={len(rows)}", flush=True)
    if any(r["returncode"] != 0 or len(r["metrics"]) != args.epochs for r in results):
        raise SystemExit("One or more arms failed/incomplete; inspect summary.json and logs.")


if __name__ == "__main__":
    main()
