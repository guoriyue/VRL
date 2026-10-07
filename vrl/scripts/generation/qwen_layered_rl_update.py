"""Run one reviewed, reward-driven Flow-GRPO update on layered SDE rollouts.

This small-batch integration runner reuses the repository GRPO objective. It is
not a replacement for iterative rollout collection or held-out quality testing.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import torch

from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.rollouts.evaluators.types import FlowSDESignal, TrajectorySignalBatch
from vrl.scripts.generation.qwen_layered_rl_probe import (
    load_transformer,
    load_upstream,
    predict_velocity,
)


def run_update(root, config):
    from diffusers import FlowMatchEulerDiscreteScheduler

    plan = json.loads((root / "plan.json").read_text())
    reward_directory = root / config.get("reward_directory", "reward_calibration")
    scores_path = reward_directory / "scores.json"
    validation = json.loads((reward_directory / "validation.json").read_text())
    assert validation["usable_for_small_update"] is True
    assert validation["scores_sha256"] == hashlib.sha256(scores_path.read_bytes()).hexdigest()
    scored = json.loads(scores_path.read_text())
    assert scored["status"] == "complete"
    cases = {case["name"]: case for case in plan["cases"] if case["split"] == "train"}
    rows = [row for row in scored["rows"] if row["kind"] == "candidate"]
    assert rows and all(row["valid"] and row["case"] in cases for row in rows)
    groups = torch.tensor([list(cases).index(row["case"]) for row in rows])
    assert all((groups == group).sum() >= 2 for group in groups.unique())
    rewards = torch.tensor([row["reward"] for row in rows], dtype=torch.float32)
    assert torch.isfinite(rewards).all()
    algorithm = GRPO(
        GRPOConfig(clip_ratio=config["clip_ratio"], kl_coef=config["kl_coef"], flow_kl_use_dt=True)
    )
    advantages = algorithm.compute_advantages_from_tensors(rewards, groups)
    assert torch.isfinite(advantages).all() and advantages.abs().max() > 0, "No reward signal"
    output = root / "update01"
    output.mkdir(exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    upstream = load_upstream(upstream_path)
    model = load_transformer(plan)
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    initial = torch.load(root / "initial_adapter.pt", weights_only=True)
    assert initial.keys() == parameters.keys()
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(initial[name])
    optimizer = torch.optim.AdamW(parameters.values(), lr=config["learning_rate"], weight_decay=0)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        plan["base_path"], subfolder="scheduler", local_files_only=True
    )
    indices = config["timesteps"]
    assert len(set(indices)) == len(indices) and all(
        0 <= index < plan["steps"] for index in indices
    )
    count = len(rows) * len(indices)
    logs = []
    optimizer.zero_grad(set_to_none=True)
    for row_index, row in enumerate(rows):
        directory = root / "rollouts" / row["case"] / row["candidate"]
        assert hashlib.sha256(Path(row["view"]).read_bytes()).hexdigest() == row["view_sha256"]
        trajectory = torch.load(directory / "trajectory.pt", weights_only=True)
        assert trajectory["noise_level"] == plan["noise_level"] > 0
        assert trajectory["condition_case"] == row["case"]
        assert trajectory["initial_adapter_seed"] == plan["adapter_seed"]
        condition = torch.load(
            root / "conditions" / row["case"] / "condition.pt", weights_only=True
        )
        assert condition["source_sha256"] == cases[row["case"]]["sha256"]
        condition = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in condition.items()
        }
        scheduler.sigmas = trajectory["sigmas"].cuda()
        scheduler.timesteps = trajectory["timesteps"].cuda()
        for index in indices:
            latent = trajectory["observations"][index].cuda()
            action = trajectory["actions"][index].cuda()
            timestep = scheduler.timesteps[index]
            velocity = predict_velocity(model, latent, timestep, condition, upstream)
            replay = sde_step_with_logprob(
                scheduler,
                velocity,
                timestep[None],
                latent,
                prev_sample=action,
                noise_level=plan["noise_level"],
                step_index=index,
                return_dt=True,
            )
            old = trajectory["old_log_density_mean"][index].cuda()
            parity = float((replay.log_prob.detach() - old).abs().max())
            assert parity < 1e-5, f"Pre-update replay mismatch: {parity}"
            with torch.no_grad():
                reference = sde_step_with_logprob(
                    scheduler,
                    trajectory["old_velocity"][index].cuda(),
                    timestep[None],
                    latent,
                    prev_sample=action,
                    noise_level=plan["noise_level"],
                    step_index=index,
                    return_dt=True,
                )
            signal = FlowSDESignal(
                name="layers",
                distribution="flow_sde_mean_element_density",
                log_prob=replay.log_prob,
                old_log_prob=old,
                mask=torch.ones_like(old),
                ref_log_prob=reference.log_prob,
                prev_sample_mean=replay.prev_sample_mean,
                std_dev_t=replay.std_dev_t,
                dt=replay.sqrt_neg_dt,
                sigma=replay.sigma,
                ref_prev_sample_mean=reference.prev_sample_mean,
            )
            loss, metrics = algorithm.compute_loss(
                AlgorithmInput(
                    signals=TrajectorySignalBatch(
                        segments={"layers": signal},
                        group_ids=groups[row_index : row_index + 1].cuda(),
                    ),
                    advantages=advantages[row_index : row_index + 1].cuda(),
                    model=model,
                    timestep_index=index,
                )
            )
            assert torch.isfinite(loss)
            (loss / count).backward()
            logs.append(
                {
                    "case": row["case"],
                    "candidate": row["candidate"],
                    "step": index,
                    "reward": row["reward"],
                    "advantage": float(advantages[row_index]),
                    "loss": float(loss.detach()),
                    "preupdate_logprob_error": parity,
                }
            )
            print(json.dumps(logs[-1]), flush=True)
            (output / "replay.json").write_text(json.dumps(logs, indent=2))
            del velocity, replay, reference, signal, loss, metrics
    gradients = [value.grad for value in parameters.values() if value.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    norm = torch.nn.utils.clip_grad_norm_(
        list(parameters.values()), config["max_grad_norm"], error_if_nonfinite=True
    )
    assert float(norm) > 0
    optimizer.step()
    saved = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    assert all(torch.isfinite(value).all() for value in saved.values())
    differences = {
        name: float((value - initial[name]).abs().max()) for name, value in saved.items()
    }
    assert max(differences.values()) > 0
    torch.save(saved, output / "adapter.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    # Test an actual recorded transition after update and after disk reload.
    with torch.no_grad():
        updated_velocity = predict_velocity(model, latent, timestep, condition, upstream)
        for value in parameters.values():
            value.zero_()
        restored = torch.load(output / "adapter.pt", weights_only=True)
        for name, value in parameters.items():
            value.copy_(restored[name])
        reloaded_velocity = predict_velocity(model, latent, timestep, condition, upstream)
    assert torch.equal(updated_velocity, reloaded_velocity)
    record = {
        "optimizer_updates": 1,
        "gradient_norm_before_clip": float(norm),
        "gradient_tensors": len(gradients),
        "changed_tensors": sum(value > 0 for value in differences.values()),
        "max_parameter_difference": max(differences.values()),
        "reload_velocity_exact": True,
        "velocity_difference_from_initial_rms": float(
            (updated_velocity.cpu() - trajectory["old_velocity"][indices[-1]])
            .square()
            .mean()
            .sqrt()
        ),
        "adapter_sha256": hashlib.sha256((output / "adapter.pt").read_bytes()).hexdigest(),
        "scores_sha256": validation["scores_sha256"],
        "initial_adapter_sha256": hashlib.sha256(
            (root / "initial_adapter.pt").read_bytes()
        ).hexdigest(),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "logprob_convention": "Mean element Gaussian log density, matching repository Flow-GRPO; not joint log probability.",
        "scope": "One real group-reward optimizer update. Quality improvement and generalization remain untested.",
    }
    (output / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with Path("/tmp/vrl-preservation-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_update(args.root.resolve(), json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
