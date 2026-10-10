"""Hand off at a completed checkpoint, then favor long training intervals."""

import json
import os
import subprocess
import sys
import time

import lora_campaign as campaign

ROOT = campaign.ROOT
OLD_UNIT = "qwen21-aesthetic-campaign.service"


def wait_for_completed_stage():
    run = ROOT / "train_to_5"
    deadline = time.monotonic() + 7200
    while True:
        # Only four explicit success results and a final checkpoint allow
        # replacing the old supervisor. A live training process is never stopped
        # solely because a poll or observation timeout expired.
        try:
            campaign.require_success(run)
            break
        except (FileNotFoundError, RuntimeError):
            state = subprocess.check_output(
                ["systemctl", "--user", "show", OLD_UNIT, "-p", "ActiveState", "--value"],
                text=True,
            ).strip()
            if state not in ("active", "activating", "deactivating"):
                raise RuntimeError(
                    "Existing campaign terminated before a successful round 5"
                ) from None
        if time.monotonic() > deadline:
            raise TimeoutError("Handoff observation expired; original job left untouched")
        time.sleep(5)
    state = json.loads((ROOT / "campaign_status.json").read_text())
    if state.get("phase") == "training" and state.get("target_step", 0) > 5:
        raise RuntimeError("Old campaign already started later training; leave it untouched")
    # Any remaining old-unit workers now belong to cleanup or optional round-5
    # evaluation. Stop that unit before allocating the same four GPUs again.
    subprocess.run(["systemctl", "--user", "stop", OLD_UNIT], check=True)
    return run


def main():
    sys.path.insert(0, str(campaign.REPO))
    os.environ.update(
        HF_HOME="/mnt/nvme/hf",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(campaign.REPO),
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    initial_plan = json.loads((ROOT / "long_campaign_plan.json").read_text())
    if initial_plan.get("resume_run"):
        run = ROOT / initial_plan["resume_run"]
        campaign.require_success(run)
        completed = int(initial_plan["completed_round"])
        print(f"Resuming completed round {completed} from {run}", flush=True)
    else:
        print(
            "Waiting for successful round 5; current four-GPU training remains untouched",
            flush=True,
        )
        run = wait_for_completed_stage()
        completed = 5
    while True:
        # The active agent can revise later stages without interrupting a live run.
        plan = json.loads((ROOT / "long_campaign_plan.json").read_text())
        pending = [stage for stage in plan["stages"] if stage["end_round"] > completed]
        if not pending:
            campaign.status("awaiting_quality_review", checkpoint=str(run / "checkpoint-final"))
            return
        stage = min(pending, key=lambda item: item["end_round"])
        end = stage["end_round"]
        previous = run
        run = ROOT / stage.get("run_name", f"long_to_{end}")
        run.mkdir(exist_ok=False)
        campaign.status("training", target_step=end, resume=str(previous / "checkpoint-final"))
        overrides = stage.get("overrides", [])
        command = [
            str(ROOT / "launch.sh"),
            f"trainer.resume_from={previous}/checkpoint-final",
            f"trainer.output_dir={run}",
            f"trainer.total_epochs={end}",
            "trainer.save_freq=5",
            "rollout.n_samples_per_prompt=8",
            "rollout.samples_per_generation_batch=2",
            "actor.training_microbatch_size=2",
            *overrides,
        ]
        (run / "launch_arguments.json").write_text(json.dumps(command, indent=2) + "\n")
        with (run / "service.log").open("w") as log:
            subprocess.run(
                command, cwd=campaign.REPO, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        campaign.require_success(run)
        campaign.evaluate(run, end, 0, build_gallery=False)
        completed = end


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Keep the live campaign's status authoritative while merely observing it.
        (ROOT / "long_campaign_error.json").write_text(
            json.dumps({"time": time.time(), "error": repr(error)}, indent=2) + "\n"
        )
        raise
