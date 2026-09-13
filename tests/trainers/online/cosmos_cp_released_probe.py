"""Opt-in released-weight execution of the native CP composition test harness."""

import argparse
from pathlib import Path

import torch
import torch.multiprocessing as mp

from tests.trainers._strategy_policies import free_port
from tests.trainers.online.test_cosmos_cp_composition import _worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("control", "resume"), required=True)
    args = parser.parse_args()
    model_path = Path(args.model_path).resolve(strict=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() < 3:
        parser.error("requires two CP training GPUs and one disjoint rollout GPU")
    if args.phase == "control" and any(output.iterdir()):
        parser.error("control output must be empty; existing evidence will not be overwritten")
    if args.phase == "resume" and not (output / "checkpoint-1").is_dir():
        parser.error("resume requires the completed control checkpoint")
    mp.spawn(
        _worker,
        args=(free_port(), str(output), True, args.phase, str(model_path)),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    main()
