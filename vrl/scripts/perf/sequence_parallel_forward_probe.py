"""One-forward numerical probe of the SD3 sequence-parallel install on real weights.

The end-to-end acceptance (``sequence_parallel_acceptance``) compares decoded
images after a whole denoise schedule, where bf16 kernel noise is amplified by
CFG and ten steps. This probe isolates ONE transformer forward: the same
random latents / text embeddings / timestep through the plain transformer on
one GPU and through the Ulysses install on N ranks (NCCL), in a chosen dtype.
A float32 run that agrees to ~1e-5 relative error while bf16 disagrees at
~1e-2 pins the end-to-end gap on precision order, not on the exchange.

Usage (one GPU per rank plus none extra: rank 0 also computes the reference):

    python -m vrl.scripts.perf.sequence_parallel_forward_probe \\
        --model stabilityai/stable-diffusion-3.5-medium \\
        --revision b940f670f0eda2d07fbb75229e779da1ad11eb80 \\
        --world 2 --dtype float32 --height 512 --width 512
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import socket
import tempfile
from pathlib import Path
from typing import Any

import torch


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_transformer(args: argparse.Namespace, device: torch.device) -> Any:
    from diffusers import SD3Transformer2DModel

    dtype = getattr(torch, args.dtype)
    model = SD3Transformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        revision=args.revision,
        torch_dtype=dtype,
        local_files_only=True,
    )
    return model.to(device).eval()


def _build_inputs(args: argparse.Namespace, model: Any, device: torch.device) -> dict[str, Any]:
    config = model.config
    generator = torch.Generator().manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    latent_h = args.height // 8
    latent_w = args.width // 8
    return {
        "hidden_states": torch.randn(
            args.batch, config.in_channels, latent_h, latent_w, generator=generator
        ).to(device, dtype),
        "encoder_hidden_states": torch.randn(
            args.batch, args.text_tokens, config.joint_attention_dim, generator=generator
        ).to(device, dtype),
        "pooled_projections": torch.randn(
            args.batch, config.pooled_projection_dim, generator=generator
        ).to(device, dtype),
        "timestep": torch.full((args.batch,), float(args.timestep), device=device, dtype=dtype),
    }


def _rank_main(rank: int, port: int, args: argparse.Namespace, scratch: str, queue: Any) -> None:
    """Compute on one rank; results go through files, not the queue (a tensor
    put on a multiprocessing queue is a shared-memory handle that dies with the
    child before the parent reads it)."""

    try:
        import torch.distributed as dist

        from vrl.models.sequence_parallel import install_sd3_sequence_parallel

        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=args.world,
        )
        try:
            model = _load_transformer(args, device)
            inputs = _build_inputs(args, model, device)
            reference = None
            if rank == 0:
                with torch.no_grad():
                    reference = model(**inputs).sample.float().cpu()
            install_sd3_sequence_parallel(model, dist.group.WORLD)
            with torch.no_grad():
                parallel = model(**inputs).sample.float().cpu()
        finally:
            dist.destroy_process_group()
        path = Path(scratch) / f"rank{rank}.pt"
        torch.save({"reference": reference, "parallel": parallel}, path)
        queue.put((rank, str(path)))
    except BaseException as error:  # pragma: no cover - transported to parent
        queue.put((rank, f"error: {error!r}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--text-tokens", type=int, default=333)
    parser.add_argument("--timestep", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.world < 2:
        raise ValueError("--world must be >= 2")
    if torch.cuda.device_count() < args.world:
        raise ValueError("one visible GPU per rank is required")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    payloads: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="sp_forward_probe_") as scratch:
        procs = [
            context.Process(target=_rank_main, args=(rank, port, args, scratch, queue))
            for rank in range(args.world)
        ]
        for proc in procs:
            proc.start()
        try:
            for _ in procs:
                rank, payload = queue.get(timeout=1800)
                payloads[rank] = payload
        finally:
            for proc in procs:
                proc.join(timeout=60)
                if proc.is_alive():
                    proc.kill()
        errors = {rank: p for rank, p in payloads.items() if p.startswith("error:")}
        if errors:
            raise RuntimeError(f"rank failures: {errors}")
        results = {rank: torch.load(path, weights_only=True) for rank, path in payloads.items()}
    reference = results[0]["reference"]
    report: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "world": args.world,
        "shape": list(reference.shape),
        "reference_abs_mean": float(reference.abs().mean()),
        "ranks": {},
    }
    for rank in range(args.world):
        parallel = results[rank]["parallel"]
        diff = (parallel - reference).abs()
        report["ranks"][rank] = {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "rel_l2": float(diff.norm() / reference.norm()),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
