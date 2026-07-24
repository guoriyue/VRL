# FSDP2 2×1 cross-node run (Cosmos Predict2.5 + Kling DiffusionNFT)

> **Hardware acceptance runbook:** collective-safe DTensor/optimizer/EMA
> parking, resume, and checkpoint export are implemented and covered by CPU
> Gloo tests. A real two-server run remains the acceptance boundary for this
> exact topology.

Symmetric colocated **FSDP2** variant of the DDP 2×1 example. The intended topology has
two servers, one GPU each, and one torchrun rank per server, with one node-local Ray
cluster for rollout plus an in-process Kling reward — but the trainer **shards**
params/grads/optimizer as DTensor across the two ranks (ZeRO-3) instead of
replicating them. The two ranks draw **disjoint** prompt slices (genuine data
parallelism); only the per-layer all-gather / reduce-scatter crosses the network.

```
 node A (rank0 / master)              node B (rank1)
 +-------------------------------+    +-------------------------------+
 | torchrun rank0                |    | torchrun rank1                |
 |  FSDP2 shard 0  (cuda:0)      |<==>|  FSDP2 shard 1  (cuda:0)      |
 |  Ray-backed rollout           | NCCL|  Ray-backed rollout           |
 |  in-process Kling reward      |    |  in-process Kling reward      |
 |  writes metrics/ckpt/eval     |    |  (no IO — rank0 only)         |
 +-------------------------------+    +-------------------------------+
        master_addr = node A IP, master_port = 29500
```

## Launch (run on BOTH nodes)

```bash
# node A (master, rank0):
./fsdp_2x1_launch.sh 0
# node B (worker, rank1) — same env, rank 1:
./fsdp_2x1_launch.sh 1
```

Override per run via env vars (see the script header), e.g. `OUT=...`,
`EXTRA_OVERRIDES="sampling.width=832 sampling.height=480 ..."`, `NODE_A=<A_IP>`,
`IFACE=<nccl_interface>`.

## Prerequisites

- **Both nodes on THIS branch.** The `fsdp` gate only opened in
  `vrl/scripts/common/online.py` on `feat/fsdp-online-orchestration`; an
  out-of-date node B fail-fasts at startup. Sync the repo before launching.
- HF models present on each node's NVMe cache (`HF_HOME=/mnt/nvme/hf`,
  `HF_HUB_OFFLINE=1`).
- The two servers can reach each other on `--master_port` and the NCCL ports; set
  `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` to the shared interface if multi-homed.
- Do **NOT** start or attach a shared Ray cluster. The recipe uses
  `ray.init(address="local")`, which starts a fresh cluster rather than following
  `RAY_ADDRESS` or Ray's current local-cluster state. The old bare `ray.init()` behavior
  did not prove local ownership and must not be restored.
- Do **NOT** use `ray stop` to clean a shared host. It scans machine-local Ray process
  names rather than targeting one cluster address, so it can terminate another visible,
  killable Ray cluster. Explicit local init prevents accidental attachment, not an external
  process kill; prefer dedicated hosts/containers for unattended runs.

## Differences from the DDP launcher

- **Resume is supported**: pass `trainer.resume_from=<checkpoint>` through
  `EXTRA_OVERRIDES` when resuming an interrupted run.
- **No `find_unused_parameters`**: a DDP reducer knob; FSDP has no reducer.
- **EMA on, torch.compile off**: the recipe inherits EMA from the base NFT
  config; compile remains unsupported with FSDP2 reshard-after-forward.

## Note on cost

For this 2B transformer + LoRA (fits on one card), **DDP is faster** — cross-node
FSDP all-gathers params over the network every forward. This FSDP variant exists to
validate the sharded path and is the right tool only when a model does not fit on
one card, or for full-param training. See `SPRINT_multi_gpu_training.md`.
