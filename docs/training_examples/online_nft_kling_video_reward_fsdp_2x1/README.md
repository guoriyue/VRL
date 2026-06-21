# FSDP2 2×1 cross-node run (Cosmos Predict2.5 + Kling DiffusionNFT)

Symmetric colocated **FSDP2** variant of the DDP 2×1 example. Same topology — two
servers, one GPU each, one torchrun rank per server, each running its own local
Ray + colocated rollout + Kling reward — but the trainer **shards**
params/grads/optimizer as DTensor across the two ranks (ZeRO-3) instead of
replicating them. The two ranks draw **disjoint** prompt slices (genuine data
parallelism); only the per-layer all-gather / reduce-scatter crosses the network.

```
 node A (rank0 / master)              node B (rank1)
 +-------------------------------+    +-------------------------------+
 | torchrun rank0                |    | torchrun rank1                |
 |  FSDP2 shard 0  (cuda:0)      |<==>|  FSDP2 shard 1  (cuda:0)      |
 |  local ray.init + rollout     | NCCL|  local ray.init + rollout     |
 |  Kling reward (time-share)    |    |  Kling reward (time-share)    |
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
- Do **NOT** `ray start` a shared cluster — each rank's local `ray.init()` must own
  its own GPU.

## Differences from the DDP launcher

- **No `trainer.resume_from`**: FSDP2 optimizer-state export/load is not implemented
  yet (`build_strategy` fail-fasts on resume), so this launcher does not auto-resume.
- **No `find_unused_parameters`**: a DDP reducer knob; FSDP has no reducer.
- **EMA off, torch.compile off**: both fail-fast under FSDP2 in this first version
  (set in `online_nft_kling_video_reward_fsdp_2x1.yaml`).

## Note on cost

For this 2B transformer + LoRA (fits on one card), **DDP is faster** — cross-node
FSDP all-gathers params over the network every forward. This FSDP variant exists to
validate the sharded path and is the right tool only when a model does not fit on
one card, or for full-param training. See `SPRINT_multi_gpu_training.md`.
