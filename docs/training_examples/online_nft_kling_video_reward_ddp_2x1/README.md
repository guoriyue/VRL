# Cosmos Predict2.5 + Kling DiffusionNFT — symmetric colocated DDP 2x1

> **Capability-gated:** this historical 2x1 runbook is not currently runnable.
> Shared-GPU phase leasing now requires collective-safe DDP state parking, which
> `DDPStrategy` rejects before model/Ray launch until reducer-bucket, optimizer,
> EMA/scaler, and live-gradient parking is implemented. Current DDP runs need a
> rollout GPU pool disjoint from the trainer.

Runbook for `experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1`,
run unattended across **two servers, 1 GPU each**, with the resume-capable launcher
[`ddp_2x1_launch.sh`](./ddp_2x1_launch.sh).

This is the intended **symmetric colocated** topology — distinct from the
[cross_node](../online_nft_kling_video_reward_cross_node/README.md) one. Each node runs a
full RL replica (rollout + reward + train on its own GPU); only the gradient all-reduce
crosses the network.

## Topology

```text
 node A (rank0 / master)              node B (rank1)
 +-------------------------------+    +-------------------------------+
 | torchrun rank0                |    | torchrun rank1                |
 |  DDP replica + Ray rollout    |gAR | DDP replica + Ray rollout     |
 |  + in-process Kling reward    |<-->| + in-process Kling reward     |
 |  GPU 0                        |NCCL| GPU 0                         |
 +-------------------------------+    +-------------------------------+
        gAR = gradient all-reduce only (NCCL over `enp39s0`)
```

- **One explicitly local Ray cluster per node.** The recipe uses
  `ray.init(address="local")` for this non-cross-node topology, so it starts a fresh cluster
  instead of following `RAY_ADDRESS` or Ray's current local-cluster state. The old bare
  `ray.init()` behavior did not prove ownership and must not be restored. (Contrast:
  cross_node intentionally attaches both hosts to one explicitly addressed cluster.)
- **Genuine 2× batch.** The data loader draws `world_size × prompts_per_batch` prompts and
  hands each rank a **disjoint** slice, so rank0 + rank1 cover 32 distinct conditions for
  `rbs=16` — the all-reduced gradient is over 2× the data, not a duplicated run.
- **NFT requires `find_unused_parameters=true`** — its previous/reference adapters
  `require_grad` but run under `no_grad`, so the DDP reducer would otherwise flag them.

## Run it

Explicit local init prevents accidental attachment; it does not protect processes from a
different workload's machine-local lifecycle commands. Prefer dedicated hosts/containers,
and do not use `ray stop` to "clean" a shared host: it is a process-name scan, not an
address-scoped teardown command.

On **each** node (from the repo root), pass the node rank:

```bash
# node A (master, rank0)
NODE_A=<A-ip> OUT=outputs/my_run \
  EXTRA_OVERRIDES="sampling.width=832 sampling.height=480 sampling.num_frames=33 \
    rollout.prompts_per_batch=16 rollout.sample_batch_size=4 \
    rollout.trajectory_storage.device=cpu rollout.trajectory_storage.dtype=bfloat16 \
    model.torch_compile.enable=false" \
  bash docs/training_examples/online_nft_kling_video_reward_ddp_2x1/ddp_2x1_launch.sh 0

# node B (worker, rank1) — same env, rank 1
... ddp_2x1_launch.sh 1
```

The launcher is **resume-capable**: rank1 rsyncs rank0's `checkpoint-*` first, then both
resume from the latest. Output is never deleted, so re-running after a crash continues.
All run-specific knobs go in `EXTRA_OVERRIDES` (the example above is the 480p_33f speed
config from the 2026-06-20 run; the recipe defaults are 512p/93f paper params).

## Unattended self-heal (what kept the 2026-06-20 run alive)

The launcher is the durable, in-repo half. The driving loop that ran it unattended was a
**session-only** hourly health-check + a 30-min metrics monitor — they are NOT committed
(they live in the agent session, not the repo). To reproduce unattended operation, wrap the
launcher in your own supervisor with this logic:

- **every ~hour:** check both ranks alive (`pgrep -f node_rank=0` on A / `node_rank=1` on B),
  `ep` advancing (rows in `metrics.csv`), no `OutOfMemory`/`Traceback` in `rank*.log`. On a
  dead rank → kill leftovers on both nodes, re-launch (it resumes from checkpoint). On an
  **OOM** → lower `rollout.sample_batch_size` one notch and re-launch.
- **note (node B):** worker processes do not always survive ssh-backgrounding; kill by PID
  and verify `nvidia-smi` is clear before re-launching, or the next run hits a stale GPU.

See the engineering history in
[`docs/sprints/done/SPRINT_symmetric_colocated_ddp.md`](../../sprints/done/SPRINT_symmetric_colocated_ddp.md)
and [`SPRINT_ddp_2x1_first_run_findings.md`](../../sprints/planned/SPRINT_ddp_2x1_first_run_findings.md),
and a full run record (config + reward trend + cost profile + the walls hit) in
[`docs/runs/cosmos_predict25_nft_kling_480p33f_rbs16_20260620/`](../../runs/cosmos_predict25_nft_kling_480p33f_rbs16_20260620/).

## Known gotchas (from the 2026-06-20 run)

- **512p/93f OOMs on host RAM**, not GPU — rollout trajectories fill ~50 GB. Fix with
  `rollout.trajectory_storage.dtype=bfloat16`; at 480p_33f it's unnecessary (host RAM ~13 GB).
- **torch.compile cold-compile desyncs the two ranks** at 512p (one compiles while the other
  waits at the all-reduce). Disabled for that reason; re-enabling at 480p is the main untaken
  ~1.37× speedup.
- **`algorithm.global_std=true` under DDP** was per-rank-local until fixed — prefer the default
  `global_std=false` (per-group, correctly rank-local) unless on a build with the cross-rank
  all-reduce fix.
