# Cosmos Predict2.5 + Kling DiffusionNFT across 2 servers (1 GPU each)

> Evaluation is no longer embedded in online training. Saved checkpoints are
> evaluated separately with
> `python -m vrl.scripts.eval.cosmos_predict25_kling_eval`; this runbook covers
> training placement only.

Runbook for `experiment/cosmos_predict2_5/online_nft_kling_video_reward_cross_node`.

> Status as of 2026-07-11: reward placement follows the current in-process
> runtime. Kling scores on the driver; because reward has no separate GPU
> reservation, the resolved topology automatically gives it a CuMem phase lease
> on the trainer GPU and requires a release proof after every score. Only rollout
> is scheduled on the worker. The trainer+Kling memory fit on node A is still a
> deployment prerequisite for this two-host layout.

Runs one RL job across **two hosts over a Ray cluster**: the trainer on one
server and generation on the other. This is the cross-node
("rollout on one GPU, train on another, on different servers") topology.

## Topology

```text
 node A (HEAD / driver)            node B (WORKER)
 +---------------------+           +----------------------------+
 | single_process      |  Ray RPC  | rollout worker (cosmos 2B) |
 | trainer (cosmos 2B) |<--------->| GPU 0                      |
 | + in-process reward |  weights  +----------------------------+
 | GPU 0 (NOT in Ray)  |
 +---------------------+
```

- **node A** runs the driver = the `single_process` trainer. Its GPU is held
  **out of Ray's pool** (`ray start --head --num-gpus=0`) so no actor lands on
  the trainer card.
- **node B** runs the rollout worker. The previous Kling reward pool worker no
  longer exists; reward scoring now runs in the driver process on node A.
- Trainer→rollout weight sync goes over the network via the Ray object store
  (`ray.put` → `update_weights.remote`), LoRA-only (~tens of MB per sync).

### The 2-GPU caveat

The old design put Kling in a Ray reward pool on node B, sharing that GPU with
rollout and forcing rollout to reload each epoch. That transport is gone. The
current recipe explicitly declares no reward GPU reservation, so Kling loads
in-process on node A next to the trainer model and token 1 remains
rollout-only. Do not set `distributed.resources.reward.devices=[1]`: in
cross-node mode that ordinal is a Ray scheduling token, not a driver-local CUDA
device, and the runtime rejects it before loading either model.

## Prerequisites (both nodes)

- Same VRL code + Python env installed on **both** servers.
- Network reachability from node B to node A on the Ray ports (6379 + workers).
- A dedicated host/container/process-ownership boundary for this cluster. On a
  shared host, do not use this runbook unless the Ray processes are isolated
  from other users' lifecycle commands.
- **node A**: cosmos Predict2.5 2B weights + the VideoPhy prompt dataset.
- **node B**: cosmos Predict2.5 2B weights (rollout loads them).
- **node A**: the `KlingTeam/VideoReward` model must be in the local HF cache on
  the driver because reward scoring is in-process there. Confirm the trainer
  and Kling model fit on this GPU with the configured sleep-offload policy.
- Pick the same `trainer.output_dir` on a path that exists on both, or a shared
  filesystem, so checkpoints/eval/reward artifacts land where you expect.

## Run

**1. Start the Ray head on node A** (trainer host). Keep its GPU out of Ray:

```bash
# on node A
ray start --head --num-gpus=0 --port=6379
# note node A's IP, e.g. 10.0.0.1
```

**2. Join node B as a GPU worker:**

```bash
# on node B
ray start --address=10.0.0.1:6379 --num-gpus=1
```

**3. Verify the cluster** (from either node):

```bash
ray status --address=10.0.0.1:6379
# expect: 1.0 GPU total in the cluster, all on node B; head shows 0 GPU.
```

**4. Launch training from node A** (the driver must run on the head):

```bash
# on node A
RAY_ADDRESS=10.0.0.1:6379 \
  vrl-train --config experiment/cosmos_predict2_5/online_nft_kling_video_reward_cross_node
```

Always pass the intended cluster address explicitly. The recipe rejects a
missing, `auto`, or `local` `RAY_ADDRESS` when
`distributed.resources.cross_node=true`, then attaches with
`ray.init(address=<concrete address>)`. This avoids Ray's current-local-cluster
and auto-discovery paths. The topology flag describes placement; the concrete
address identifies the operator-owned cluster.

### Quick smoke (fewer epochs) before the full run

```bash
RAY_ADDRESS=10.0.0.1:6379 \
  vrl-train --config experiment/cosmos_predict2_5/online_nft_kling_video_reward_cross_node \
    trainer.total_epochs=4 trainer.save_freq=2 trainer.eval.freq=2
```

## What to check it worked

- **Preflight passes.** `cross_node` runs a preflight after Ray init. If node B
  has not joined yet you get a clear error: *"cross_node rollout needs 1 GPU(s)
  on non-driver Ray nodes…"* — start node B first. If the head still exposes a
  GPU you get: *"the driver/head node exposes N Ray GPU(s)…"* — restart the head
  with `--num-gpus=0`.
- **Placement log.** Look for `GlobalRayPlacementOwner created: bundles=(1,) …`
  with rollout on the worker bundle. The resolved resource plan should report
  `reward=[]`; Kling follows the driver's rank-local trainer device instead of
  consuming a Ray bundle.
- **Rollout lands off-head.** Node-aware validation (`require_actor_gpu_ids`,
  cross-node path) raises if a worker lands on the head node, so a clean start
  means generation is on node B.
- **Training proceeds** with weight syncs between epochs; judge learning by
  `eval_reward_mean` in `eval_metrics.csv` (see the fixed-eval sprint), not the
  rotating training `reward_mean`.

## Teardown

Do **not** run unconditional `ray stop` on a shared host. In Ray 2.55.1 the
command scans local processes for Ray process names; it is not scoped by
cluster address, namespace, port, or session directory. It can therefore stop
Ray processes belonging to another cluster that are visible and killable by
the caller.

On dedicated nodes or inside a dedicated PID namespace, the cluster owner may
run `ray stop` on each node after confirming no unrelated Ray workload shares
that boundary. On a shared node, teardown must be performed by the cluster
owner through its host/container/service lifecycle. Driver cleanup should use
`ray.shutdown()`; when attached to this externally started cluster it
disconnects the driver and does not stop the cluster.

## Upgrade path

The old 3-GPU advice assumed a Ray reward actor pool. The current resource
schema intentionally cannot turn a remote scheduling token into a local reward
device. Moving Kling off node A therefore requires a real remote reward runtime
and transport boundary, not another `reward.devices` ordinal. If that boundary
is added later, it can own a dedicated non-head GPU while rollout remains
resident on node B.

## Notes

- This is an infrastructure runbook, not a benchmark or paper reproduction. The
  learning result for this recipe is tracked separately (single-GPU runs to date
  showed only noise — see the cosmos+Kling reward-curve sprint).
- Single-GPU colocated runs use the default config
  (`online_nft_kling_video_reward`); only switch to this cross-node variant when
  you actually have the 2-host cluster up.
