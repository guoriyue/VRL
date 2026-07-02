# Cosmos Predict2.5 + Kling DiffusionNFT across 2 servers (1 GPU each)

Runbook for `experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_cross_node`.

> Status as of 2026-07-01: this runbook is stale for reward placement. The Ray
> reward actor pool was removed; rewards now score in-process on the driver via
> `LocalRewardRuntime`, and heavyweight rewards use
> `reward.kwargs.<name>.sleep_offload=true` to park on CPU between scores. The
> rollout-on-worker cross-node half remains relevant, but this exact 2-host
> Kling reward placement needs a redesign or a new remote reward transport
> before running again.

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

This recipe predates the reward execution rewrite. The old design put Kling in
a Ray reward pool on node B, sharing that GPU with rollout and forcing rollout
to reload each epoch. That transport is gone. As composed today, Kling would
load in-process on node A next to the trainer model, so the recipe is
memory-unverified and should not be treated as a ready cross-node Kling run.
Use it only as a reference for the rollout-on-worker placement mechanics.

## Prerequisites (both nodes)

- Same VRL code + Python env installed on **both** servers.
- Network reachability from node B to node A on the Ray ports (6379 + workers).
- **node A**: cosmos Predict2.5 2B weights + the VideoPhy prompt dataset.
- **node B**: cosmos Predict2.5 2B weights (rollout loads them).
- **node A**: if you still run this stale Kling composition, the
  `KlingTeam/VideoReward` model must be in the local HF cache on the driver
  because reward scoring is in-process there.
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
ray status
# expect: 1.0 GPU total in the cluster, all on node B; head shows 0 GPU.
```

**4. Launch training from node A** (the driver must run on the head):

```bash
# on node A
vrl-train --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_cross_node
```

No code change and no `RAY_ADDRESS` needed: the driver runs on node A, which
already started the cluster, so the plain `ray.init()` in the recipe finds the
local cluster (`/tmp/ray/ray_current_cluster`) and attaches to it — node B
included. Only set `RAY_ADDRESS=10.0.0.1:6379` if you launch the driver from a
machine that is *not* part of the cluster (no local `ray start`).

### Quick smoke (fewer epochs) before the full run

```bash
vrl-train --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_cross_node \
  trainer.total_epochs=4 trainer.save_freq=2 trainer.eval.freq=2
```

## What to check it worked

- **Preflight passes.** `cross_node` runs a preflight after Ray init. If node B
  has not joined yet you get a clear error: *"cross_node rollout needs 1 GPU(s)
  on non-driver Ray nodes…"* — start node B first. If the head still exposes a
  GPU you get: *"the driver/head node exposes N Ray GPU(s)…"* — restart the head
  with `--num-gpus=0`.
- **Placement log.** Look for `GlobalRayPlacementOwner created: bundles=(1,) …`
  with rollout on the worker bundle.
- **Rollout lands off-head.** Node-aware validation (`validate_actor_gpu_ids`,
  cross-node path) raises if a worker lands on the head node, so a clean start
  means generation is on node B.
- **Training proceeds** with weight syncs between epochs; judge learning by
  `eval_reward_mean` in `eval_metrics.csv` (see the fixed-eval sprint), not the
  rotating training `reward_mean`.

## Teardown

```bash
# on each node
ray stop
```

## Upgrade path

The old 3-GPU advice also assumed a Ray reward actor pool. With the current
in-process reward runtime, a real cross-node reward run needs either a new
remote reward transport or an explicit redesign that keeps reward memory off the
trainer GPU.

If such a remote transport is added later, the intended placement is still:
trainer on node A, rollout on node B, reward on a dedicated non-head GPU so
rollout stays resident:

```yaml
distributed:
  resources:
    reward:
      share_with_rollout: false   # reward gets its own GPU bundle
```

Join the extra GPU to the cluster (`ray start --address=<head>:6379 --num-gpus=1`
on node B's second GPU or on node C). The resolver then keeps rollout
**resident** (no reload), with trainer=A, rollout and reward on the two
non-head GPUs.

## Notes

- This is an infrastructure runbook, not a benchmark or paper reproduction. The
  learning result for this recipe is tracked separately (single-GPU runs to date
  showed only noise — see the cosmos+Kling reward-curve sprint).
- Single-GPU colocated runs use the default config
  (`online_nft_kling_video_reward`); only switch to this cross-node variant when
  you actually have the 2-host cluster up.
