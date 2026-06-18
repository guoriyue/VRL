# Cosmos Predict2.5 + Kling DiffusionNFT across 2 servers (1 GPU each)

Runbook for `experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_cross_node`.

Runs one RL job across **two hosts over a Ray cluster**: the trainer on one
server, generation (and reward scoring) on the other. This is the cross-node
("rollout on one GPU, train on another, on different servers") topology.

## Topology

```text
 node A (HEAD / driver)            node B (WORKER)
 +---------------------+           +-------------------------------+
 | single_process      |  Ray RPC  | rollout worker (cosmos 2B)    |
 | trainer (cosmos 2B) |<--------->|  + Kling reward pool worker   |
 | GPU 0  (NOT in Ray) |  weights  | share GPU 0 (time-shared)     |
 +---------------------+           +-------------------------------+
```

- **node A** runs the driver = the `single_process` trainer. Its GPU is held
  **out of Ray's pool** (`ray start --head --num-gpus=0`) so no actor lands on
  the trainer card.
- **node B** runs the rollout worker *and* the Kling reward pool worker, which
  **time-share node B's one GPU**.
- Trainer→rollout weight sync goes over the network via the Ray object store
  (`ray.put` → `update_weights.remote`), LoRA-only (~tens of MB per sync).

### The 2-GPU cost (why rollout reloads each epoch)

The Kling reward is **pool-only** (`vrl/rewards/base.py` rejects anything but
`execution: pool`): it must own a GPU slice on a Ray node and cannot run inside
the trainer process. With only 2 GPUs, it shares node B's card with rollout.
Sharing makes the resolver mark rollout **`on_demand`**, so the rollout worker is
killed after each collect and **reloads the cosmos 2B model on the next epoch**
(`vrl/generation/ray/runtime.py`: `release()` → `kill_actors`; next `generate()`
reloads via `load_policy`).

That is acceptable to **validate** the cross-node mechanism, but the per-epoch
reload is expensive over 256 epochs. For a real training run, give the reward
its own GPU — see [Upgrade to 3 GPUs](#upgrade-to-3-gpus-rollout-resident).

## Prerequisites (both nodes)

- Same VRL code + Python env installed on **both** servers.
- Network reachability from node B to node A on the Ray ports (6379 + workers).
- **node A**: cosmos Predict2.5 2B weights + the VideoPhy prompt dataset.
- **node B**: cosmos Predict2.5 2B weights (rollout loads them) **and** the Kling
  `KlingTeam/VideoReward` model in the local HF cache — the reward preset sets
  `local_files_only: true`, so node B must have it pre-downloaded.
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
- **Placement log.** Look for `GlobalRayPlacementOwner created: bundles=(1,) …
  roles={'rollout': (0,), 'reward': (0,)}` — one GPU bundle, both roles on it.
- **Rollout/reward land off-head.** Node-aware validation (`validate_actor_gpu_ids`,
  cross-node path) raises if a worker lands on the head node, so a clean start
  means generation + reward are on node B.
- **Training proceeds** with weight syncs between epochs; judge learning by
  `eval_reward_mean` in `eval_metrics.csv` (see the fixed-eval sprint), not the
  rotating training `reward_mean`.

## Teardown

```bash
# on each node
ray stop
```

## Upgrade to 3 GPUs (rollout resident)

To drop the per-epoch reload, give the reward its own GPU so rollout stays
resident. Add a second GPU to node B (or a third host node C with 1 GPU), then in
the config set the reward to a dedicated card instead of sharing:

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
