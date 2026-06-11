# SPRINT: Cosmos Performance

状态：P1.4 implemented, awaiting live gate（基于 Cosmos Predict2.5 + DiffusionNFT + Kling VideoReward 单卡 run，2026-06-09）。

进度：
- 2026-06-09 晚：P1.4（deferred per-epoch reward scoring）代码 + 测试落地
  （tests/trainers 97 passed，tests/rollouts+rewards+ray 195 passed）。
- P1.5（warm offload）降级为 P1.4 之后再测；P2 被 P1.4 取代。
- 2026-06-10：P1.4 live gate 通过（motion run 全程每 epoch 恰好 1 次
  `reward worker built model`，epoch 13.4 → 12.35 min）。P0 trace 落盘并
  分析（见 P0 Results：launch-bound，elementwise 47%）。
- compile smoke 并入 multi-GPU 全参 bring-up：单卡 cosmos NFT 配方已随
  LoRA 移除而退役（2026-06-10），在退役配方上烧 GPU 验证 compile 不再有
  对象；P0 的结论（elementwise 融合收益）直接转入全参路径首跑时验证。

## 0. Core Decision

当前 Cosmos run 的主要问题不是 active CUDA kernel 段没有吃满 GPU，而是端到端流程被单卡共享 reward/rollout 的生命周期切碎了。

结论分三层：

1. **GPU active 段已经很忙。** `nvidia-smi` 采样看到 training PID 在 active 段 `SM=99-100%`，显存约 `31GB`，功耗约 `570W`。
2. **端到端 wall-clock 仍然不满。** 当前配置让 rollout 和 Kling reward 共享同一张 GPU，每轮 reward scoring 都会释放 rollout actor、加载 reward actor、score、再释放 reward actor。这个阶段切换产生明显空洞。
3. **没有保存完整 profiler trace。** 当前 evidence 来自 `metrics.csv`、reward debug JSONL、训练日志和临时 `nvidia-smi` 采样。真正的 sprint 第一项是跑一个短 profiling run，把 trainer + rollout trace 落盘。

这套 shared-GPU 生命周期作为单卡防 OOM/debug fallback 是合理的；作为 Cosmos + Kling 的性能路径不合理。性能路线（按落地顺序）：P1.4 把打分推迟到每 epoch 一个 reward 会话（已实现，单卡上把 churn 从 6×/epoch 降到 1×/epoch）；P0 短 profiler trace 量化 active 段内部；多卡到位后 P1 给 reward 独立 GPU 让 actor 常驻；最后才是 rollout/trainer 分离和 NCCL weight sync（P3）。

## 1. Current Baseline

Run：

```text
PID: 2232676
config: experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward
output_dir: outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313
sampling: 416x240, 33 frames
rollout: rollout_batch_size=6, n=12, sample_batch_size=1
reward: KlingTeam/VideoReward@main, local_files_only=true
```

当前保存的 artifacts：

```text
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/metrics.csv
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/resolved_config.yaml
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/reward_debug/kling_video_reward_requests.jsonl
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/reward_debug/kling_video_reward_results.jsonl
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/reward_artifacts/manifest.jsonl
```

当前没有保存：

```text
*.nsys-rep
*.qdrep
torch_profiler trace
nvidia dmon/pmon log
```

### Training Signal

截至 2026-06-09 21:38（5 个 epoch）：

```text
epoch 0: reward_mean=-4.7853   epoch 3: reward_mean=-4.7180
epoch 1: reward_mean=-4.7943   epoch 4: reward_mean=-4.7726
epoch 2: reward_mean=-4.7404
```

这说明 `RewardModelWorker.load_model` 卡住的问题已经解除，但 reward signal 仍在 `-4.79` 附近震荡，还没有可判断的趋势（需要 15-20 epoch）。性能 sprint 不应该把 reward 曲线问题和系统吞吐问题混在一起：前者是 reward/algorithm diagnosis，后者是 rollout/reward/trainer scheduling diagnosis。

### Epoch Wall-Clock Structure（从日志时间戳实测）

每个 epoch 的 generation-placement 事件间隔呈固定模式：

```text
[48, 49, 49, 49, 50, 557] 秒/epoch ≈ 13.4 min/epoch
```

- 6 个 collect 周期 × ~49.7s ≈ 5.0 min（37%）：重启 rollout actor → batched
  生成 12 videos → 释放 → Kling actor 冷启动（queue_wait ~7.3s）→ 打分
  （12 × ~93ms）→ 释放
- 训练 ~9.3 min（63%）：6 microbatch × 9 timestep 切片 × 3 次前向
  （policy/previous/reference），这是 DiffusionNFT 的算法形状
- actor churn 税 = 6 × (7.3s reward 冷启动 + ~10-15s rollout 重启) ≈
  1.5-2 min/epoch，占 11-15%

**Amdahl 上限**：训练占 63%，所以全部调度优化（P1.4/P1.5/P1）合计只能把
epoch 从 ~13.4 min 压到 ~11.8-12 min。对比旧 run（11.45 min / 12 videos /
1 microbatch），当前 run 单位样本吞吐已提升约 5×（batched 生成 + 换载摊薄）。

50 epochs 预计 ~11 小时（2026-06-09 20:23 启动，预计 06-10 早晨结束）。

### Reward Runtime Evidence

截至当前检查：

```text
reward requests: 18
reward results: 216
queue_wait_ms: min=7229.1 mean=7326.6 max=7496.1
inference_ms:  min=70.1   mean=92.7   max=338.7
latency_ms:    min=7299.8 mean=7419.2 max=7830.0
```

解释：

- `queue_wait_ms` 主要反映 reward actor 启动、加载模型、Ray scheduling 这类每个 scoring request 付一次的开销。
- `inference_ms` 是 Kling model 对单个 artifact 的实际 scoring 时间，平均约 `93ms`，不是主要瓶颈。
- 当前 `rollout_batch_size=6`、`n=12`，每个 epoch 是 6 个 prompt group，每组 12 个 videos。因为当前数据样本带 metadata/reference，collector 会逐 prompt collect，所以 reward scoring 被切成 6 次 request，而不是一次 72 videos 的 request。

## 2. Why The Actor Churn Happens

关键配置来自 `configs/reward/kling_video_reward.yaml`：

```yaml
distributed:
  resources:
    reward:
      share_with_rollout: true
  rollout:
    release_before_reward_model: true
  reward:
    release_after_score: true
```

关键行为：

```text
generate videos
release rollout runtime memory before reward
load Kling reward actor
score batch
release reward actor after score
relaunch rollout actor for next collect
```

代码路径：

```text
vrl/rollouts/collector/core.py
  collect()
    if runtime.should_release_memory_before_reward():
        await self.release_runtime_memory()

vrl/ray/runtime.py
  RayActorMethodRuntime.map()
    finally:
        if self.release_after_call:
            await self.shutdown()

vrl/generation/ray/runtime.py
  ReleasableRayGenerationRuntime.release_memory()
    await runtime.shutdown()
```

资源解析也证明：只要 `reward.share_with_rollout=true`，即使 `visible_devices` 有多张卡，reward 仍会跟 rollout 共享 GPU，而不会自动拿独立 reward GPU。

```text
visible 2 -> trainer (0), rollout (1), reward (1), shared=True
visible 3 -> trainer (0), rollout (1), reward (1), shared=True
visible 4 -> trainer (0), rollout (1), reward (1), shared=True
```

所以“有多 GPU”本身不解决问题；必须显式把 reward 放到非 rollout GPU，并关闭 release-after-score。

## 3. Performance Hypotheses

### H1: Active denoise/train compute is already near saturated

Evidence:

```text
nvidia-smi: utilization.gpu=100%, utilization.memory=61%, memory.used=31087 MiB, power.draw=574W
pmon: PID 2232676 SM=99%, mem=60-66%
```

Interpretation:

active 段的 GPU compute 已经打满。下一步不是先调更大的 batch，而是保存 profiler trace，确认 active 段内部是 GEMM/attention/elementwise 哪一块占比最高。

### H2: End-to-end utilization is limited by phase switching

Evidence:

```text
reward worker built model: total_s ~= 7.1
reward queue_wait_ms mean ~= 7.3s
reward inference_ms mean ~= 0.093s/artifact
```

Interpretation:

reward forward 本身不是瓶颈。瓶颈是 shared-GPU lifecycle：每个 scoring request 都要重新拿 GPU、加载 model、释放 actor。这个问题不会靠微调 Kling inference batch 解决，应该先改资源放置和 actor residency。

### H3: Prompt grouping makes actor churn visible

Evidence path:

```text
vrl/rollouts/orchestration/prompt_collection.py
  PromptExample path calls collector.collect([prompt], ...)
```

With `rollout_batch_size=6` and `n=12`:

```text
6 collect calls/epoch
6 reward requests/epoch
72 videos/epoch
```

Interpretation:

如果 reward actor 常驻，这个 split 主要影响 scheduling overhead；如果 reward actor 每次释放，这个 split 会把 7s startup tax 放大 6 次。

## 4. Gates

### P0: Save a real short profiler trace

目标：把当前“终端采样 + JSONL debug”升级为可复查的 profiler artifact。

Command shape (note: `/profile=torch_profiler` does NOT work here — defaults
overrides only substitute an existing group entry, and this experiment has no
`profile` group in its defaults; use dotlist overrides instead):

```bash
python -u -m vrl.scripts.train \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward \
  trainer.profile=true \
  trainer.torch_profiler.enabled=true \
  trainer.torch_profiler.max_steps=1 \
  rollout.torch_profiler.enabled=true \
  rollout.torch_profiler.max_steps=1 \
  rollout.rollout_batch_size=1 \
  sampling.width=416 \
  sampling.height=240 \
  sampling.num_frames=33 \
  trainer.total_epochs=1 \
  trainer.output_dir=outputs/cosmos25_perf_profile_short
```

Expected profiler output:

```text
outputs/cosmos25_perf_profile_short/torch_profiler/trainer/
outputs/cosmos25_perf_profile_short/torch_profiler/generation/<worker_id>/
```

Success criteria:

```text
trainer trace exists
rollout generation trace exists
summary.txt exists for at least trainer or generation
metrics.csv has one epoch row
reward_debug JSONL exists
```

Do not run a 50-epoch production job with torch profiler enabled; profiler should be a short diagnostic run only.

#### P0 Results（2026-06-09 深夜，real bs=6 config，1 step trace 落盘）

Artifacts:

```text
outputs/cosmos25_perf_profile_bs6/torch_profiler/trainer/*.pt.trace.json   (6.0G, 完整 step)
outputs/cosmos25_perf_profile_bs6/torch_profiler/generation/rollout-0/     (trace + summary.txt)
outputs/cosmos25_perf_profile_bs6/trainer_trace_analysis.txt               (流式解析输出)
```

训练 step（54 次切片迭代）GPU kernel 分布：

```text
GEMM         250.4s   50.7%   (279k launches)
elementwise  231.9s   46.9%   (1.02M launches  ← 主要发现)
norm          11.8s    2.4%
attention       ~0s     ~0%   (240p 序列太短；SDPA backward 合计 2.8s)
```

结论：

1. 训练阶段是 launch-bound：GPU 在训练 span 内只有 ~64% 时间在忙
   （494s kernel / 775s span），一个 step 发射 1.3M+ kernel，elementwise
   占了近一半 GPU 时间（LoRA 缩放 / RoPE / 调制 / NFT 混合算术的碎 kernel）。
2. attention 在 240p 下不是成本——attention-kernel 类优化对此配置无关。
3. generation 侧同病：一组 12 视频纯前向 GPU 只有 ~2.8s（denoise 1.8s +
   VAE 0.9s），每组 ~25s 周期的其余部分是 CPU 序列化/同步/落盘
   （Command Buffer Full 44% self-CPU，37 万次 copy/to）。
4. 行动含义：torch.compile（Inductor elementwise 融合 + launch 削减）正中
   此画像，优先级上调；compile smoke 是下一个 GPU 空闲窗口的第一件事。

### P1: Separate reward GPU and keep reward actor resident

Goal: remove repeated Kling model reload from the hot path.

Since 2026-06-09 the release lifecycle flags derive from the resolved GPU
topology in `resolve_distributed_resources` (shared GPU -> release between
phases, dedicated GPU -> resident actors), so the target shapes only state the
placement decision:

Target resource shape for 3 GPUs (reward + rollout both resident):

```yaml
distributed:
  resources:
    visible_devices: [0, 1, 2]
    trainer:
      devices: [0]
    rollout:
      devices: [1]
      num_gpus: 1
      num_workers: 1
    reward:
      devices: [2]
      num_gpus: 1
      num_workers: 1
    allow_overlap: false
```

Target resource shape for 2 GPUs (reward resident on its own GPU; rollout
still hands GPU0 to the trainer between collect and train):

```yaml
distributed:
  resources:
    visible_devices: [0, 1]
    trainer:
      devices: [0]
    rollout:
      devices: [0]
      num_gpus: 1
      num_workers: 1
    reward:
      devices: [1]
      num_gpus: 1
      num_workers: 1
    allow_overlap: true
```

Success criteria:

```text
resolved resources show reward_shared_with_rollout=False
logs show reward worker loads once at startup, not once per score request
reward_debug queue_wait_ms drops from ~7.3s toward scheduling-only latency
no OOM during first epoch
```

### P1.4: Single-GPU first step — defer scoring to one reward session per epoch

Status: implemented 2026-06-09 (unit-tested; live GPU smoke pending — the
current 50-epoch run still uses the old per-group flow and must not be
restarted for this).

What landed (v2, 2026-06-09 night — merged scoring; the earlier keep_alive
session mechanism was replaced and deleted the same evening):

All prompt groups score through ONE reward call per epoch. Rollout prompts and
metadata are already per-sample at the rollout/artifact level (each
RewardRollout carries its own prompt + group metadata), so merging groups
needs no wire-format change — `release_after_score` then yields exactly one
actor lifecycle per epoch with its original semantics, no lifecycle flagging
anywhere.

```text
vrl/rollouts/collector/rewards.py       RewardScorer.score_many(requests) —
                                        build rollouts across groups, one
                                        score_batch call, split scores back by
                                        group size; score() delegates to it
vrl/rollouts/collector/core.py          collect_unscored() (generate only) +
                                        score_rollouts() (release rollout once,
                                        score all groups via score_many, build
                                        batches); UnscoredRollout carries groups
vrl/rollouts/orchestration/prompt_collection.py
                                        generate all groups first, then one
                                        collector.score_rollouts(...) call
tests/rollouts/orchestration/test_prompt_collection.py  order + remap pins
tests/rollouts/collector/test_runtime.py                score_many merge pin
```

Side benefits over the keep_alive variant: MultiReward components each load
once per epoch (no per-group reload, no multi-model VRAM stacking — components
still score sequentially), and per-request fixed costs (sharding, queueing,
debug rows) amortize. reward_debug now logs 1 request row per epoch with all
artifact ids instead of 6 rows.

Verification: tests/rollouts+rewards+ray+trainers+scripts+generation 369
passed. Live gate left: next GPU run must show `reward worker built model`
once per epoch (not 6×) and a reward curve consistent with the per-group
baseline.

P1 needs a second GPU. On the current 1×5090 box the cheapest large win is to
stop scoring per prompt group: collect all groups' trajectories first (rollout
stays resident the whole time — nothing else needs the GPU between groups),
then release rollout once, score all 72 artifacts in ONE reward request, split
rewards back per group, and build the batches.

The seam already exists — scoring and batch building are separate steps inside
`collect()` (`vrl/rollouts/collector/core.py:146-155`):

```python
rewards = await self.reward_scorer.score(
    batch_builder.reward_scoring_input(collector_request.metadata),
)
batch = batch_builder.build(rewards)
```

Change shape:

```text
1. Split RolloutCollector.collect() into collect_trajectories() (generate, no
   release, no score; returns the batch builder) and score+build.
2. collect_prompt_batches() accumulates builders for all groups, then does:
   release rollout once -> one reward request over all artifacts -> split
   rewards by group sizes -> build() each batch.
3. release_before_reward_model moves from per-collect to per-epoch.
```

Effect: actor churn drops 6x -> 1x per epoch (~2 min -> ~20s tax;
epoch ~13.4 min -> ~11.8 min). No actor lifecycle or Ray resource-accounting
changes; rollout->reward ordering stays strictly sequential, so no new OOM or
staleness surface. Risks are plain data-plumbing (reward order/group split) —
gate by diffing per-sample scores against a per-group-scoring run.

### P1.5: Single-GPU warm release — offload to CPU instead of killing actors

Status: deprioritized — re-measure only after P1.4. With one cycle per epoch
the remaining churn is ~20s/epoch; warm offload can recover at most ~15s of it,
and it carries real structural cost (see below).

Note the hidden cost beyond the 4 code touch points: Ray GPU accounting is
logical. The reward actor is schedulable today only because rollout shutdown
frees its 1.0-GPU placement. An offloaded-but-alive rollout actor keeps its
reservation, so reward cannot schedule — fixing that means fractional GPU
quotas (0.5/0.5) or num_gpus=0 with manual device pinning, i.e. fighting Ray's
resource model.

Today both releases are process kills:

```text
vrl/ray/runtime.py:63-64
  RayActorMethodRuntime.map()
    finally:
        if self.release_after_call:
            await self.shutdown()        # kills actor group + placement group

vrl/generation/ray/runtime.py:150-155
  ReleasableRayGenerationRuntime.release_memory()
    await runtime.shutdown()             # kills generation actors
  _ensure_runtime()                      # full relaunch + LoRA weight re-push
```

Measured cold-start tax per cycle (from the 2026-06-09 run):

```text
reward actor cold start: ~7.3s  (process spawn + import 1.75s + model build 5.4s)
rollout actor relaunch:  ~10-15s (respawn + pipeline load + update_weights re-push)
6 cycles/epoch -> ~1.5-2 min of a 13.4 min epoch (11-15%)
```

Change shape:

```text
1. Reward worker actor: add release_gpu_memory() -> model.to('cpu') + empty_cache,
   and restore-on-next-score (worker checks param device before scoring).
2. RayActorMethodRuntime: release_mode: shutdown | offload (default shutdown,
   preserving today's behavior); offload calls release_gpu_memory() on actors
   instead of shutdown().
3. Generation worker: same offload/restore pair; ReleasableRayGenerationRuntime
   keeps the actor and skips the LoRA re-push when the actor survived.
4. Config: rollout.release_before_reward_model / reward.release_after_score
   gain mode "offload"; kling_video_reward.yaml switches to offload on 1 GPU.
```

Expected effect: reward 7.3s -> ~2-3s (H2D of ~16GB), rollout 10-15s -> ~2-4s
(no respawn, no re-push). Epoch ~13.4 min -> ~12 min. Host RAM is not a
constraint (94GB total; CPU copies need ~20GB).

Risks / gates:

```text
CUDA fragmentation when alternating 24GB rollout and 16GB reward allocations;
gate on a 2-epoch smoke with no OOM and identical reward values vs shutdown mode.
Weight staleness: offloaded rollout actor must still apply update_weights()
pushed while it was offloaded (keep _last_state replay on restore).
```

Ceiling check: training is 63% of the epoch, so P1.5 (and P1) cap out at
~15% end-to-end. Worth doing once, not worth iterating on.

### P2: Batch reward requests — superseded by P1.4

The single-GPU variant of this ("score accumulated artifacts in one reward
request per epoch") is exactly P1.4 and is now the recommended first step.
On a multi-GPU box with a resident reward actor (P1), per-group requests cost
only scheduling latency, so no further batching work is planned there.

Worker-internal mini-batching (one forward over N videos instead of batch=1
per artifact) is a separate, parked sprint with explicit start triggers:
docs/sprints/SPRINT_reward_batched_inference.md — currently 0.8% of epoch,
not worth doing on this box.

### P3: Move weight sync off Ray object store

This is not the immediate blocker for the current single-GPU shared reward run, but it becomes important after reward and rollout have dedicated GPUs.

Direction:

```text
Ray remains control plane
NCCL becomes data plane for model/LoRA weight sync
```

Use the prior scaling sprint for implementation details:

```text
docs/sprints/SPRINT_cosmos_rl_scaling_learnings.md
```

## 5. Non-Goals

- Do not treat this document as proof of full end-to-end GPU utilization. There is no complete profiler trace yet.
- Do not tune Kling model internals before removing actor reload. Current evidence says `inference_ms` is small compared with `queue_wait_ms`.
- Do not flatten the reward/rollout lifecycle abstractions just to remove files. The shared-pool release behavior is a real protocol boundary, even if it is not the desired performance path.
- Do not build a full stage pipeline until trace data shows encode/decode/reward bubbles are large enough to hide.
- Do not enable torch profiler on long production runs.

## 6. References

Repository paths:

```text
configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml
configs/reward/kling_video_reward.yaml
configs/profile/torch_profiler.yaml
vrl/rollouts/collector/core.py
vrl/rollouts/orchestration/prompt_collection.py
vrl/generation/ray/runtime.py
vrl/ray/runtime.py
vrl/ray/resources.py
vrl/rewards/ray/runtime.py
vrl/utils/profiling.py
```

Run artifacts:

```text
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/metrics.csv
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/resolved_config.yaml
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/reward_debug/kling_video_reward_requests.jsonl
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/reward_debug/kling_video_reward_results.jsonl
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/reward_artifacts/manifest.jsonl
```

