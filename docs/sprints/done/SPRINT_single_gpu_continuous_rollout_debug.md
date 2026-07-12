# SPRINT: 单卡 continuous rollout debug harness

状态：已完成 — 单卡 continuous rollout debug harness 全部落地于 061cfb2（main）：新增 config online_grpo_ocr_single_gpu_async_debug.yaml + persistent_colocated_workers flag（vrl/generation/ray/config.py、vrl/ray/resources.py 完整校验），5 项验证目标均有通过的测试覆盖。

> **Superseded（2026-07-11）**：本文件保留当时的 debug 结果作为历史记录，但它定义的
> same-GPU resident continuous 产品路径已经由
> `docs/sprints/planned/SPRINT_miles_phase_lease_and_one_continuous.md` 删除。当前 shared
> trainer/rollout GPU 只允许 `strict_on_policy` phase lease；production `continuous` 必须使用
> disjoint GPUs。本文中的 preset、`persistent_colocated_workers`、role `memory_fraction` 和
> `require_separate_gpus` 均已删除，下面的命令与配置不再是可运行接口。

## 结论

要做一个能在 **1 张 GPU** 上验证 rollout/training 异步调度的例子，不能复用
`online_grpo_ocr_crossnode_debug.yaml`。那个配置验证的是 cross-node Ray 路径：
trainer 和 rollout GPU 在不同节点；它不是单卡 colocated 场景。

本 sprint 的正确目标是一个很小的 orchestration harness：

- trainer replay model 和 Ray rollout worker 都驻留在同一张 GPU；
- continuous producer/ready queue/consumer/weight-sync barrier 真实启用；
- workload 足够小，避免把这个 harness 变成大模型内存能力测试；
- 默认安全策略不变，大模型单卡仍走 release-after-collect。

## 已落地的最小形态

新增实验：

`configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml`

关键设置：

```yaml
defaults:
  - /recipe/online/flow_matching_grpo
  - /base/rollout/orchestration/continuous

sampling:
  height: 128
  width: 128
  num_steps: 4

distributed:
  rollout:
    release_after_collect: false
    persistent_colocated_workers: true

trainer:
  rollout_orchestration:
    require_separate_gpus: false
    continuous:
      max_inflight_groups: 2
      max_ready_groups: 4
      max_stale_policy_versions: 1

rollout:
  rollout_batch_size: 2
  n_samples_per_prompt: 2
  sample_batch_size: 1
```

运行方式：

```bash
CUDA_VISIBLE_DEVICES=0 vrl-train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug
```

## 为什么需要新 flag

现有单卡 colocated preset 默认使用 `RayGenerationRuntime` 的 release-after-collect
模式：rollout collect 后杀掉 Ray worker，把 GPU 交还给 trainer replay/backward。
这对大模型是正确的默认安全策略。

但 continuous rollout 要验证的是 background producer 在 trainer 训练时继续持有 rollout
worker。如果每次 collect 后都释放 worker，单卡上就看不到真实 persistent async 形态。

所以新增了显式 debug flag：

```yaml
distributed.rollout.persistent_colocated_workers: true
```

这个 flag 很窄：

- 只有 trainer/rollout GPU overlap 时允许；
- 需要 `release_after_collect=false`；
- 不改变默认 colocated 行为；
- 不允许同时把 Ray reward worker 也放进同一个 rollout GPU 池。

## 验证目标

1. 配置能 load，并解析成 `mode=continuous`。
2. 资源层能解析成 single-GPU colocated persistent rollout。
3. Ray config 允许这个显式 debug 模式，但默认仍拒绝
   `overlap + release_after_collect=false`。
4. continuous schedule 默认仍拒绝 colocated runtime；只有
   `require_separate_gpus=false` 时才允许。
5. 运行日志能显示：
   - `rollout_persistent_colocated_workers=True`
   - `continuous.producer_*`
   - `continuous.queue_*`
   - `continuous.stale_policy_versions`

## 非目标

- 不承诺单卡 persistent async 对大模型安全。
- 不把普通 single-GPU debug 从 release-after-collect 改成 persistent。
- 不做 memory auto-tune。
- 不把这个当 throughput benchmark；它只验证 async orchestration 真的能在一张卡上跑通。

## 后续判断

如果这个 harness 能跑通，再看指标决定是否值得继续：

- `continuous.queue_ready_groups` 是否经常大于 0；
- `continuous.producer_inflight` 是否能在训练时维持非零；
- `continuous.stale_policy_versions` 是否稳定在 0/1；
- replay ratio / approx KL 是否没有因为 stale rollout 明显变坏。
