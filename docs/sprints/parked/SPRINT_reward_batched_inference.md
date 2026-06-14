# SPRINT: Batched Reward Inference (parked)

状态：parked / future（2026-06-09 记录）。当前**不做**——见 Trigger 一节的启动条件。

## 0. Core Decision

reward worker 内部目前是逐个 artifact、batch=1 的串行前向。框架已经留好了批量
钩子（`score_request`），但按实测数字，现在填这个钩子每个 epoch 只能省约 5
秒（0.6%），不值得动 Kling 模型内部。本 sprint 把"将来什么时候做、做什么、
怎么验证"一次性写清，避免后人重新调研。

## 1. Current Reality（2026-06-09 实测）

调度层（P1.4，已落地）把每个 epoch 的打分合并成一个请求：6 组 × 12 = 72 个
artifact 装进一个 `RewardInferenceRequest`，一次 actor 生命周期。但 worker
内部的执行是逐个循环（`vrl/rewards/inference.py:308-314`）：

```python
for artifact in request.artifacts:        # 72 个逐个来
    raw_scores = model(artifact=artifact, request=request)   # batch=1 前向
```

分发路径上可选的批量钩子已存在（`vrl/rewards/inference.py:292-306`）：模型
实现 `score_request(request) -> list[Mapping]`（与 `request.artifacts` 对齐）
即自动走批量路径，逐 artifact `__call__` 是 fallback。

Kling 模型（`vrl/rewards/models/kling_video_reward.py:251`）只实现了单
artifact `__call__`；`_prepare_batch` 实际支持多视频输入（processor 接受
列表），所以批量化主要是组织工作不是模型手术。

实测成本（RTX 5090，240p×33f，KlingTeam/VideoReward）：

```text
inference_ms ≈ 93ms/视频（batch=1）
72 视频/epoch ≈ 6.7s，占 13.4 min epoch 的 0.8%
批量前向估计可压到 ~2s → 每 epoch 省 ~5s（0.6%）
```

多 worker 并行已经存在且与本 sprint 无关：`shard_reward_request` 按 worker
数切片（`vrl/rewards/ray/runtime.py:53-64`），单 worker = 单分片串行。

## 2. Trigger — 什么时候启动本 sprint

满足任一条即启动，否则保持 parked：

```text
1. phase_times 中 collector.reward_score 占 epoch wall-clock > 5%
   （来源：trainer phase_times 或 reward_debug requests.jsonl 的
   inference_total_ms / epoch 时长）
2. 多 reward 组件叠加后打分总时长 > 30s/epoch
   （例如 kling + videocon_physics 双组件、或更高 n/rollout_batch_size）
3. reward 拿到独立常驻 GPU（P1）且打分与 rollout/训练重叠执行——
   那时打分延迟直接决定流水线气泡大小
```

## 3. Change Shape

1. `KlingVideoRewardModel.score_request(request)`：按
   `worker_config.inference_batch_size`（默认 8）把 artifacts 切成 mini-batch，
   每批一次 `_prepare_batch(多视频) → model(...)` 前向，返回与
   `request.artifacts` 对齐的 score map 列表。
2. config：`reward.kwargs.<name>.worker_config.inference_batch_size`，
   缺省 1 = 维持现状（逐个），>1 才走批量。
3. 计时语义：`score_artifacts_with_model` 的批量路径已把
   `per_artifact_ms` 摊平（`inference.py:303`），reward_debug 行为不变。
4. VideoCon（mPLUG-Owl 7B）同样的钩子形状，但显存更紧，batch 上限单独测。

Gates：

```text
同一组 mp4 上 batch=1 vs batch=N 的 selected_score 逐 artifact diff < 1e-3
  （注意 VLM 的 padding/attention 差异可能引入数值漂移，超阈值要查 padding）
reward GPU 显存峰值在 batch=N 下不 OOM（240p 与 512p 各测一次）
72-artifact 请求 wall-clock 下降 ≥ 2×
```

## 4. Non-Goals

- 不改请求/传输层：`RewardInferenceRequest`、分片、`validate_reward_results`
  原样。
- 不改 artifact 落盘路径（mp4 materialize）。
- 不删逐 artifact fallback——它是没有实现钩子的 reward 的默认路径，也是
  数值对照的基准。
- 不在单卡共享 GPU 的当前配置下启动本 sprint（0.8% 不值得）。

## 5. References

```text
vrl/rewards/inference.py:292-314    批量钩子 + 逐 artifact fallback + 计时
vrl/rewards/models/kling_video_reward.py:251,276   __call__ / _prepare_batch
vrl/rewards/ray/runtime.py:53-64    shard_reward_request 多 worker 切片
docs/sprints/info/SPRINT_cosmos_performance.md   P1.4（调度层合并，已落地）
outputs/cosmos25_nft_240p_bs6_reward_retry_20260609_202313/reward_debug/
                                    inference_ms 实测来源
```
