# SPRINT：UnifiedReward adaptive batching

状态：**parked（2026-07-21）**。触发条件：

1. [Versioned lookahead](SPRINT_continuous_versioned_lookahead.md) 已完成；并且
2. stage telemetry 证明 reward inference 或 reward queue wait 是可见流水线瓶颈。

父 program：[Continuous three-stage pipeline](../planned/SPRINT_continuous_three_stage_pipeline_program.md)

## 0. 结论先行

分两层做 batching，不能混为一个旋钮：

```text
request batching:
  reward pump 把多个 unscored group 合为一次 score_rollouts()/HTTP request

model micro-batching:
  UnifiedRewardVideoModel.score_request() 把多个 artifact 组织为真正的 batched forward
```

第一层减少 HTTP、validation 和调度开销，也让 service 看见足够多 artifact；第二层才可能提高
GPU 3 的 inference throughput。两层都必须保持 artifact/result 一一对应和 batch=1 数值 oracle。

这份 sprint 专用于当前 UnifiedReward video path。旧的
[Batched reward inference](SPRINT_reward_batched_inference.md) 保留 Kling/VideoCon 的历史 trigger，
不作为 UnifiedReward 的实现 source of truth。

## 1. Root cause / current behavior

reward service 已接收 `RewardInferenceRequest`，并通过 semaphore 限制并发；一个 request 调用
`score_batch(request)`。generic inference layer 也会优先调用模型的：

```python
score_request(request) -> list[Mapping[str, float]]
```

但 `UnifiedRewardVideoModel` 目前只有逐 artifact `__call__()`。每个 artifact 都独立完成 frame
decode、processor、`model.generate()` 和 parser；服务 `max_concurrency=1` 时，多个 group 会串行。

简单提高 service concurrency 不是解法：同一个 7B 模型在一张 L4 上并发多个 generate 可能
增加显存峰值、上下文切换和 OOM，却不保证更高吞吐。

## 2. Goal and ownership boundary

### Reward pump policy

reward pump 拥有：

```text
max_reward_batch_groups
max_reward_batch_artifacts
max_reward_batch_bytes
max_reward_batch_wait_ms
```

它只在已有 unscored item 中组 batch，不延迟 current trainer-critical group 超过 bounded wait，
也不把不同 reward schema/model/version 的请求合并。

### Model policy

`UnifiedRewardVideoModel.score_request()` 拥有 model micro-batch：

```text
inference_microbatch_size
frame decode / processor collation
batched model.generate
per-row prompt-length slicing and decode
per-row score parsing
```

service/collector 不读取模型私有 tokenizer 或 padding 逻辑。

## 3. Correctness and resource invariants

1. request artifact order和返回 score-map order 完全一致。
2. 每个 artifact 使用自己的 prompt/rubric framing，不能复用第一行 prompt。
3. batched decode 必须按每行 attention/input length 去掉 prompt tokens，不能使用一个全局
   `prompt_len` 错切所有行。
4. parser 对缺 axis、非法 range、越界值继续 fail closed；不能因 batch 中一行失败而给默认分。
5. 一个 row 失败时整 request 的原子性与 service contract 保持一致，除非先显式设计并测试
   typed per-row error protocol；本 sprint 不偷偷改变协议。
6. microbatch OOM 可以缩小 batch 后重试，但不能改变 artifact identity/order；batch=1 OOM 为
   terminal capacity failure。
7. batch wait 受 current-batch priority 限制；不能为填满 GPU 3 让 trainer 长时间等一个未满 batch。
8. model/version/rubric path 必须同一 request 一致。

## 4. Implementation stages

### T0 — Profiling KILL-gate

对同一批 robotics artifact 分解：

```text
artifact validation
frame decode
processor
host-to-device
model prefill/decode
parse
service queue wait
```

如果 model inference 不是主成本，或 microbatch=2 已无吞吐收益，保留 request batching 并关闭
model batching。Negative result 写入 `docs/sprints/info/`，不为了 sprint 完整性强推。

### T1 — Reward pump request batching

- 按 current-batch priority 从 unscored queue 取 item。
- 在 group/artifact/byte/wait 四个 cap 中最先达到者触发提交。
- `collector.score_rollouts(list)` 仍是唯一 batch build owner。
- output 通过 input identity/order 映射，不依赖 completion order。

### T2 — UnifiedReward `score_request`

- frame sampling 保持现有算法。
- 构造 per-artifact messages/text/images，processor 一次处理 microbatch。
- 明确 padding side、attention mask 和每行 prompt boundary。
- batched deterministic `generate()` 后逐行 decode/parse。
- microbatch=1 直接复用同一代码路径，避免维护两套 inference semantics；原 `__call__` 可作为
  protocol fallback facade，不复制实现。

### T3 — Safe microbatch fallback

- 从配置上限开始，typed CUDA OOM 才触发二分缩小。
- 清理 partial CUDA state 后重试同一 ordered slice。
- 记录实际 batch size 和 OOM fallback；非 OOM exception 不吞掉。

### T4 — Service integration

- service `max_concurrency=1` 保持默认；吞吐来自内部 batching。
- `/info` 暴露实际 model batch capability/schema version，而不是仅显示配置请求值。
- client timeout、request cache/fingerprint、cancellation 和 artifact validation 保持现有协议。

## 5. Failure, cancellation and recovery semantics

- reward pump 尚未提交的 item 仍由 unscored queue 拥有。
- request 提交后由 reward task 拥有整批 artifact lease，取消要等 service/client cleanup settled。
- service retry 必须使用相同 request payload/fingerprint；不同 payload 复用 request id 继续 409。
- microbatch OOM fallback 是同一 request 内部动作，不产生第二个 logical reward attempt。
- parse failure 记录 artifact identity 和短错误摘要，不记录完整 prompt/model output。

## 6. Telemetry

```text
reward.request_batch_groups
reward.request_batch_artifacts
reward.request_batch_bytes
reward.request_batch_wait_ms
reward.model_microbatch_requested
reward.model_microbatch_actual
reward.model_microbatch_oom_fallbacks
reward.frame_decode_ms
reward.processor_ms
reward.prefill_decode_ms
reward.parse_ms
reward.artifacts_per_second
```

## 7. Verification and acceptance gates

- 固定 artifact 集上 batch=1 与 batch=N 的 axis map、selected score、order 完全一致；如果底层
  deterministic generate 存在可解释数值差异，selected score 仍必须一致，否则 model batching
  不转默认。
- 不同 prompt length、不同 video frame count、padding side、单 row parse failure 有测试。
- OOM split 后结果数量/order 与 batch=1 一致，非 OOM error 原样传播。
- request batching 与 per-group control 的 `RolloutBatch.rewards`、components 和 group ids 一致。
- 真实 L4 上记录 batch size 1/2/4/... 的 latency、throughput 和 peak memory；选择不 OOM 且吞吐
  最优的 bounded profile。
- 只有当 reward critical-path wall 或 artifacts/s 有实质改善才进入下一 sprint。

## 8. What changes / what stays

### 改变

- reward pump 可合并多个 unscored group。
- UnifiedReward 获得实际消费的 microbatch config 和 `score_request()`。
- capacity/OOM telemetry 进入 reward timing。

### 保持

- service HTTP schema、request validation、fingerprint 和 result validation。
- selected reward axis 与 robotics rubric。
- one dedicated reward GPU、service concurrency default 1。
- batch=1 oracle/fallback。

## 9. Non-goals

- 不同时改 Kling/VideoCon。
- 不用多个模型副本争抢同一张 L4。
- 不改变 reward rubric 或训练目标。
- 不让 reward batcher读取 eval 数据。
- 不把 eval 结果混入 reward components。

## 10. Definition of Done

- [ ] request batching 保序且通过 collector parity。
- [ ] UnifiedReward `score_request()` 通过 batch=1/N parity。
- [ ] OOM fallback 和 cancellation 有确定性测试。
- [ ] 真实 L4 capacity/throughput profile 已归档。
- [ ] 默认 batch 来自测量，不是猜测。
- [ ] 没有收益时已执行 negative exit，不留下无效默认。

## 11. References

- `vrl/rewards/models/unified_reward_video.py`
- `vrl/rewards/inference.py`
- `vrl/rewards/service/server.py`
- `vrl/rewards/service/client.py`
- `vrl/rollouts/collector/core.py`
- `vrl/config/reward_service/unified_reward_robotics.yaml`
- `tests/rewards/unified_reward_video/test_unified_reward_video.py`
- `tests/rewards/service/test_service.py`
- [Reward service](../done/SPRINT_reward_service.md)
- [Historical batched reward inference](SPRINT_reward_batched_inference.md)
