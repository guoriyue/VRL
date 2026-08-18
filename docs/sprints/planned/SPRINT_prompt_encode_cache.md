# SPRINT：prompt embedding 缓存 —— 冻结 text encoder 不该每个 chunk 重算

状态：**planned（2026-08-17）**。基线 main @ `abb8e4da`。

## 0. 结论先行

text encoder 在本仓是**证明冻结**的（`requires_grad_(False)`，不在
`trainable_modules`，不在 `policy_cores`），也就是说**同一段 prompt 在整个训练
run 里的 embedding 是常量**。但 `_forward_chunk` 每个 chunk 都重新
`encode_prompt` 一次。

GRPO 每个 prompt 采 G 条轨迹；当 `sample_batch_size=1`（wan 的实测配置，
`SPRINT_cross_model_performance.md` 记的 25% 利用率就是这个），G 条轨迹 = G 个
chunk = **同一个 prompt 被 T5/umT5 编码 G 次**。跨 epoch 复现同一批 prompt 时
还要再乘一遍。

这是纯浪费，且**没有数值风险**（常量函数的缓存），也**不引入 replay drift**
（embedding 逐位相同）。这是 diffusion RL 版的 prefix caching。

## 1. 证据

### 1.1 text encoder 确实冻结

```python
pipeline.text_encoder.requires_grad_(False)
```
`vrl/models/families/wan_2_1/model.py:196`

训练面只有 transformer：

```python
def trainable_modules(self) -> dict[str, Any]:
    modules = self._wan_transformers()
    return {name: modules[name] for name in self._trainable_transformer_names}
```
`vrl/models/families/wan_2_1/model.py:357-359`

优化面同样只有 transformer（`vrl/models/steps/denoise/base.py:433-443`
的 `policy_cores` 返回 `{"transformer": ...}`）。权重同步推的也只是
trainable state（`vrl/trainers/weight_sync.py:112-127`）。

**结论：没有任何路径会改动 text encoder 的权重。** 缓存的失效条件只剩
「模型被换掉」，而那等于换 run。

### 1.2 每个 chunk 重算一次

```python
started = time.perf_counter()
with profile_range("generation.prompt_encode"):
    encoded = self.encode_prompt_for_batch(
        generation_request=request,
        video_request=video_request,
        params=params,
        batch=batch,
    )
stage_durations["encode"] = time.perf_counter() - started
```
`vrl/generation/bindings/full_sequence_denoise/executor.py:272-280`

`_forward_chunk` 是 per-chunk 入口，所以 encode 是 per-chunk 成本。

### 1.3 成本已经在生产里被测量了

上面那段已经把耗时写进 `stage_durations["encode"]`，并且有
`profile_range("generation.prompt_encode")`。**baseline 不需要新建测量设施** ——
直接从现有 telemetry 取数即可，这也是本 sprint 风险低的原因之一。

### 1.4 为什么这笔钱不小

video 家族的 text encoder 不是小模型：wan 用 umT5-XXL、cosmos 系用 T5 量级。
`SPRINT_cross_model_performance.md` §0 把「12 个单样本 chunk 串行的**重复固定
成本**」列为 cosmos 生成段 25s/组里的大头，prompt encode 正是这些固定成本中
可以被完全消除的一项。

## 2. 范围

**P1 — 进程内缓存。** 在 rollout worker 进程里挂一个 prompt→encoded 的缓存。

- key：`(prompt_text, negative_prompt_text, 影响编码的 sampling 字段)` 的稳定
  哈希。**key 必须由编码的真实输入构成**，不能只用 prompt 字符串 —— 若某家族
  的 `encode_prompt` 还吃 `max_sequence_length` / dtype / task 变体，这些都得进
  key。落地前先逐家族核对 `encode_prompt` 的签名（18 个家族都实现了它）。
- value：编码结果的 tensor（GPU 常驻还是 CPU 常驻见 P2）。
- 容量：LRU，条目数上限走配置；默认值由 §4 的实测显存占用决定，不拍脑袋。

**P2 — 驻留策略。** embedding 不小（序列长 × hidden），G 条轨迹共享一份时
GPU 常驻最划算；但 prompt 集合大时会吃显存。两种都实现，默认 GPU 常驻 +
容量上限，超限退化到重算（不退化到 CPU，避免引入 H2D 抖动）。

**P3 — 与 parking 的关系（必须查）。** rollout worker 有 sleep/wake parking，
且 parking 强制 `empty_cache()` 并对残留 >256MiB 硬报错
（`vrl/utils/cuda_memory.py:25-27, 300-305`，见
`SPRINT_gemm_utilization.md` 对 CUDA graph 的同一处分析）。**GPU 常驻的
embedding 缓存会计入这个残留预算**，必须在 `sleep()` 里显式清空或搬走。
这条不做，parking 会直接报错——这是本 sprint 唯一的真实集成风险。

## 3. 验收标准

- **逐位等价**：同一 prompt 走缓存与不走缓存，encoded tensor
  `torch.equal` 为真（不是 allclose —— 常量缓存就该逐位相同）。
- **命中率**：单个 prompt group（G 条轨迹）内命中率 = (G-1)/G；跨 step 复现
  prompt 时命中。用一次真实 wan 或 cosmos rollout 取数。
- **端到端**：`stage_durations["encode"]` 的每组总和下降 ≥ (G-1)/G；
  组 wall time 下降幅度进 §5。
- **parking 存活**：跑 sleep/wake 循环，确认 P3 的清理生效、不触发
  `cuda_memory.py` 的残留硬报错。
- **不引入 drift**：rollout-vs-replay logprob parity 与开缓存前一致
  （embedding 逐位相同 → parity 必须完全不动；若动了说明 key 漏了字段）。
- 既有测试全绿；`make verify` 绿。

## 4. 非目标

- 不缓存 latent / VAE 输出（不是常量函数，随采样噪声变）。
- 不做跨进程 / 跨 run 的持久化缓存（磁盘 embedding 库）—— 先证明进程内够用。
- 不动 `encode_prompt` 的家族实现签名。
- 不为 text encoder 做量化或 offload（另一条正交的线）。
- 不改 chunk 切分策略本身（那是
  `planned/SPRINT_rollout_finalize_overlap_ga.md` 的范围）。

## 5. 执行记录

（待填：命中率、encode 耗时 before/after、组 wall time before/after、
缓存显存占用、逐家族 key 字段核对表）

## 6. 相关

- 固定成本来源：`docs/sprints/info/SPRINT_cross_model_performance.md` §0
- parking 残留预算的先例分析：`docs/sprints/done/SPRINT_gemm_utilization.md`
  「为什么即使收益为正也不能开」一节
- 现有 telemetry：`vrl/generation/bindings/full_sequence_denoise/executor.py:273`
