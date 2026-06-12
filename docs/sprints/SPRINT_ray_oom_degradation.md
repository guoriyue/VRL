# SPRINT: Ray chunk OOM degradation (split-on-OOM)

状态：implemented（2026-06-11，确定性测试 5/5 绿；真 GPU 验收 gate 未跑，见 §5）。

## 0. Core Decision

chunk OOM 不再整跑报废：driver 把 OOM 的 chunk 用已有的 `SampleChunk.split()` 对半裂开、
重新提交给**同一个** worker，递归直到跑通或裂到单样本仍 OOM（硬错）。它不是性能优化——
成功路径零变化；它把"试探显存边界"从赌一次整跑变成免费动作（Wave 2 两次 live OOM
证伪了"自动 split 存在"的安全网假设，所有容量实验都站在那个假设上）。

## 1. 设计决策（实现前想清楚的三件事）

### 1.1 降级循环放 driver 侧 executor，不放 actor_pool

推荐稿里说 actor_pool 的派发循环是"天然位置"，实现时否决：`vrl/ray` 是通用 Ray 基建层，
不知道 chunk 语义；让它理解 `SampleChunk.split()` 会把 generation 语义漏进基建层。
降级循环放 `RayGenerationExecutor._degrade_oom_chunks`（`vrl/generation/ray/executor.py`），
按"轮"运行：收齐一轮结果 → 裂开 OOM 项 → 作为新一轮 `run_actor_jobs` 提交。
代价是轮与轮之间有 barrier（裂出的子 chunk 不与本轮仍在跑的 job 重叠）；OOM 是罕见事件，
这个简化值得。若未来 OOM 高频到 barrier 可见，再考虑往 pull 派发循环里做流式重入。

### 1.2 gather 重组问题自然消解（本 sprint 最关键的事实）

推荐稿预判的难点——"一个 job 位置上回来两个子结果"——不存在：

```text
vrl/generation/diffusion/layout.py:163 ordered_chunks()
  按 (prompt_index, sample_start) 元数据排序，并对照 sample_rows 精确校验样本覆盖
```

gather 从来不按 job 位置重组。子 chunk 自带正确的 (prompt_index, sample_start,
sample_count)，直接混进结果列表即可。gather 端**零改动**；executor 原来的
`len(results) != len(assignments)` 整数校验保留为首轮提交完整性检查，最终覆盖完整性
由 `ordered_chunks` 的逐样本校验承担。

### 1.3 子 chunk 绑回 OOM 的那个 worker

`max_inflight_per_actor=1` 下两个子 chunk 在该 worker 上顺序执行——刚证明放不下
N 个样本的 GPU 不会同时收到两个 N/2。代价：该 worker 慢一点；不代价：其它 worker 的
内存安全。pull 派发的 job 在结果里也带具体 `worker_id`，统一用它。

## 2. 行为

```text
OOM 判定:  error 字符串含 "out of memory"（大小写不敏感）—— worker 把异常压成 str
           （execution/worker.py:168 error=str(exc)），CUDA/HIP/torch.OutOfMemoryError
           的消息都含这个短语；其它错误照旧立刻 raise
裂开:      SampleChunk.split()（execution/chunks.py:67，对半）；envelope 用
           dataclasses.replace 换 chunk，request/plan_id/stage 原样继承
终止:      sample_count == 1 仍 OOM → RuntimeError（带 worker/chunk/原始错误）
遥测:      output.extra["ray_chunk_oom_splits"] = [{chunk_key, worker_id,
           sample_count, children}]，按裂开顺序；无 OOM 时键不存在（零开销）
policy_version 校验: 对每个最终结果照常执行（子 chunk 也查）
```

## 3. 改动文件

```text
vrl/generation/ray/executor.py       _degrade_oom_chunks + _is_oom_error；
                                     原 error-raise 块移入循环
tests/generation/ray/test_oom_split.py  5 个确定性用例（fake 容量 worker 注入 OOM）
```

## 4. 确定性验收（已过）

```text
8 样本 chunk 在容量 2 的 worker 上 → 裂 3 次成 4×2，样本覆盖精确，遥测记录 3 行
单样本 OOM → RuntimeError，不死循环
非 OOM 错误 → 不重试，执行恰好 1 次
健康路径 → 结果与遥测和改动前逐位一致（无 oom_splits 键）
分类器 → CUDA/HIP 命中，ValueError 不命中
```

## 5. 真 GPU 验收 gate（待跑，需要 wm-infra 的 5090）

```text
配置:  predict2 sbs=8（Wave 2 已知必 OOM 的 live gate 配置）
通过:  run 跑完不崩；training_debug/extra 里出现 ray_chunk_oom_splits；
       reward 曲线与 sbs=4 基线同量级（split 只改 batch 边界不改数值）
```

## 6. Non-goals

```text
不做 chunk 合并回升（降级后不在同一 run 内试探回大 chunk）
不把 split 语义下沉进 vrl/ray（分层，见 §1.1）
不做跨 worker 重分配裂出的子 chunk（同 worker 顺序跑是内存安全的有意选择）
不改 gather / layout / worker —— 它们不需要知道这件事
```

## 7. 顺手同步

`SPRINT_cross_model_performance.md` Wave 3 #1（predict2 VAE tiling 接线）实际已于
2026-06-10 随 cleanup 落地（`vae_decode_memory.py` 镜像接线 + 配置），本次标记 done。

## 7. 真 GPU gate 结果（2026-06-11 22:31，predict2 sbs=8 @512p93f num_steps=2）

**降级机制本身：行为上通过。** 已知必 OOM 配置只发生 1 次
`torch.OutOfMemoryError`（1.5GiB @ transformer），随后 driver 侧静默 split，
generation 全部完成、Kling 打分完成——OOM 不再杀 run。

**但 gate 整体 fail（exit=1），死因是一个无关的新回归**：trainer 重放阶段
`restore_eval_state` KeyError `'init_latents'`
（`vrl/models/diffusion/cosmos/predict2/model.py:470`）。复现：上面的 gate
命令（任何 predict2 GRPO 真训练步都会踩；n=1 的 probe 因零优势跳过训练步而
幸免）。范围：G1 parity run（6/10 11:48）同路径正常 ⇒ 回归窗口为其后的
提交（wire-diet fbd9234 / stage pipeline b224383 / pull dispatch 0e8ec69）
或工作区进行中改动。线索：`init_latents` 经 forward `extra` dict
（model.py:378）进入重放段，restore 侧读 `replay_tensors["init_latents"]`
——查 `extra` 持久化到 replay tensors 的链路在重构中是否丢键。

telemetry 验收（training_debug 里的 `ray_chunk_oom_splits`）因 run 在写
debug 文件前死亡而未确认，待 init_latents 修复后复跑确认。
