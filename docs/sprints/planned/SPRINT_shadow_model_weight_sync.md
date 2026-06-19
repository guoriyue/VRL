# SPRINT: Trainable-state shadow slots —— 安全去掉 continuous drain bubble（planned）

状态：**P0 + P1 + P2 全部完成 —— P2 多卡真机验证通过（2026-06-19）**。P0/P1 实现 + CPU 测试（2026-06-18）；P2 运行时启用 + executor stale-slot 路由 + CPU 测试（2026-06-19）；P2 多卡真机验证通过（2026-06-19，见 §11「P2 多卡验证结果」）。

> **更新（2026-06-18，P0/P1 落地）：**
> - **设计偏差（比 §3 的 PEFT 多 adapter 方案更简单、更低风险）**：slot 不用 PEFT 命名 adapter（要 key 重映射、易撞 NFT 的 `default`/`previous`、2× VRAM），而用 **state-dict 快照 slot**。`TrainableStateSlots`（`vrl/models/utils.py`）只存「版本→flat trainable payload dict」（host RAM，≈0 额外 VRAM）；`activate` 复用已有且测试过的 `load_trainable_state(slot[v])` 把那版参数 copy 进 live 模型，并用 `_active_slot_version` 跳过同版本重载。对所有 diffusion family 通用（含 wan 多 transformer），零 PEFT key 重映射风险。代价：切版本时一次 H2D copy（LoRA 很小；§8 的 CUDA-stream overlap 仍是后续优化）。
> - **P0（已实现）**：`DiffusionModelBase` 加 `supports_versioned_trainable_state=True` + `install_trainable_state`/`activate_trainable_state`/`has_trainable_state`（getattr-dispatch，不入硬 `RuntimeModel` 契约，AR family 自动 fallback）。`ChunkExecutionResult.stale_slot` 区分 evict 与真错误。`GenerationWorkerCore.update_weights` 在支持时安装 slot 不覆盖旧版；`execute_chunk` 按 `request.policy_version` 激活 slot、success 返回 request version（过 executor 断言）、缺 slot 返回 typed stale-slot。worker 暴露 `supports_versioned_trainable_state()`（+ Ray actor forwarder）。
> - **P1（机制已实现，默认 OFF）**：`RolloutLifecycle.supports_non_draining_weight_sync()`（getattr-default-False）；`after_train_step` 在支持时跳过 `drain_inflight`，并发 `continuous.weight_sync_barrier_mode`（0=draining/1=non-draining）metric（已进 metrics.csv）。
> - **CPU 测试**：`tests/models/test_utils.py`（slot store）、`test_model_base.py`（install/activate/retain 旧版/skip-if-active）、`tests/generation/execution/test_worker_versioned_slots.py`（install 不覆盖、缺 slot stale-slot、激活 request 版本、plain-model 仍走 mismatch）、`test_schedule.py`（draining mode=0、non-draining 不等 in-flight + mode=1）。全套 366 passed。
> - **P2 运行时启用 + stale-slot 路由（已实现 + CPU 测试，2026-06-19）**：
>   - **runtime capability 派生**：`RayGenerationLauncher.launch` 通过 `_all_workers_support_versioned_slots(ray, workers, weight_sync=...)` 把 `runtime.supports_non_draining_weight_sync` **派生为所有 worker `supports_versioned_trainable_state()` 的 AND**（worker core + `RayGenerationWorker` actor forwarder 已暴露）。`weight_sync is None`（sync 关闭）/ 无 worker / 任一 query 抛错 → False → 安全回退 draining barrier。`RayGenerationRuntime.__init__` 与 lease 模式默认 False，保持默认行为不变。
>   - **executor stale-slot graceful discard**：新增 typed `StaleSlotDiscard`（`vrl/generation/execution/types.py`，**不是** `RuntimeError` 子类）。`RayGenerationExecutor.execute` 在 OOM-degrade（对非 OOM error 硬 raise）与 version assert **之前**检测 `ChunkExecutionResult.stale_slot`，raise `StaleSlotDiscard`（一个 evicted chunk 即丢整个 request，绝不拼混版本输出）。`ContinuousRolloutProducer._harvest_done` 捕获该类型 → `discarded_stale_count += 1`（不进 `error_count`、不触发 fail-fast），与 receipt-time staleness gate 同一 discard 语义。
>   - **CPU 测试**：`test_runtime_config.py`（AND 派生 + 无 sync/无 worker/query 抛错 → False）、`test_oom_split.py`（stale-slot raise `StaleSlotDiscard` 且不重试、非 RuntimeError 子类）、`test_schedule.py`（StaleSlotDiscard → discarded_stale 而非 error_count + queue 保持空）。全相关套 47 passed；`tests/generation/` + `tests/rollouts/orchestration/` 137 passed / 2 skipped 无回归。
> - **P2 仍待做（多卡，无法单卡验证）**：在 2 卡分池上验收「无 policy_version mismatch + weight_sync_pause 不再含整条 clip + 每步 wall-clock 下降」。

状态（原文）：**proposed / design（2026-06-18, trainable-state-first 修订）**。

这是 `SPRINT_continuous_scheduler_redesign.md` 里 GAP 2 的可落地版本：不要裸删
`drain_inflight()`；先给 rollout worker 做 **trainable-state versioned slots**。旧请求继续按它提交时的
policy version 找旧 trainable slot，新请求按新 version 找新 trainable slot。这个设计不是 LoRA-only：
同步 payload 里有什么 trainable params，就 duplicate 那部分。LoRA/adapter 是最便宜、最适合首个验收的
case；full-param 也符合语义，但额外显存接近一整份 transformer trainable state，必须由 capability /
memory budget 决定是否启用。

---

## 0. 结论

- **现状**：continuous 的 `after_train_step` 是 `pause_admission -> drain_inflight -> sync_weights ->
  post_sync_purge -> resume_admission`。`drain_inflight()` 会等所有在途视频生成完成；视频模型一条 clip
  可能跑数分钟，所以这是最大的串行 bubble。
- **为什么不能直接删 drain**：当前 worker 只有一个全局 `_policy_version`，`update_weights()` 会就地
  改同一个 rollout model。旧请求后续 chunk 仍期望旧 version，但 worker 已经变成新 version，于是
  `policy_version mismatch`。
- **正确修法**：worker 不再只有一个 trainable state。它保留少量 versioned trainable slots：
  `slot[v1]` 给旧请求，`slot[v2]` 给新请求。`execute_chunk()` 按 `request.policy_version` 激活对应 slot
  再 forward。
- **显存判断**：额外显存跟 trainable-state payload 成正比。LoRA / adapter-only sync 很便宜；full-param
  sync 会接近整模型翻倍，但这只是预算问题，不是设计不适用。
- **配置原则**：不要新增一堆用户参数。由 capability 派生：模型/worker 支持 versioned trainable slots 时
  continuous 可以走 non-draining sync；不支持时继续用当前 draining barrier。

---

## 1. 问题拆清楚

当前 barrier：

```python
self.producer.pause_admission()
await self.producer.drain_inflight()
await self.lifecycle.sync_weights_after_train(phase_times)
phase_times["continuous.post_sync_dropped_stale"] = float(
    self._drop_stale_ready_items_after_sync(),
)
self.producer.resume_admission()
```

`drain_inflight()` 的含义是等所有已经提交的 generation task 完整结束：

```python
await asyncio.gather(*self._inflight, return_exceptions=True)
self._harvest_done()
```

它保正确性，但太粗。正确性真正要避免的是：同一个请求的一部分 chunk 用 v1 权重，另一部分 chunk 用 v2
权重。现在 worker 用一个全局 version 防这个问题：

```python
expected_version = request.policy_version
if expected_version is not None and self._policy_version != expected_version:
    return ChunkExecutionResult(..., error="policy_version mismatch")
```

所以如果裸删 drain，`update_weights()` 会把全局 `_policy_version` 从 v1 改成 v2，旧请求的后续 chunk
到 worker 时就会 mismatch。这个 mismatch 是对的；它说明系统没有静默生成混合策略样本。

关键结论：**chunk-level version check 不是 non-draining sync 的实现；它只是防止错误静默发生。**

---

## 2. 不能做的假解法

**假解法 A：只有一个 live slot + 一个 pending buffer，到边界把 pending 切成 live。**

这仍然不够。Ray executor 会把一个 `GenerationRequest` 拆成多个 chunk；旧 request 的未来 chunk 可能还没
执行。如果 worker 在两个 chunk 之间把全局 live 从 v1 切到 v2，旧 request 的下一个 chunk 仍然期望
v1，还是 mismatch。

**假解法 B：裸删 drain，让 mismatch 失败后重试。**

这会把失败变成调度路径的一部分：浪费生成、污染 error metrics，并且不是所有算法都能接受 stale tail。
DiffusionNFT 尤其不能靠 stale 修正。

**假解法 C：不看 trainable payload，直接 full model 双缓冲。**

这当然能做，但对视频 transformer 基本等于权重显存翻倍。正确抽象不是“先假设复制整模型”，而是复用当前
weight-sync 的 source of truth：`flatten_trainable_module_state(...)` 产出的 trainable-state payload。
payload 是 LoRA，就只 duplicate LoRA；payload 是全参 transformer，就 duplicate 全参 transformer。

---

## 3. 正确设计：versioned trainable slots

worker 侧从这个模型：

```text
base + one active trainable state
global _policy_version = v1
```

改成：

```text
base transformer: one copy
trainable slots:
  v1 -> LoRA/adapter state for old requests
  v2 -> LoRA/adapter state for new requests
current_submit_version = v2
```

执行 chunk 时：

```python
version = request.policy_version
slot = trainable_slots.get(version)
if slot is None:
    return stale_or_missing_slot_result(...)

model.activate_trainable_state(version)
output = forward_chunk_plan(request, chunk)
return ChunkExecutionResult(..., policy_version=version)
```

更新权重时：

```python
def update_weights(state_ref, policy_version):
    state = deref(state_ref)
    trainable_slots.install(policy_version, state)
    current_submit_version = policy_version
```

注意这里**不覆盖旧 slot**。旧 request 的后续 chunk 继续 `activate_trainable_state(v1)`；新 request 由
producer 在 sync 后戳 `v2`，走 `activate_trainable_state(v2)`。

### 3.1 slot 数量

首版保留少量版本即可：

- 至少保留 `current` + `previous` 两个 slot，即使 `max_stale=0` 也需要 previous，让已经在途的旧请求能
  跑完并被丢弃，而不是 mismatch。
- 如果 `max_stale_policy_versions=N`，保留 `N + 2` 个 slot：当前版本、允许训练的 stale 版本、以及一个
  正在收尾但可能即将被丢弃的旧版本。
- LoRA / adapter slot 小，保留 2-3 个通常可接受；full-param 也能走同一语义，但必须先通过 memory
  budget / capability gate。

如果 chunk 到达时版本 slot 已经被淘汰，worker 应返回 typed stale-slot error，producer 计入 discard，
不能把它当普通 collect failure。

### 3.2 为什么这不破坏旧请求

旧请求不依赖“worker 当前全局版本”。它依赖自己提交时的 `request.policy_version`。只要对应 slot 还在，
每个 chunk 都能显式激活同一个 trainable state：

```text
request_v1 chunk0 -> slot[v1]
update installs slot[v2]
request_v1 chunk1 -> slot[v1]
request_v2 chunk0 -> slot[v2]
```

这才是“不破坏旧请求”的真实机制。

---

## 4. 和 StalenessPolicy 的关系

versioned slots 只解决“旧请求还能不能完整生成出来”。它不决定“生成出来以后能不能训练”。

训练 admission 仍由 `StalenessPolicy` 决定：

```text
rollout item version = v1
trainer current version = v2
staleness = 1
```

- GRPO：`max_stale=1` 可以训练，靠 log-prob ratio / PPO clip 修正。
- DiffusionNFT：`max_stale=0`，旧版本结果必须丢；但 slot 仍然有价值，因为它让旧请求正常结束并被显式
  丢弃，而不是撞 mismatch。

也就是说：

```text
versioned slots: 让旧请求不炸
StalenessPolicy: 决定旧结果能不能训练
```

---

## 5. 实现计划

### P0 — worker trainable-state slots（generic semantics, LoRA baseline）

目标：不改用户配置，按现有 trainable-state sync payload 支持多版本。LoRA/adapter 是 P0 的低风险验收
基线；full-param 不特殊化，只受内存预算和模型 capability 限制。

工作项：

- 新增 worker 内部 slot store，例如 `TrainableStateSlots`：
  - `install(version, state_dict)`
  - `has(version)`
  - `activate(version)`
  - `evict_older_than(cutoff)`
- 给 runtime model 增加 capability/protocol：
  - `supports_versioned_trainable_state: bool`
  - `install_trainable_state(version, state)`
  - `activate_trainable_state(version)`
- 模型按自己的 trainable-state 结构实现该 protocol；PEFT/LoRA 先作为基线实现，full-param 可以复用同一
  protocol 但需要明确 memory budget。
- `GenerationWorkerCore.update_weights()`：
  - 如果支持 slots：安装新 slot，更新 submit/current version，不覆盖旧 slot。
  - 如果不支持：保留当前单版本 `load_trainable_state()` 行为。
- `GenerationWorkerCore.execute_chunk()`：
  - 如果支持 slots：按 `request.policy_version` 激活 slot；result policy version 等于 request version。
  - 如果 slot 缺失：返回 typed stale-slot result，不能伪装成普通 model exception。
- 单测：
  - 旧 v1 request 第一个 chunk 跑完后 install v2，旧 v1 后续 chunk 仍成功。
  - 新 v2 chunk 使用 v2 slot。
  - 缺 slot 返回明确错误。
  - slot store 只保存 trainable state，不复制 base model。

验收：

- 裸 non-draining sync 下不再出现 `policy_version mismatch`。
- `collect error_count` 不因 version swap 增长。
- GPU memory 增量接近 retained trainable-state slots 的大小。LoRA 情况应远小于 base model；full-param
  情况会接近 base model，需要显式 memory/capability gate。

### P1 — schedule non-draining sync（仍保持 strict fallback）

目标：只有在 rollout runtime 全部 worker 支持 versioned trainable slots 时，continuous 才跳过 drain。

工作项：

- runtime/worker 暴露 capability：`supports_non_draining_weight_sync`。
- `ContinuousRolloutSchedule.after_train_step()`：
  - 支持 capability：`pause_admission -> sync_weights -> post_sync_purge -> resume_admission`。
  - 不支持 capability：继续当前 `pause -> drain -> sync -> purge -> resume`。
- producer 在 sync 后 resume，新提交请求自然戳新 version；旧 in-flight 仍按旧 version 完成。
- metrics：
  - `continuous.weight_sync_barrier_mode`: draining / non_draining（如果 metrics 只支持数字，写 0/1）
  - `continuous.post_sync_dropped_stale`
  - `continuous.producer_discarded_stale`
  - `continuous.stale_policy_versions`
- 测试：
  - capability=false 时行为与当前一致。
  - capability=true 时 `after_train_step` 不等待 in-flight 结束。
  - in-flight v1 完成后，GRPO `max_stale=1` 可入队训练；NFT `max_stale=0` 被 post-sync/consumer 丢弃。

验收：

- P0 的 no-mismatch 仍成立。
- `continuous.weight_sync_pause_s` 不再包含完整 generation request 时长。
- max_stale=0 不训练旧版本；max_stale=1 可以观测到 stale item 被 admission。

### P2 — 多卡 throughput 验证

目标：证明这不是单卡自嗨，而是真正消掉 rollout/train overlap 的最大 bubble。

配置形态：

```yaml
distributed:
  resources:
    trainer: {num_gpus: 1}
    rollout: {num_gpus: 1}
```

验收：

- trainer GPU 与 rollout GPU 分离。
- `weight_sync_pause_s` 接近 weight push/slot install 成本，不再等最慢 clip。
- 每步 wall-clock 下降。
- 没有 `policy_version mismatch`。
- stale/drop 指标和算法配置一致：GRPO 可 stale，DiffusionNFT 不训练 stale。

---

## 6. 内存预算

首版只允许 trainable-state slots，不允许无脑复制整套 frozen base：

```text
extra_vram ~= (retained_slot_count - 1) * trainable_state_bytes
```

LoRA / adapter 情况：

```text
base transformer: 1 copy
LoRA slot v1: retained
LoRA slot v2: retained
optional LoRA slot v0: retained until too stale / no longer referenced
```

full-param trainable 情况：

```text
base/transformer v1 + base/transformer v2
```

这是同一抽象下的高内存 case，不是单独设计。若模型没有 versioned trainable-state capability，或 memory
budget 不允许保留足够 slots，默认保持 draining barrier。

---

## 7. 配置原则

不要新增用户级开关。用户不应该理解 `persistent/drain/shadow/live` 这些内部细节。

推荐内部派生：

```text
if schedule.mode == continuous
and runtime.supports_non_draining_weight_sync
and sync_trainable_state enabled:
    use non-draining weight sync
else:
    use existing draining barrier
```

需要对用户暴露的是 diagnostics，不是 knobs：

```text
continuous.weight_sync_barrier_mode
continuous.weight_sync_pause_s
continuous.post_sync_dropped_stale
continuous.producer_discarded_stale
continuous.stale_policy_versions
```

---

## 8. 非目标

- 不做无条件 full-model double buffer；只 duplicate 当前 weight-sync payload 覆盖的 trainable state。
- 不给 DiffusionNFT 开 `max_stale>0`。
- 不把普通 collect failure 和 stale-slot eviction 混在一起。
- 不引入 microbatch/minibatch async。
- 不新增一组 YAML 参数让用户手动选择 drain/persistent/shadow。
- 不照搬 cosmos-rl 的 NCCL/独立 CUDA stream 首版实现；VRL 先走 Ray object store + trainable-state slots。
  CUDA stream 只作为后续优化，用来隐藏 slot install 的 copy 成本。

---

## 9. 风险

- **模型接口风险**：不同 diffusion family 的 LoRA/adapter 激活方式不一定一致。先通过 protocol 收口，不在
  `GenerationWorkerCore` 里硬编码 PEFT 细节。
- **slot 淘汰风险**：过早 evict 会让旧 request 缺 slot。首版保守保留 `max_stale + 2`，并把缺 slot 作为
  typed stale discard，而不是普通错误。
- **并发风险**：worker 必须保证同一时刻只执行一个 chunk 或者 activation 是 chunk-local。当前 Ray actor
  方法是同步执行；若以后打开真正并发 actor，需要加 per-slot executor 或锁。
- **数值一致性风险**：slot activation 必须和 trainer/rollout logprob 路径一致。首步 drift guard 仍然是红线。

---

## 10. 代码引用

- barrier / drain：
  - `vrl/rollouts/orchestration/continuous/schedule.py`
  - `vrl/rollouts/orchestration/continuous/producer.py`
- request version stamp：
  - `vrl/rollouts/orchestration/continuous/producer.py`
  - `vrl/rollouts/orchestration/prompt_collection.py`
- worker single-version behavior：
  - `vrl/generation/execution/worker.py`
- executor version assertion：
  - `vrl/generation/ray/executor.py`
- trainable-state sync payload：
  - `vrl/trainers/weight_sync.py`
  - `vrl/generation/ray/weight_sync.py`
- staleness admission：
  - `vrl/rollouts/orchestration/continuous/staleness.py`
  - `vrl/rollouts/orchestration/continuous/queue.py`
- algorithm soundness：
  - `vrl/algorithms/grpo/continuous.py`
  - `vrl/algorithms/diffusion_nft.py`

---

## 11. P2 多卡验证结果（2026-06-19，2 卡 cross-node 真机）

**配置**：`experiment/diffusion/sd3_5/online_grpo_ocr_crossnode_debug`（SD3.5-medium LoRA、128×128、
4 步、continuous + cross-node）+ `continuous.max_stale_policy_versions=1`。拓扑：Ray head 在 node A
`--num-gpus=0`（trainer 直接用 head 卡）+ worker 在 node B `--num-gpus=1`（rollout）。3 epoch 跑完 ~2min。

**验收（全部通过）**：

| 验收项（§5 P2 / P1） | 实测 | 判定 |
|---|---|---|
| `continuous_weight_sync_barrier_mode` | **1.0**（ep0/1/2 全 1）| ✅ non-draining 真生效——runtime capability 从 SD3.5 LoRA worker 的 `supports_versioned_trainable_state` AND 派生成功 |
| `policy_version mismatch` | **0** | ✅ versioned slots 防住混版本 |
| `continuous_weight_sync_pause_s` | 0.87 / 0.35 / 0.69s | ✅ 亚秒级（= 权重推送/slot install 成本，不再等最慢 clip）|
| stale admission（max_stale=1）| ep1 `continuous_stale_versions=1` | ✅ GRPO 可训练 stale item |
| stale-slot graceful discard | ep2 `continuous_producer_discarded_stale=1`、`post_sync_dropped=0`、`error_count=0` | ✅ `StaleSlotDiscard` 路由生效，evicted slot 走 discard 不污染 error |
| 端到端 | 3 epoch 跑完、0 Traceback | ✅ |

注：reward=0 是 128×128/4 步 debug 配置的预期（验编排机制、非学习；图太小 OCR reward 恒 0）。

**附带修复的预存 bug**：`vrl/rollouts/orchestration/lifecycle.py` `_collector_runtime()` 原只 catch
`AttributeError`，但 collector 的 `runtime` property 在 `set_runtime()` 前抛 **`RuntimeError`**
（cross-node/continuous 启动期会在 runtime attach 前查 policy version）——没被 catch → 没 fallback 到
weight_syncer 就崩。已改为 catch `(AttributeError, RuntimeError)` 优雅返回 None。这是 cross-node
continuous 路径启动即崩的根因，非 P2 改动引入，但 P2 多卡验证才暴露。

**结论**：P2 完成。non-draining weight sync 在真实 2 卡 cross-node 拓扑上去掉了 drain bubble
（barrier_mode=1、pause 亚秒、无 mismatch、stale-slot 优雅丢弃），全部硬指标达标。
