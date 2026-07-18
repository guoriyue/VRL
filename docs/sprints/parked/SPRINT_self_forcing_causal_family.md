# SPRINT：Self-Forcing causal family — 因果流式视频模型进入 native engine

日期：2026-07-13

状态：**parked**。触发事件是
[FlashDreams execution provider](../planned/SPRINT_flashdreams_execution_provider.md) 的 F0–F3
generic primitive、trainable-state installer 与 fake-adapter contract 完成；不等待
Self-Forcing 自己关闭 provider production conformance。wm-infra 侧接入还要求
`SPRINT_ray_rollout_fault_tolerance` 的阶段 3–6 先完成。

## 0. 结论先行

wm-infra 继续拥有 generation runtime、Ray 生命周期、policy version、trajectory、replay
和 trainer。FlashDreams 受控 fork 只提供 causal Wan transformer、BlockKVCache 与可组合
denoise primitive；本 sprint 是第一个真实 family consumer，不 vendor 第二份模型栈，也不
创建第二个顶层 rollout engine。

首个目标是 **Self-Forcing Wan2.1-T2V 1.3B**。其 pinned recipe 做 4 次 flow forward，
但只有 3 个 policy-dependent Gaussian transition：初始噪声与 policy 无关，forward
0/1/2 的 clean prediction 分别参数化下一次 renoise；forward 3 只产生 deterministic
terminal clean，没有后继随机节点。不能把 4 次 forward 写成 4 个 action，也不能声称
这里计算的是最终视频密度。

## 1. 顺序 KILL gates

### P0 — RENOISE likelihood 数学门（CPU）

对 forward `i = 0, 1, 2`：

```text
clean_i = noisy_i - sigma_i * flow(noisy_i, timestep_i)
noisy_{i+1} = (1 - sigma_{i+1}) * clean_i + sigma_{i+1} * epsilon_i
```

因此 trainable transition 是：

```text
noisy_{i+1} | noisy_i
  ~ Normal((1 - sigma_{i+1}) * clean_i(noisy_i), sigma_{i+1}^2 * I)
```

P0 必须同时证明：

1. 4-forward recipe 恰好产生 3 个 stochastic actions；
2. sample-time 与 replay-time log-prob 逐元素相等；
3. gradient audit 证明 flow forward 0/1/2 得到非零 policy gradient；
4. 首版固定把 forward 3 作为 non-trainable terminal computation：不创建 action/log-prob
   slot，不对它单独加 policy loss，只保存 terminal clean；共享网络参数仍会由前三个
   transition 更新；
5. trajectory 保存网络实际收到的 warped timestep，以及该 transition 实际使用的
   `sigma_{i+1}`，replay 不从 `[1000, 750, 500, 250]` 二次推导；cache finalization 的
   `context_noise/context timestep` 也来自同一 resolved recipe。首版 Self-Forcing 固定并
   验证 `context_noise == 0`；非零值在定义 finalize-noise trajectory contract 前 fail closed。

`[1000, 750, 500, 250]`、`shift=8` 是 pinned integration recipe/config contract，
不是 checkpoint 元数据。checkpoint 只提供权重。

**任一 likelihood、mask、gradient 或 schedule round-trip 不闭合，本 sprint 停止。**

### P1 — Trainable state + 真权重生成门

实现前先固定四个不同边界，禁止把它们叫作“同一个 mapping”：

1. FlashDreams fork 的 source checkpoint normalization：解开 `generator_ema/generator` 与
   wrapper prefix；
2. FlashDreams fork 的 post-load module normalization：例如 Wan modulation 从
   `[1, 6, D]` 变为 `[6, D]`；
3. wm-infra family adapter 的 trainer canonical keys → provider raw module names；
4. FlashDreams fork 的 raw-key-only in-place installer。

四条边界汇入同一个经过验证的 raw-module destination schema，但各自有独立输入、错误
信息与正反测试。P1 必须在 compile/CUDA capture **之前**安装 LoRA，并证明：

- target modules 由真实 raw transformer module discovery/typed build 派生，不维护
  `LORA_TARGETS` 一类手写大表；
- trainer replay model 与 rollout raw module 的 trainable key/schema/digest 完全一致；
- adapter enable/disable 能产生 train/reference 两条明确前向；
- reference path frozen，train path 的 LoRA gradient 非零；
- missing/unexpected/shape/dtype mismatch fail closed；
- hot update 后总是清空旧权重产生的 AR KV session；是否 reset compile/CUDA graph
  只由 eager/compiled/captured parity 证明，不做 blanket reset。

FlashDreams 当前没有现成 PEFT surface，因此 P1 的明确实现目标是先增加 LoRA surface。
若无法在 compile 前挂载并通过梯度/key-parity 门，本 sprint 停止；full-parameter 训练必须
经过独立容量证据后另行解 park，不能作为这里的静默 fallback，更不能用未定义的
`transformer.*` 映射继续。

trainer 必须在自己的进程构造 gradient-capable `SelfForcingReplayModel`，复用同一 pinned
FlashDreams raw module/config；禁止通过 rollout worker、HTTP 或子进程远程调用来假装
保留 autograd。

真权重门使用 Self-Forcing 1.3B checkpoint，生成可人工查看的 mp4，记录峰值显存；只看
邻帧差统计不算通过。

### P2 — Causal cache replay 门

现有 trainer 按 timestep 独立调用 `replay_forward()`。如果每个 `(temporal_chunk,
transition)` 都从 chunk 0 重建 prefix，完整 update 是
`O(num_temporal_chunks^2 * num_flow_steps)`，不可接受。`replay_samples_per_chunk` 只切
sample batch，不能解决 temporal history 重建。

现有 FlashDreams `BlockKVCache.update()` 会原地覆盖同一 physical slot：

```python
self._k[sl_write] = k[sl_read]
self._v[sl_write] = v[sl_read]
```

这对 no-grad inference 合理，但不能让多个带梯度 flow forward 在一次 backward 前共享该
mutable storage；后一个 forward 会修改前一个 backward 保存的 tensor version。仅在四次
forward 后 `detach()` 无法修复。

P2 因此在 FlashDreams fork 增加 **trainer-only functional replay cache**，而不是在
wm-infra 重写依赖内部：past temporal chunks 是 immutable/detached prefix；每个带梯度
flow forward 用 fresh current-slot K/V view，函数式组合 prefix + current slot，不写入任何
被另一个 forward graph 保存的 storage。rollout 继续使用原 in-place `BlockKVCache`。

trainer 以一个 causal replay session 线性推进：

```text
for each temporal chunk k, in order:
  hold one immutable/detached prefix for chunks < k
  run flow forwards 0..2 with three independent functional current-slot views
  compute the three Gaussian losses and backward this temporal chunk
  under no_grad, run denoise terminal forward 3 from stored noisy_3
  compare terminal clean with stored chunk_clean_latent
  discard denoise-forward-3 current-slot K/V
  run a separate functional finalize_kv_cache forward on terminal clean
    at the resolved context timestep; discard its flow output
  commit only this finalization forward's K/V, then cache.finalize bookkeeping
  detach the committed prefix for k+1
```

这次额外 finalization forward 不能和 denoise forward 3 合并：public
`DiffusionModel.finalize()` 会在 scheduler 结束后把 terminal clean（首版
`context_noise=0`，所以不再加噪）与 context timestep 交给 `finalize_kv_cache()`，丢弃 flow，
再调用 `cache.finalize()`。grouped replay 必须复刻这个 lifecycle，下一 temporal chunk 才会
看到与 rollout 相同的 prefix。

prefix 每个 temporal chunk 只构建/commit 一次，目标复杂度是
`O(num_temporal_chunks * num_flow_steps)`，graph memory 也只保留一个 temporal chunk。
trainer 增加按 capability dispatch 的窄 `GroupedTrajectoryEvaluator` session：每次返回一个
temporal chunk 的三个 signals，loss 按整条 trajectory 的
`num_temporal_chunks * 3` 归一化并立即 backward；其他 evaluator 继续走现有逐 timestep
loop，禁止按 family name 分支。`SelfForcingReplayModel` 暴露 family-specific functional
cache/replay method，不把它塞进共享 `ReplayModel`。这个薄协议边界是必要的，因为它改变
replay 迭代单位并拥有 causal cache 生命周期；在第二个 family 出现相同语义前，不扩成
通用 temporal-engine framework。

P2 的首个 KILL test 是 tiny Wan block 的两条 cache path parity：同一 prefix/current input
下 functional replay 与 in-place inference 输出相等；连续 4 个 denoise forward、独立
finalization forward、至少 2 个 temporal chunk 在
`torch.autograd.set_detect_anomaly(True)` 下 backward，不出现 version-counter 错误且
trainable grad 非零。每个 chunk 还要逐 tensor 比较 functional path 与 public
`generate() + finalize()` **完成后的 committed cache**，不能只比较 denoise output。随后
覆盖 AR0、cache filling 最后一个 chunk、首次 rolling-window 更新和 steady state。若只能
得到二次重放、mutable-cache backward、跨进程 autograd，或 output/cache parity 不闭合，
本 sprint 停止。

## 2. Trajectory contract

不要把 native `SampleChunk`（prompt/sample batch 切片）和模型的 temporal AR chunk 混为
一谈。trajectory 使用三根有真实消费者的逻辑轴：

```text
sample
temporal_chunk       kind=custom
renoise_transition   kind=denoise_step, length=3
```

trainable segment 中保存：

- `observations[sample, temporal_chunk, renoise_transition]`：`noisy_i`；
- `actions[...]`：`noisy_{i+1}`；
- `old_log_prob[...]` 与 trainable mask；
- `autoregressive_indices[temporal_chunk]`：rollout 实际传给 provider 的 index，grouped replay
  调 `cache.start/finalize`；continuation rollout 不假设从 0 开始，因此不能只从 axis length
  猜值；
- `flow_step_indices[renoise_transition]`：明确 action 由 forward 0/1/2 参数化；
- `prediction_timesteps[temporal_chunk, renoise_transition]`：实际 warped timestep；
- `transition_sigmas[...]`：实际 `sigma_{i+1}`；
- `cache_context_timesteps[temporal_chunk]`：实际 finalization context timestep；首版验证
  全为 0，grouped replay 的独立 `finalize_kv_cache` forward 消费；
- `chunk_clean_latents[sample, temporal_chunk, ...]`：rollout executor 直接保存
  `finish_denoising` 返回、decode 前的 terminal clean；grouped replay 用它验证 terminal
  forward 并推进下一 chunk cache。

`ReplayInput.tensor_refs` 明确引用这些 tensor；resolver 读取整个 segment，不使用错误的
`replay_tensor_dict(axis="denoise")` 伪切片。若某处确实按轴切片，必须同时传
`axis` 和 `axis_index`。

KV cache、provider session、Ray handle 不进入 trajectory。最后 deterministic forward
只产出 `chunk_clean_latents`，不塞假 action，也不在首版增加 auxiliary objective。

## 3. Execution path

新增 family-specific `SelfForcingChunkExecutor`，复用 native 外层 request planning、
`SampleChunk` OOM split、gather、launch contract 与 policy-version discipline，但**不直接
复用** `DiffusionChunkExecutorBase` 的单一连续 denoise 主循环。

一个 native `SampleChunk` 内部执行 temporal state machine：

```text
prepare request conditioning once
for each temporal AR chunk:
  derive fresh initial noise from native request seed
  start_denoising
  4 flow forwards / 3 recorded renoise transitions
  finish_denoising
  finalize before the next temporal chunk
    (extra finalize_kv_cache forward on terminal clean, then cache.finalize)
assemble clean chunks
decode one video
```

同一 `FinalState` 至多 finalize 一次；进入下一 temporal chunk 前必须 finalize；最后一个
chunk 若不会再被读取可以不 finalize，保持上游 public lifecycle 语义。验收统计
`start/finish/finalize` 次数、fresh-noise seed、cache index 与 assembled frame layout。

## 4. 实施产物

- `vrl/models/diffusion/self_forcing/`：family model/replay facade、FlashDreams adapter、
  custom executor；不复制 WanDiT、scheduler 或 BlockKVCache。
- `vrl/math/diffusion/`：renoise sample/logprob math 与 CPU gradient tests。
- family registry entry + FlashDreams provider binding：选择 custom executor 与 grouped
  evaluator/replay builder；recipe 必须显式选择 FlashDreams，不能伪造 native fallback，
  也不新增平行 `SUPPORTED_MODELS` 表。
- 一个最小 DROID-overfit 风格配方：4 个固定 prompt、BLOCK/motion guard、现有
  Kling/VideoScore2 reward；只有 P0–P2 全绿后才进入训练 smoke。

## 5. Architecture hygiene

### 应改变

- 新增 causal temporal executor 与 grouped replay，因为现有 single-denoise/per-step replay
  无法表达 cache 状态机且会产生二次复杂度。
- 在 FlashDreams fork 新增 trainer-only functional cache；inference 的 in-place cache 保持。
- 分离 checkpoint normalization、post-load normalization、trainable-name mapping 与
  runtime installer；validation 由 raw module/schema 派生。
- trajectory 增加三个行为消费轴和 `chunk_clean_latents` producer/consumer contract。

### 保持不变

- native `GenerationRuntime`、Ray worker、policy version、trajectory 与 trainer ownership。
- FlashDreams public `generate/finalize` facade 与 transformer `predict_flow`；它们分别是
  public cache lifecycle 和跨 family execution hook。
- wm-infra adapter 与 family executor 即使薄也保留：前者是跨仓协议边界，后者拥有
  temporal cache state machine。
- FlashDreams `_FP32_BUFFERS` 被 `_apply` 消费，用于 schedule 精度，是合理 module-state
  taxonomy，不删除。
- `BlockKVCache` 的 in-place rollout path 保持不变；它是 inference 性能路径，但禁止直接
  进入带梯度 replay。

### 非目标

- 不嵌入 FlashDreams WebRTC/serving control plane；不建第二个 runtime。
- 不 vendor 平行 WanDiT/BlockKVCache/scheduler。
- 不做 OmniDreams/LingBot；不做蒸馏训练。
- 首版不启用 CUDA graph rollout；后续只有 hot-update parity 与性能数据支持时再开。
- 不做 full-parameter fallback 或 terminal auxiliary objective；它们需要各自的新证据与门。
- 不为减少 LOC flatten protocol/family facade；一致性与真实 lifecycle ownership 更重要。

## 6. Definition of Done

- [ ] P0 证明 3 个 transition 的 sample/replay likelihood、mask 与 gradient 正确。
- [ ] actual warped timestep 和 transition sigma 来自同一 resolved recipe 并写入 trajectory。
- [ ] LoRA trainable state 在 compile 前构造，trainer/rollout key-schema/digest 一致。
- [ ] `SelfForcingChunkExecutor` 正确区分 native sample chunk 与 temporal chunk。
- [ ] grouped replay 为线性复杂度，覆盖 filling、首次 rolling update 与 steady state。
- [ ] functional cache 与 public generate+finalize 后的 committed cache parity 通过，
  anomaly-mode backward 无 version-counter error。
- [ ] old/fresh log-prob 在真权重上过 drift guard，terminal clean round-trip 通过。
- [ ] 单卡 collect → advantage → backward 的 trainable grad norm 非零。
- [ ] 人工检查真实 mp4；结果与峰值显存写入长期测量档案。
- [ ] provider/family failure 关闭 admission，不留下 cache/session。

GPU 被占用期间只执行 CPU math/config/contract tests；P1/P2 的真实模型门不能用 skip 或
mock 冒充通过。

## 参考

- `docs/sprints/SPRINT_native_generation_engine_program.md`
- `docs/sprints/planned/SPRINT_flashdreams_execution_provider.md`
- `docs/sprints/SPRINT_ray_rollout_fault_tolerance.md`
- `vrl/generation/execution/chunks.py`
- `vrl/generation/diffusion/executor.py`
- `vrl/trajectory/resolver.py`
- `vrl/trainers/online/trainer.py`
- `vrl/rollouts/evaluators/diffusion/sde_logprob.py`
- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/infra/diffusion/model/base.py`
- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/core/attention/kvcache.py`
- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/infra/diffusion/scheduler/fm.py`
- `/home/mingfeiguo/Desktop/flashdreams/flashdreams/flashdreams/infra/pipeline/base.py`
- `/home/mingfeiguo/Desktop/flashdreams/integrations/self_forcing/self_forcing/config.py`
