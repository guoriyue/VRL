# SPRINT: Shadow-model weight sync —— 安全去 drain barrier（planned）

状态：**proposed / design（2026-06-17）**。这是 `SPRINT_continuous_scheduler_redesign.md` §5 P1
「async 双缓冲 weight sync」的落地设计 + 单卡实证。结论先行：**continuous 的每步全-drain barrier 是吞吐
瓶颈（video 每步停数分钟），但裸去掉它会让在途请求横跨版本 bump → 撞 executor 同版本硬断言 → 请求失败
丢样本（本 sprint 已实测确认）。唯一安全的去 drain 办法是 shadow-model：新权重先收进 buffer，在途请求继续
用旧权重跑完,在请求边界才热切换。**

**触发**：correctness 里程碑（无 mismatch + pause 下降）**单卡即可验**；真 wall-clock overlap 收益需
**≥2 卡**（trainer 常驻一卡、rollout actor 常驻另一卡——单卡时间片下生成与训练抢同一批 SM,拿不到真重叠）。

来源：本仓库 + `~/Desktop/cosmos-rl` 逐文件读出（2026-06-17 workflow,file:line 已核）+ 单卡去 drain 实测。

---

## 0. 结论（TL;DR）

- **现状**：`after_train_step` = pause → **drain 全部在途** → sync → resume
  (`continuous/schedule.py:119-122`)。drain 等最慢一整条请求跑完才换权重,video clip ~5-6 分钟/条 →
  每步停数分钟。
- **为什么不能裸去 drain（已实测）**：删掉 `await drain_inflight()` 后跑单卡 GRPO smoke,每个 sync 步打
  `policy_version mismatch: expected=N, actual=N+1`。进程不崩（mismatch 被当 collect failure 捕获）,但
  **每步静默丢横跨 swap 的 rollout、`stale_versions` 仍 0、训练跑残缺集**。
- **唯一安全解 = shadow-model**：worker 把新权重收进 buffer,**在途请求继续用 live（旧）权重跑完**,在
  请求/生成边界把 buffer→live 热切换。请求全程不横跨版本 → 无 mismatch;完成的旧版本 group 留存 → 下一步
  `max_stale≥1` 当 stale 训 → GRPO 的 IS 比值补偿。
- **收益拆分（别混）**：① 去 drain-pause（correctness + `weight_sync_pause_s`↓,单卡可验）
  ② 真 overlap（生成与训练 wall-clock 重叠,`pause`→~0,epoch 时间↓,**≥2 卡 only**)。

## 1. 问题：drain 围栏 + 裸去 drain 的失败模式（实测）

drain barrier(`continuous/schedule.py:113-123` `after_train_step`):

```python
self.producer.pause_admission()
await self.producer.drain_inflight()                 # 等所有在途请求跑完
await self.lifecycle.sync_weights_after_train(...)   # 版本 bump（lifecycle.py:60-70）
self.producer.resume_admission()
```

`drain_inflight`(`producer.py:122-132`)docstring 自陈其存在理由：
> Mutating worker weights while a generation request is running could mix two policies inside one
> request, so the barrier always drains rather than cancels.

正确性闸（不可绕过）：executor 对每个返回 chunk 断言 `result.policy_version == request.policy_version`,
否则 raise(`executor.py:127-137`);请求版本在 submit 时一次性钉死(`producer.py:167`),worker 在
`execute_chunk` 入口再拒一次版本不符的 chunk(`worker.py:118-136`)。worker 是单线程 Ray actor →
`update_weights`(`worker.py:73-84`,就地 `self._policy_version=...`)只在两次 `execute_chunk`
(`worker.py:113`)之间跑,chunk 内部原子。

**实测(2026-06-17,去 drain 单卡 GRPO smoke,6 步,RTX 5090)**:

```
policy_version mismatch: expected=1, actual=2   (sample_start=1 = group 的第 2 个 chunk)
policy_version mismatch: expected=2, actual=3 / expected=3, actual=4 / ...（每步一条）
```

一个 group(`n_samples_per_prompt=2`,`sample_batch_size=1`)= 2 个 chunk;第 1 个 sample 在版本 N 跑完 →
swap → 第 2 个在 N+1 跑 → 该请求触发断言失败。每步 ~1 个 `collect failed`(error_count 累加)、
`stale_versions` 仍 0、`weight_sync_pause_s` 0.5→0.15s(跳过工作然后失败)。

**根因 = "submit 时钉死版本 vs 执行时已 swap",与模型大小、chunk 粒度无关**:即使把 group 压成单 chunk
(`sample_batch_size=2`),在途 group 只要在 swap 后才执行,照样 mismatch。所以 drain 不能裸去,也不能靠
"小 rollout" 规避。

## 2. 设计：shadow-model 双缓冲

核心：把"换权重"与"请求完成"解耦。

**worker 侧**(`vrl/generation/execution/worker.py`):
1. `update_weights` 不再就地 `load_state_dict` 到 live model,而是把新权重收进一个 **shadow buffer**
   (LoRA 只收 trainable adapter,buffer 小;`weight_sync.py:50-59` 已是 trainable-only flat state dict)。
2. 维护 `pending_version`(buffer 里的)与 `live_version`(正在跑的,即现 `self._policy_version`)。
3. 在**请求/生成边界**(两次 `execute_chunk` 之间,worker 单线程天然原子)检查:有 pending 且当前无在途
   请求依赖 live → `live ← buffer` 热切换、`live_version = pending_version`。
4. 请求全程按其 submit 版本在 live 上执行;executor 断言(`executor.py:127-137`)仍成立(请求执行期间
   live_version 不变)。

**schedule 侧**(`continuous/schedule.py` `after_train_step`):
- 去掉 `await self.producer.drain_inflight()`;改为非阻塞 `push_weights_to_shadow()`(推 buffer + bump
  逻辑版本),`pause_admission/resume_admission` 仍保留(避免 admit 风暴)。
- 在途请求不再被 drain;它们用旧 live 权重跑完,落 ready 队列时带旧版本戳 → 下一步 `staleness.admit`
  (`staleness.py:38-44`)以 `max_stale≥1` 收。

**producer/queue 侧**:基本无需改(版本戳 `producer.py:167`、StalenessPolicy 已就位);只需确认 resume 后
producer 用新 `live_version` 重交。

## 3. cosmos-rl 对标（最小可搬集）

- `WeightSyncThread`:独立 CUDA stream(`weight_sync.py:265-294`)、NCCL recv 进 buffer(`:371-413`)、
  `sync_buffer_to_live` 在生成边界用 `inf_stream.wait_event` 跨流定序后 copy(`:178-234`)。
- 生成中途抢占:patch `llm_engine.step` 每 N 步 `consume_command`(`vllm_rollout.py:93-119`),
  `enable_prefix_caching=False`(`:319`)保证换权重后无依赖旧权重的 cache 残留。
- **VRL 最小集**:**不必照搬独立 stream + NCCL**——VRL 走 Ray object store 推权重(非 NCCL),且单线程
  actor 让"请求边界"天然存在(execute_chunk 之间),比 vLLM 的连续 batching 简单。只需 buffer + 边界热
  切换;独立 stream 隐藏 copy 延迟是后续优化,不在首版。

## 4. soundness 闸

- **GRPO sound**:stale=1 时 surrogate `ratio = exp(log_prob − old_log_prob)`(`grpo/continuous.py:85`),
  `old_log_prob` = rollout 时 behavior 版本,PPO clip 兜底 → 跨版本偏差正是 IS 设计要纠正的。
- **DiffusionNFT 不 sound**:likelihood-free,loss 是 normalized_mse on positive/negative x0 预测,无任何
  `exp(log_prob − old_log_prob)`(`diffusion_nft.py`),无处安放 stale 修正。**DiffusionNFT 必须
  `max_stale=0`**——shadow-model 仍可去 drain-pause(纯吞吐),但**不开 staleness 窗**。

## 5. 分阶段（先单卡 correctness,后多卡 throughput）

**P0 — shadow buffer + 边界切换(单卡,correctness 里程碑)**
实现 buffer + `pending/live_version` + 边界热切换,`after_train_step` 去 drain 改推 buffer。
验收:去 drain 后跑单卡 GRPO smoke,**`policy_version mismatch` 消失**、collect error=0、训练不丢样本、
`weight_sync_pause_s` 下降。纯正确性,不依赖 overlap。

**P1 — 开 staleness 窗(单卡,staleness 里程碑)**
`max_stale=1` + 让 producer 跑在 trainer 前面。验收:`continuous_stale_versions` 偶发 0→1(证明完成的旧
版本 group 被当 stale 训)+ GRPO reward 不退化。
**注意**(`SPRINT_continuous_scheduler_redesign.md` 实测):单卡 colocated 下 consumer "同版本最新优先"
选择(`queue.py:149-159`)+ 快速重填会把 stale 压回 0;可能需要更深 `max_ready_groups` / 更慢生成才能稳定
观察到 stale=1。若单卡始终压不出 stale>0,记录之并把 staleness 实测移到 P2。

**P2 — 真 overlap(≥2 卡,throughput 里程碑)**
trainer 常驻 GPU0、rollout actor 常驻 GPU1;验收每步 wall-clock 下降、`weight_sync_pause_s` 被隐藏到接近
0、rollout 与 train 时间真重叠。这是 parked doc(`SPRINT_async_rollout_train_overlap.md`)Option A 的 host。

## 6. 非目标

- **不裸去 drain**(实测每步丢样本);**不 cancel-resubmit**(取消在途重交同样丢部分工作、且不产生
  staleness,是更差的中间方案)。
- 不照搬 cosmos 的独立 CUDA stream / NCCL(VRL 走 Ray object store);独立 stream 隐藏 copy 是后续。
- **不给 DiffusionNFT 开 staleness**(无补偿,parked doc §1);DiffusionNFT 只用 shadow-model 去 pause。
- P0/P1 不追求 wall-clock overlap(单卡时间片做不到,别把 stale=1 当吞吐赢)。

## 7. 验收 metrics（复用已落地的 P0 4 列）

`metrics.csv` 的 `continuous_stale_versions / continuous_ready_groups / continuous_weight_sync_pause_s /
continuous_producer_max_gap_s`(`online.py` `_write_metric_row` 已 flush)+ 日志 grep `policy_version
mismatch`:
- **P0**:mismatch=0、`weight_sync_pause_s`↓、collect error=0。
- **P1**:`stale_versions` 偶发=1、reward 与 strict baseline 持平(同 seed/prompt/reward,block-test)。
- **P2**:每步 wall-clock↓、`weight_sync_pause_s`→~0、producer 不饿。

## 8. 关键文件引用

- barrier / drain:`vrl/rollouts/orchestration/continuous/schedule.py:113-123`(after_train_step)、
  `vrl/rollouts/orchestration/continuous/producer.py:122-132`(drain)、`:167`(submit 版本戳)
- 正确性闸:`vrl/generation/ray/executor.py:127-137`(同版本断言)、
  `vrl/generation/execution/worker.py:73-84`(update_weights + `_policy_version` 戳)、
  `:113-136`(execute_chunk + chunk 拒绝)
- 权重推送:`vrl/generation/ray/weight_sync.py:50-59`(CPU state_dict → ray.put → fan-out)
- 版本 bump:`vrl/rollouts/orchestration/lifecycle.py:60-70`(sync_weights_after_train)
- staleness:`vrl/rollouts/orchestration/continuous/staleness.py:38-44`(admit)、
  `vrl/rollouts/orchestration/continuous/queue.py:149-159`(最新优先选择)
- soundness:`vrl/algorithms/grpo/continuous.py:85`(IS 比值)、`vrl/algorithms/diffusion_nft.py`(无比值)
- 可观测(P0,已落地):`vrl/scripts/common/online.py`(_write_metric_row / _prepare_metrics_csv 的 4 个
  `continuous_*` 列)、`vrl/rollouts/orchestration/schedule.py`(`_build_continuous_schedule` 启动告警)
- 上游:`SPRINT_continuous_scheduler_redesign.md` §5 P1、`SPRINT_async_rollout_train_overlap.md`
  (parked,Option A / DiffusionNFT 约束)
- cosmos:`~/Desktop/cosmos-rl/cosmos_rl/rollout/worker/weight_sync.py:178-234,265-294,371-413`、
  `rollout/vllm_rollout/vllm_rollout.py:93-119,319`
