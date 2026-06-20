# SPRINT: continuous rollout scheduler —— 真重叠 + 统一调度器（done）

状态：P0 / P1(GAP1+GAP2) / P2 统一调度器均已落地(见下方 2026-06-17 / -18 / -20 更新);唯一 parked 的是 P2
更大子步「真 wall-clock 重叠(独立线程/进程 rollout owner)」,需 ≥2 卡验证、作为独立 PR。这是对
`vrl/rollouts/orchestration/continuous/` 异步调度的诊断 + 重设计方案。
findings 全部带证据(代码 path:line),对标 cosmos-rl 单控制器。**先做 P0/P1 小杠杆,统一调度器(P2)
是更大的可选步,不是从零重写。**

**Scope guard（2026-06-17）**：本文里的 async 只指 **rollout producer / reward / ready queue** 与
**trainer** 的跨阶段、跨 step wall-clock overlap；不再包含 microbatch/minibatch prefetch。同步
microbatch 只由 `SPRINT_streaming_rollout_accumulation.md` 与
`SPRINT_memory_budgeted_microbatch.md` 维护。

> 方法:1 个 workflow(3 agent)逐项核实 overlap/drain/staleness + 结构对称性,对照 cosmos-rl 源码
> (`/home/mingfeiguo/Desktop/cosmos-rl`)。我独立复核了 driver loop 时序(trainer.py)。

> **更新(2026-06-17,P0 落地 + 单卡 A/B 实测):**
> - **P0 已实现 + 验证**:启动告警 `continuous async prefetch ENABLED / max_stale=0 → no prefetch`
>   (`schedule.py` `_build_continuous_schedule`);并把 4 个 async 诊断量 flush 进 `metrics.csv`
>   (`online.py` `_prepare_metrics_csv`/`_write_metric_row`):`continuous_stale_versions / ready_groups /
>   weight_sync_pause_s / producer_max_gap_s`(strict 模式自动写 0)。这让"是否真 async"per-step 可见。
> - **单卡实测**(sd3_5 OCR GRPO,colocated resident,RTX 5090,`max_stale=1` vs `0` A/B,同 seed,
>   `max_ready_groups=8`,6 步):`ready_groups` 在 `max_stale=1` 下 = 1(预取缓冲生效)、`max_stale=0` 下 = 0
>   (无预取,strict-equiv)——**窗口开关有可观测效果**。但 **`stale_versions` 两组恒为 0**:即使开窗 + 加深
>   队列,consumer 仍消费 fresh group。无崩溃、无 `policy_version mismatch`(barrier 不变量成立)。
> - **§2.2/§2.3 经实测确认 = H1(结构性,非 smoke 太小)**:drain 在 version bump 前 await 全部在途
>   (`producer.py:130-132` + bump 在 `lifecycle.py:67-70`)+ consumer "同版本最新优先"选择(`queue.py:149-159`),
>   把消费围栏到 current 版本;单卡 colocated 下 producer 重填 fresh 太快,stale group 永远选不上。
>   **`max_stale=1` 必要不充分**。
> - **P1 真实形态修正(与 §5 P1 的乐观判断相左)**:"去掉/放松 drain" **不是单卡安全增量**——executor
>   硬断言同请求同版本(`executor.py:127-137`),放松会让横跨版本的请求**崩溃**而非让 StalenessPolicy
>   吸收尾巴。真 P1 = ≥2 卡 + shadow-model(cosmos `WeightSyncThread`)或 cancel-resubmit(producer/schedule
>   层,取消在途剩余 chunk + 新版本重交),是独立研究 PR。单卡只能做到"机制可观测"(已做);真 staleness/
>   overlap 解 park 于 ≥2 卡。
> - **soundness 闸**:`max_stale=1` 对 GRPO sound(IS 比值补偿,`grpo/continuous.py:85`),对 DiffusionNFT
>   不 sound(无比值,parked doc §1);DiffusionNFT 单卡只跑 `max_stale=0`。
>
> **更新(2026-06-18,算法无关 async 基础设施落地 —— 不依赖 ≥2 卡的 CPU 可测部分):**
> - **GAP 1 落地:producer 侧 receipt-time freshness gate + schedule 侧 post-sync purge**(此前只在 consumer
>   端事后丢弃,见 parked `SPRINT_async_rollout_train_overlap.md:79`)。`producer.py` `_enqueue_result`
>   顶部用同一个 `StalenessPolicy.too_stale(result["version"], current_version)` 自检:如果在途期间
>   `current_version` 已经越过窗口,该组在收货时直接丢弃,不入队。新增 `discarded_stale_count` 计数
>   (`types.py` `ContinuousRolloutProducerState`)+ `continuous.producer_discarded_stale` metric
>   (`schedule.py` `_attach_producer_metrics`)。同时补上真实 barrier 顺序下更常见的路径:标准
>   `after_train_step` 是 `drain_inflight -> sync_weights_after_train`,所以训练期间已经入队的旧版本 item
>   会在 **sync 之后** 才变 stale；`schedule.py` 在 sync 后立即调用 `queue.drop_too_stale(...)`,通过
>   `continuous.post_sync_dropped_stale` 与 queue `dropped_stale` 记录,避免把旧 ready queue 拖到下一次
>   consumer wait 才丢。**算法无关**:两处闸门都只读 `max_stale_policy_versions` 配置 —— NFT(=0)不训练跨版本
>   item,GRPO(≥1)只丢真超窗的;`None` 版本与 future(staleness<0,bug)仍由 queue/consumer fail-fast。
>   CPU 单测:`test_contracts.py` `test_producer_discards_group_too_stale_at_receipt` / `_past_stale_window`
>   + `test_schedule.py` `test_after_train_step_purges_stale_ready_items_after_sync`。
> - **soundness 闸从"概念"变成"代码 fail-fast"**:算法声明能力属性
>   `tolerates_off_policy_staleness`(`diffusion_nft.py` 显式 `False`,GRPO 走安全默认 `True`,镜像现有
>   `uses_evaluator` pattern)。`build_rollout_schedule` 收一个 **bool**(不 import 算法层,守住架构边界)
>   传到 `_build_continuous_schedule`:`max_stale>0 且算法不容忍` → `ValueError` fail-fast,而非静默有偏跑。
>   trainer 在 `__init__` 用 `getattr(self.algorithm, "tolerates_off_policy_staleness", True)` 接线。
>   CPU 单测:`test_schedule.py` 三个(intolerant+窗口>0 拒绝 / tolerant+窗口>0 放行 / intolerant+窗口=0 放行)
>   + `test_diffusion_nft.py` 能力声明断言。这把用户的"机制算法无关 + staleness soundness 是 per-algorithm
>   config"两半都落实了:机制对所有算法一视同仁,soundness 由算法属性 × 配置共同判定。
> - **仍 parked 于 ≥2 卡**:GAP 2(shadow-model de-drain,消除 barrier drain 气泡)是多卡优先、改动大的
>   独立 PR(见 §5 P1 修正 + `SPRINT_shadow_model_weight_sync.md`)。本次只做单卡 CPU 可测、对所有算法
>   自动就绪的那部分。
>
> **更新(2026-06-20,P2 统一调度器落地 —— `RolloutScheduler` 收编 admission/budget/staleness):**
> - **新 `RolloutScheduler`(`continuous/scheduler.py`)= §4 目标里的「单一 owner」**。把此前散在 producer 的
>   双计数器(`len(inflight)<max_inflight` 且 `queue.size()+inflight<capacity`)与 queue 独立 caps 的 admission
>   决策收进**一个 `can_admit()`**:in-flight 上限 + 跨 in-flight/ready 的单一条目预算 + ready 字节预算 +
>   admit 时预测版本节流。producer/queue/consumer 仍保留各自的 *mechanism*(Ray dispatch / 容器 / iteration
>   build),`RolloutScheduler` 持有 *decision*。
> - **真正的单一 owner(第二轮收编)**:第一版只搬了 admit 一处决策,显得薄。复核后把**全部 5 处 policy-version
>   决策**都路由进 scheduler —— 除 `can_admit` 外再加 `discard_at_receipt`(producer receipt 闸)、`drop_stale`
>   (post-sync purge + select 前置)、`select_iteration`(同版本选择,含 `_candidate_versions`/`_take_distinct`)。
>   `queue.py` 因此**彻底不再 import `StalenessPolicy`**,退化成纯容器(deque + 字节/条目安全网),只对外暴露
>   `snapshot`/`remove`/`note_dropped_stale` 三个 mechanism 接口;`consumer.py` 也不再持 `StalenessPolicy`,改持
>   scheduler 并驱动 `scheduler.select_iteration`。eviction(`_pick_victim`/`drop_policy`)经核查**不涉及
>   policy-version**(按 `completed_at` 墙钟挑最旧的字节安全网 tiebreak),故合理留在 queue。现在「什么算 stale /
>   选哪个版本 / 收货丢不丢」只有一个答案出处。
> - **修掉诊断出的两个真漏洞**(详见诊断):
>   (1) **字节维度「两 owner 打架」**——producer 的 `_admit` 此前只看条目数,queue 的 `_enforce_caps` 却按
>   `max_bytes` 静默 evict 掉 producer 自以为已收下的新条目;现在 `can_admit` 的字节预算让 admission 看见
>   字节,queue 字节驱逐降级成罕见安全网(in-flight 字节未知,仅作 burst 兜底)。
>   (2) **staleness 全是事后丢、缺事前节流**——新增 admit 时预测落点版本
>   `predicted = (inflight+ready)//groups_per_iteration`(cosmos `controller.py:273-303` 的 per-iteration 改写,
>   一个被消费的 iteration = 一次 version bump):若现在发出的组会落在窗口外就**不发**(主动硬节流),取代纯
>   reactive 的 receipt/select 丢弃。证明无回归且不饿死:`predicted==0` 直到攒满一整个 iteration,所以
>   admission 总能到达恰好 `(max_stale+1)` 个 iteration 的预取深度再停;不减少被训练的数据,只消除注定被
>   receipt-gate 丢弃的浪费生成。**`max_ready_groups` 超过 `(max_stale+1)×一个iteration` 的部分被 staleness
>   capped**(印证 §2.3「单独调大 max_ready_groups 没用」)。
> - **可观测**:producer state 新增 `predicted_admit_staleness` + `admit_blocked_reason`("inflight_full" /
>   "item_budget_full" / "byte_budget_full" / "would_land_too_stale"),flush 进 `continuous.predicted_admit_staleness`
>   与 `continuous.admit_blocked_on_staleness` metric —— 「串行假象」现在 per-step 可诊断,无需附 debugger。
> - **测试**:新 `test_scheduler.py`(20 例,覆盖三类预算 + 预测节流窗口缩放 + 不饿死一个 iteration + reason
>   优先级 + prompt-swap,以及从 `test_queue.py` 迁来的 drop-stale / 同版本 select / future fail-fast /
>   receipt 闸);`test_queue.py` 瘦身为纯容器测试(backpressure/stats/snapshot-remove);`test_contracts.py` 的
>   `_producer`/`_consumer` 改为注入 scheduler。`continuous/` + soundness 全套 66 例通过、`tests/rollouts/` +
>   schema 186 例通过、`ruff` 干净(1 例 pre-existing 失败与本次无关:
>   `test_sd35_single_gpu_async_debug_uses_persistent_colocated_rollout`,断 colocate.memory_fraction,属
>   distributed 配置漂移)。
> - **Review 修复(2 项)**:
>   - **[High] prompt-set ownership**:select 此前只按 `rollout_policy_version` 分组,而 `group_key` 只是 slot 序号,
>     两个 prompt set 都从 0 编号 —— prompt 换批时,上一批同版本的 ready items 会被当成新批 iteration 训练(reviewer
>     用 fake schedule 复现:`['p0','p1']` 后调 `next_iteration(['p2','p3'])` 实际返回旧队列的 `[['p0'],['p1']]`)。
>     根因修:`ContinuousRolloutItem` 新增 submit 时刻戳的 `prompt_set_id`;`ContinuousRolloutSchedule` 持有单调
>     generation id,prompt 换批时 +1 并经 `update_prompts(prompts, prompt_set_id=...)` 下发;
>     `RolloutScheduler.select_iteration(..., prompt_set_id)` **只选当前 set** 的 items(同时不混版本、不混 prompt
>     set)。复核后补上第二个死锁边界:旧 set ready items 如果留在队列里会占满 capacity,让新 set admission
>     因 `item_budget_full` 卡死;现在 prompt swap 时 scheduler purge obsolete ready items,in-flight 旧 set 结果
>     receipt 时丢弃。回归测试 `test_select_only_serves_the_requested_prompt_set` /
>     `test_prompt_set_update_purges_old_ready_items_before_wait`。
>   - **[Med] admit observability 未进 metrics.csv**:`continuous.predicted_admit_staleness` /
>     `continuous.admit_blocked_on_staleness` 此前只在 phase_times,固定列 CSV writer 没写。补 `online.py`
>     header + row 映射 + 写列,并扩 `test_online_metrics_csv_includes_logprob_mismatch_metrics` 断言两列。
> - **仍 parked(§5 P2 的更大子步)**:真 wall-clock 重叠 —— 让生成编排不与训练共用同一线程/事件循环跑同步
>   backward(独立线程 / cosmos 形状的独立进程 rollout owner)。文档原文「单列评估,不在本轮」,本次未做:它需要
>   ≥2 卡验证收益且改动最大,作为独立 PR。统一调度器(P2 主体)已落地。

---

## 1. 核心结论 (TL;DR)

**当前 continuous 调度在默认配置下"名义异步、实际串行"。三个原因叠加:(a) 单事件循环 + 同步
backward 不可能真重叠;(b) 每步全 drain 的 weight-sync barrier 把在途生成全部围栏掉;(c) 默认
`max_stale=0`,每次同步权重就让整个预取队列失效。** generation 确实跑在独立 Ray actor 进程上、
*本可以*重叠,但被上面三点锁死成"先生成窗口、再训练窗口"。

关于你问的"2 queue 对不对":**两段结构(in-flight set + ready queue)本身是对的、合理的**——它们是
两个不同生命周期阶段的不同对象。你"感觉怪"的直觉指向的是另外两件真问题:**(1) 容量记账横跨两个
结构、两个 owner 会打架;(2) 没有任何一个组件拥有整条流水线**——admit/staleness 决策散落在
producer/queue/consumer/schedule/trainer 五处,且 staleness 只在 select 时**事后丢弃**,从不在 admit 时
**事前节流**。这正是 cosmos-rl 单控制器(Controller)补上的那个洞。

---

## 2. 三个发现(带证据)

### 2.1 真重叠?—— 默认下 overlap 上限≈0(串行)

driver loop 是**单线程 asyncio**:

```
trainer.step -> collect_training_batch:  await next_iteration   # 事件循环可跑,producer admit/harvest
             -> train_on_rollout_batch:  for ppo×microbatch×timestep { self._backward(loss); await asyncio.sleep(0) }
             -> after_train_step:         pause -> drain ALL inflight -> sync -> resume
```

- `self._backward(loss)` 是**同步** CUDA 调用(`trainer.py:855-856`,`strategy.py:69-73` `loss.backward()`),
  整个 backward 期间**独占线程**,producer 一步都进不了。
- 唯一让步点是每个 timestep 的 `await asyncio.sleep(0)`(`trainer.py:876`,注释自己写了"yield once per
  timestep so producer admit/harvest can progress")——但它只在两个 backward **之间**的亚毫秒缝隙里让
  producer 跑,**不能跨越一个 backward**。
- generation 确实在**独立进程**的 Ray actor 上(`generation/ray/launcher.py:83-92`
  `RayActorGroup.launch`,`executor.py` `execute_chunk.remote` + `await ref`)——所以*物理上*能重叠;
  但单循环 + 每步 drain-to-zero 把它围栏掉。
- 而且**默认 `mode='strict_on_policy'`**(`trainers/core/types.py:124`),continuous 默认根本没开。
  即使开了,默认 knob(`max_inflight_groups=1, max_ready_groups=2, max_stale=0`)下,重叠上限也只是"一个
  在途 group 的远程算力漏进 sleep(0) 缝隙",远不是 generate‖train 流水。

**结论:默认串行;`sleep(0)` 做的是本该由"独立线程/进程的生成 owner"做的事。**

### 2.2 weight-sync barrier —— 每步全 drain(视频上每步停几分钟)

`after_train_step`(`schedule.py:115-119`)= `pause_admission → drain_inflight → sync → resume`;
`drain_inflight`(`producer.py:122-132`)= `await asyncio.gather(*所有 inflight)`。也就是**权重同步要等最慢
的在途生成跑完**;视频 clip ~5-6 分钟/条 → **每步停数分钟**(reading 里点名的 vrl P1-1,
`SPRINT_framework_lessons_vrl.md:100-117`)。

drain 的理由是正确性(权重在请求中途变会让一条生成混两个策略版本,`producer.py` 注释写明)。**但它比
不变量更严**:`generation/execution/worker.py:115-129` 已经做了 **chunk 级版本拒绝**——所以换权重只需在
**chunk 边界**换,不必等**整条请求**完。

cosmos-rl 证明 drain 没必要,三件套(都可对标):
1. **独立 CUDA stream + 双缓冲 shadow model**:`WeightSyncThread`(`cosmos_rl/rollout/worker/weight_sync.py:265-294`
   自带 `self._stream = torch.cuda.Stream()`,`:371-413` 在该 stream 上 NCCL recv/broadcast 进 buffer,live
   model 不动),`:178-234` `sync_buffer_to_live` 在生成边界用 `inf_stream.wait_event(...)` 跨流定序后 copy。
2. **生成中途抢占**:patch `llm_engine.step` 每 N 步 `consume_command`(`vllm_rollout.py:93-119`),
   靠 `enable_prefix_caching=False`(`:319`)保证换权重后没有依赖旧权重的 cache 残留。
3. 传输 P2R/R2R(`collective/collective.py`)。

**vrl 已具备前置条件(submit 时戳版本 + StalenessPolicy + worker 级 chunk 版本拒绝),所以这不是重写,
是把"换权重的粒度"从 request-complete 放松到 chunk-boundary,让 StalenessPolicy 吸收混版本的尾巴。**

### 2.3 staleness 默认 0 —— 异步是摆设

`schedule.py:44-46` 默认 `max_stale_policy_versions=0`;`staleness.py:38-44` `admit()` 只收 `version==current`
(`0<=staleness<=0`)。配合每步 barrier drain,**每次 bump 版本就把整个 ready 队列判定为过期丢掉**
(`queue.select_iteration` 先 `drop_too_stale`)。所以预取缓冲永远用不上 → 每步等新鲜生成 → 串行。

**只有把 `max_stale_policy_versions >= 1` 才真正开启 off-policy 预取**(`max_ready_groups`/`max_inflight`
单独调大没用,admit 仍拒绝旧版本)。

---

## 3. 你"感觉怪"的那两段结构:对在哪、错在哪

| | `producer._inflight` | `ContinuousRolloutQueue` |
|---|---|---|
| 类型 | `set[asyncio.Task]` | `deque[ContinuousRolloutItem]` |
| 阶段 | 在途生成(WIP) | 生成完待训(done) |
| 带的语义 | 无(没版本/没排序/没 staleness) | 有(版本戳、homogeneous-version select、staleness、byte 背压、drop policy) |

**拆成两段本身是对的**:在途 task 还没有"落定的版本/字节/可丢弃性",对它谈 stale/drop/select 没意义;硬
塞进一个结构才是错的。你的不适感其实指向两件真问题:

1. **容量记账横跨两个 owner、会打架。** `_admit`(`producer.py:152-156`)用
   `len(inflight)<max_inflight_groups` **且** `queue.size()+len(inflight)<capacity` 两个计数器管"总在途
   工作量";而 queue 又自己独立执行 `max_items`/`max_bytes`(`queue.py:187-199`)并可能 **silently
   drop_backpressure** 掉 producer 以为已收下的 item。**admission 和 eviction 由两个会分歧的 owner 管。**
2. **没有单一生命周期 owner。** `prompts→admit→inflight→harvest→ready→select→train→weight-sync` 散在
   schedule(编排+barrier)、producer(admit/inflight/harvest)、queue(ready/staleness/drop)、
   consumer(select/build)、OnlineTrainer(外层循环)**五处**。没有任何组件看到整条流水线;staleness 只在
   consumer 的 select **事后丢**,producer 从不在 admit 时问一句"生成已经领先这么多了,这个 prompt 还该不该发"。

---

## 4. 目标:统一的全局调度器(对标 cosmos-rl Controller)

cosmos-rl 的单 `Controller` 就是 vrl 缺的那个"全局更大更准确的 scheduler":

- **admit 时戳"预测版本"**:`weight_version = current_step + total_pending_rollouts()//rollouts_per_global_batch`
  (`dispatcher/controller.py:273-276`)——发任务时就知道它会落在哪个权重版本,**事前**就 staleness-aware。
- **软 + 硬节流**:软 `allowed_outdated_steps`(`controller.py:291-303`)、硬 `max_inflight_steps` 返回空批直接
  stall 生成(`controller.py:348-352`)——**事前**背压,不是事后丢。
- **per-replica 状态机**(READY/RUNNING/REDUCED,`status.py:98-148`)+ **统一触发**
  `try_trigger_data_fetch_and_training`(`status.py:1362-1456`)接在 `trigger_weight_sync` 之后。

**给 vrl 的目标形状:一个 `RolloutScheduler` 组件**,把现在散落的 admit/inflight/ready/select/trigger 收进
一个 owner,具备:
1. **单一"在途工作量"预算**(取代横跨两计数器 + queue 独立 caps 的记账),admission 与 eviction 同一个 owner。
2. **staleness 上移到 admit**:戳预测的训练后版本,预测漂移超界就**不发**(主动硬节流),取代当前的
   reactive drop-at-select。
3. **async 双缓冲 weight sync**:换权重在 chunk 边界(复用 worker.py:115-129),与生成重叠,**去掉全 drain**。
4. **打开 staleness 窗口**(`max_stale>=1` + 更深队列):一步训预取集、同时下一集在生成 → 真正跨步流水。
5. (更大)**真 wall-clock 重叠**要让生成编排不与训练共用同一线程/事件循环跑同步 backward——要么 producer
   loop 独立线程,要么走 cosmos-rl 的独立进程 rollout owner。这是最大的一步,放最后。

调度 item 是 rollout group / ready rollout batch / policy-versioned producer work，不是 training
microbatch。不要把 `_run_streaming_optimizer_update` 改成 microbatch prefetch 来冒充 continuous async。

---

## 5. 分阶段计划(先小杠杆,后大重构)

**P0 — 先让默认配置别自欺(零风险,改默认/文档)**
- 把"continuous 默认等价 strict-on-policy"写进配置注释/日志告警:开 continuous 但 `max_stale=0` 时打一条
  "no off-policy prefetch in effect"提示,避免误以为在异步。
- 加可观测:把 §2.1 的"backward 期间 producer tick gap"暴露成 metric(已有 `producer_max_tick_gap_s`),
  让"是否真重叠"可量。

**P1 — 两个真痛点(中等,各自独立 PR)**
- **async 双缓冲 weight sync**:replace 全 drain → shadow model + chunk-boundary swap(对标
  `weight_sync.py:178-234,265-294`)。验收:视频每步的 `continuous.weight_sync_pause_s` 从分钟级降到秒级。
- **打开 staleness 窗口**:量出 GRPO 能吃几步 stale(一般 1-4),设 `max_stale_policy_versions>=1` +
  `max_ready_groups` 加深;验收:`continuous.queue_ready_groups` 在训练步间不再清零。

**P2 — 统一调度器(大,可选,先验证 P1 收益再决定)**
- 抽 `RolloutScheduler`:单一在途预算 + 事前预测版本 admission + 统一 trigger,收编 producer/queue/consumer
  的记账。**不是从零重写**:复用现有 queue 的 select/staleness、producer 的 Ray dispatch、consumer 的 build;
  只把"决策权"集中。
- (更大)评估独立线程/进程的 rollout owner(cosmos-rl 形状)以拿到真 wall-clock 重叠——单列评估,不在本轮。

---

## 6. 非目标

- **不从零重写 continuous 调度。** 现有 producer/queue/consumer 的对象模型基本正确(两段结构合理);改的是
  记账归属、staleness 时机、weight-sync 粒度。
- **不改 strict-on-policy 默认的安全性。** 全 drain 是 text/短生成下正确且廉价的保守默认;P1 只在 video 长
  生成路径上放松。
- **不照搬 sglang-omni 的 stage 流水**——那是 serving 内部一次前向的拆分,与 RL 生成/训练调度无关。
- **不在没量出 staleness 容忍度前**就把 `max_stale` 拍一个大值。
- **不做 microbatch/minibatch async。** `microbatch_size` 继续只表示同步内存切片和梯度累积；本 sprint 不在
  `_run_streaming_optimizer_update` 内实现 prefetch，也不把 continuous consumer 边界降到 microbatch。

---

## 关键文件引用

- vrl 调度:`vrl/rollouts/orchestration/continuous/{producer.py:122-179, queue.py:109-199, consumer.py:71-97, schedule.py:44-46/115-119, staleness.py:38-48}`
- driver loop:`vrl/trainers/online/trainer.py:437-441/855-876/948`、`vrl/trainers/strategy.py:69-73`、`vrl/scripts/common/online.py:347-368`
- 生成进程边界:`vrl/generation/ray/launcher.py:83-92`、`vrl/generation/execution/worker.py:115-129`(chunk 版本拒绝)
- 默认值:`vrl/trainers/core/types.py:99-124`
- cosmos-rl 对标:`cosmos_rl/dispatcher/controller.py:273-276/291-303/348-352`、`dispatcher/status.py:98-148/1259-1264/1362-1456`、`rollout/worker/weight_sync.py:178-234/265-294/371-413`、`rollout/vllm_rollout/vllm_rollout.py:93-119/319`
- 既有诊断:`docs/sprints/reading/SPRINT_framework_lessons_vrl.md:100-117`(P1-1)、`docs/sprints/reading/cosmos-rl.md`
