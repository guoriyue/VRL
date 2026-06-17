# SPRINT: 跨请求 step 调度器（统一推理引擎的终局形态）

状态：**parked / direction-decided（2026-06-16）**。这是对"blocking stage + inference
for all models from scratch"这个框架级问题的方向裁决。结论：**终局形态是一个 family-neutral
的跨请求 `StepScheduler`（Angle C），但它现在不做**——它优化的不是当前瓶颈，且当前硬件
（单卡 `num_workers:1`）跑不出它的收益，而决定项目成败的"RL 到底学不学"还没解。

**触发事件（两条都满足才解 park）**：
1. 真实 recipe 证明 GPU 在 forward 期间被填不满（小 per-request batch，如 wan sbs=1 / AR
   token-decode 的 underutilization），**且**存在真正的 workload 异构（多卡 rollout bring-up，
   或 serving 式混合负载）——即"有第二个请求可拼"。今天 `num_workers:1` + 每个 config
   `ar_scheduler_batch_size: null`，不成立。
2. 主导空闲**不再是** weight-sync drain / actor relaunch（`schedule.py:110-120` 全局 drain
   barrier、`runtime.py:95-175` 每周期重启 actor 才是实测几分钟级停顿，是 P0/P1 项，性价比碾压）。

唯一允许提前做的：下面 §4 的 **AR-first 探针**，它便宜、可逆、能证伪——但其结果**不允许**把
多阶段 ladder 拉到 RL 信号和 P0 停顿之前。

---

## 1. 为什么是 Angle C（不是自研 forward，也不是接通 stage pipeline）

根因只有一个：`EnginePlanner.build()` 每请求产出一个**不可变 EnginePlan**
（`vrl/generation/execution/planner.py:100`），`RayGenerationExecutor.execute(request)` 一进一出
（`vrl/generation/ray/executor.py:43`）——**两个请求永远不能共享一次 forward**。

- **Angle A（自己拥有 transformer forward / native executor）**：最底层承重件，但回报最靠后
  （拥有 forward 之后还要再叠 kernel/block-causal 才兑现），且触发条件最不成立——"attention 占
  34%"是 **fp32 老数据**，bf16+`torch.compile`（31f6843，1.25–1.37x，inductor 自带 fusion）上线后
  从未重测。现在自研 forward 去打一个可能已被 compile fuse 掉的瓶颈 = 镀金。
- **Angle B（接通休眠的 `vrl/generation/pipeline/*` stage 契约层）**：启动最便宜，但 Phase-1 是
  **数值惰性搬运**（把同一个 fused 调用塞进 payload 契约，trajectory 逐位不变），真正的 win 需要
  和 C 的 Phase-3 完全相同的"可恢复 denoise 循环"手术——绕远路到同一终点。
- **Angle C** 打的就是 per-request 边界本身，且**已 90% 存在**：

  | 已有原语 | 路径 | 作用 |
  |---|---|---|
  | `TokenScheduler` 按 position 分组弹 batch | `vrl/generation/ar/decode_loop.py:105` | **这就是 continuous batching，只是今天只喂单请求** |
  | `ARCacheRows.gather/scatter` 行索引批状态 | `vrl/nn/layers/attention/cache_rows.py` | 跨请求合并状态可原样复用 |
  | `ARSequenceKey`（family/task/tokenizer/dtype/max_new_tokens） | `decode_loop.py:20` | 现成的跨请求分组 key |
  | `batch_group_key` / `batch_signature()` / `capability_key` | `planner.py:65` / `capabilities.py:179` / `types.py:133` | **算好存好、零消费者的孤儿 key**——调度器是它们的第一个读者 |
  | runner 契约 `init_ar/step_ar/finalize_ar`、diffusion `encode/prepare/forward_step/decode` | `janus_pro/runner.py`、`models/diffusion/base.py` | family 插件接缝，不动 |

---

## 2. 现状成熟度（来自 12-agent map，全部经路径核对）

- 执行核 = **contract-only**：planner 有 `ResolvedAxis(batchable/chunkable)` / `ExecutionStage`
  （含 `cache_read/cache_write` 槽位但**无任何 KV store/lookup 实现**）/ `SampleChunk` 调度，但
  staging 严格**请求内、chunk 内**；唯一动态适配是 OOM 时 chunk 减半（`chunks.py:153-180`）。
- stage pipeline 契约层 = **contract-only，零生产消费者**：`PipelineTopology/SerialPipelineRunner/
  RayPipelineStageWorker`（12 测试过）休眠;diffusion executor 已把工作拆成 typed stage
  （`run_prompt_encode_stage/run_prepare_stage/run_denoise_stage/run_decode_stage`,
  `diffusion/executor.py:437-454`),但被同一个 fused 串行协调器**内联消费**,没走 topology。
- native transformer executor（Angle A）= **none**：所有 family 仍 `self.transformer = diffusers
  对象`，唯一接触点是 `DiffusionBackboneCaller._call_transformer`（`backbone.py:152`）。

---

## 3. 阶段阶梯（每阶段独立可逆、藏在 flag 后、永不碰 trainer/RL 层）

- **Phase 1（唯一近期项，详见 §4）**：AR 两请求合并探针。
- **Phase 2**：AR 调度器后挂**共享 block pool + free-list**——今天 `free()` 是 `pass`
  （`vrl/nn/layers/attention/paged.py:202`，单请求够用、共享池致命）。vLLM/SGLang 的复杂度都花在这，
  独立 gated 阶段。
- **Phase 3**：泛化到 diffusion，**仅当**重测证明异构存在（多并发请求共享 resolution bucket + step_idx）。
  把 `run_denoise_steps`（`diffusion/executor.py:633` 的闭合 `for step_idx`）重构成 `advance_one_step`,
  藏在 Phase-1 parity harness 后,保 trajectory 写入逐位不变(`executor.py:723-746`)+ 不跨 policy version
  不变量(`worker.py:115-129` 已强制)。
- **Phase 4–6**：AR/diffusion 合一个 `pop_batch()`;从 continuous producer 队首 feed
  `RayGenerationExecutor.execute`;暴露 serving。

⚠️ **诚实**:diffusion 收益是**条件性的**——reading 文档否决 diffusion 的跨步 KV 复用;跨请求 step
合并只在并发请求共享 resolution + step_idx 时才赢,若 RL recipe 单分辨率,diffusion 收益退化成已有的
请求内 batching。**AR 的赢稳,diffusion 的赢看 workload。所以以 AR 切入。**

## 4. AR-first 探针（唯一现在可做，一次性 spike）

把 `TokenScheduler` 的 pool/group/pop/push-back（`decode_loop.py:105-157`）抽到
`vrl/generation/scheduler/step_scheduler.py`,喂**两个并发 janus_pro 请求**,用现成 `ARSequenceKey`
当 key,`ARCacheRows` 原样复用;藏在 `ARChunkExecutorBase._ar_runner` 一个 config flag 后,family 不动。

**通过 = 两条都满足**:
- (a) finalized tokens 与单请求路径**逐位相同**（先断言这条,RL 轨迹不能变,去风险）;
- (b) 两请求合并时**实测 SM 利用率上升**。

**SM 没升 → 停**:赌注不成立,你用一次提取的代价买到了答案。`SPRINT_runtime_block_policies` 是反面
教材(b16 比 b8 慢 3.9%,已撤回)——**不跳过实测闸**。

## 5. 排序（相对 RL 信号,无条件靠后）

引擎在 forward 期优化一个还产不出有用梯度的 loop = 负价值。正确顺序:
**(a) 修 RL regime 拿可信学习曲线（diffusion-loss 正则进 GRPO、更多 inner step、全参——纯 trainer
工作,引擎零参与）→ (b) 落 P0 KL CPU-swap（`online.py:74-80`,不修 KL 静默消失）+ NCCL weight-sync
（`weight_sync.py:56-62`,顺带解锁全参）→ (c) P1 消停顿 → (d) 引擎。**

## 6. 明确不要做
- 不现在自研 forward（A）——除非新 bf16+compile profile 证明 attention 仍是 inductor 关不掉的瓶颈。
- 不把 `vrl/generation/pipeline/*` 提升成生产路径当政绩(Phase-1 数值惰性搬运)。
- 不跳过实测 SM 闸。
- 不为 diffusion 建 paged-KV 当目标(跨步 KV 复用已被否决)。
- 不 big-bang——每阶段藏 flag、可逆、不碰 trainer。

## 关键文件引用
- `vrl/generation/execution/planner.py:100`（per-request EnginePlan）、`vrl/generation/ray/executor.py:43`
- `vrl/generation/ar/decode_loop.py:105,20`、`vrl/nn/layers/attention/cache_rows.py`、`paged.py:202`
- `vrl/generation/capabilities.py:179`、`vrl/generation/execution/types.py:133`、`planner.py:65`
- `vrl/generation/diffusion/executor.py:633,437-454,723-746`、`vrl/generation/diffusion/backbone.py:152`
- `vrl/generation/execution/worker.py:115-129`
- `vrl/rollouts/orchestration/continuous/schedule.py:110-120`、`vrl/generation/ray/runtime.py:95-175`
- `vrl/scripts/common/online.py:74-80`、`vrl/trainers/weight_sync.py:56-62`
- 相关 parked：`SPRINT_generation_scheduler.md`（chunk 派发层,不同层）、`SPRINT_physical_stage_runtime.md`
  （Angle B 物理管线）、`SPRINT_diffusion_native_transformer_executor.md`（Angle A）
- 方向研究：`docs/sprints/reading/SPRINT_framework_lessons_vrl.md`、`reading/{vllm,sglang,slime}.md`
