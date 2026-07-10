# SPRINT: Diffusion rollout stage pipeline

> # ⭐ 架构定论（2026-06-28 最终,实测驱动）：chunk == minibatch(sbs);max-sbs 是真赢;step-streaming 比 chunk-pipeline 慢 → 留 chunk-pipeline、砍 step-streaming
>
> **chunk 就是 minibatch,不要额外抽象。** 一个 "chunk" = 一次 denoise 的 sbs 个样本。N 个样本 → `ceil(N/sbs)` 个 minibatch。OOM → 减小 sbs(= 减小 minibatch)→ 自动更多 minibatch。装得下时 sbs=N → 1 个 minibatch(实测 `sbs=8,n=8 → chunks=1`)。这是处理 N 样本的固有机制(+ 多卡分发单位 + OOM-retry),不是可去掉的抽象层。
>
> **唯一干净确立的赢 = max sbs(1 minibatch)= ~1.95x**(sbs 1→8 阶梯,见 [[project_diffusion_multigpu_dataparallel]])。两个 copy-hiding pipeline 都是小效应(~几 s on 65s),实测对比反复被 cold/warm 编译态污染。
>
> **两种 copy-hiding(都藏 trajectory D2H,粒度不同):**
> - **chunk-pipeline**(`forward_chunks_pipelined`/`execute_request_pipelined`,粒度=整 minibatch):末尾一次 bulk D2H 藏到下个 minibatch 的 denoise 后。需 ≥2 minibatch。开销小。
> - **step-streaming**(`sampling.stream_trajectory_copy`,`_TrajectoryCopyStreamer`,粒度=单 denoise step):loop 内逐步 D2H。1 minibatch 也生效,但**每 chunk 35 步 ×~7 buffer 的 async copy + event + 整条 trajectory 的 pinned 分配 = 逐步开销大**。
> - **实测(sbs=4,release triton 3.6.0):chunk-pipeline 64.965s vs step-streaming 77.6s(扣冷编译后 ~72)→ step-streaming 慢 ~7s。** 早先"step-streaming subsumes chunk-pipeline"是错的(reasoning 跑在 data 前):它的逐步开销 > 它藏的 copy。
>
> - **决定(最终):** ① **chunk == minibatch(sbs)**,保留核心 minibatch 处理 + OOM-retry + gather(删不得)。② **单卡最优 = sbs=max-that-fits(趋 1 minibatch)** —— 这是真赢。③ **留 chunk-pipeline**(多-minibatch 兜底时藏 inter-minibatch copy,实测有效、开销小、不碰 hot loop)。④ **砍 step-streaming**(`sampling.stream_trajectory_copy` + `_TrajectoryCopyStreamer` + denoise-loop 集成 + 其测试)—— 碰 hot denoise loop、逐步开销大、实测更慢,是 session 里基于错前提加的过度设计。保留 generation-wall 计时器 + forward_plan 4参 bug 修复。
> - 注:copy-pipeline 的效应小且测量受编译态干扰;若要更确,需同编译态 warm-vs-warm 重测。但方向(max-sbs 主赢、step-streaming 不值)够清楚。
>
> ## ⭐ 多卡 rollout-专用卡的 idle:已有 vs 待建(2026-06-28,代码核实)
>
> **rollout 专用卡只要 idle 就是纯浪费。idle 分两类,大小差很多:**
> ```
> 大 idle = 这批生成完、等【训练步+权重同步】跑完才生成下一批（strict 下 rollout 卡空整个训练步)
> 小 idle = 一批内 per-chunk 的 encode/copy/pack bubble + 每 group 最后一个 chunk 的 tail copy
> ```
> **代码核实(`vrl/rollouts/orchestration/continuous/producer.py` docstring + `ray/executor.py`):**
> - **大 idle → continuous 模式已能填(✅ 已有)**:producer background 保持 `max_inflight_groups` 个 collect 在飞,生成 group N+1 时训练在跑 group N —— "enough to overlap rollout with training on a cross-node setup"。这是 rollout 专用卡不空的第一要务,现成的。**前提:部署用 continuous + rollout/train 分卡(strict+单卡 dev 没有这个填充)。**
> - **小 idle → 未自动处理:**
>   - **② per-worker 流水(每张卡内部藏自己 chunk 的 copy)= ❌ 未建。** chunk-pipeline 仍 gate 在 `len(workers)==1`(executor.py),多 worker 时每张卡 per-chunk 串行、不藏。机制 = 把现有 chunk-pipeline 放开成 per-worker(每个 worker 跑自己那批 chunk 的 `forward_chunks_pipelined`)。**值(多卡+大 shape+sbs=1,每 worker 多 chunk → ~33% reclaim,见 [[project_real_run_profiling]] 的 36% idle/33% 编排 gap);但 1 卡验证不了,推迟到多卡。**
>   - **③ "continuous 把 chunk 连成流 → tail 被下一批自动藏" = ❌ 我的概念错,不是这样。** 每个 group 是独立的 collect/execute(各自 dispatch+gather+tail);continuous 是 group 层 generate∥train 重叠,不是 worker 内连续 chunk 流。要藏跨-group tail 得做 worker 级跨-group 流水 = 大改、只为一个 tail copy → **砍,不建。**
>   - **cross-worker(不同卡 A 的 copy ∥ B 的 denoise)= 无意义**:不同 GPU 本来并行,藏了不让任一卡更快。要填的是"同卡上永远有下一个 denoise"(continuous 下一批 + per-worker 流水)。
>
> **决定:** ③ 砍。② 推迟到多卡 + 测量 gate(上多卡先测 per-worker GPU-idle:~33% 才建 per-worker 流水,~few% 跳过)。现在 1 卡不建任何一个——单卡杠杆已完成(max sbs + chunk-pipeline);大 idle 靠 continuous(已有),多卡 denoise 吞吐靠 data-parallel(已有)。
>
> ---
> # ⭐ NEXT-STEP 方向（2026-06-27 复核版。这一段是当前 plan，覆盖下方旧 §）
>
> **修正:sglang-omni 不是 AR-only——它是异构-bottleneck 多模态服务框架**(compute-bound thinker + memory-bound talker + latency-sensitive codec;`~/Desktop/sglang-omni` README、`StageConfig`、`SimpleScheduler`、relay/{shm,nccl,nixl}、same-GPU stream target / `fused_stages`)。核心思想:**每个 stage 按自己的 bottleneck 调度;不同 bottleneck 的 stage 只有放到不争同一硬件的 placement 上才真正并发;同卡优先 fuse / same-process / CUDA IPC,跨卡优先 NCCL/NIXL GPU P2P。**注意:默认 `shm` relay 仍可能走 CPU 参与的数据路径,所以不能把"跨卡一定不是 GPU→CPU→GPU"写成无条件事实。早先把它说成"AR 专属 / transport=CPU-bounce / stage-runtime 死脚手架"是错的;可借的是 stage runtime 形状,不是照搬 serving stack。
>
> **实测校准过的 diffusion 事实:**
> - denoise = compute/tensor-bound（NCU 43%≈bf16 天花板)。
> - copy + CPU orchestration = rollout 的 33%(实测),bottleneck 与 denoise 不同 → **单卡能藏(实测 compute∥(copy+CPU)=20%)**。← 单卡 stage 重叠的大头。
> - VAE decode ~5%、text encoder ~1%(cosmos):合成测太小没测清是否 memory-bound;即便重叠也小(SD3.5 的 T5 encoder 大,那边值)。
> - denoise 吞吐:denoise 占 58-94% → 把 denoise 单独切一卡,另一卡只 ~19% 活、被 denoise 卡住(~1.2x)。**denoise 的【吞吐扩展】该 data-parallel(每卡跑完整 rollout、分 chunk,近线性),不是 stage-split。**
>
> **修正后的目标形态 = 混合(不是"纯 stage-split"也不是"REJECT"):**
> ```
> ① 同卡:  fuse 生成 stage（encode+denoise+decode）+ I/O 流水重叠 → 藏 copy+CPU(33%)  ← sglang-omni fusion 模式
> ② 跨卡:  reward 切专卡（已建 async-reward enabler: reward.gpu_pool=dedicated)→ 藏 14%   ← per-stage placement + nccl/nixl
> ③ 吞吐:  denoise 用 data-parallel 复制扩展(denoise 主导,stage-split 扩不动 denoise)
> ```
>
> **NEXT STEPS（增量、单卡可验、按序）:**
> 1. **[1卡] 同卡生成-stage I/O 流水**:目标 `produce(N+1) ∥ teardown+copy(N)` on side stream + 延迟 sync,藏 copy+CPU。借 sglang-omni 的"同卡 fuse"——别为 decode/encode 拆独立 actor。
>    - **✅ 1a 机制已落 + bit-exact(2026-06-27)**:`forward_chunks_pipelined`（pipeline.py)——produce(N+1) 默认流 ∥ teardown(N) copy 流 + `Event` 防 torn-read + 单一 join。`tests/generation/pipeline/test_chunks_pipelined.py` 4 测试证 pipelined 结果逐位=串行、chunk 序、produce/teardown 各一次（共 21 passed）。
>    - **⚠️ 1a 实测:copy 本身藏不出大头。** 单卡量(produce=DiT denoise 1.68s/chunk + D2H trajectory)：SERIAL 6735ms vs PIPELINED 6729ms = ~0%。copy 相对 denoise compute 太小（~几%）。
>    - **🎯 1b 大头定位(关键):那 30s Python orchestration 是 nsys 在 WORKER 进程抓到的**（`forward_chunk_plan` 的 Python + `worker._to_cpu` + 打包），**不是 driver dispatch、不是 copy**。`max_inflight_chunks_per_worker=2` 没用——`RayGenerationWorker` 是 **sync Ray actor**(`def` 非 `async def`,worker.py:17)→ 一次一 task,不内部并发。要藏 worker 侧 30s = **改 dispatch 成 per-request + worker 内部用 `forward_chunks_pipelined`**。
>    - **✅ 1b executor 入口已落**:`DiffusionExecutor.forward_plan_pipelined`（executor.py，用 `forward_chunks_pipelined` + 同一 `gather_chunks`，bit-exact by composition）+ 顺手修了死的 `forward_plan` 4参 bug 当串行基线。185 tests passed,**gate 默认关 = live rollout 行为零改动(安全)**。
>    - **✅ 1b Ray 生产路径已接好 + version-safe + tested(2026-06-28)**:
>      - `GenerationWorkerCore.execute_request_pipelined`(worker.py)——**对整个 request 复制 `execute_chunk` 的版本安全**(slot 模式 `has_trainable_state`→`activate_trainable_state`;evicted→`StaleSlotDiscard`(graceful discard,不训 off-policy);非-slot version mismatch→raise)然后调 `forward_plan_pipelined`。`tests/generation/execution/test_execute_request_pipelined.py` 5 测试锁住:stale 不运行、mismatch 不运行、通过才运行。
>      - `RayGenerationWorker.execute_request_pipelined`(actor 暴露)→ core。
>      - `RayGenerationExecutor`:加 `pipelined` flag(默认 False)+ `execute()` 在 `pipelined and len(workers)==1` 时走 `_execute_request_pipelined`(否则原 per-chunk 路)。`tests/generation/ray/test_oom_split.py` 3 routing 测试:单worker→走 pipelined、多worker→回退 per-chunk、默认→per-chunk。
>      - config 接好:`distributed.rollout.pipelined`(config.py 解析 + launcher.py 传参)。
>      - **290 generation/ray/continuous tests passed,默认 off = live rollout 零回归(安全)。** OOM:pipeline depth=1(峰值≈2 chunk),仅当 2 chunk 装得下才开;version-stamp request 级。
>    - **✅ 1b 真 cosmos A/B 已跑(2026-06-28,1×5090,clean generation-wall timer in `ray/executor.py`,warm compile 两边一致):**
>      - **合理 shape = 240p_33f(416×240×33,真实 cosmos config shape),8 samples,sbs=1 → 8 chunks,Kling reward:**
>        ```
>        FALSE (per-chunk dispatch):  generation wall 130.19s（warm exec 15.6s/chunk × 8 串行;cold 首 chunk +5.3s)→ warm-equiv ≈124.8s
>        TRUE  (per-request pipeline): generation wall 80.07s（warm),无 OOM（240p 流水装得下 1 卡)
>        → ~1.56x / 省 ~36%（~45s on 8 chunks)
>        ```
>      - **分解:** warm per-chunk 15.6s ≈ denoise ~10s(GPU,藏不了)+ overhead ~5.6s(trajectory `_to_cpu` + Ray 序列化 + 8× 串行 dispatch + 打包,可藏)。pipelined 把 ~5.6s/chunk overhead 藏到下个 denoise 后 → 有效 ~10s/chunk。**denoise 是地板,overhead 是省下的。**
>      - **正确性(真 run 确认):** tiny-shape(256×256×9)那次完整 run `metrics.csv`:`group_size=8.00` + `reward_mean=-4.9158` + `logprob_abs_diff=0.000412`(on-policy)→ 真 8 样本不丢活;240p run 计时器 log `chunks=8` 无 OOM;同一代码路 + 22 bit-exact/版本安全单测。
>      - **shape 趋势(诚实):** overhead 大致固定、denoise 随 shape 涨 → shape 越大赢越小。240p_33f=**36%**;tiny 256×256×9(denoise 仅 ~3-4s)被放大到 ~2-3x(不代表生产);更大 704p_93f denoise 主导 → 赢更小,且 704p 流水在 1 卡 OOM(只能多卡)。
>    - **⚠️ 关键修正(2026-06-28):sbs(sample_batch_size)是更好的单卡杠杆,DOMINATES pipelining。** 同 240p_33f/8-sample,generation wall warm-equiv:
>      ```
>      sbs=1 per-chunk:      124.8s  (15.6s/sample,8 dispatch)
>      sbs=1 + pipeline:      80.1s  (1.56x — 只藏 overhead,denoise 地板没动)
>      sbs=4 per-chunk:       71.6s  (1.74x — 批 denoise:MFU↑ 15.6→8.95s/sample + overhead 少)
>      sbs=4 + pipeline:      64.9s  (sbs 批 + 藏 chunk0 teardown — 比 sbs=4 单独快 6.7s)
>      sbs=8 per-chunk(1块):  ~64s   (1.95x — 顶满批:最佳 MFU + 零 per-chunk overhead)
>      ```
>      **两条杠杆正交,可叠加,但在显存允许顶满批的 shape 上汇聚到同一天花板:** sbs 批 denoise 降地板(MFU),pipeline 藏 teardown overhead。`sbs=4+pipeline=64.9s` ≈ `sbs=8=64s`(打平)——sbs=8 用纯批(1 块零 overhead)到顶,sbs=4+pipeline 用半批+藏到同一顶。**结论 = 组合是通用最优,decision rule:**
>      ```
>      ① 顶满批(1 chunk)装得下 → 用它（240p:64s,最优,pipeline 无 chunk 可藏=自动 no-op)
>      ② 1 chunk OOM 但 2 chunk 装得下 → 用能装 2 chunk 的最大 sbs + pipeline（藏 teardown → 逼近 ① 的顶)
>         ← 这才是 pipelining 真正赚钱的窗口:显存逼小批的【生产大 shape】
>      ③ 连 2 chunk 都 OOM(512p93f sbs=1)→ 单卡无解 → 多卡 data-parallel
>      ```
>      所以早先"pipelining 单卡窗口很窄/被 sbs 支配"要修正:**在显存允许顶满批的 shape(240p)上 pipeline 确实多余;但在【1 块 OOM、2 块 fit】的内存受限大 shape 上,sbs+pipeline 是单卡能达到的最优,pipeline 在这里真加值(把受限小批拉回接近满批的顶)。** pipeline 代码(正确+version-safe+默认 off)的定位 = 与 sbs 组合的内存受限优化 + 多卡 stage 地基。
> 2. **[已补,待多卡量] async-reward recipe / 正确性**:`configs/experiment/diffusion/cosmos_predict2/online_grpo_async_reward.yaml` 已把 cosmos continuous + `reward.gpu_pool=dedicated` 组合起来;`tests/ray/test_resources.py::test_cosmos_async_reward_recipe_resolves_resident_reward_overlap` 锁住 resolver 行为;`tests/rollouts/orchestration/continuous/test_contracts.py::{test_late_reward_finishes_before_version_bump_under_draining,test_late_reward_group_dropped_under_non_draining_max_stale_0}` 锁住 late-reward draining / non-draining 正确性。剩下的是 ≥2/3 GPU 实测 14% reward hide + barrier 消失。
> 3. **[≥2卡] data-parallel denoise 吞吐**:摸 VRL 的 continuous + rollout num_workers 现状（已支持到哪、差什么),给 ≥2 卡近线性 rollout 的落地路径。← denoise 吞吐的真杠杆。
> 4. **[gating probe]** 任何把 denoise 自己切 stage 跨卡之前,先证它打得过 N 个 data-parallel monolithic actor(denoise 94-98% bound 下大概率打不过)——别把 denoise 拆 stage。
>
> 已落地基:`vrl/generation/diffusion/pipeline.py`(topology + per-stage gpu_ids + run_chunk_through_pipeline,bit-exact,17 测试)。async-reward 配方与 late-reward tests 也已落。下方 §4.4/§5 仍有效;旧的"REJECT/AR"与"T0-only/discussion"措辞以本段为准。**不要把单卡 reward/rollout 共享 GPU 常驻混进这条主线;该 opt-in 已删除。async reward 只由 disjoint reward pool 表达。**

状态（2026-06-27 实测复核 → 重定向）：**async-reward / per-stage-placement 这条线已经基本建好且已测**，不是一个待从零开工的 build。下面 §4.4 用代码 + 通过的测试逐条核实：placement+release 契约（disjoint reward GPU → resident、reward 独占 bundle）已建且有测试；`distributed.resources.reward.gpu_pool: auto|rollout|dedicated` 配置语法已建且有测试；continuous producer 在 `max_inflight_groups>=2` 时已经把 reward(N) ∥ generate(N+1) 重叠；late group 的正确性由 version-stamp + max_stale + drop_stale 兜住。**配方与 late-reward 正确性测试现已补上**：cosmos continuous + `reward.gpu_pool=dedicated` recipe 已存在并有 resolver test; draining / non-draining late reward 都有测试。真正剩下的是 ≥2/3 GPU 吞吐验证,以及单卡 `forward_chunks_pipelined` 的 worker/pool 层机制验证。教训见 §4.4 末。

> **2026-06-27 晚 — T2 地基已落 + 单卡并发管线的关键架构事实（实测+读码核实）:**
> - **T2 落地**：`vrl/generation/diffusion/pipeline.py` 新增 `build_diffusion_pipeline_topology`（encode→prepare→denoise→decode[→reward]，每 stage 带 `gpu_ids` per-stage placement 钩子）+ `run_chunk_through_pipeline`（把 executor 的 4 个 stage 方法包成 handler 走 `SerialPipelineRunner`，**bit-exact**：与 monolithic `forward_chunk_plan` 同序同 threading，fake-executor 测试证明）。`tests/generation/pipeline/` 17 passed（12 旧 + 5 新）。这是把 monolithic 链重表达成可路由 stage 的地基,本身**不带重叠**。
> - **关键架构事实(决定剩余工作形态)**：单卡省时间的 **copy+CPU 重叠在 worker 层**（`worker._to_cpu` 的 GPU→CPU 拷贝 + Python 编排），**不在 executor stage 层**。executor 内的 stage 重叠(decode ∥ denoise)是 **compute∥compute = NEUTRAL**（decode=VAE compute 和 denoise 抢 tensor core,micro-benchmark 实测;copy∥compute 才省 20%）。
> - **现有两个 runner（Serial/RayPipelineRunner）是顺序的**（一个 payload 走完所有 stage），还没并发 pipelining。Ray dispatch 是"一次派一个 chunk、每 call 返回完整结果"(`ray/executor.py` 的 `run_actor_jobs(max_inflight_per_actor=1)`),逼着 sync 在返回前发生 → chunk 在 pool 层串行。
> - **并发 pipeline(真正省时间的核心)= 两条真架构路之一**：(A) sglang-omni stage actors + bounded queue（denoise-actor(N+1) ∥ decode/copy-actor(N),stage 间 payload 异步重叠）;(B) dispatch 改 per-request + worker 内 software-pipeline（produce(N+1) ∥ teardown+copy(N) on side stream + 延迟 sync）。**两者都需改 Ray worker/pool。** workflow 原计划 target 的 `forward_plan` 是 dead code(0 调用者),已排除。
> - **诚实结论**：T2 地基 + 拓扑 + placement 钩子已实落地、已测;并发 pipelining 是真架构活(改 worker/pool)。下一步 = 选 A/B,先非-Ray in-process 验机制(bit-exact + 单卡量 copy 重叠),再接 Ray 生产路径。

历史状态（旧，已被顶部 2026-06-27 复核版覆盖）：部分落地（设计 + 已建基座）。T1（DiffusionExecutor 内 typed stage payloads + 方法对齐）已随 b224383 "Add generation stage pipeline foundation" 落地——forward_chunk_plan 现以 DiffusionPromptStageInput→DiffusionPromptStageOutput→DiffusionPreparedStageOutput→DiffusionDenoisedStageOutput 串接 build_prompt_stage_input/run_prompt_encode_stage/run_prepare_stage/run_denoise_stage/run_decode_stage（executor.py:437-454），且同 commit 落地 vrl/generation/pipeline/ 契约层（PipelineTopology/PipelineStage/SerialPipelineRunner，tests/generation/pipeline 12 passed）。T0（model.memory.vae_decode.batch_size YAML 旋钮）仍是独立 memory-control 小项;不再代表本 sprint 的唯一 immediate 项。T2–T6（generic stage_pipeline config / serial_staged / ray_staged 物理管线）仍不进生产主线,除非后续 profiling 证明它们打得过 data-parallel rollout。

状态说明：旧的 "discussion; only T0 immediate" 判断已被顶部 NEXT-STEP 覆盖。当前优先级是
单卡 worker/pool I/O overlap、多卡 reward 吞吐验证、以及 data-parallel rollout 摸底。

本文记录 diffusion rollout 的 stage-pipeline 方向。目标不是迁移到 SGLang-Omni；值得借的是
它的 stage-runtime 形状。VRL 只有在 physical stage pipeline 打得过更简单的 worker/pool
overlap 与 data-parallel rollout 路线时，才应该进入完整物理管线实现。

当前 immediate work 以顶部 NEXT-STEP 为准：单卡 worker/pool I/O overlap、多卡 reward
吞吐验证、data-parallel rollout verification。`docs/sprints/parked/SPRINT_runtime_block_policies.md`
仍是独立的 memory/config-policy track，不是本 sprint 当前执行路径。

## 0. Core Decision

Do not build the full physical stage pipeline now.

VRL already has the high-level rollout system:

```text
continuous producer/consumer
Ray generation workers
sample chunk planning
OOM retry
policy_version / weight sync barriers
trajectory gather
```

Today `DiffusionExecutor.forward_chunk_plan()` is already a coordinator with
logical method boundaries and per-phase timings. It still runs one sample chunk
as a single physical unit:

```text
prompt encode -> prepare latent -> denoise loop -> VAE decode -> chunk result
```

That shape could become limiting in the future because it prevents
stage-specific batch sizing and placement. In particular, VAE decode may need a
smaller mini-batch or a separate GPU while denoise keeps the largest stable
batch size possible.

But the current SD3.5 OCR profile does not justify building a full stage
pipeline yet:

```text
encode:   0.499s total
decode:   0.571s total
denoise: 78.664s total
```

`encode + decode` is less than 1.5% of denoise. That means the current rollout
does not have a meaningful encode/decode bubble to hide. The main bottleneck is
still `generation.denoise_forward`, so the active performance path remains the
denoise transformer path from `SPRINT_rollout_performance.md`.

Future target shape, only if the gate opens:

```text
prompt_encode -> denoise -> vae_decode -> reward -> gather
```

Each stage gets its own:

```text
batch size
max inflight count
placement
memory budget
profiling label
payload contract
```

Expected benefit if the gate opens:

```text
reduce visible stage bubbles
let memory-heavy stages stop blocking denoise batch size
separate denoise / VAE / reward placement on multi-GPU rollout
```

旧 pre-profile draft 的 immediate work:

```text
T0: expose VAE decode mini-batch config.
```

T0 仍值得做，因为它是小的 memory-control 修复，可能解锁更大的 denoise batch 或避免 VAE
decode OOM。但它不能证明 full stage pipeline 该启动，也不再是当前主 next step。当前 next
steps 是顶部块：worker/pool I/O overlap、多卡 reward 吞吐验证、data-parallel rollout 摸底。

## 1. What To Borrow From SGLang-Omni If The Gate Opens

Borrow the stage-runtime shape, not the model/runtime implementation.

### 1.1 Pipeline worker schema

SGLang-Omni has a useful stage declaration shape:

```python
class StageConfig(BaseModel):
    name: str
    factory: str
    next: str | list[str] | None = None
    terminal: bool = False
    gpu: int | list[int] | None = None
    runtime: StageRuntimeConfig = Field(default_factory=StageRuntimeConfig)
    wait_for: list[str] | None = None
    stream_to: list[str] = Field(default_factory=list)
    relay: RelayConfig | None = None
```

VRL should use a diffusion-specific version of this idea, not this exact class.
It should also avoid introducing a generic `StageSpec` name because VRL already
uses `ExecutionStage` for planner-visible execution labels:

```python
class ExecutionStage:
    """One planner-visible execution stage and profiler label."""
```

Naming decision:

```text
ExecutionStage
  Existing planner/profiler concept.
  Keep it as request-plan metadata and profiler labeling.
  Do not add placement, queues, or worker lifecycle to it.

PipelineWorkerSpec / PipelineTask / PipelinePayload
  Future physical pipeline runtime concepts.
  These own placement, batch sizing, max inflight, queueing, and payload flow.
```

`ExecutionStage` can keep recording profiler labels for the future pipeline, but
the physical worker concept needs a separate name and boundary.

Future config surface:

```yaml
rollout:
  stage_pipeline:
    enabled: true
    stages:
      denoise:
        gpu: [0, 1]
        batch_size: 16
        max_inflight: 2
      vae_decode:
        gpu: [2]
        batch_size: 2
        max_inflight: 2
      reward:
        gpu: [3]
        batch_size: 8
        max_inflight: 2
```

### 1.2 Simple scheduler contract

SGLang-Omni's `SimpleScheduler` has the right knobs for non-AR pipeline
workers:

```text
batch_compute_fn
max_batch_size
max_batch_wait_ms
request_cost_fn
max_batch_cost
max_concurrency
```

VRL should mirror this at the pipeline worker level:

```text
denoise cost    = samples * num_steps * latent_tokens
vae_decode cost = samples * frames * height * width
reward cost     = decoded artifacts
```

This lets each pipeline worker batch according to its own memory and throughput
curve.

### 1.3 Placement validation

SGLang-Omni validates per-worker GPU memory fractions before starting workers.
VRL needs the same idea because denoise, VAE, and reward can share or split GPUs.

Minimal VRL version:

```text
PipelineWorkerPlacement(worker_name, gpu_ids, max_memory_fraction)
GpuPipelinePlacement(gpu_id, worker_names, total_memory_fraction)
```

Reject invalid placement before launching Ray actors.

### 1.4 Relay/backpressure idea

SGLang-Omni uses relay credits to prevent upstream workers from producing
unbounded payloads. VRL needs the same backpressure behavior, but the first
implementation should use bounded Ray tasks/queues instead of a custom relay.

Initial rule:

```text
denoise must not produce more latent payloads than decode can drain
decode must not produce more decoded artifacts than reward can drain
```

Only add CUDA IPC / NCCL / NIXL-style transport after profiling shows Ray object
or CPU transfer is the bottleneck.

## 2. What Not To Borrow

Do not copy these SGLang-Omni parts:

```text
OmniScheduler
AR KV cache logic
prefill/decode token scheduling
tree/prefix cache
SGLang server args
model registry
ZMQ control plane
NIXL/NCCL relay as the first transport
```

Those solve AR/multimodal serving problems. SD3.5/Wan/Cosmos diffusion rollout
needs dense denoise batching, pipeline-worker placement, and backpressure.

## 3. Current VRL Boundaries

Current `forward_chunk_plan()` is not an undifferentiated monolith. It is
already a coordinator over existing logical boundaries:

```python
def forward_chunk_plan(...):
    encoded = self.encode_prompt_for_chunk(...)
    stage_durations["encode"] = ...

    chunk_encoded = self.build_chunk_encoded(...)
    prepare_kwargs = self.build_prepare_kwargs(...)
    config = self.build_denoise_config(...)
    state = self.prepare_denoise_state(...)
    stage_durations["prepare_latent"] = ...

    denoise_result = self.run_denoise_steps(...)
    stage_durations["denoise"] = ...

    chunk_result = self.decode_denoise_result(...)
    chunk_result.stage_durations["decode"] = ...
    return chunk_result
```

So the existing serial logical boundaries are:

```text
encode_prompt_for_chunk
build_chunk_encoded / build_prepare_kwargs / build_denoise_config
prepare_denoise_state
run_denoise_steps
decode_denoise_result
```

The missing part is not "split the method into stages." That is mostly already
done. The missing part is typed payloads plus optional physical pipeline workers:

```text
typed payload/result contracts
pipeline-step name alignment with the existing methods
bounded queues
placement
worker lifecycle
```

Current denoise output already exists:

```python
@dataclass(slots=True)
class DiffusionDenoiseResult:
    state: Any
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    peak_memory_mb: float | None = None
    engine_counters: dict[str, Any] = field(default_factory=dict)
```

This is the first useful split point. It already separates trainable denoise
from VAE artifact decode.

Current final chunk contract:

```python
@dataclass(slots=True)
class DiffusionChunkResult:
    prompt_index: int
    sample_start: int
    sample_count: int
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    video: Any
    replay_tensors: dict[str, Any]
    context: dict[str, Any]
```

This final contract should stay stable at first so `DiffusionChunkGatherer` and
the trainer do not change.

## 4. Profiling Gate

The full stage pipeline stays blocked until a new profile shows at least one
real bottleneck outside denoise forward.

Gate opens if one of these is true after T0 and the current denoise-forward
optimization path:

```text
decode, reward, text encode, or queue wait is >= 10% of rollout wall time
VAE decode memory still prevents the target denoise batch size
multi-GPU rollout shows denoise workers idle while decode/reward drains
Ray transfer or inter-stage payload movement is a measured top bottleneck
```

Gate stays closed while this remains true:

```text
generation.denoise_forward is still the global dominant rollout cost
encode/decode together remain around 1-2% of denoise
no larger denoise batch is blocked by VAE decode memory
```

This mirrors `SPRINT_rollout_performance.md`: optimize denoise first; only then
consider batch-level staged rollout.

### 4.1 2026-06-27 cosmos video profile — gate result + single-GPU verdict

A real Ray run was profiled with nsys (fork-flag to capture the worker) + the
kernel-interval UNION method (NOT nsys projection — projection double-counts
async launches and misleads; see memory `project_real_run_profiling`). cosmos
predict2 2B, 240p_33f, n_samples=8, single 32GB GPU:

```text
rollout window 134.1s, kernel-union GPU-busy = 64%, idle = 36%
  denoise loop                96-98% GPU-busy  (GPU-BOUND; compile 1.25x = fusion, not idle-fill)
  encode + VAE decode + prep  ~6% of wall      (matches the SD3.5 "<1.5-2%" reading above)
  UNATTRIBUTED between-chunk gap = 44.3s (33%)  <- the real bubble
```

So the §4 gate's ">= 10% bubble" condition IS numerically met for video — but the
bubble is NOT encode/decode (those stay small, as on SD3.5). It is the
**between-chunk orchestration gap**: 7 chunk boundaries (sbs=1 -> 8 chunks),
~7.9s each, 81% GPU-idle, filled with cudaMemcpyAsync (latent/observation
transfer) + cudaLaunchKernel overhead + pure host Python (build next chunk /
trajectory-buffer write / decode setup). This maps to the gate's "queue wait /
inter-stage payload movement" line, not the "decode/reward/encode" line.

**Single-GPU verdict — the cheap lever subsumes most of it WITHOUT a physical
pipeline:** `rollout.sample_batch_size` reduces the chunk COUNT (chunks =
ceil(n/sbs), boundaries = chunks-1), so each boundary's cost is paid fewer times.
Measured scaling (single GPU, fits 32GB):

```text
sbs  chunks  boundaries  rollout wall  GPU-busy  speedup
1    8       7           134.1s        64%       1.00x
2    4       3           103.0s        77%       1.30x
4    2       1            89.2s        89%       1.50x   <- landed in config
```

**Why the full physical stage pipeline (T3-T6, SGLang-Omni shape) on a single GPU
is bounded — NCU-measured, NOT assumed:** an earlier draft claimed "all stages
contend for the same SMs, so single-GPU staging cannot help." That was too
strong. NCU on the real cosmos denoise kernels (synthetic real dims, 240p_33f,
time-weighted over 45 kernels) shows the SMs are NOT saturated:

```text
sm__throughput (compute SM util)        = 42%
sm__warps_active (achieved occupancy)   = 15%   (cutlass GEMM runs few, heavy warps)
sm__pipe_tensor_cycles_active           = 43%   (the dominant cutlass GEMM is 82% of time)
```

So there IS spare SM capacity in principle (only 42% compute throughput, 15%
occupancy). BUT the bottleneck pipe is the **tensor core**, and 43% ≈ the RTX 5090
consumer **bf16+fp32-accumulate half-rate ceiling** (~47%) — i.e. the tensor
cores ARE maxed; the ~58% "headroom" is mostly the half-rate penalty plus the
NON-tensor pipes (FMA/ALU/LSU/memory) sitting below 43%. Consequence for
single-GPU concurrent multi-staging (CUDA streams / MPS):

```text
co-run another TENSOR-heavy stage (VAE conv / reward GEMM) with denoise GEMM
   -> both want the already-maxed tensor pipe -> NO speedup (serializes at half-rate)
co-run NON-tensor work (memcpy on copy engine, host Python, elementwise/norm on FMA/ALU)
   -> uses the idle non-tensor pipes -> REAL but bounded overlap  (= P2 / serial_staged T2)
```

So single-GPU staging is a **bounded** lever, not a forbidden one: it can hide the
boundary's memcpy + CPU + non-GEMM work behind the next chunk's denoise, but it
cannot make two GEMM-heavy stages run concurrently for free (the tensor cores are
at ceiling). The full per-stage **compute** concurrency that the SGLang-Omni shape
buys — running denoise GEMMs and VAE/reward GEMMs at the same time — still needs
**separate GPUs** (denoise on GPU0-1, VAE on GPU2, reward on GPU3), because each
tensor-heavy stage wants its own un-contended tensor cores. On one GPU: `sbs`
removes most boundaries for free (sbs=4 -> 89% busy), and the residual ~9% is the
non-tensor boundary overlap (T2/P2). The one-shot NCU probe was retired after
recording this gate result; it ran with
`ncu --launch-skip 400 --launch-count 45 --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,...`.

**Updated gate verdict:**

```text
single GPU  -> sbs (reduce boundary count, landed sbs=4 = 1.50x) + optional T2/serial_staged
               for the last boundary. Full physical pipeline stays CLOSED here.
multi GPU   -> the SGLang-Omni-shape physical pipeline (T3-T6) opens: separate
               denoise/VAE/reward placement is the only way to run the stages as
               concurrent compute. THIS is where §1's StageConfig/placement/relay pays off.
```

So this sprint's full build is gated to the **multi-GPU** path; the single-GPU
bubble is a `sample_batch_size` + (optional) serial-staged-overlap problem, not a
physical-pipeline problem.

### 4.2 GATE OPENS — the reward stage is the measured justification (2026-06-27)

The §4 gate condition "decode/reward/text-encode/queue-wait >= 10% of rollout wall"
is now met by a stage I had NOT measured until now: the **reward stage**. It is
scored after rollout (outside the denoise window), already instrumented as
`collector.reward_score` (core.py:182). Measured (cosmos, sbs=1 and sbs=4 alike):

```text
reward stage = 12.6s wall per group, 0% busy on the rollout GPU, = 14% of reward+denoise
```

Root cause of the 0%: `reward execution: pool` (kling_video_reward.yaml) runs reward
in a separate pool actor, and `release_rollout_before_reward` (core.py:222)
OFFLOADS the rollout 2B model first because a single 32GB GPU cannot hold the
rollout model + the reward model at once. So on ONE GPU rollout and reward are
forced SERIAL by **memory** (offload -> reward -> restore barrier), not by tensor
contention. This is the clean, measured >= 10% bubble — and unlike the generation
boundary, `sbs` does NOT touch it.

**Gate verdict (final):** OPEN for the multi-GPU path, on the reward stage.
Placing the reward pool on a 2nd GPU runs reward(group N) concurrently with
rollout(group N+1) AND removes the offload/restore barrier — hiding the full 14% +
the barrier. This is exactly §1's per-stage placement; the reward stage is the
first and best-justified stage to split off (it is already an independent pool
actor — only placement + async scheduling are missing).

### 4.3 Goal (how to set it)

```text
North star metric : end-to-end samples/sec (rollout + reward + train cycle wall),
                    NOT per-kernel SM occupancy (§7).
Measured prize    : hide the 14% reward stage + delete the release_rollout_before_reward
                    offload/restore barrier. (VAE/encode are small ~6%, lower priority.)
Mechanism         : reward pool actor -> dedicated GPU; bounded-async schedule so
                    reward(group N) overlaps rollout(group N+1).
Throughput model  : 1 / max(T_rollout, T_reward, ...) once stages are placed on
                    separate GPUs (single GPU degenerates to the serial sum — see 4.1).
Correctness gate  : reward values / advantage / old_log_prob BIT-IDENTICAL to the
                    synchronous baseline. Async only moves WHEN reward runs; group N's
                    reward MUST complete before group N trains (bounded async, NO
                    staleness, NO off-policy). drift guard + reward-curve parity must hold.
```

**Validation split (single machine now -> multi-GPU later):**

```text
on 1 GPU now  : build + UNIT-test the async-reward scheduler and per-stage placement
                contracts; correctness is verifiable (serial fallback must stay
                bit-identical). NO throughput win here (memory forces serial).
on >= 2 GPU   : measure the 14% reward hide + barrier removal; validate the 1/max
                throughput model and reward-curve parity.
```

So: the sprint can START now scoped to **async reward + per-stage placement**
(reward stage first), correctness-validated on one GPU; the throughput payoff is
gated to a second GPU. Do NOT build the generic encode/VAE/relay machinery first —
the reward stage is the measured prize; everything else is small until proven.

### 4.4 2026-06-27 实测复核 — async-reward / per-stage-placement 基本落地（核实清单）

§4.2 / §4.3 把 reward stage 定为 gate-open 的 prize，并把 sprint 范围收到「async
reward + per-stage placement」。本节是**对照已有测试与配置**的复核：当时的 do-next
其实**严重过度规划**——它提议「建」的东西大部分**已经存在并已被测试覆盖**。本轮又补上
cosmos async-reward recipe 与 late-reward correctness tests。下面每条都给代码/测试证据。

**(a) placement + release 契约：已建，已测。** disjoint reward GPU 会
derive 出 `release_rollout_before_reward=False` + reward resident；reward 独占自己的
bundle。证据：

```text
tests/ray/test_resources.py:579-665
  test_dedicated_reward_gpu_derives_resident_lifecycle_when_unset
    reward 独占 GPU2 -> rollout.mode == "resident"
                     -> release_rollout_before_reward is False
                     -> release_reward_after_score is False
  test_lifecycle_plan_resident_when_roles_disjoint
    trainer/rollout/reward 三者 GPU 全 disjoint -> 每个 role resident, 无 handoff barrier
tests/ray/test_resources.py:867-917
  test_colocated_reward_on_dedicated_gpu_owns_its_own_bundle
    reward 独占 GPU1 -> layout.reward_bundle_indices 与 rollout_bundle_indices disjoint
  test_shared_single_gpu_reward_reuses_rollout_bundle
    单卡 shared reward -> 复用 rollout bundle, total_bundles == 1
本 session 运行：15 passed（reward and (release|dedicated|bundle|gpu_pool|resident)）。
```

即 §4.2「placing the reward pool on a 2nd GPU ... removes the offload/restore barrier」
所需的**契约推导本身已经存在**：把 reward 放到 disjoint GPU，resolver 自动关掉
`release_rollout_before_reward`，barrier 自动消失。这不是要新建的代码。

**(b) 配置语法 `distributed.resources.reward.gpu_pool: auto|rollout|dedicated`：已建，
已测。** 证据：

```text
tests/ray/test_resources.py:1019-1113
  test_reward_gpu_pool_rollout_shares_rollout_gpu          (=rollout 落 rollout GPU)
  test_reward_gpu_pool_rollout_matches_legacy_share_with_rollout (新旧等价)
  test_reward_gpu_pool_auto_prefers_spare_gpu              (=auto 抢空闲卡)
  test_reward_gpu_pool_both_keys_is_an_error               (gpu_pool + share_with_rollout 互斥)
  test_reward_gpu_pool_rejects_unknown_value               (只收 auto/rollout/dedicated)
```

所以 §1 设想的「per-stage placement 配置面」对 reward 这一 stage **已经有了**，连
auto-pick-spare-GPU 都已实现。

**(c) continuous producer 已经 reward(N) ∥ generate(N+1) 重叠。** producer 用 bounded
asyncio inflight 集合，在 `max_inflight_groups>=2` 时允许第 N 组（含 reward 打分）与第
N+1 组的 generate 同时在飞：

```text
vrl/rollouts/orchestration/continuous/producer.py
  self._inflight: set[asyncio.Task[Any]]   # 每组一个 task，含 generation+reward
  _run 循环按 max_inflight_groups 维持多组在飞 -> N 与 N+1 重叠
configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml
  /base/rollout/orchestration/continuous, continuous.max_inflight_groups: 2
  -> 已有一份可跑的单卡 async debug 配方（gpu_pool: trainer 借显存）
```

即「bounded-async schedule so reward(group N) overlaps rollout(group N+1)」（§4.3
Mechanism）**已经是 continuous 模式的既有行为**，不需要新写调度器。

**(d) late group 正确性已兜住（version-stamp + max_stale + drop_stale）。** reward 与
policy 无关，迟到的组由调度层按版本丢弃，不会污染 on-policy 训练：

```text
vrl/rollouts/orchestration/continuous/staleness.py
  StalenessPolicy.admit: 0 <= (current_version - item_version) <= max_stale_policy_versions
  too_stale / is_future 显式拒绝
vrl/rollouts/orchestration/continuous/scheduler.py:8-10
  "submit 时 stamp 版本，consumer/queue 丢弃过期组"
```

**已补上的缺口（本轮落地/验证）：**

```text
(1) 配方：configs/experiment/diffusion/cosmos_predict2/online_grpo_async_reward.yaml
    已组合 cosmos continuous + rollout.gpu_pool=dedicated + reward.gpu_pool=dedicated。
(2) 配方解析/拓扑测试：
    tests/ray/test_resources.py::test_cosmos_async_reward_recipe_resolves_resident_reward_overlap
    断言 3-GPU disjoint layout 下 reward 与 rollout disjoint、release_rollout_before_reward=False、
    reward resident、continuous.max_inflight_groups>=2。
(3) late-reward 正确性测试：
    tests/rollouts/orchestration/continuous/test_contracts.py::
      test_late_reward_finishes_before_version_bump_under_draining
      test_late_reward_group_dropped_under_non_draining_max_stale_0
    分别锁住 draining barrier 等 reward 完成后再 bump version、non-draining/max_stale=0 下迟到组
    被 drop_stale 丢弃而不会进入训练。
```

**真正剩下的缺口：**

```text
(1) 吞吐验证（需要 >=2/3 GPU）：实测 ~14% reward stage 是否被 reward(N) ∥ generate(N+1)
    藏住、offload/restore barrier 是否消失、1/max throughput model 是否成立。
(2) 单卡生成侧 I/O 流水：在 worker/pool 边界实现/验证 forward_chunks_pipelined，目标是藏
    worker._to_cpu + Python orchestration；executor 内部 stage 再拆不是生产瓶颈。
(3) denoise 吞吐扩展：先摸清 continuous + rollout num_workers/data-parallel 的现状，再决定
    是否需要更重的 physical stage pipeline。任何 denoise stage-split 必须先打赢 N 个完整
    data-parallel rollout actor。
```

**非主线提醒：** 单卡共享 reward/rollout GPU 的常驻 overlap 配置已删除。共享 reward pool
只表示 phase handoff；async reward 的配置语义是 `distributed.resources.reward.gpu_pool=dedicated`
或显式 disjoint reward devices。不要把同卡 reward/rollout 常驻当成当前 async-reward 主线的依赖。

**教训（写给后续 workflow）：先核对已有测试与配置，再 scope build。** 早先的 do-next
之所以**过度规划**（提议建已经存在的 placement/release 契约、gpu_pool 语法、async 重叠
调度），是因为它的 reader 把**源码**映射了一遍，却**没有映射既有测试套件**——
`tests/ray/test_resources.py` 的 placement/lifecycle 测试 + 那份 async debug 配方
本就证明这些机制是活的。结论：scope 任何 build 之前，必须先 grep tests/ 与 configs/
确认「要建的」是否已经被测被配，否则会把已完成的工作重新立项。

把 §6 的 T3-T6（generic encode/VAE/relay 物理管线）继续保持 gated：它们针对的是
encode/VAE 这些 ~6% 的小 stage 与跨 stage compute 并发，仍需 multi-GPU 才有意义，且
不是当前 prize。当前值得动手的是单卡 worker/pool I/O overlap、≥2/3 GPU reward 吞吐验证、
以及 data-parallel rollout 现状摸底。

## 5. Future Target Architecture

This section is a design reference for the gate-open path, not a current build
plan.

### 5.1 Pipeline specs

Add a diffusion pipeline spec layer. Keep it narrow and typed.

```text
vrl/generation/pipeline/
  specs.py          # PipelineWorkerSpec, PipelineRuntimeSpec, PipelinePlacementSpec
  placement.py      # validate pipeline worker -> GPU/resource mapping
  scheduler.py      # bounded batching scheduler for non-AR workers
  runner.py         # in-process pipeline worker contract

vrl/generation/diffusion/pipeline.py
  DiffusionEncodePayload
  DiffusionDenoisePayload
  DiffusionDecodePayload
  DiffusionRewardPayload
  diffusion_pipeline_graph(...)
```

Do not add an abstract framework wider than the first diffusion use case needs.

### 5.2 Pipeline payloads

Suggested payload flow:

```text
DiffusionEncodePayload
  request
  chunk
  params
  video_request

DiffusionPreparePayload
  request
  chunk
  params
  video_request
  encoded

DiffusionDenoiseOutput
  request_id
  prompt_index
  sample_start
  sample_count
  denoise_result
  config
  stage_durations

DiffusionDecodeOutput
  request_id
  prompt_index
  sample_start
  sample_count
  decoded_video
  replay_tensors
  context
  denoise_result
  stage_durations
```

The final adapter converts `DiffusionDecodeOutput` back into the existing
`DiffusionChunkResult`.

### 5.3 Execution modes

Support two execution modes behind one config flag:

```text
serial_staged
  same process, same worker, explicit pipeline payload boundaries
  used to validate contracts and metrics first

ray_staged
  physical Ray pipeline workers with per-worker placement
  used for multi-GPU rollout
```

Keep the current fused path:

```text
fused_chunk
  current forward_chunk_plan behavior
  fallback path for debugging and model-family bring-up
```

## 6. Gated Implementation Plan

### T0: Add explicit VAE decode mini-batch config

Status: immediate candidate.

Goal: make the immediate memory lever official without building a stage runtime.

Current latent decode already supports `decode_batch_size`, but SD3.5 does not
expose a canonical YAML knob for it.

Add:

```yaml
model:
  memory:
    vae_decode:
      batch_size: 2
```

Extend the existing VAE memory policy parser so it owns:

```text
tiling
slicing
batch_size
```

Acceptance:

```text
SD3.5 decode uses model.memory.vae_decode.batch_size
Wan/Cosmos existing tiling/slicing behavior is unchanged
config tests reject unknown keys
latent decode tests cover batch_size
```

### T1: Add typed payloads and align naming inside `DiffusionExecutor`

Status: gated by profiling.

Goal: keep behavior identical while making the existing logical boundaries
explicit enough to feed a future physical pipeline.

Do not add a parallel set of `run_encode_stage()` / `run_denoise_stage()`
wrappers on top of the existing methods. The current methods are already the
logical boundaries. If names change, treat it as a rename/alignment of existing
methods, not as duplicate wrappers.

Current logical boundary methods:

```python
def encode_prompt_for_chunk(...)
def build_chunk_encoded(...)
def build_prepare_kwargs(...)
def prepare_denoise_state(...)
def run_denoise_steps(...)
def decode_denoise_result(...)
```

Future T1 work is therefore:

```text
add typed payload/result objects around the existing method boundaries
align naming with future PipelineTask/PipelinePayload concepts
keep `forward_chunk_plan()` as the serial coordinator
avoid introducing new methods that duplicate existing boundaries
```

Acceptance:

```text
existing diffusion generation tests pass
stage_durations preserve encode/prepare_latent/denoise/decode
no trainer or gather contract changes
new payload names do not conflict with ExecutionStage
```

### T2: Add serial pipeline executor mode

Status: gated by profiling.

Goal: validate the pipeline payload API without Ray placement complexity.

Add config:

```yaml
rollout:
  stage_pipeline:
    enabled: true
    mode: serial
```

In serial mode, the executor still runs in one worker, but the code path uses
pipeline payload/result classes and per-worker metrics.

Acceptance:

```text
fused_chunk and serial_staged outputs match for fixed seed
policy_version behavior is unchanged
precision drift guard still passes
pipeline metrics report per-worker wall time and tensor bytes
```

### T3: Add pipeline-worker-aware Ray planning

Status: gated by profiling.

Goal: make physical placement possible.

Extend planning from:

```text
chunk -> worker
```

to:

```text
chunk pipeline task -> pipeline worker pool
```

Initial physical pipeline workers:

```text
denoise
vae_decode
reward
```

Do not split every denoise timestep. Diffusion timestep dependency is serial:

```text
x_t -> transformer -> scheduler -> x_{t-1}
```

The useful pipeline is across chunks, not inside one denoise chain.

Acceptance:

```text
pipeline worker placement validates configured GPU ids
denoise worker can feed decode worker through bounded queue
decode worker can feed reward path without changing final trajectory shape
policy_version mismatch drains or rejects stale pipeline payloads
```

### T4: Add bounded pipeline queues and backpressure

Status: gated by profiling.

Goal: prevent the physical pipeline from creating new memory pressure.

Rules:

```text
max_inflight per pipeline worker is enforced
upstream worker blocks when downstream queue is full
request abort or policy_version change cleans queued payloads
OOM in one pipeline worker reports worker name and payload identity
```

Acceptance:

```text
denoise cannot enqueue unlimited latent payloads
decode cannot enqueue unlimited videos
OOM retry still splits at the sample-chunk boundary
metrics include queue wait time by pipeline worker
```

### T5: Multi-GPU SD3.5 OCR rollout validation

Status: gated by profiling.

Goal: prove the pipeline improves capacity or throughput on the real target.

Minimum scenarios:

```text
baseline fused_chunk on one GPU
serial_staged on one GPU
ray_staged with denoise and VAE decode separated
ray_staged with denoise, VAE decode, and reward separated
```

Metrics:

```text
images/sec
GPU active time by pipeline worker
queue wait time by pipeline worker
peak memory by pipeline worker
OOM rate
policy staleness
reward curve parity
precision drift guard result
```

Acceptance:

```text
stage pipeline does not regress reward correctness
stage pipeline enables at least one larger denoise batch that fused_chunk cannot run
or stage pipeline improves end-to-end rollout throughput at the same batch size
```

### T6: Transport upgrade gate

Status: gated by profiling.

Goal: avoid premature relay complexity.

Only implement CUDA IPC / NCCL / NIXL-style tensor relay if profiling shows:

```text
inter-stage transfer is a top rollout bottleneck
or Ray object transfer creates measurable GPU idle bubbles
```

Until then:

```text
use Ray object store / CPU transfer
keep tensor payloads explicit and measured
```

## 7. Throughput Model

After warmup, stage pipeline throughput is limited by the slowest normalized
stage:

```text
throughput ~= 1 / max(
  T_denoise / num_denoise_workers,
  T_vae_decode / num_decode_workers,
  T_reward / num_reward_workers
)
```

The stage pipeline is useful when it lets us tune each stage independently:

```text
denoise: large batch, trainable transformer, compile/low precision
vae_decode: smaller batch, memory-heavy artifact decode
reward: OCR/reward batch, possibly separate device
```

Do not evaluate this sprint by asking whether every kernel reaches 100% SM
occupancy. Evaluate it by rollout throughput, GPU idle bubbles, OOM rate, and
training correctness.

## 8. Non-goals

```text
Do not migrate to SGLang-Omni.
Do not copy OmniScheduler or AR KV-cache logic.
Do not split individual denoise timesteps into separate pipeline stages.
Do not mark this as an implementation sprint while denoise forward dominates.
Do not replace diffusers transformer forward in this sprint.
Do not replace VAE implementation in this sprint.
Do not change trainer trajectory semantics.
Do not add custom relay transport until transfer is proven to bottleneck.
Do not remove the current fused_chunk fallback.
```

## 9. Source References To Follow

VRL:

```text
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/diffusion/executor.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/diffusion/gather.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/execution/planner.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/execution/scheduler.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/executor.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/common/latent_decode.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/common/vae_decode_memory.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/sd3_5/model.py
/home/mingfeiguo/Desktop/wm-infra/docs/sprints/info/SPRINT_rollout_performance.md
/home/mingfeiguo/Desktop/wm-infra/docs/sprints/reading/SPRINT_diffusion_rollout_system.md
```

SGLang-Omni:

```text
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/config/schema.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/config/placement.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/pipeline/stage/runtime.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/scheduling/simple_scheduler.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/relay/base.py
/home/mingfeiguo/Desktop/sglang-omni/sglang_omni/pipeline/relay_io.py
```
