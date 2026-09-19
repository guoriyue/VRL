# SPRINT: Rollout finalize-overlap GA — 把已有的 chunk 分相重叠补满并转默认

**日期**: 2026-07-13  **状态**: PLANNED
**来源**: FlashDreams 引擎评审教训 ①（`StreamInferencePipeline` 的 generate/finalize
分相：热路径只做 denoise+decode，KV/簿记推迟到 finalize 并可后台化——
`docs/.../inference_pipeline_overview.rst:57-63`）。NVIDIA 的设计验证了 vrl
已走的方向；本 sprint 不是新建，是把现有实现**补覆盖、量化残差、转默认**。

**vrl 现状（2026-07-13 核实）**：`forward_plan_pipelined`
（`vrl/generation/bindings/full_sequence_denoise/executor.py::forward_plan_pipelined`）已实现同一模式——chunk N 的
teardown（GPU→CPU 结果拷贝 + host 打包，copy stream）与 N+1 的
produce（encode→prepare→denoise→decode，默认流）重叠，BIT-EXACT，经
`vrl/generation/execution/pipeline.py::forward_chunks_pipelined` 复用同一批 stage 方法。但它：
- **opt-in**（`distributed.rollout.pipelined`，默认 false）；
- **仅单 worker + 多 chunk 时生效**（`RolloutWorkerSection` 注释）；
- 实测只回收 ~20%（nsys 档案：rollout 64% busy / 36% 空转，其中 ~33% 是
  between-sample 编排间隙——copy+CPU 只是其中一部分）。

## 目标

单卡视频 rollout 的 chunk 间空转从 ~36% 压到可归因的低残差；pipelined 路径
在图像/视频全家族默认开启；残差逐项归因（Ray 调度 / Python 编排 / 无法重叠的
同步点），写进测量档案。

## KILL-gate（先测后做）

P0：在真实 cosmos V2W run 上重跑 nsys（`VRL_PROFILE=1` +
`--trace-fork-before-exec`，GPU 空转按 kernel-interval UNION 判，不看投影），
对比 pipelined on/off 的逐段归因。**若 copy+CPU 之外的残差主体是 Ray dispatch /
weight-sync 栅栏等本 sprint 够不着的项，则只做「转默认」部分，扩展项降级为
记录。**

## 变更清单

1. **残差归因**（P0 输出物）：pipelined on 状态下按
   `stage_durations` + nvtx 把剩余空转拆成：①下一 chunk 的 encode/prepare
   （已在 produce 内，应已重叠——验证）②trajectory/context 的 host 侧 dict
   构建 ③gather_chunks 的串行 cat ④Ray executor 的 chunk 派发间隙。
2. **补覆盖**（按 P0 结论裁剪）：
   - host 打包若仍在主线程占位，参照 FlashDreams "finalize 可后台线程化"，
     把纯 Python 打包挪 worker 线程（GPU 侧已在 copy stream，无正确性风险；
     打包只读已拷贝完成的 CPU 张量，以 stream event 为界）。
   - `gather_chunks` 的 `torch.cat` 若可见，评估边收边 cat（保序前提下）。
3. **转默认**：`RolloutWorkerSection.pipelined` 默认改 true；planner 对
   多 worker / 单 chunk 情形保持现行自动回退（行为已存在，补一条显式测试）。
   Bit-exact 性质是转默认的前提，已由现有对比测试钉住——补一条
   video 形状（多帧）用例。
4. **测量档案**：结果并入 `docs/sprints/info/`（遵守 perf sprint 永久保留惯例），
   更新 `project_real_run_profiling` 记忆中的 64%/36% 数字。

## 非目标

- 跨 chunk 的 denoise 级流水（那是 parked stage-pipeline sprint 的领域）。
- reward 重叠（已由 async-reward / sleep_offload 路线单独覆盖）。
- 多 worker 间的 chunk 窃取/动态放置（`chunk_placement_strategy=dynamic` 已存在）。

## 验证

- pipelined on/off 输出逐张量相等（现有 bit-exact 测试 + 新增视频形状用例）。
- 真实 run 前后 nsys 对比：gen 段 wall-clock 与空转占比，写明测量口径。
- 全量 pytest 基线对齐。

## 2026-09-18 更新：侧流拷贝已拆除，per-request 循环保留

`forward_batches_pipelined` 不再把第 N 批的 D2H 拷贝放到 copy stream 上与第 N+1
批的 produce 重叠。现在它是一个普通循环：`forward_batch` → 同步的 pinned 拷贝
（`vrl/trajectory/device.py::copy_tensor_tree_to_pinned_cpu`，与 Ray worker 逐批路径
共用同一个助手）→ 发布 fence。`BatchProduceFence` 不再携带 CUDA event，
`RayGenerationWorker.pipelined_progress` 直接读快照，不再轮询 event。

依据是本仓库已有的两次测量，而不是新测：

- 侧流重叠单独计量为零：串行 6735 ms 对重叠 6729 ms
  （`reading/SPRINT_diffusion_rollout_stage_pipeline.md` 1a，2026-06-27）。
- 36% 的收益来自"整个 request 一次 RPC"消掉的逐批派发开销
  （同文 1b，cosmos 240p_33f：124.8 s → 80.1 s），与拷贝重叠无关。

拆除的代价是每批多一次 `torch.cuda.synchronize()`，按上面的量级是毫秒级；
去掉的是 copy stream、每批两个 event、`record_stream`、pinned 生命周期规则和
OOM 路径里"先 join 侧流再清帧"的顺序约束，以及第 N 批 payload 在第 N+1 批
produce 期间多占的一份显存（planner 从未为它记账）。

对照 miles-diffusion（2026-09-18 读本地 sglang-diffusion 源码）：引擎在 denoising
stage 末尾就是同步 `.cpu()`（`runtime/pipelines_core/stages/denoising.py:888`），
没有侧流；打包与序列化在 scheduler / HTTP 进程，反序列化在 Miles 的 parser
actor 池。他们三个层次全靠进程分离，没有线程或 CUDA 流。

本 sprint 剩余部分的前提不变：循环结束后 `merge_generation_batches` 与 Ray
序列化仍在 GPU worker 进程内串行，P0 的 nsys 归因才决定是否值得把它们搬出
worker（附录 B 的 CPU actor 形状）。

## 2026-09-18 更新（二）：合并出 GPU worker，request 之间不再串行

同日第二批改动把上一节"剩余部分"里的进程拆分做掉了，形状与附录 B 一致，
未新增配置键。

- **合并与序列化出 GPU worker。** `pipelined=true` 时，rank 在循环里每批
  `ray.put` 一次（`forward_batches_pipelined` 的 `stage_result` 钩子），返回的是
  `PipelinedBatchRefs`（每批一个 object ref），不再在 rank 进程里
  `merge_generation_batches` 也不再把整条轨迹 pickle 进返回值。合并跑在每个
  engine 旁边一个 `RayGenerationFinalizer`（`vrl/generation/ray/finalizer.py`，
  `num_cpus=0`，钉在该 engine 的 primary bundle 上）：`ray.get` 各批、gatherer
  合并并构建轨迹、把 reward media 装箱成 `MediaReference`。rank 在最后一批
  staged 之后即空闲。
- **driver 侧反序列化。** driver 现在收到的是 finalizer 返回的一个
  `GenerationOutput`（tensor wire 带外字节），不再在 per-batch 路径之外多做一次
  合并。训练进程仍要把最终轨迹读进内存一次，这一步与之前相同。
- **Reward 按组打分。** 核实后发现 strict 模式早已有 `PER_GROUP_STREAMING`
  （`vrl/rollouts/collector/core.py`，reward 隔离验证通过时默认启用）。本批未改；
  共卡（reward 与 rollout 时分同一张卡）仍是 `BATCHED_SERIAL`，逐组打分意味着
  逐组 park/wake，是否值得需要按配置测量，没有证据前不动。
- **下一个 request 立即开始。** 两处：① `RayGenerationExecutor` 去掉了 driver
  侧的单飞 asyncio 锁，准入交给 dispatcher 的每 engine 一个槽位（deadline 在拿到
  槽位后才起算，语义不变）；② strict 模式的 `prepare_training_batches` 提前一组
  提交下一次生成（`PER_GROUP_SERIAL` 对照臂除外），使 engine 在上一组的批 staged
  后立刻接到下一组，与该组的合并、打分重叠。continuous 模式本就并发提交，自动受益。
- **多 engine。** `pipelined` 不再要求恰好一个 engine：每个 engine 按 round-robin
  拿到自己那份批，一次 RPC 跑完，finalizer 按 plan 顺序合并所有 engine 的 refs。
  任一 engine 返回 OOM 则整个 request 退回逐批拆分路径。

测试：`tests/generation/ray/test_finalizer.py`（含真实 Ray 往返）、
`test_oom_split.py` 的多 engine 与 OOM 回退用例、`test_runtime_config.py` 的
finalizer 启动用例、`tests/rollouts/orchestration/test_prompt_collection.py` 的
提前提交与取消用例。

仍未做：P0 的 nsys 归因（本批的收益上限仍要在真实 Cosmos run 上量）；共卡场景
的逐组打分节奏。
