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
`run_chunk_through_pipeline` 复用同一批 stage 方法。但它：
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
