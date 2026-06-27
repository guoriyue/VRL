# SPRINT: Trustworthy profiling API（planned）

状态：proposed / planned（2026-06-27）。本 sprint 只改 profiling API 的可信边界与验证，
不做新的性能优化结论。目标是让一次 profile run 产出的 trace、summary、NVTX 归因能被信任：
知道采了什么、没采什么、每个数字能不能当 wall-time 百分比读，以及失败时不会静默给出半真结果。

## 0. 现状问题

当前 `vrl/utils/profiling.py` 有实用价值，但 API 名字和职责混在一起，导致 profiler 结果看起来像
“能用但不够可靠”。

第一处是 `record_function` 名字太窄，实际做了两件事：

```python
ctx = torch.profiler.record_function(name)
emit_nvtx = os.environ.get("VRL_PROFILE") == "1"
if emit_nvtx:
    torch.cuda.nvtx.range_push(name)
```

这不是单纯的 PyTorch `record_function` wrapper，而是“VRL stage range 标注”。它同时影响
torch profiler trace 和 nsys/ncu NVTX 归因，但 NVTX 行为藏在 `VRL_PROFILE` 里，调用点看不出来。

第二处是 `torch_profiler_step` 一口气承担了采集、目录布局、worker 命名、TensorBoard trace handler、
summary 写文件：

```python
with torch.profiler.profile(
    activities=activities,
    record_shapes=bool(config.record_shapes),
    profile_memory=bool(config.profile_memory),
    with_stack=bool(config.with_stack),
    with_flops=bool(config.with_flops),
    on_trace_ready=handler,
) as active_prof:
    yield
    active_prof.step()
```

这个函数不是通用 profiler 框架，只是“包住一个 trainer/rollout step 并写 trace”。这个方向是对的，
因为历史 sprint 明确要求不要维护第二套 wall-time profiler，而是直接读 `record_function` ranges。
问题是它没有把“采集窗口正确性”“导出文件完整性”“summary 是否可信”拆开验证。

第三处是活动选择过于经验化。现在只看 `device.type` 和 `torch.cuda.is_available()`：

```python
if "cuda" in requested and device_type.startswith("cuda") and torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)
```

PyTorch 已提供 `torch.profiler.supported_activities()`，而且官方说明 CUDA 可用但 CUPTI 不可用时，
表格里可能有 CUDA time，JSON trace 里却没有完整 kernel 事件。这里如果只 warning 或静默降级，
会让用户拿着“不完整 trace”做结论。

## 1. 设计原则

1. **标注和采集分层。**
   - stage 标注函数只负责给代码块命名：torch range + 可选 NVTX。
   - trace 采集函数只负责开启 PyTorch profiler、推进 step、导出 trace、写 manifest。
   - nsys 仍然由外部 `nsys profile ...` 管理；Python 代码只提供 NVTX ranges，不假装能管理 nsys。

2. **默认短诊断，深挖显式 opt in。**
   - 默认 profile preset 应低开销：CUDA kernel + 必要 CPU ranges。
   - `record_shapes/profile_memory/with_stack/with_flops` 都应是 deep-dive 选项，因为这些会改变开销，
     甚至影响 tensor 生命周期和优化机会。

3. **失败不能伪装成功。**
   - profiler disabled 时，标注函数可以是 no-op。
   - profiler enabled 但 requested activity 不被支持时，要 fail fast 或在 manifest 里写清楚
     `requested` / `effective` / `missing`，不能只输出一个看似正常的 summary。
   - trace handler 没有生成 trace 文件时，测试和 smoke run 必须失败。

4. **summary 只能总结，不能替代 trace。**
   - `key_averages().table(...)` 是快速入口，不是唯一证据。
   - CUDA/NVTX 百分比必须标注语义：nsys 的 NVTX range kernel summary 会把同一个 kernel 归到
     所有包含它的 range，嵌套 range 不能直接相加成 wall-time 百分比。

## 2. 目标 API

保留一个薄的公共边界，但名字要准确。

```python
with profile_range("generation.denoise_forward"):
    step_output = model.forward_step(state, step_idx)
```

`profile_range` 是 `record_function` 的语义后继：它是 VRL 的 stage range，不是裸 PyTorch API。
迁移期保留 `record_function = profile_range` 兼容旧调用点，但新代码只导入 `profile_range`。

trace 采集入口改成表达真实职责：

```python
with capture_torch_trace(
    config,
    output_dir=...,
    step=...,
    device=...,
    worker_name=...,
    trace_subdir="generation/rollout-0",
):
    ...
```

`torch_profiler_step` 作为兼容 alias 保留一轮，最后删除。它是薄函数，但保留迁移期是 public API
facade，不是为了少改几行。

## 3. 配置面

`TorchProfilerConfig` 继续作为 typed config 边界保留。它是 YAML 到 runtime 的 schema 边界，
不是无意义 thin dataclass。

需要调整的字段：

| 字段 | 处理 |
|---|---|
| `enabled` | 保留，控制是否采集 trace。 |
| `output_dir` | 保留，覆盖默认 `<trainer.output_dir>/torch_profiler`。 |
| `activities` | 保留，但用 `torch.profiler.supported_activities()` 校验 requested vs effective。 |
| `record_shapes` | 默认应考虑改为 false；深挖 profile 再打开。 |
| `profile_memory` | 默认应考虑改为 false；memory profile 另开 preset。 |
| `with_stack` / `with_flops` | 保留，显式 deep-dive 开关。 |
| `skip_first` / `max_steps` | 迁移到 PyTorch `schedule(wait/warmup/active/repeat/skip_first)` 语义，或明确文档化“只包单 step”。 |

配置文件只保留 `configs/profile/torch_profiler.yaml` 这一个短诊断 preset。若需要 deep profile，
用本次 run 的 dotlist overrides 临时打开 `record_shapes/profile_memory/with_stack/with_flops`，
不要维护第二个几乎全是布尔开关的 preset。

明确不进入 `TorchProfilerConfig` 的项：

- `emit_nvtx`：NVTX 是 process-wide tracing signal，继续由 `VRL_PROFILE=1` 控制；`profile_range`
  保留函数级 override 只用于测试/特殊调用，不进 YAML。
- `fail_on_missing_activity`：requested activity 不受支持永远 fail fast；best-effort trace 会制造半真结果。
- `write_manifest`：manifest 是 trust anchor，profile capture 永远写；不提供关闭开关。

## 4. Manifest

每个 trace 目录必须写 `profile_manifest.json`，这是信任链的核心产物。字段最少包括：

```json
{
  "schema_version": 1,
  "worker_name": "...",
  "step": 0,
  "trace_subdir": "generation/rollout-0",
  "requested_activities": ["cpu", "cuda"],
  "effective_activities": ["cpu", "cuda"],
  "missing_activities": [],
  "record_shapes": false,
  "profile_memory": false,
  "with_stack": false,
  "with_flops": false,
  "emit_nvtx": true,
  "trace_files": ["...pt.trace.json"],
  "summary_file": "...summary.txt",
  "torch_version": "...",
  "cuda_available": true,
  "cuda_device_name": "...",
  "hostname": "..."
}
```

任何后续 sprint 引用 profile 结果时，先引用 manifest，再引用 summary 或 trace。没有 manifest 的
历史 artifact 可以保留，但不能作为新结论的唯一证据。

## 5. 正确性测试

### P0 — 单元测试：无 GPU 也能证明 API contract

- `profile_range` 在 `torch` import 失败或 profiler disabled 时不会影响业务返回值。
- `profile_range` 在 `emit_nvtx=false` 时不调用 `torch.cuda.nvtx`。
- `profile_range` 在 `emit_nvtx=true` 且 `range_push` 成功时，异常路径也保证 `range_pop` 成对执行。
- `safe_worker_name` / label sanitizer 对空值、空格、路径分隔符、unicode 输入稳定。
- `TorchProfilerConfig` normalize 后没有手写重复 allow-list；如果要校验字段，从 dataclass fields 派生。

### P1 — 单元测试：activity 解析可信

- requested `["cuda"]` 但 `supported_activities()` 不含 CUDA 时，默认 fail fast。
- requested `["cpu", "cuda"]` 在 CPU-only 环境能给出明确 missing CUDA，而不是输出“正常 trace”。
- `activities` 包含未知值时 fail fast，而不是忽略。
- `device` 是 `cuda:0` / `torch.device("cuda")` / string 的路径都一致。

### P2 — 真实 PyTorch smoke：CPU trace

用一个小矩阵乘或 tensor op 跑 CPU-only trace：

```python
with capture_torch_trace(... activities=("cpu",)):
    with profile_range("test.outer"):
        x = torch.ones(4, 4)
        y = x @ x
```

验收：

```text
trace json exists
summary exists
manifest exists
manifest.effective_activities == ["cpu"]
trace contains "test.outer"
summary contains at least one torch op
```

这条测试应进入常规 CPU CI，因为它不需要 CUDA。

### P3 — CUDA smoke（有 GPU 才跑）

用小 CUDA matmul 跑 `activities=("cpu","cuda")`：

```text
manifest.effective_activities contains cuda
trace json contains CUDA activity or manifest explicitly says CUDA trace unavailable
summary has CUDA section
profile_range("test.cuda_matmul") encloses the matmul range
```

如果 CUDA available 但 CUPTI/trace 不完整，测试必须暴露这个事实。可以 skip GPU 测试，但不能让
profile run 看起来成功。

### P4 — nsys/NVTX runbook smoke（人工或 nightly）

目标不是让 Python 管理 nsys，而是证明 NVTX range 名能被 nsys 读到。最小命令：

```bash
VRL_PROFILE=1 nsys profile --trace=cuda,nvtx --output outputs/profile_smoke/nsys_profile \
  python -m vrl.scripts.perf.profile_smoke
```

验收：

```text
nsys report exists
nsys stats can show NVTX range names
document warns nested NVTX ranges are attribution aids, not additive wall-time percentages
```

## 6. 执行顺序

- [x] **P0 — 写 profile smoke 脚本和 CPU contract tests。** `vrl/scripts/perf/profile_smoke.py`
  + `tests/utils/test_profiling.py`（P0/P1 contract + P2 真实 CPU trace）。
- [x] **P1 — 引入 `profile_range`，保留 `record_function` alias。** 主入口
  `trainer.step` / `worker._profile_forward_chunk` 改用 `profile_range` / `capture_torch_trace`；
  深层 `record_function(...)` ranges 走 alias，按非目标不一次性机械迁移。
- [x] **P2 — 重写 activity resolver。** `_resolve_activities` 基于 `supported_activities()` 派生
  effective set（合法名从 `ProfilerActivity.__members__` 派生，未知名 fail fast），
  requested/effective/missing 全部写进 `profile_manifest.json`。
- [x] **P3 — 拆 `capture_torch_trace` 的内部职责。** 私有 helper：`_resolve_activities`、
  `_write_summary`、`_write_manifest`、`_discover_trace_files`，各有测试或直接消费者。
- [x] **P4 — 复核 schedule 语义。** 确认是单 step capture（每个合格 step 开一个独立 profiler），
  在 `capture_torch_trace` / `_should_profile_step` docstring 写明，不引入半套手写 schedule。
- [x] **P5 — 更新 docs 和 profile preset。** `configs/profile/torch_profiler.yaml` 保持唯一短诊断
  preset；重开销深挖通过 dotlist overrides 临时打开，不维护第二个 YAML。
- [x] **P6 — 跑真实 smoke 三件套 + nsys NVTX runbook。** 见下方完成记录。

### 完成记录（2026-06-27）

```text
pytest tests/utils/test_profiling.py                      -> pass
pytest .../test_cli_overrides_reach_typed_trainer_config  -> 1 passed
profile_smoke --activities cpu                            -> trace+summary+manifest, effective=['cpu']
profile_smoke --activities cpu,cuda  (RTX 5090)           -> effective=['cpu','cuda'], device_name 写入 manifest
VRL_PROFILE=1 nsys profile --trace=cuda,nvtx ...          -> nvtx_pushpop_sum 读到 test.outer / test.cuda_matmul
唯一 profile preset find_unknown_keys -> []
```

NVTX 注意：上面 nsys 报告里 `test.outer` 99.6% 与 `test.cuda_matmul` 0.4% 是嵌套 range，
matmul 也被 outer 包含；不能把两者百分比相加当 wall-time 拆分。summary 头部已写同样的告警。

## 7. 验收标准

本 sprint 完成时必须满足：

```text
pytest -q tests/utils/test_profiling.py
pytest -q tests/config/test_load_all_experiments.py::test_cli_overrides_reach_typed_trainer_config
python -m vrl.scripts.perf.profile_smoke --activities cpu --output-dir outputs/profile_smoke_cpu
```

有 GPU 的机器额外跑：

```text
python -m vrl.scripts.perf.profile_smoke --activities cpu,cuda --output-dir outputs/profile_smoke_cuda
```

真实系统验收：

```text
短 trainer profile 目录含 trace + summary + manifest
短 rollout profile 目录含 trace + summary + manifest
manifest 的 requested/effective/missing activities 与实际机器状态一致
trace 中能看到 trainer.loss / trainer.replay / generation.denoise_forward 等现有 ranges
summary 文档明确不能把嵌套 NVTX range 百分比相加当 wall-time
```

## 8. 非目标

- 不新增第二套 wall-time profiler/counter。阶段归因仍走 torch profiler ranges 和 NVTX ranges。
- 不把 nsys CLI 包进 Python runtime。nsys 是外部采集器，VRL 只提供 NVTX 标注和 runbook。
- 不把所有现有 `record_function(...)` 调用一次性机械改完。先保 alias，之后按模块迁移。
- 不用 hardcoded ALL_CAPS allow-list 维护合法 config key；字段集合从 dataclass/schema 派生。
- 不为了减少 LOC 合并所有私有 helper。`capture_torch_trace` 下的 thin helpers 如果各自承担
  activity 解析、manifest 写入、summary 导出、文件校验边界，就应该保留。

## 9. 参考

- 当前 profiling 实现：`vrl/utils/profiling.py`
- Trainer 调用点：`vrl/trainers/online/trainer.py`
- Generation worker 调用点：`vrl/generation/execution/worker.py`
- Profile preset：`configs/profile/torch_profiler.yaml`
- 历史性能方法论：`docs/sprints/info/SPRINT_rollout_performance.md`
- PyTorch profiler API：https://docs.pytorch.org/docs/2.12/profiler.html
- Nsight Systems Analysis Guide：https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
