# SPRINT: 退役 rollout sleep A/B 探针（planned）

状态：**planned / ready**。删除 `vrl/scripts/perf/rollout_sleep_probe.py`（198 行），不增加长期 GPU run。

## 目标

该脚本是 [[SPRINT_frozen_component_preservation]] 的一次性验收产物。它比较 naive CPU round-trip 与 vLLM CuMem sleep/wake，结果已经完整记录；当前生产又由 `97b3a89f` 收窄为 declared-CuMem family 必须使用 `CumemPool`，naive 臂不再是生产能力。

删除条件已经满足：

- 零 importer、零测试、零 preset、零 registry/entry point。
- 用 `SimpleNamespace` 手搓旧 `ModelBuild` 形状；当前 `from_build` 读取 `pretrained_kwargs`，脚本在分配模型前已经失效。
- 直接驱动 raw `CuMemAllocator`，绕过生产 `CumemPool.building/sleep/wake/close` 生命周期。
- `tests/generation/execution/test_worker_sleep.py` 已有真实 CuMem 小张量 round-trip、值保持、one-shot scope 与 terminal release 测试。
- 生产每次 park 都通过 `WorkerMemoryParkingSnapshot.validate()` 检查 physical residual。

事实边界：生产只持续校验 **residual**。probe 记录的 fragmentation delta 与 sleep/wake latency 没有生产 consumer；它们是已归档的一次性测量，不应误写成在线硬校验。

## 改动

1. 删除 `vrl/scripts/perf/rollout_sleep_probe.py`。
2. 在 `docs/sprints/done/SPRINT_frozen_component_preservation.md` 的两处脚本引用旁注明：探针在产出表格后退役，表格是最终 provenance。

## 保持不变

- 不修改 `CumemPool`、`WorkerMemoryParkingSnapshot`、memory parking owner 或 residual limit。
- 不删除 `tests/generation/execution/test_worker_sleep.py` 的 GPU contract test。
- 不增加多 GiB 真模型常驻测试。它慢、依赖 checkpoint/GPU 空闲状态，而且不比生产 residual gate 提供更稳定的 correctness contract。
- 不删除其他 perf probe；它们必须各自按“答案是否归档、是否仍可跨环境复用、是否经过生产边界”判定。

`CumemPool` 的薄方法应保留：它们是 vLLM framework adapter、one-shot protocol guard 与 fail-closed terminal release 边界。`CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT` 也是环境变量/运行协议边界，保持不变。

## 验收

```bash
rg -n 'rollout_sleep_probe' \
  vrl tests README.md docs/sprints/planned docs/sprints/parked
# expected: no matches

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  tests/generation/execution/test_worker_sleep.py \
  -q -m 'not gpu and not e2e and not slow_test'
```

有空闲卡时只运行现有短门，不加载真模型：

```bash
.venv/bin/python -m pytest \
  tests/generation/execution/test_worker_sleep.py \
  -q -m gpu -k real_cumem_one_shot_scope_sleep_wake_in_subprocess
```

本 sprint 只删除 Python 文件并编辑 Markdown，无需对未修改的 Python 文件跑 Ruff。

## References

- `vrl/scripts/perf/rollout_sleep_probe.py`
- `docs/sprints/done/SPRINT_frozen_component_preservation.md`
- `vrl/utils/cuda_memory.py`
- `vrl/generation/execution/memory_parking.py`
- `vrl/generation/execution/types.py`
- `tests/generation/execution/test_worker_sleep.py`
- `97b3a89f` (`refactor(generation): require CuMem for declared cumem parking`)
