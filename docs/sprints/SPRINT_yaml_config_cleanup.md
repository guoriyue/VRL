# SPRINT: YAML config cleanup — 死 key / 假选项 / 孤儿设置

状态：批次 1-3 implemented（2026-06-09）；批次 4 deferred（与在途 reward 放置重构撞车，见 §4）。

## 0. 一句话

对全部 84 个 YAML / 254 个 key 做了多代理审计（8 域清点 + 8 域对抗复核，workflow
`wf_84e8047e-fcb`）：23 个 key 有问题。根因是 `vrl/config/schema.py` 的
`extra="ignore"`（×10 处）让没人读的 key 静默通过——用户以为配置生效了，实际什么都没发生。

## 1. 已落地（批次 1-3，692 tests passed）

### 批次 1 — 删除死掉的 `eval.*` 整树（14 key，9 个 experiment YAML）

`eval.enable/freq/manifest/max_prompts/num_steps/sample_batch_size/noise_level/seed/
use_ema/fixed.*/prompts_file/seeds/eval_only/sanity_gates.*` —— 代码里**零读者**
（grep + cfg_get + getattr + schema 字段 + dataclass 字段全查过，亲自复核）。
用户看到 `eval.freq: 60` 会以为每 60 步评估一次；实际不存在评估子系统。

- 删除 9 个 experiment YAML 的 `eval:` 块。
- 同步删 `tests/config/test_load_all_experiments.py` 的 `cfg.eval.fixed` 结构性断言。
- **不混淆**：`data.eval_manifest` 是另一个 key、有真读者（`cosmos/anima/generate.py`），保留。

### 批次 2 — 删除假选项 `distributed.backend`

3 个 base YAML 全写 `backend: ray`，代码（`generation/ray/config.py:60-62`）对其它值直接
raise——这不是选项，是要求用户抄写的唯一答案。删 YAML 行 + 删代码读取/校验 + 清 4 处测试
fixture + 删 1 个失去意义的默认值测试。backend 即 Ray，不再可配。

### 批次 2b — orchestration「唯一合法值」改为派生

`weight_sync_barrier` 在每种 mode 下只有一个合法值（strict→`before_sync`，
continuous→`pause_admission_and_drain_inflight`），原来却要求 YAML 逐字抄写。
现在 `RolloutOrchestrationConfig.weight_sync_barrier` 默认 `None` → 按 mode 派生
（`vrl/trainers/core/types.py`）；显式写错仍 fail loudly。`strict.yaml` 删 2 行冗余、
`continuous.yaml` 删 1 行。

### 批次 3 — torch_profiler 孤儿设置搬家

`configs/base/trainer.yaml` 里躺着 11 行 profiler 调参，但同文件 `profile: false`——基线
永远用不上，纯属误导。整块删除；`builders.py` 把 `trainer.torch_profiler` 从 require 改为
optional（缺省走 `vrl/utils/profiling.py` 的 dataclass 默认值，**与原 YAML 逐项相同**，
行为零变化）。要 profiling = 组合 `/profile/torch_profiler` preset（早已存在且自带
`profile: true`）。

## 2. 验证

```bash
pytest -m "not e2e and not slow_test" -q   # 692 passed
```

包含与另一会话在途改动（reward 放置重构）共存的全量回归。

## 3. 明确不动的（审计确认健康）

- 科研超参：`algorithm.*` 33 key、`sampling/**` 37 key、`model.*` 的 lora/compile/memory。
- `trainer.rollout_orchestration.mode`（strict vs continuous 是真实选择）、
  `max_pending_rollouts`（continuous 下是真实选择）、continuous 的 7 个 backpressure key。
- `fail_fast_errors`：不在任何 YAML，但 `schedule.py:99` 默认 3 是真读者（test-only 调参），保留。
- `torch_profiler.output_dir` 空串自动派生逻辑（`profiling.py:137`）本来就对，随块搬家。

## 4. 推迟项

### 4a. `share_with_rollout` 三态化（批次 4，audit 唯一中风险项）

结论已定：unset = 按 GPU 拓扑自动推导（够卡就给 reward 专卡，不够回落共享），
`true`/`false` 保留为显式 override——消除「多卡机器上显式 true 仍强制挤单卡」的
footgun（实录见 `SPRINT_cosmos_performance.md`）。
**为什么推迟**：另一会话正在重构同一区域（`vrl/ray/resources.py`、release 生命周期已改为
从 `share_with_rollout` 派生，30 文件在途）。现在动 = 撞车。他们落地后做这步反而更简单
（届时 `share_with_rollout` 已是唯一放置旋钮）。

### 4b. 根因修复：`extra="ignore"` → `extra="forbid"`

schema.py 第 5 行自己写着 ignore 是「migration 期临时态」。整棵 eval 死树就是它的产物。
分两步：先加 CI 对照测试（resolved config vs schema 字段集 diff，新增未知 key 即 fail）
跑稳一轮，再统一翻 forbid 并修暴露的存量。不先清存量直接 forbid 会炸出大量无关失败。

## 5. 关键参考

- 审计 workflow：`wf_84e8047e-fcb`（254 key 清点 / 30 findings / 8+8+1 agents）
- `vrl/config/schema.py`（extra="ignore" ×10）、`vrl/config/builders.py`
- `vrl/trainers/core/types.py`（barrier 派生）、`vrl/utils/profiling.py`（profiler 默认值）
- `vrl/generation/ray/config.py`（backend 读取已删）
- `configs/base/trainer.yaml`、`configs/base/rollout/orchestration/{strict,continuous}.yaml`、
  `configs/base/distributed/*.yaml`、`configs/profile/torch_profiler.yaml`
