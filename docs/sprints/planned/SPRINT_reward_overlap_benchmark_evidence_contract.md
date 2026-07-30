# SPRINT: Reward-overlap benchmark evidence contract（planned）

状态：**planned / CPU-only verdict implementation**。不运行昂贵训练 campaign。

## 根因

`vrl/scripts/perf/reward_overlap_benchmark.py` 会直接决定 overlap 特性是否达到 acceptance，
但当前 evidence reader 是 fail-open：

```python
if verdict_path.exists():
    ...

row.get("collect.reward_wall", 0.0)
```

并且 resume 只看 `rollout_stats.jsonl` 是否存在：

```python
if (... / "rollout_stats.jsonl").exists():
    print("already complete")
```

CPU 合成数据已复现：

- A/C 共 10 个 run 全部缺 `run_verdict.json`，仍 `accepted=True`；
- C 臂部分 row 缺 `collect.reward_wall`，仍 accepted，且
  `realized_fraction_of_theoretical=1.5`；
- 文档声称检查 `rollout.prompts_per_batch >= 2`，实现没有该 preflight；
- 不同 resolved workload 的 run 可以混在同一个 verdict。

这是 correctness bug：错误 acceptance 比没有 benchmark 更危险。

## Evidence contract

### Run completion

每个 run 必须同时有：

- `run_verdict.json`：object、`schema_version == 1`、`verdict == "success"`；
- `rollout_stats.jsonl`；
- `resolved_config.yaml`。

目录存在但 evidence 不完整时不得当作 completed，也不得在同目录继续追加 attempt；
benchmark fail loud，保留现场给操作者检查。

### 承重 metrics

每个含 `collect.wall` 的 steady-state row 必须同时含：

```text
collect.generation_wall
collect.reward_wall
collect.generation_reward_overlap
```

值必须是 finite、non-negative number；`collect.wall` 必须正数。`step` 必须是唯一的
非负整数。`reward.queue_wait_s` 继续可选，因为它只用于 diagnostic；存在时也必须
finite/non-negative。

### Workload identity

从每个 run 的 `resolved_config.yaml` 解析 canonical workload，只删除两个有意变化的
字段：

```text
trainer.output_dir
trainer.rollout_orchestration.reward_collection_mode
```

其余 config 必须 byte-canonical 等价。SHA-256 写入 `acceptance.json`，让 verdict
带可追溯 workload identity；任何 model revision、batch size、sampling、reward
identity 或训练长度变化都拒绝混合。

### Launch preflight

启动 subprocess 前验证：

- `iterations > warmup_iterations >= 0`；
- `rollout.prompts_per_batch >= 2`；
- 请求/已有 successful run 的组合最终含 A 与 C 判定臂。

## 审计裁决

| Suspect | 裁决 | 原因 |
|---|---|---|
| optional run verdict | **FIX** | explicit training completion protocol，不可猜测 |
| missing metric 默认 0 | **REMOVE** | 伪造 denominator / theoretical saving |
| resolved workload duplication | **DERIVE** | 从每个 run 的 canonical resolved config 派生 hash |
| `ARMS` A/B/C mapping | **KEEP** | deliberate benchmark taxonomy；value 从 `RewardCollectionMode` enum 派生 |
| threshold constants | **KEEP** | public acceptance protocol |
| `_T_ONE_SIDED_95` | **KEEP** | 刻意隔离的统计表，不是 workflow vocabulary |
| 独立 benchmark script | **KEEP** | 长期 perf acceptance entrypoint，不是 one-shot probe |

## True / false regressions

- 完整、同 workload 的 A/C campaign 通过；
- 缺 verdict、错误 schema、failed verdict、缺任一承重 metric、NaN/Inf/负数、
  duplicate step 均拒绝；
- output dir 与 arm mode 不同仍可比较；
- batch size/model revision/reward/sampling 任一变化后拒绝；
- incomplete existing run 不会被跳过；
- `prompts_per_batch=1` 在 `subprocess.run` 前失败；
- `reward.queue_wait_s` 缺失仍合法，证明 diagnostic 与承重 evidence 分离。

## 非目标

- 不运行 A/B/C GPU campaign；
- 不创建统一 benchmark result framework；
- 不改变 overlap 调度、reward runtime 或 acceptance thresholds；
- 不删除 checkpoint discard：它是这个长期 benchmark 的明确空间上界。

## 验收

```bash
.venv/bin/python -m pytest -q tests/scripts/perf/test_reward_overlap_benchmark.py
```

## References

- `vrl/scripts/perf/reward_overlap_benchmark.py`
- `tests/scripts/perf/test_reward_overlap_benchmark.py`
- `vrl/scripts/train.py::write_run_verdict`
- `vrl/trainers/checkpointing.py::save_resolved_config`
- `docs/sprints/done/SPRINT_reward_service.md`
