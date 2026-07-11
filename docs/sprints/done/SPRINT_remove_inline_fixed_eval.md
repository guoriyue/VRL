# SPRINT: Remove inline fixed eval and restore evaluation boundaries（planned）

状态：DONE（2026-07-11）。本 sprint 已删除 online training loop 内的 fixed eval，
保留可复现的 checkpoint evaluation 能力，并把它放回 `vrl/scripts/eval/` 的独立进程边界。

落地摘要：inline implementation、`EvalConfig`、`trainer.eval` presets、training CSV writer 和
global eval tests 已删除；`data.eval_manifest` 与现有 Cosmos/Kling standalone evaluator 保留；
遗留 `trainer.eval` 现在 hard-fail 并给出迁移命令。针对性验证为 129 passed / 2 deselected；两项
deselected 是与此次 fixed-eval 改动无关的 `samples_per_chunk="auto"` 类型比较失败。Standalone evaluator
CLI 已验证可启动，但未运行需要真实 checkpoint/GPU 的实验作业。

历史 fixed eval 解决了一个真实问题：rotating training prompts 的 `reward_mean` 不能直接比较不同
epoch。后来它被扩展成全 rank global phase，复用训练中的 collector、reward、distributed process group
和 colocated rollout runtime。这个执行位置让一个可选指标开始了解 training schedule 的 GPU handoff，
并在 on-demand runtime 引入后漏掉 `activate()`，形成回归。

本 sprint 不修补这条旁路。根因是 evaluation 被嵌进 training orchestration；处理方式是删除旁路，
让 training 只生产 checkpoint，让独立 evaluator 加载 checkpoint 后执行固定 prompt/seed 协议。

---

## 0. 核心结论

Fixed evaluation 包含两个被错误绑定的概念：

```text
fixed protocol:
  same held-out prompts + same seeds + same sampling + same metrics

inline execution:
  pause a live training job and evaluate its in-memory weights
```

前者对 checkpoint 比较有用；后者不是训练正确性、fast probe 或普通 checkpoint 产出的必要条件。
VRL 当前没有基于 fixed eval 自动选择 best checkpoint、early stop 或控制 optimizer，因此没有足够收益
抵偿它引入的生命周期和 collective 复杂度。

目标结构：

```text
training process:
  train -> save complete checkpoint -> continue/exit

evaluation process:
  load checkpoint -> fixed prompt/seed grid -> generate -> score -> report -> shutdown
```

禁止的旧结构：

```text
online.py
  -> trainer.rollout_schedule.lifecycle
  -> partial GPU handoff
  -> collector.generate()
  -> on-demand runtime rejects generate before activate()
```

---

## 1. 已核实的现状与根因

### 1.1 Inline eval 是可选观测，不参与学习

`EvalConfig.enabled` 默认是 `false`。启用后，online loop 在训练前跑 baseline，在指定 epoch 后再次跑
固定 prompt/seed grid，最终写 `eval_metrics.csv`。它不 backward、不更新 optimizer、不更新 EMA，
训练结果不依赖它。

因此文档和 docstring 中的 “the learning signal” 应纠正为 “an evaluation signal”。学习信号来自训练
rollout/reward/loss；fixed eval 只测量 checkpoint。

### 1.2 当前耦合是旁路，不是 engine 能力

`vrl/scripts/common/online.py::_fixed_eval_and_log` 直接取得
`trainer.rollout_schedule.lifecycle`，手动调用 driver-model offload，然后绕过 schedule 调
`_run_distributed_fixed_eval`。后者经 collector 直接进入 runtime `generate()`。

正常 strict schedule 的资源顺序是：

```text
offload driver -> activate rollout -> generate -> offload rollout -> restore driver
```

fixed eval 只复制了部分顺序，没有 activate on-demand runtime。这次异常是边界泄漏后的必然结果，
不是 kernel 或 generation engine 应该新增 `fixed_eval` 分支的理由。

### 1.3 分布式 fixed eval 放大了可选功能的故障面

`vrl/scripts/common/fixed_eval.py` 当前负责：

- eval prompt rank sharding；
- world-size-independent seed grid；
- collector generation 和 reward scoring；
- reward sufficient statistics；
- NCCL/Gloo all-reduce；
- 空 shard collective participation。

任何 rank 在 generation/reward 中失败，都可能让其他 rank 等在 all-reduce 或 barrier。一个只写评估
CSV 的可选功能不应扩大 FSDP/DDP training job 的 collective failure surface。

### 1.4 已有 standalone evaluator 可作为迁移起点

`vrl/scripts/eval/cosmos_predict25_kling_eval.py` 已经支持：

- 一个或多个 checkpoint；
- manifest 或 `data.eval_manifest`；
- 固定 seed 和 samples-per-prompt；
- generation-only 或 Kling VideoReward scoring；
- 独立输出目录。

原始 Cosmos/Kling fixed-eval 用例不需要再造一个 training hook。该 evaluator 应成为此次迁移的第一条
canonical path，并按缺口做小幅完善，而不是新增第二套通用 eval framework。

---

## 2. 目标

### 2.1 Training 边界

- Online training 不加载 eval prompts。
- Online training 不创建或写 `eval_metrics.csv`。
- Online training 不因为 evaluation 执行额外 generation、reward scoring、all-reduce 或 barrier。
- Fast probe 只受其训练/rollout 配置控制，不隐式运行 evaluation。
- Checkpoint 保存保持现有 all-rank gather / primary writer 语义。

### 2.2 Evaluation 边界

- Evaluation 从完整 checkpoint 启动，不读取 live trainer model。
- Evaluation 可以使用 `data.eval_manifest` 或显式 manifest。
- Checkpoint 间使用相同 prompt order、seed derivation、sampling 参数和 reward 配置。
- Evaluation 独立拥有 generation/reward 资源；失败不阻塞 training collectives。
- 单 GPU 下允许 evaluator 自己按阶段释放 generation model，再加载 GPU reward model，但这个阶段切换
  属于 evaluator，不属于 rollout schedule。

### 2.3 Runtime 边界

- Generation runtime 继续只暴露通用 `activate/generate/offload/shutdown` 协议。
- Kernel、engine、worker 和 scheduler 不出现 `fixed_eval`、eval frequency、eval CSV 或 baseline epoch。
- 独立 evaluator 优先使用 resident runtime 或直接 family runtime；进程结束时 shutdown。
- 不允许为了 evaluation 在 `generate()` 内加入隐式 auto-activate。资源 owner 必须显式建立阶段。

---

## 3. 删除范围

### P0. 删除 inline training surface

从 `vrl/scripts/common/online.py` 删除：

- `fixed_eval` import；
- eval config 读取和 `eval_examples` 加载；
- `_fixed_eval_and_log`；
- pre-RL baseline eval；
- periodic epoch eval；
- `eval_metrics.csv` prepare/write methods 和 run-state 字段；
- eval-only barrier、logging 和 lifecycle probing。

删除 `vrl/scripts/common/fixed_eval.py`。它的 global-phase implementation 不应作为长期库保留；独立
evaluation 不使用 training ranks，因此 rank sharding/all-reduce/empty-shard handling 全部失去生产调用者。

### P1. 删除 training config surface

删除：

- `vrl.trainers.core.types.EvalConfig`；
- trainer schema 的 `eval` block；
- builders/validation 中只服务 `trainer.eval` 的接线；
- experiment presets 中的 `trainer.eval` blocks 和声称训练会写 `eval_metrics.csv` 的注释。

旧配置不做 silent ignore。删除 schema 后，遗留 `trainer.eval` 应作为 unknown key fail fast，提示用户改用
standalone evaluator。

### P2. 删除或迁移 inline-only tests

删除不再对应生产行为的测试：

- distributed fixed-eval sharding/all-reduce smoke；
- online loop baseline/periodic eval ownership；
- “fixed eval never backward” 的 inline hook 测试；
- `EvalConfig` schema/build tests。

更新 rank-ownership 测试，使其只覆盖仍然存在的 training rollout、training collective 和 checkpoint
writer ownership。不要保留一个 test-only fixed-eval module。

---

## 4. 保留与迁移范围

### 4.1 保留 `data.eval_manifest`

`data.eval_manifest` 是 dataset 的 held-out split 边界，不是 inline training feature。它还被 standalone
evaluation、数据 bootstrap 和数据完整性 validation 使用，因此保持 schema、loader 和数据生成工具不变。

不要因为删除 `trainer.eval` 而删除 eval manifests 或把 held-out rows 混回 training manifest。

### 4.2 保留 fixed protocol，不保留 global training phase

以下语义保留在 standalone evaluation：

```text
prompt order: manifest order
prompt limit: truncate before evaluation
group seed: base_seed + global_prompt_index * samples_per_prompt
comparison: identical sampling/reward config across checkpoints
```

这套协议应在 evaluator 的 CLI help、result metadata 或 run manifest 中显式记录。可复现性来自输出记录，
不是来自 training process ownership。

### 4.3 完善现有 Cosmos evaluator

以 `vrl/scripts/eval/cosmos_predict25_kling_eval.py` 为原始 Cosmos/Kling 用例的 canonical entrypoint。
实现时核对并补齐：

- checkpoint、resolved config、manifest path/hash；
- prompt index、sample index 和实际 seed；
- generation/sampling overrides；
- reward model和 score key；
- per-checkpoint aggregate mean/std/stderr/count；
- machine-readable summary，供 curve/checkpoint comparison 工具读取。

如果现有输出已包含某项，不重复建立第二个 source of truth。

### 4.4 处理现有消费者

审计所有读取 `eval_metrics.csv` 的长期工具。例如
`vrl/scripts/eval/sana_aesthetic_curve_verdict.py` 必须二选一：

- 迁移为读取 standalone evaluator 的 documented summary；或
- 如果它只服务已完成的历史实验，则明确标成 legacy 并拒绝新 run。

`docs/runs/` 中已归档的历史 `eval_metrics.csv` 不删除、不改写；历史结果是证据，不是当前 production
API。`docs/runs/README.md` 应区分 legacy inline-eval artifact 与新的 standalone evaluation output。

---

## 5. 执行顺序

- [x] **P0 — Freeze the replacement contract.** 记录 standalone evaluator 的 CLI、fixed seed derivation、
  result metadata 和 aggregate schema；确认原 Cosmos/Kling checkpoint 能独立加载。
- [x] **P1 — Remove inline execution.** 删除 `online.py` eval hook、global phase、CSV writer 和
  `vrl/scripts/common/fixed_eval.py`。
- [x] **P2 — Remove training config.** 删除 `EvalConfig` / `trainer.eval` schema 和 presets；保留
  `data.eval_manifest`。
- [~] **P3 — Migrate the original use case.** Existing Cosmos evaluator 与 CLI 已保留并验证可启动；用 baseline 与 trained
  checkpoint 跑同一 manifest/seed grid，生成可比较 summary。
- [x] **P4 — Migrate or retire consumers.** 更新 curve/verdict 脚本与文档，不让新工具依赖 training run
  内的 `eval_metrics.csv`。
- [x] **P5 — Delete obsolete tests and add boundary tests.** 测 standalone checkpoint comparison，确认
  online loop 不加载 eval manifest、不执行 eval collective、不创建 eval CSV。
- [x] **P6 — Config compatibility gate.** 遗留 `trainer.eval` 配置 hard-fail，并给出迁移错误信息；所有
  bundled presets 移除旧 block。

---

## 6. 验收标准

1. Online training 和 fast probe 均不读取 `data.eval_manifest`，不执行 fixed eval。
2. `online.py` 不引用 `fixed_eval`，不访问 schedule/coordinator 来执行 evaluation。
3. `vrl/scripts/common/fixed_eval.py`、`EvalConfig` 和 `trainer.eval` public surface 被删除。
4. Training run 不再创建新的 `eval_metrics.csv`。
5. Kernel、engine、generation runtime、rollout scheduler 没有新增 evaluation-specific branch。
6. 原 Cosmos/Kling 用例能由 standalone evaluator 对 baseline/trained checkpoints 使用同一 grid 评估。
7. Standalone result 记录 checkpoint、manifest、sampling、seed、reward identity 和 aggregate stats。
8. Evaluation failure 只使 evaluation process 失败，不会让 live training ranks 卡在 collective。
9. `data.eval_manifest`、dataset validation 和历史 run artifacts 保持可用。
10. 遗留 `trainer.eval` 配置给出明确迁移错误，而不是静默失效。

---

## 7. Thin functions/files 与 hardcoded data 审计

### 应删除

- `vrl/scripts/common/fixed_eval.py`：它是只被 online hook 使用的业务模块；删除 hook 后没有 production
  caller，保留会变成 test-only dead code。
- `_run_distributed_fixed_eval` 和 stats helpers：它们只为 training process-group global phase 服务，
  standalone evaluator 不需要。
- Online eval CSV methods：仅被一个已删除的调用链使用，不是公共 API facade。

### 应保留

- Existing evaluator 的 parser/main 边界：CLI 是真实 public entrypoint，不属于无意义薄函数。
- Dataset eval-manifest loader/validation：它们是 schema 和数据完整性边界。
- Generation runtime 的 `activate/generate/offload/shutdown`：它们是通用 protocol boundary，不能为了
  fixed eval 删除或内联。
- Family-specific evaluator：checkpoint loading、generation request 和 reward artifact 具有真实 family
  差异时，明确的脚本比装饰性的通用 manager 更容易审计。

### Hardcoded data

当前 fixed-eval module 没有需要迁移的模块级大型 ALL_CAPS taxonomy。Seed derivation 是 protocol，应该
作为一个有测试的函数或 evaluator 内明确公式存在，不复制成多个 magic constants。CLI defaults 可以保留，
但实际运行值必须写入 result metadata。

### 非目标

- 不为了减少 LOC 把不同 family 的 checkpoint loader 强行 data-ize。
- 不在只有一个 consumer 时创建 `EvaluationManager`、`EvaluationLifecycle` 或 `eval_utils.py`。
- 不删除历史 sprint、历史 `eval_metrics.csv` 或 held-out datasets。
- 不实现 asynchronous checkpoint watcher、job queue、best-checkpoint controller 或 early stopping。
- 不改变 reward model 数学、generation kernels、scheduler policy 或 on-demand lifecycle。
- 不把 standalone evaluator 接进 fast probe；需要评估时显式运行独立命令。

---

## 8. 风险与处理

### 8.1 Checkpoint I/O 成本

Standalone evaluation 必须先保存再加载 checkpoint。这是有意接受的隔离成本。不要为了省一次 I/O 把
evaluation 放回 live trainer。若成本实测成为瓶颈，后续单独设计 checkpoint snapshot/export，不污染
training scheduler。

### 8.2 Evaluation 与训练配置漂移

Evaluator 必须读取 checkpoint 对应的 resolved config，并把 CLI overrides 写入结果。比较多个 checkpoint
时，如果 model family、sampling 或 reward identity 不一致，应 fail fast，而不是生成不可比较的曲线。

### 8.3 单 GPU generation/reward 共存

若 generation model 与 learned reward model 无法同时驻留，evaluator 分成两个显式 stage：先生成并持久化
必要 artifacts，再释放 generation model并评分。这是 evaluator-owned handoff，不复用 training schedule。

### 8.4 历史分析兼容

旧 run 的 `eval_metrics.csv` 继续可读；新 evaluation output 不应伪装成 training epoch row。需要统一分析时，
在 reader 层兼容 legacy/new schema，不让 training writer 永久背负旧格式。

---

## 9. 关键文件

删除/修改：

- `vrl/scripts/common/online.py`
- `vrl/scripts/common/fixed_eval.py`
- `vrl/trainers/core/types.py`
- `vrl/config/schema.py`
- `vrl/config/builders.py`（仅删除 `trainer.eval` 接线，若存在）
- 含 `trainer.eval` 的 bundled experiment presets
- `tests/trainers/online/test_fixed_eval_distributed.py`
- `tests/trainers/online/test_reward_update_flow.py`
- fixed-eval ownership/config tests

保留/迁移：

- `vrl/scripts/eval/cosmos_predict25_kling_eval.py`
- `vrl/trainers/data/prompts.py` 的 eval-manifest loader（供 standalone scripts 使用）
- `vrl/config/schema.py::DataConfig.eval_manifest`
- `vrl/config/validation.py` 的 dataset split integrity validation
- `vrl/scripts/eval/sana_aesthetic_curve_verdict.py`（迁移 reader 或明确 legacy）
- `docs/runs/` 历史 artifacts

历史依据：

- `docs/sprints/done/SPRINT_cosmos_kling_fixed_eval_signal.md`
- `docs/sprints/done/SPRINT_distributed_fixed_eval.md`
