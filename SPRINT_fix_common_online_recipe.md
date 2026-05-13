# SPRINT：Common online recipe

状态：active family train script 迁移已完成。SD3.5、Janus-Pro、NextStep、Wan、Cosmos 入口都通过 `run_online_recipe(...)` 共享 collector/runtime/evaluator/algorithm/trainer glue；family script 只保留模型构建和少量 hook。

## 目标

把各 family train script 中重复的 online RL glue 收敛到 common recipe 层。

当前 `vrl/scripts/train.py` 已经是统一 dispatcher，真正重复的是每个 family recipe 内部的这条链路：

```text
YAML -> typed config -> reward -> collector/runtime -> evaluator -> algorithm -> OnlineTrainer -> loop/checkpoint/CSV
```

这个 sprint 的目标是让 family train script 只保留模型构建和少量 hook，不再手写 rollout/evaluator/algorithm/trainer glue。

执行时机：这个 sprint 必须放在 `SPRINT_fix_rollout_extras_thinning.md`、evaluator strict cleanup、`SPRINT_fix_algorithm_strict_input.md`、`SPRINT_fix_engine_plan_execution.md` 之后。否则 common recipe 会把当前 transitional adapter 一起抽象进去，形成新的 legacy。

## 不做的事

- 不迁移 DPO offline trainer；`wan_2_1/train_dpo.py` 后续可以单独做 offline recipe。
- 不把尚未 strict 的 evaluator/algorithm 适配逻辑藏进 common recipe。
- 不复制旧 script 到 `factory.py` 后宣称统一。
- 不改变 active experiment YAML 行为。
- 不承担 trajectory/evaluator/algorithm/engine contract 迁移；这里只整理 family train glue。

## 当前重复点

典型重复：

```python
collector.set_runtime(
    build_rollout_backend_from_cfg(
        cfg,
        driver_bundle=bundle,
        runtime_spec=rollout_runtime_inputs.runtime_spec,
        gatherer=rollout_runtime_inputs.gatherer,
    ),
)
```

和：

```python
trainer = OnlineTrainer(
    algorithm=algorithm,
    collector=collector,
    evaluator=evaluator,
    model=model,
    weight_syncer=build_runtime_weight_syncer(...),
    config=trainer_config,
    device=device,
    stat_tracker=stat_tracker,
)
```

重复范围：

- config load、profile env、resume、dtype/device/seed。
- reward construction。
- collector config construction。
- runtime inputs 和 rollout backend setup。
- evaluator/algorithm dispatch。
- `OnlineTrainer` construction。
- loop、checkpoint、CSV、eval hooks。

## 新增结构

新增：

```text
vrl/scripts/recipes/__init__.py
vrl/scripts/recipes/types.py
vrl/scripts/recipes/factory.py
vrl/scripts/recipes/online.py
tests/scripts/recipes/test_online_recipe_factory.py
tests/scripts/recipes/test_online_recipe_runner.py
```

当前实现状态：

- 已新增 `vrl/scripts/recipes/types.py`，定义 family hook、device context 和 common stack。
- 已新增 `vrl/scripts/recipes/factory.py`，把 YAML + family registry 投影为 collector config、reward、algorithm、evaluator。
- 已新增 `vrl/scripts/recipes/online.py`，提供 common runner 骨架，覆盖 bundle 构建、collector/runtime、trainer、CSV、checkpoint、prompt loop。
- `vrl/scripts/wan_2_1/train.py` 已迁移为第一条 golden path，只保留 Wan bundle / gradient checkpointing / KL ref model / LoRA export hooks。
- 已新增 `tests/scripts/recipes/*`，覆盖 active online experiment 的 dispatch 构造，以及 runner 的 fake golden path。
- 还没有迁移所有 family train script；SD3.5、Janus、NextStep、Cosmos 仍按后续 phase 单独切，避免把固定 eval、EMA、reference image、eval-only、optimization probe 这些 family hook 混进第一条 golden path。

核心类型：

```python
OnlineRecipeDefinition(
    family: str,
    build_bundle: Callable[..., RuntimeBundle],
    export_module_getter: Callable[..., Any] | None,
    reference_provider: Callable[..., Any] | None,
    prompt_provider: Callable[..., Any] | None,
    loop_hooks: OnlineLoopHooks,
    csv_extra_fields: tuple[str, ...],
)
```

核心函数：

```python
build_collector_config_from_cfg(cfg, family_entry)
build_reward_from_cfg(cfg, built)
build_algorithm_and_evaluator(cfg, family_entry)
build_online_trainer(cfg, definition, bundle, collector, algorithm, evaluator)
run_online_recipe(cfg, definition)
```

## 实施阶段

### Phase 1：抽 factory，不改行为

编辑或迁移：

```text
vrl/scripts/eval_common.py
vrl/scripts/recipes/factory.py
vrl/rollouts/family_registry.py
vrl/rollouts/collector/factory.py
```

要求：

- 把 `_collector_config_from_cfg(...)` 这类训练也需要的逻辑移到 recipe factory 或更底层 rollout helper。
- reward builder 收敛到同一入口，NextStep 不再手写 `OCRReward` 特例。
- algorithm/evaluator dispatch 只有一个实现点。
- factory 只依赖 config 和 family registry，不硬编码 family script。

### Phase 2：迁移 Wan GRPO 作为黄金路径

编辑：

```text
vrl/scripts/wan_2_1/train.py
```

要求：

- Wan family script 只保留 model/bundle builder 和 family-specific hook。
- runtime setup、trainer construction、loop、checkpoint、CSV 走 common recipe。

状态：已完成。

### Phase 3：迁移 SD3.5

编辑：

```text
vrl/scripts/sd3_5/train.py
```

要求：

- 保留 SD3 fixed eval hook、EMA eval、driver frozen module offload hook。
- SD3.5 OCR 入口和行为保持兼容。
- 不为了抽象 common recipe 删除 SD3.5 OCR 回归测试。

### Phase 4：迁移 Janus / Janus R1 / NextStep

编辑：

```text
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
```

要求：

- `janus_pro` / `janus_pro_r1` 通过 family registry entry 派生，不在 train script 用 `r1_mode` 写两套 collector branch。
- NextStep reward 和 continuous evaluator 走 common factory。
- AR family 的 prompt provider、export module getter 作为 recipe hooks。

### Phase 5：迁移 Cosmos Predict2.5 / Predict2

编辑：

```text
vrl/scripts/cosmos/train.py
```

要求：

- Predict2.5 DiffusionNFT 接入 common collector/runtime/trainer。
- 保留首步 optimization probe hook。
- Predict2 的 reference image、eval-only、checkpoint eval sample hook 作为 recipe hooks，不回到 family hardcode。

### Phase 6：清理 family scripts

要求：

- family train script 不再直接手写 `build_rollout_runtime_inputs(...)`。
- family train script 不再直接手写 `build_rollout_backend_from_cfg(...)`。
- family train script 不再直接构造 `OnlineTrainer(...)`。
- duplicated CSV/checkpoint/resume loop 删除。

## 测试计划

新增或编辑：

```text
tests/scripts/recipes/test_online_recipe_factory.py
tests/scripts/recipes/test_online_recipe_runner.py
tests/config/test_load_all_experiments.py
tests/rollouts/test_family_registry.py
tests/rollouts/test_runtime_inputs.py
tests/trainers/test_online.py
```

新增断言：

- 每个 active experiment 能构造正确 collector config、algorithm 类型、evaluator 类型、runtime family。
- common runner 调用 `OnlineTrainer.step`，写 CSV header，保存 final checkpoint，resume 起点正确。
- common factory 只依赖 registry，不在 recipe 层 hardcode rollout family。
- SD3.5 OCR config load 和 smoke gate 不变。

## 完成标准

- 所有 online family train scripts 不再直接手写 runtime backend setup 和 `OnlineTrainer(...)`。
- collector config 从 YAML + family registry 自动映射。
- algorithm/evaluator dispatch 只有一个实现点。
- SD3.5、Wan、Janus、Janus R1、NextStep、Cosmos active experiment YAML 入口都能 load、dispatch、构造 recipe。
- 通过：

```bash
pytest tests/config/test_load_all_experiments.py \
  tests/rollouts/test_family_registry.py \
  tests/rollouts/test_runtime_inputs.py \
  tests/trainers/test_online.py \
  tests/scripts/recipes/test_online_recipe_factory.py \
  tests/scripts/recipes/test_online_recipe_runner.py
```

## 参考路径

- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/sd3_5/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/wan_2_1/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/janus_pro/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/nextstep_1/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/cosmos/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/eval_common.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/family_registry.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/collector/factory.py`
