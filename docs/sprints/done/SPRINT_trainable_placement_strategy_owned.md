# SPRINT: 可训练权重的放置归训练策略所有（删除 `defer_trainable_device_move`）

状态：已完成。两步分别提交为 `ef61daca` 和 `ce645413`。验证记录见 §7、§8。

## 1. 问题

`ModelBuild.defer_trainable_device_move` 回答的是"谁把可训练 transformer 搬上卡"。
这是训练策略的事实（FSDP 用 `fully_shard` 逐块搬，单进程整体搬），但现在由注册表
用六个条件推出来，塞进 `ModelBuild`，穿过 Ray wire 边界，再由模型层三处
`if not flag: .to(device)` 消费：

```python
# vrl/models/families/registry.py:357
defer_trainable_device_move=(
    isinstance(self.family_build, DenoiseFamilyBuild)
    and self.family_build.replay_cls is not None
    and not for_rollout
    and root.distributed is not None
    and root.distributed.training is not None
    and root.distributed.training.strategy == "fsdp"
),
```

```python
# vrl/models/steps/denoise/base.py:539
def apply_full_finetune(self, build):
    transformer.requires_grad_(True)
    if not build.defer_trainable_device_move:
        transformer.to(self.device)
```

FSDP 策略自己已经在搬（`vrl/trainers/fsdp.py:210` 逐块 `fully_shard`；
`vrl/trainers/strategy.py:576` 的 `shard_trainable_only` 分支还显式
`handle.to(device)`）。模型层的搬运只是在和它抢。

### 1.1 之前一个错误判断的更正

方案早期版本认为 Wan 的 pipeline CPU offload 是训练路径上的例外。读了
`_resolve_wan_offload_mode`（`wan_2_1/model.py:1382`）之后确认它只读
`build.rollout.pipeline_offload_mode`，而 replay 构建的 `rollout` 恒为 `None`，
`ModelBuild.__post_init__` 也拒绝在 rollout 构建上设这个标志
（`interfaces/runtime.py:319`）。所以 pipeline offload 只存在于 rollout 路径，
与本方案要改的训练路径无关；Wan 的钩子覆写在 replay 路径上退化为基类默认。

## 2. 两条路径的边界

| 路径 | 构建入口 | 现在谁搬 | 方案之后谁搬 |
|---|---|---|---|
| replay（训练） | `assemble_replay_bundle` → `apply_lora` / `apply_full_finetune` | 模型层，除非标志为 True | **策略层** `prepare_model` |
| rollout（采样） | `build_denoise_runtime_bundle` → `move_to_device` 序列 | 模型层，依量化 / offload 决定时机 | 不变，条件改成从 `build.rollout` 直接读 |

DPO 走 rollout 路径（`vrl/scripts/families/wan_2_1/train_dpo.py:193` 用
`for_rollout=True` 取全 bundle），不受影响。

## 3. 改动清单

### 3.1 策略层：`prepare_model` 负责放置（`vrl/trainers/strategy.py`）

- `SingleProcessStrategy.prepare_model`（368 行）：从 `return model` 改为对
  `_trainable_module_handles(model)` 的每个 handle 执行
  `handle.to(self.context.device)`，再返回。
- `DDPStrategy.prepare_model`（963 行）：在 `DistributedDataParallel(handle, ...)`
  之前同样搬上 `self.context.device`。DDP 要求模块已经在 `device_ids` 对应的卡上，
  现在这一步是模型层替它做的。
- `ContextParallelStrategy.prepare_model`（834 行）：同上，在广播参数之前搬。
- `FSDPStrategy.prepare_model`：不动。`fully_shard` 从 CPU 逐块搬，
  `shard_trainable_only` 分支已有显式搬运。
- 三处共用一个 `_UnshardedStateStrategy` 上的放置步骤（三个调用方，不是单调用方
  helper）。

### 3.2 模型层：replay 路径不再搬（`vrl/models/steps/denoise/base.py`）

- `apply_lora`（455 行）：`defer_device_move = self._defer_trainable_device_move(build) or (rollout and quantization)`
  改为只在 rollout 路径且不需要延迟时搬：

  ```python
  move_now = (
      build.rollout is not None
      and not build.precision.quantization
      and build.rollout.pipeline_offload_mode is PipelineOffloadMode.NONE
      and not type(self).trainable_roots_preplaced
  )
  ```

  replay 路径（`build.rollout is None`）永远不搬，交给策略。
- `apply_full_finetune`（539 行）：同一个条件。
- 删除 `_defer_trainable_device_move` 钩子（421 行）。
- 新增类属性 `trainable_roots_preplaced: ClassVar[bool] = False`，语义是
  "可训练根已经由 family 自己放好，构建路径不要再搬"。这是 family 的事实，
  放在模型类上，不从配置推导，不过 wire 边界。

### 3.3 family 覆写

- `wan_2_1/model.py`
  - 删除 `_defer_trainable_device_move` 覆写（253 行）。它的两个分支分别被
    3.2 的通用条件（rollout offload）和 3.1（FSDP）覆盖。
  - `apply_full_finetune`（266 行）保留 dtype 归一化，设备搬运改用 3.2 的条件。
    dtype 是模型的事，设备不是。
- `cosmos/anima/model.py:175`：`apply_full_finetune` 同上，保留 `dtype=self._dtype`。
- `minimax_h3/partitioned_generation.py:73`：`_defer_trainable_device_move`
  覆写改为 `trainable_roots_preplaced = True`。这个类是 rollout 生成模型，
  冻结底座按块分散在多张卡上，走 `build_denoise_runtime_bundle` → `apply_lora`，
  仍需要这个信号。
- `minimax_h3/runtime.py:87`：`replace(build, defer_trainable_device_move=True)`
  删除。见 3.5。

### 3.4 `ModelBuild` 与注册表

- `vrl/models/interfaces/runtime.py`：删除字段（257-260 行）和
  `__post_init__` 里的两条校验（316-323 行）。
- `vrl/models/families/registry.py:357-365`：删除六条件表达式。
- Ray wire：`worker.py:826` 用 `dict(launch_contract.model_build)` 展开，
  字段删掉后 payload 自然少一个键，无需改序列化代码。

### 3.5 H3 分区 replay（`block_devices`）：保留

`load_h3_replay_components(build, block_devices=...)` 在 `vrl/` 里没有生产调用方，
但它不是死代码：2026-09-13 的 H3 预检报告
（`docs/research/h3_four_l40s_preflight_20260912.md` §"explicit partitioned replay
loader"）把它作为 4×L40S H3 计划的显式加载器交付，并在两张 GPU 上验证了
"分区生成 → 分区 replay" 的 log-prob 一致性（`test_device_dispatch.py` 的两个
两卡用例）。删掉它会抹掉一条有 GPU 证据的能力。

所以按备选方案处理：`build_minimax_h3_replay_runtime_bundle` 在
`block_devices` 非空时给模型实例设 `trainable_roots_preplaced = True`，
策略层的放置步骤看到它就不搬。加载器本身、`placement.py` 和测试都不动。

## 4. 测试

- 删除 / 改写：
  - `tests/trainers/test_fsdp.py:791-800`：断言注册表算出的标志 → 改为断言
    FSDP `prepare_model` 后参数在目标设备且为 DTensor（已有类似断言可复用）。
  - `tests/models/steps/denoise/common/test_lora_fp8_build.py:160-180`：
    `test_fsdp_replay_lora_attach_defers_device_move` →
    `test_replay_lora_attach_never_moves`（replay 路径 `events == []` 不再依赖标志）；
    `test_fp8_config_replay_build_does_not_defer_device_move` 删除。
  - `tests/models/families/wan_2_1/test_model_loading.py`、
    `tests/trainers/test_wan_fsdp_distributed.py:174,206`：去掉构造参数。
  - `tests/models/families/minimax_h3/test_device_dispatch.py:133,413`：
    去掉对已删字段的断言；`block_devices` 用例按 §3.5 保留。
- 新增：
  - `tests/trainers/test_strategy.py`：单进程 / DDP `prepare_model` 把 CPU 上的
    trainable root 搬到 `context.device`（CPU 上用 `torch.device("cpu")` 验证
    调用发生即可，参考现有 fake-handle 测试）。
  - rollout 路径：H3 分区生成模型 `apply_lora` 后各块仍在原卡
    （`test_device_dispatch.py:415` 的断言已经覆盖，保留）。
- 验证顺序：`ruff check --fix` / `ruff format` 只跑触碰的文件；
  `pytest tests/trainers tests/models/steps/denoise tests/models/families/wan_2_1 tests/models/families/minimax_h3 tests/models/families/cosmos`；
  最后在 5090 上重跑 `experiment/sd3_5/online_grpo_pickscore`
  （`samples_per_generation_batch=1`，两轮）确认 parity 仍为 0。

## 5. 非目标

- 不改 rollout 路径的时机（先量化再上卡、pipeline offload 钩子装在最后）。
- 不合并 rollout 模型与 replay 模型。两进程各一份权重是刻意设计，和 miles 一致。
- 不动 Codex 正在编辑的 LoRA 语义（`lora_init_weights_default`、
  `autocast_adapter_dtype`、previous-policy adapter）。`apply_lora` 只改
  设备搬运那一行及其条件；实施前先等 Codex 的 LoRA 提交落地再 rebase。

## 6. 参考

- `vrl/models/families/registry.py:343-365`
- `vrl/models/interfaces/runtime.py:220-330`
- `vrl/models/steps/denoise/base.py:412-470, 539-547`
- `vrl/models/steps/denoise/build.py:44-63, 91-135`
- `vrl/trainers/strategy.py:343-371, 501-580, 834-880, 963-990`
- `vrl/trainers/fsdp.py:180-225`
- `vrl/models/families/wan_2_1/model.py:253-280, 1382-1397`
- `vrl/models/families/cosmos/anima/model.py:175-178`
- `vrl/models/families/minimax_h3/runtime.py:36-90`、`partitioned_generation.py:72-76, 142-155`
- miles_diffusion 对照：`miles/backends/fsdp_utils/actor.py:113-152`
  （训练侧只有一条放置路径，模型层从不 `.to(device)`）

## 7. 实施记录（2026-09-17）

- 触碰文件：`vrl/models/steps/denoise/base.py`、`vrl/models/interfaces/runtime.py`、
  `vrl/models/families/registry.py`、`vrl/models/families/wan_2_1/model.py`、
  `vrl/models/families/cosmos/anima/model.py`、
  `vrl/models/families/minimax_h3/{runtime,partitioned_generation}.py`、
  `vrl/trainers/strategy.py`，以及 6 个测试文件。
- 与 §3 的差异：
  - Wan `apply_full_finetune` 在 replay 路径上除 `requires_grad_` 外什么都不做，
    连 dtype 归一化也交给 FSDP（原注释的理由：提前触碰整套参数会破坏逐块构建）。
    rollout 路径保持"无 offload 则搬上卡并归一化 dtype，有 offload 则只归一化 dtype"。
  - 放置步骤落在 `_UnshardedStateStrategy.place_trainable_roots`，
    单进程 / DDP / CP 三个 `prepare_model` 调用；没有 `trainable_modules` 的
    普通 `nn.Module`（测试假模型）原地训练。
- 验证：ruff 通过；`tests/trainers tests/models/steps/denoise tests/models/families/{wan_2_1,minimax_h3,cosmos} tests/models/interfaces tests/config tests/scripts`
  2218 passed / 42 skipped；全量 CPU 套件与 5090 真跑结果见对话记录。

## 8. 第二步（2026-09-17）：rollout 放置也收敛到一处，删掉 `trainable_roots_preplaced`

第一步之后采样路径的放置仍分散在两处：`apply_lora` 在不量化时搬，
`build.py` 的 `move_to_device` 在量化后搬。第二步把它收敛：

- `apply_lora` / `apply_full_finetune` 不再搬任何东西，只挂适配器 / 设
  `requires_grad`。基类的 `_place_trainable_roots_at_build` 判定删除。
- `build_denoise_runtime_bundle.move_to_device` 是采样路径唯一的放置点：
  量化之后、compile 之前，把每个 host 上的 trainable root 搬到 `model.device`；
  pipeline offload 时不搬（Accelerate 钩子随后接管）。
- `trainable_roots_preplaced` 删除。两处放置（builder、策略）改为只搬
  **仍在 CPU 上的根**（`vrl/models/parking.py::module_on_host`）。分区 H3 的
  transformer 由 `device_map` 直接加载到各卡，不在 host 上，自然不动；
  不再需要 family 声明。
- anima 的 `apply_full_finetune` 覆写删除（transformer 加载时已是目标 dtype，
  覆写退化为基类行为）。Wan 的覆写只剩 rollout 路径的 dtype 归一化。
