# SPRINT: Single-caller inlines & form-4 hoist — 单调用内联与同体上提

**日期**: 2026-07-13  **状态**: EXECUTED and archived

**Source:** the superseded planned copy was removed after execution. This file
is the canonical record of the landed changes and the corrections made while
implementing them.

## 结论

计划中的真正重复和单调用裂片已收敛，不保留无效 wrapper，也没有为了减少行数拆掉
protocol、framework adapter 或跨 family 一致性边界。执行时还从全仓快测发现并修复了一个
`generation -> rollouts` 反向依赖：唯一 family registry 已迁到中立的 `vrl/families/`，
Ray launch contract 仍只传 canonical `family` 和 per-run payload，没有恢复被删除的
`task` / `generation_kind` / `runtime_builder` / `executor_cls` 镜像字段。

## 已落地变更

### 1. 跨 family 同体上提

- 将六份同体 `_encoder_device` 上提到 `DiffusersPipelineModelBase`：
  `pixart_sigma`、`mochi`、`hunyuan_image`、`hunyuan_video`、`cogvideox`、
  `qwen_image`。
- 将 `flux`、`hunyuan_image`、`qwen_image` 的同体 `_guidance_embeds` property
  一并上提。
- 保留 `FluxModel._encoder_device`；它优先读 `text_encoder_2`，与单 encoder 基类实现
  不同，是真正的 override。
- 新增基类行为测试：主 encoder device、无参数 encoder fallback、
  `guidance_embeds` true/false/missing，以及 FLUX 双 encoder 优先级。

### 2. 真正的单调用内联

- `trainers/data/prompts.py`: 删除 `_load_prompt_examples_from_config`，将唯一活跃的
  `data.manifest` 加载流程并入公开 `load_prompt_examples_from_config`；同步修正
  `config/schema.py` 的指向。
- `trainers/offline/dpo.py`: 删除两行 `_trainable_forward_model`，原样内联到
  `wan_forward`；保留“`transformer` 属性存在但为 `None` 时回退原 model”的语义。
- `nn/quantization/targeting.py`: 删除 `is_mlp_linear_path`，将 predicate 并入
  `matches_linear_target`；测试转向公开 `MLP_ONLY` 分支，覆盖不降级。

### 3. 执行时与计划不同的两项

- `_reward_view_name` 不做“内联”：当前树已连同 `reward_view` config 投影整体删除。
  collector 现在要求 trajectory 恰好有一个 scoring view；恢复局部 helper 反而会复活死配置。
- `_optional_block` 不重复处理：`vrl/models/model_build.py` 已被删除，
  config-to-`ModelBuild` 投影现由 canonical family entry 统一完成。

### 4. 全仓闸门暴露的 registry 归属修正

精简后的 `GenerationRuntimeLaunchContract` 只携带 canonical `family`；worker 因此要恢复 family
wiring。旧实现从 `vrl.rollouts.families.registry` 全局反查，直接违反了
`generation` 不依赖 `rollouts` 的长期 architecture gate。

最终结构：

- 把原有唯一 registry 整体迁到 `vrl/families/registry.py`，没有分裂第二张表。
- alias taxonomy 迁到 `vrl/families/names.py`；该薄模块保留，因为 config 只做别名
  归一时不应加载完整 runtime registry，这是真实 lazy-import boundary。
- 删除 `vrl/rollouts/families/` re-export facade 和 `vrl/rollouts/family_names.py`；
  不留 compatibility wrapper。
- 同步更名过时 ownership：`RolloutFamilyEntry` -> `ModelFamilyEntry`、
  `get_rollout_family_entry` -> `get_model_family_entry`、
  `normalize_rollout_family` -> `normalize_model_family`。
- `GENERIC_DIFFUSION_EXECUTOR` 仍保留一份，但放在 registry；它是 Ray worker
  动态 import 的 protocol path，是合法的 ALL_CAPS boundary，不是重复业务词表。

## 明确保留

- `MLP_PATH_SEGMENTS`: 隔离的量化 target taxonomy，被公开 predicate 实际消费。
- `FAMILY_REGISTRY`: 23 个 model family 的唯一 wiring source，属于刻意隔离的
  protocol/config table。
- `get_model_family_entry`、`resolve_model_build`、`build_replay`、`build_rollout`、
  `new_gatherer`: 都有多个生产消费方或真实 validation/dispatch 语义，不是薄转发。
- `_diffusion_entry`、`_ar_entry`、`_register_model_family`: 统一 23 个 family 的构造形状和
  duplicate guard，跨 family consistency 比展开 LOC 更重要。
- trainer 长流程 helper、FSDP/DCP adapter、wire codec、trajectory resolver helper、
  family-specific tensor helper：按原 sprint 非目标保持不变。

## 验证

- 六 family CPU 构造/import smoke 通过，六家均继承基类实现，FLUX 保持唯一 override。
- 定向 sprint tests: `110 passed`。
- architecture + family registry: `25 passed`。
- generation/rollouts/models/config/scripts 宽套件:
  `1383 passed, 22 skipped, 26 deselected`。
- embedded-Ray registry/worker slow tests: `4 passed, 2 deselected`。
- 全仓 CPU 快测: `2175 passed, 35 skipped, 38 deselected`。
- config lint: code sweep 和全 experiment config sweep 全绿。
- registry import probe: 不加载 `torch` 或 `vrl.generation.diffusion.executor`；
  names-only import 不加载 registry。
- 残留扫描: `_load_prompt_examples_from_config`、`_trainable_forward_model`、
  `is_mlp_linear_path`、`_reward_view_name`、`_optional_block` 在 `vrl/` 和 `tests/` 零命中；
  `_encoder_device` 仅基类 + FLUX override，`_guidance_embeds` 仅基类一份。
- Ruff: 仅对本 sprint 和 registry 迁移实际触碰的 Python 文件运行
  `check --fix` / `format` / `check` / `format --check`，全绿。
