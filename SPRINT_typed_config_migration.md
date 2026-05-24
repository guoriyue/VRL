# 冲刺：Typed Config Boundary Migration

## 1. 结论

这个 sprint 要解决的是 config 入口层的真实问题：当前 `vrl/config/validation.py`
把 whitelist、required kwargs、reward 特例和跨字段规则混在一个 imperative validator
里，导致每次加 reward、algorithm 或 dataset 都要继续往中心文件塞分支。

本 sprint 的目标不是把整个 repo 改成 typed config，而是在 config 入口建立一个 typed
boundary：

```text
load_config()
  -> DictConfig
  -> OmegaConf.to_container(resolve=True, throw_on_missing=True)
  -> RootConfig.model_validate(...)
  -> existing runtime dataclasses / raw cfg
```

OmegaConf 继续负责 YAML defaults、interpolation 和 CLI dotlist override；Pydantic
只负责合并完成后的结构校验。

## 2. 明确决策

### D1: 使用 Pydantic v2

添加 `pydantic>=2` 到 `pyproject.toml` dependencies。

原因：
- `Literal` / discriminated union 可以替代 `_ALGORITHM_KINDS`、`_DATA_LOADERS`、
  `_PROMPT_SAMPLERS` 这类手写 set。
- `@model_validator` 可以承载跨字段规则。
- 每个 reward kwargs schema 可以跟 reward 自己放在一起，不再集中在
  `_REWARD_REQUIRED_KWARGS` 和 `_REWARD_KWARGS_VALIDATORS`。

### D2: Typed boundary 放在 `build_configs()`

`build_configs()` 是训练入口已经经过的边界，所以这里 parse typed root 最稳。

保留现有返回形状：

```python
{
    "trainer": TrainerConfig,
    "algorithm": GRPOConfig | TokenGRPOConfig | ...,
    "reward": (weights, kwargs),
    "raw": cfg,
}
```

`raw` 必须保留，直到后续单独迁移 runtime consumer。

### D3: runtime dataclasses 保持不变

Pydantic schema 是 config validation layer，不替代训练运行时的 dataclass。

`TrainerConfig`、`GRPOConfig`、`TokenGRPOConfig`、`MultiSegmentTokenGRPOConfig`、
`DiffusionDPOConfig`、`DiffusionNFTConfig` 继续作为 runtime type。

### D4: schema 不做 filesystem / IO 检查

`RootConfig.model_validate(...)` 只做结构、类型和跨字段规则。

例如 production video reward 的 manifest 文件存在性，不放进 Pydantic validator。
这个检查应该是显式函数：

```python
check_production_video_reward_paths(cfg)
```

并且只在 production entrypoint 或 production validation path 调用。

### D5: migration 阶段默认 `extra="ignore"`

为了不改变当前 YAML 行为，schema 初始阶段默认忽略未知字段。

后续可以对稳定 section 分批切到 `extra="forbid"`，但这不是本 sprint 的第一目标。

### D6: reward kwargs 按 component key 分发（hybrid，不为每个 reward 建 model）

当前结构是：

```yaml
reward:
  components:
    video_reward: 1.0
  kwargs:
    video_reward:
      reward_name: KlingTeam/VideoReward@main
```

所以 discriminator 不是字段，而是 dict key。

9 个 reward 里有 8 个只是“必须存在这些 key”，本身就是一张 declarative 表（现在的
`_REWARD_REQUIRED_KWARGS`），没必要各建一个 `BaseModel`——那是把 8 行表换成 8 个类，
代码更多而不是更少。只有 `video_reward` 有真正的校验逻辑（removed field、runtime 必须
是 ray、worker_config 规则）。所以采用 hybrid：

- 简单 reward：保留 required-keys 表，从 `validation.py` 迁到 `schema.py`，例如
  `REWARD_REQUIRED_KEYS: dict[str, tuple[str, ...]]`。
- 复杂 reward：只为 `video_reward` 写一个真正的 `VideoRewardKwargs` model。
- registry 把 component name 映射到“可选的” model（目前只有 video_reward 一项）：

```python
REWARD_KWARGS_MODELS: dict[str, type[BaseModel]]  # {"video_reward": VideoRewardKwargs}
```

`RewardConfig` validator 负责：
- component weight 必须是数字且 `>= 0`
- weight `> 0` 的 component 必须有 kwargs
- 有 model 的 component（video_reward）交给对应 model validate
- 没有 model 的 component 只查 `REWARD_REQUIRED_KEYS` 表

## 3. 当前需要替换的代码

主要替换目标在 `/home/mingfeiguo/Desktop/wm-infra/vrl/config/validation.py`：

```python
_ALGORITHM_KINDS = {...}
_DATA_LOADERS = {...}
_PROMPT_SAMPLERS = {...}
_REWARD_REQUIRED_KWARGS = {...}
_REWARD_KWARGS_VALIDATORS = {...}
```

以及 `validate_training_config()` 里的 kind ladder：

```python
if kind in {"grpo", "diffusion_nft"}:
    ...
if kind == "token_grpo":
    ...
if kind == "token_grpo_multisegment":
    ...
if kind == "diffusion_dpo":
    ...
```

这些规则应该迁到 `RootConfig` 或相关 section schema。

## 4. 目标结构

```text
vrl/config/
  loading.py     # 不改：OmegaConf compose + resolve
  builders.py    # parse RootConfig, return existing runtime objects
  validation.py  # 变薄：compat helpers + typed boundary wrappers
  schema.py      # RootConfig + parse_config + 所有 section model
```

先用单个 `schema.py`，不预先按 section 切成 9 个文件——当前每个 section 只有几个字段，
按 section 切是 premature file-splitting。只有当某个 section（最可能是 reward）真的长到
难以阅读时，再拆成 `schema/` 包。

`validation.py` 最终不应该再包含 whitelist set、reward kwargs table 或 per-reward branch
（required-keys 表会迁到 `schema.py`，见 D6）。

## 5. 分阶段计划

### 阶段 1：新增 schema package 和 parse boundary

改动：
- 新增 `vrl/config/schema.py`
- 新增 `parse_config(cfg: DictConfig) -> RootConfig`
- 在 parse boundary 里包装 `OmegaConf.to_container(..., throw_on_missing=True)`，
  保持当前错误信息：

```text
config missing required field: <path>
```

验证：
- 所有 experiment config 都能 `parse_config`
- `${...}` interpolation 解析后的值与现有 config 行为一致
- `tests/config/test_load_all_experiments.py` 保持通过

### 阶段 2：algorithm / data / model schema

改动：
- 用 `Literal` 替换 `_ALGORITHM_KINDS`
- 用 `Literal` 替换 `_DATA_LOADERS`
- 用 `Literal` 替换 `_PROMPT_SAMPLERS`
- `build_algorithm_config()` 继续返回现有 algorithm dataclass，但 kind validation
  走 schema

保留：
- runtime dataclass 不换
- algorithm consumer 不迁移

验证：
- invalid `algorithm.kind` 仍然 fail fast
- `algorithm.adv_estimator` 仍然报 “no longer supported”
- representative algorithm dispatch 测试继续通过

### 阶段 3：reward schema

改动（hybrid，见 D6）：
- 用 `RewardConfig` + `REWARD_REQUIRED_KEYS` 表替换 `_REWARD_REQUIRED_KWARGS`
- 简单 reward（aesthetic / clipscore / codex_image_qa / geneval / image_qa_cli /
  nsfw_safety / pickscore / ocr）**不建 model**，只在 `REWARD_REQUIRED_KEYS` 留一行
- 只为 `video_reward` 写 `VideoRewardKwargs` model，承载真正的校验逻辑
- `VideoRewardKwargs` 要区分**两个作用域**——这是当前两个函数分开做的，合并会改变行为：

  (a) kwargs 顶层非法字段（来自 `_validate_video_reward_kwargs`）：

```text
backend            # 单独的报错信息：use inference_runtime=ray
enqueue_url
fetch_url
token
poll_interval_s
max_wait_s
stub_scale
device
```

  (b) `worker_config` 内部非法字段（来自 `validate_production_video_reward_config`）：

```text
backend
backend_import_path
backend_code_dir
import_path
model_subdir
score_key_map
model_factory
```

  另外 `VideoRewardKwargs` 还要保留：`inference_runtime == "ray"`、
  `worker_config` 必须是 mapping、`scheduling` 默认/必须是 `"sync"`。

保留：
- production YAML 继续只暴露 `reward_name` 和简短 `worker_config`
- internal `model_factory` 仍然由 `VideoReward` 根据 `reward_name` 派生

验证：
- reward required kwargs 测试继续通过
- negative reward weight 测试继续通过
- video reward legacy / removed field 测试继续通过

### 阶段 4：cross-field validators 和 production split

改动：
- 把 `validate_training_config()` 的 kind ladder 迁到 `RootConfig` validators
- production structural rule 进 schema，例如：
  - `production.video_reward.enabled`
  - `reward.kwargs.video_reward.media_type == "video"`
  - `artifact_format == "mp4"`
  - `reward_name` 非空
  - `data.task_type == "text_to_video"`
- production file existence 进入独立函数：

```python
check_production_video_reward_paths(cfg)
```

保留：
- `validate_training_config(cfg)` 作为 public wrapper，内部调用 `parse_config(cfg)`
- `validate_reward_config(cfg)` 作为 public wrapper，内部走 `RewardConfig`
- `require` / `optional_none` / `path_exists` 暂时保留，给旧 consumer 使用

验证：
- `validation.py` 不再有 `_ALGORITHM_KINDS`、`_DATA_LOADERS`、
  `_PROMPT_SAMPLERS`、`_REWARD_REQUIRED_KWARGS`、
  `_REWARD_KWARGS_VALIDATORS`
- `rg -n "if kind ==" vrl/config/validation.py` 没有结果
- config 测试、script factory 测试、runtime inputs 测试通过

### 阶段 5：后续 consumer migration

这不是本 sprint 必须完成的内容。

后续可以逐个迁移高价值 consumer：
- `vrl/config/builders.py`
- `vrl/scripts/common/factory.py`
- `vrl/scripts/common/online.py`
- diffusion / AR train scripts

每迁移一个 consumer，就要求它不再直接读 raw `DictConfig`。全部完成后，才考虑从
`build_configs()` 返回值里移除 `"raw"`。

## 6. 非目标

本 sprint 不做这些：
- 不重写 OmegaConf loading / defaults overlay / CLI override
- 不改变 YAML config 行为
- 不把 runtime algorithm/trainer dataclass 换成 Pydantic
- 不迁移所有 `cfg.get()` / `OmegaConf.select()` consumer
- 不把 production path existence 放进 schema
- 不改训练流程、reward runtime、Ray runtime 或 model runtime 行为

## 7. 风险和处理

### R1: 错误信息漂移

很多测试用 `pytest.raises(match=...)` 检查错误 substring。

处理：
- 保留关键 substring，例如 `reward_name`、`remove extra loader fields`、
  `config missing required field`
- 只在 schema 行为变得更清楚时更新测试

### R2: unknown extra field 行为改变

当前 `_dataclass_payload()` 会过滤未知字段，等价于 ignore。

处理：
- migration 阶段 schema 默认 `extra="ignore"`
- 后续对稳定 section 再单独切 `extra="forbid"`

### R3: Pydantic 和 dataclass 双类型并存

短期会同时有 Pydantic schema 和 runtime dataclass。

处理：
- Pydantic 只在 config boundary 校验
- runtime 继续拿已有 dataclass
- 不把类型系统迁移扩散到训练和 runtime 代码

### R4: production path check 误伤普通 config test

普通 config tests 构造的路径不一定存在。

处理：
- schema 不检查路径存在性
- 文件存在性只在 production validation path 显式调用

## 8. 测试策略

必须保持这些测试通过：
- `/home/mingfeiguo/Desktop/wm-infra/tests/config/test_load_all_experiments.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/config/test_prompt_dataset_configs.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/scripts/test_common_factory.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/rollouts/test_runtime_inputs.py`

新增：
- `/home/mingfeiguo/Desktop/wm-infra/tests/config/test_schema.py`

`test_schema.py` 覆盖：
- algorithm kind discriminator
- data loader discriminator
- sampler type literal
- reward kwargs registry
- `VideoRewardKwargs` removed field rejection
- cross-field validators
- `???` missing field mapping
- `extra="ignore"` migration policy

## 9. 完成标准

完成后应该满足：
- `build_configs()` 入口 parse `RootConfig`
- `validate_training_config()` 和 `validate_reward_config()` 走 typed schema
- `validation.py` 不再维护 whitelist set 或 reward kwargs table
- production path existence 是独立显式检查
- 当前 YAML 行为不变
- runtime dataclass 行为不变
- config 相关测试和受影响脚本测试通过

## 10. 预期文件改动

会改：
- `/home/mingfeiguo/Desktop/wm-infra/pyproject.toml`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/config/builders.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/config/validation.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/config/schema.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/config/test_schema.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/config/test_load_all_experiments.py`

不应该改：
- model runtime
- reward runtime
- Ray runtime
- training loop behavior
- YAML config semantics
