# SPRINT：Rollout Config Contract Cleanup

## 0. Core Decision

本 sprint 的核心目标是把 rollout 配置边界收干净：

```text
YAML owns values/defaults.
Code owns schema/contract/wiring.
```

也就是说，实验默认值和 recipe 选择必须来自 YAML：

```yaml
sampling:
  width: 1280
  height: 704
  num_frames: 93
  num_steps: 35
  guidance_scale: 7.0
  fps: 16
```

代码里不应该再保存另一份同样的 family-specific 默认值：

```python
class CosmosPredict2CollectorConfig:
    num_steps: int = 35
    guidance_scale: float = 7.0
    height: int = 704
    width: int = 1280
    num_frames: int = 93
    fps: int = 16
```

但代码仍然需要知道每个 rollout family 接受哪些字段、这些字段从 YAML 哪些路径读取、哪些字段会进入 engine request、哪些字段只是 rollout runtime control。这一层是程序 contract，不是实验配置。

目标结构：

```text
vrl/models/families/*/runtime.py
  只拥有 model executor / runtime builder / runtime spec extractor

vrl/rollouts/family_registry.py
  拥有 rollout registry core、lookup API、builtin rollout contracts

vrl/rollouts/schema.py
  generic YAML projection、required field validation、derived field handling

vrl/rollouts/collector/
  generic collector machinery，不包含 family-specific config defaults
```

## 1. Why This Change

现在 `CollectorConfig` 同时做了三件事：

- 保存 family-specific 默认值。
- 作为 YAML 字段白名单。
- 作为 collector/request builder 的 runtime settings 对象。

这会造成两个问题。

第一，默认值重复。Cosmos 的采样值已经在 `configs/sampling/cosmos_v2w_704p_93f.yaml`，代码里再写一份会漂移。

第二，ownership 不清楚。`vrl/rollouts/collector/configs.py` 是 collector 目录里唯一明显的 family-specific 文件，但这些字段并不只是 collector 自己用。它们会进入 request sampling、executor kwargs、group size、metadata，所以更准确的名字是 rollout contract/settings，而不是 collector config。

当前读取逻辑说明 dataclass 的真实作用是字段白名单：

```python
for field in fields(cls):
    found, value = _select_cfg_value(cfg, field.name)
    if found:
        values[field.name] = value
return cls(**values)
```

当前 request builder 说明这些字段最后会变成 engine request：

```python
sampling = _sampling_from_config(self.config, self.sampling_fields)
request = GenerationRequest(
    sampling=sampling,
    return_artifacts=set(self.return_artifacts),
)
```

所以要删除的是代码里的值默认，不是删除 schema/contract。

## 2. What We Are Removing

### 2.1 Remove duplicated rollout defaults from Python

删除这些代码默认值作为 source of truth：

```python
num_steps: int = 35
guidance_scale: float = 7.0
height: int = 704
width: int = 1280
num_frames: int = 93
fps: int = 16
sample_batch_size: int = 8
n_samples_per_prompt: int = 4
```

这些值必须来自：

```text
configs/sampling/*.yaml
configs/base/rollout/*.yaml
configs/base/algorithm/*.yaml
configs/experiment/*.yaml
```

### 2.2 Remove family-specific defaults from `collector/`

目标是删除或清空这个 family-specific 文件：

```text
vrl/rollouts/collector/configs.py
```

如果第一阶段需要兼容，可以暂时保留 re-export，但不要继续让它定义 family-specific 默认值。

最终 `vrl/rollouts/collector/` 应该只包含：

```text
base.py
core.py
factory.py
requests.py
rewards.py
```

### 2.3 Remove rollout/collector registration from model runtime

删除 model runtime 里的这类字段：

```python
collector_config_cls="vrl.rollouts.collector.configs:CosmosPredict2CollectorConfig"
```

model runtime 不应该知道 collector config class。它只提供 executor 和 runtime builder。

### 2.4 Remove `vrl/models/registry.py` as rollout registry owner

如果 `vrl/models/registry.py` 只用于注册 rollout metadata，就删除它或把它缩小成真正 model-only registry。

这些内容不应该留在 `vrl/models`：

```python
collector_kind
collector_config_cls
sampling_fields
return_artifacts
gatherer
executor_kwargs
```

它们属于 rollout 层。

### 2.5 Remove unused config fields

`max_batch_requests` 当前只在 config dataclass/YAML 里出现，没有实际 runtime consumer。除非本 sprint 同时给它补上明确语义，否则删除它。

删除标准：

```text
rg "max_batch_requests" vrl tests
```

如果没有非 config/test consumer，就不保留。

## 3. What We Are Keeping

不要删除这些能力：

- family alias normalization，例如 `wan` -> `wan_2_1`。
- `resolve_online_family()` 里 Janus R1 的 algorithm-based family resolution。
- request sampling field 白名单。
- derived field，例如 `return_kl = algorithm.kl_reward > 0`。
- `rollout.sde.*` 到 request sampling field 的映射。
- executor/runtime import path lookup。
- gatherer metadata。
- rollout capability。
- request `return_artifacts`。
- Ray/runtime launch inputs 里对 `sample_batch_size` 的处理。

核心变化只是：值从 YAML 来，代码只描述 contract。

## 4. Target API Shape

新增：

```text
vrl/rollouts/schema.py
```

建议最小类型：

```python
@dataclass(frozen=True, slots=True)
class RolloutFieldSpec:
    name: str
    paths: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True, slots=True)
class RolloutContract:
    fields: tuple[RolloutFieldSpec, ...]
    sampling_fields: tuple[str, ...]
    derived_sampling_fields: tuple[str, ...] = ()
```

resolved object 可以先保持 attribute access，减少调用侧改动：

```python
@dataclass(frozen=True, slots=True)
class RolloutSettings:
    values: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
```

构造函数替代当前 `build_collector_config_from_cfg()`：

```python
def build_rollout_settings_from_cfg(
    cfg: DictConfig,
    entry: RolloutFamilyEntry,
) -> RolloutSettings:
    ...
```

兼容期可以保留旧名字：

```python
build_collector_config_from_cfg = build_rollout_settings_from_cfg
```

## 5. Registry Layout

第一阶段不新增 per-family rollout files。保留一个 owner：

```text
vrl/rollouts/family_registry.py
```

原因：

- YAML 已经拥有实验值和默认值，Python 不应该按 family 再保存一份值。
- 当前 family 数量少，拆很多小文件会增加跳转成本。
- 现在真正需要的不是 per-family module，而是 code-owned contract：family name、aliases、支持字段、request artifacts、executor/runtime/gatherer/capability。

也就是说，第一阶段只在 `family_registry.py` 里声明这种 contract，不保存实验默认值：

```python
register_rollout_family(
    "cosmos-predict2",
    task="v2w",
    aliases=("cosmos", "cosmos_predict2", "cosmos_predict2_2b"),
    contract=RolloutContract(
        fields=(
            RolloutFieldSpec("width", ("sampling.width",)),
            RolloutFieldSpec("height", ("sampling.height",)),
            RolloutFieldSpec("num_frames", ("sampling.num_frames",)),
            RolloutFieldSpec("num_steps", ("sampling.num_steps",)),
            RolloutFieldSpec("guidance_scale", ("sampling.guidance_scale",)),
            RolloutFieldSpec("fps", ("sampling.fps",)),
            RolloutFieldSpec("sample_batch_size", ("rollout.sample_batch_size",)),
            RolloutFieldSpec("noise_level", ("rollout.noise_level", "sampling.noise_level")),
            RolloutFieldSpec("sde_type", ("rollout.sde.type",)),
            RolloutFieldSpec("sde_window_size", ("rollout.sde.window_size",)),
            RolloutFieldSpec("sde_window_range", ("rollout.sde.window_range",)),
            RolloutFieldSpec("same_latent", ("rollout.same_latent",)),
            RolloutFieldSpec("kl_reward", ("algorithm.kl_reward",)),
        ),
        sampling_fields=(
            "num_steps",
            "guidance_scale",
            "height",
            "width",
            "cfg",
            "sample_batch_size",
            "sde_type",
            "sde_window_size",
            "sde_window_range",
            "same_latent",
            "max_sequence_length",
            "noise_level",
            "return_kl",
            "num_frames",
            "fps",
        ),
    ),
    executor_cls="vrl.models.families.cosmos.predict2.runtime:CosmosPipelineExecutor",
    runtime_builder=(
        "vrl.models.families.cosmos.predict2.runtime:"
        "build_cosmos_predict2_runtime_bundle"
    ),
    runtime_spec_extractor=(
        "vrl.models.families.cosmos.predict2.runtime:"
        "extract_cosmos_predict2_runtime_spec"
    ),
)
```

注意：上面 `35`、`7.0`、`704`、`1280`、`93`、`16` 都不在代码里。

后续当 family 数量变多，或者某个 family 的 rollout contract 明显复杂时，再把 registry 拆成可选结构：

```text
vrl/rollouts/families/__init__.py
vrl/rollouts/families/sd3_5.py
vrl/rollouts/families/wan_2_1.py
vrl/rollouts/families/cosmos_predict2.py
vrl/rollouts/families/janus_pro.py
vrl/rollouts/families/nextstep_1.py
```

这个拆分不是本 sprint 的必要条件。

## 6. YAML Migration

所有旧 dataclass 默认值必须能在 YAML 中找到来源。

### 6.1 Diffusion fields

这些应来自 sampling preset：

```text
sampling.width
sampling.height
sampling.num_steps
sampling.guidance_scale
sampling.cfg
sampling.num_frames
sampling.fps
sampling.max_sequence_length
```

这些应来自 rollout base/experiment：

```text
rollout.noise_level
rollout.sample_batch_size
rollout.sde.type
rollout.sde.window_size
rollout.sde.window_range
rollout.same_latent
```

这些应来自 algorithm base：

```text
algorithm.kl_reward
```

`return_kl` 不写 YAML，作为 derived field：

```python
return_kl = float(settings.kl_reward) > 0
```

### 6.2 AR discrete fields

这些应来自 sampling preset：

```text
sampling.image_token_num
sampling.image_size
sampling.cfg_weight
sampling.temperature
sampling.max_reflect_len
sampling.r1.final_image_policy
sampling.r1.train_segments
```

这些应来自 rollout base/experiment：

```text
rollout.n_samples_per_prompt
rollout.rescale_to_unit
rollout.max_text_length
```

### 6.3 AR continuous fields

这些应来自 sampling preset：

```text
sampling.image_size
sampling.image_token_num
sampling.num_flow_steps
sampling.noise_level
sampling.cfg_scale
```

这些应来自 rollout base/experiment：

```text
rollout.n_samples_per_prompt
rollout.rescale_to_unit
rollout.max_text_length
```

如果同一个字段同时存在于 `sampling` 和 `rollout`，需要显式写优先级。不要靠偶然的 path order。

## 7. Implementation Plan

### 7.1 Add contract primitives

新增 `vrl/rollouts/schema.py`：

- `RolloutFieldSpec`
- `RolloutContract`
- `RolloutSettings`
- `build_rollout_settings_from_cfg()`
- normalizers，例如 list -> tuple、OmegaConf node -> plain container。
- derived fields，例如 `return_kl`。

完成后先让旧 dataclass path 和新 settings path 同时跑通。

### 7.2 Move registration ownership into `vrl/rollouts`

重写 `vrl/rollouts/family_registry.py`：

- 定义 `register_rollout_family()`。
- 在同一个文件中声明 builtin rollout contracts。
- `FAMILY_REGISTRY` 从这些 builtin declarations 构建。
- 保留 `normalize_rollout_family()`、`get_rollout_family_entry()`、`registered_rollout_families()`。

删除 model runtime 里的 registration call。

### 7.3 Replace collector config construction

编辑 `vrl/scripts/common/factory.py`：

- `build_collector_config_from_cfg()` 改为 wrapper。
- 新主函数叫 `build_rollout_settings_from_cfg()`。
- 返回 `RolloutSettings`。
- missing required YAML field 要 fail-fast，并打印 family、field、searched paths。

### 7.4 Make collector generic

编辑 `vrl/rollouts/collector/factory.py`：

- `config_cls` 改成 `contract` 或直接接收 resolved `RolloutSettings`。
- `_resolve_config()` 不再做 dataclass instance type check。
- `_default_group_size()` 从 settings 读 `n_samples_per_prompt`。
- `_build_executor_kwargs()` 从 settings 读 `sample_batch_size`。

编辑 `vrl/rollouts/collector/requests.py`：

- `_sampling_from_config()` 改为 `_sampling_from_settings()`。
- 继续校验 `sampling_fields` 都能从 settings 或 derived fields 得到。

### 7.5 Delete old config classes

完成迁移后删除：

```text
vrl/rollouts/collector/configs.py
```

如果 public import 还被外部测试使用，可以短期保留兼容 shim，但 shim 不允许定义默认值。

### 7.6 Clean tests

更新测试重点：

- registry 是否包含同样 family keys。
- alias normalization 是否不变。
- 每个 recipe build 出来的 request sampling 与迁移前一致。
- YAML 缺 required field 时 fail-fast。
- `return_kl` derived behavior 不变。
- `sde_window_range` normalization 不变。
- `max_batch_requests` 删除后没有 consumer。

## 8. Acceptance Criteria

必须满足：

- Python 代码里不再保存 family-specific sampling 默认值。
- `vrl/rollouts/collector/` 不再拥有 family-specific config defaults。
- model runtime 不再写 `collector_config_cls`。
- `vrl/models/registry.py` 不再是 rollout registry owner。
- 现有 recipe 的 resolved rollout settings 与迁移前一致。
- missing YAML required field 会在 recipe build 阶段报错，而不是在 executor 里晚报错。
- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过。

## 9. Non-Goals

本 sprint 不做：

- 不改变 trainer/algorithm/reward 语义。
- 不改变 request payload 格式，除非测试证明旧字段未被消费。
- 不把 model checkpoint/loading config 移进 rollout。
- 不做 Pydantic/Hydra structured config 大迁移。
- 不重写 collector/engine/batcher。
- 不引入新的 plugin/config 框架。

## 10. Risk Notes

最大风险是 YAML 缺字段。以前 dataclass 默认值会静默兜底，迁移后会 fail-fast。这是预期行为，但需要先补齐 base/sampling YAML。

第二个风险是 derived field。`return_kl` 当前来自 `kl_reward` property，不能粗暴当成普通 YAML 字段，否则会让 recipe 多写一个容易不一致的值。

第三个风险是 field path precedence。现在 `_select_cfg_value()` 会按 alias、`sampling`、`rollout`、`algorithm`、`trainer` 顺序找。迁移时每个 field 应显式声明 paths，避免某个同名字段被错误来源覆盖。

## 11. First Patch Boundary

第一 patch 只做无行为变化迁移：

- 新增 `vrl/rollouts/schema.py`。
- `family_registry.py` 拥有 rollout contract declarations。
- `build_collector_config_from_cfg()` 返回的新 settings 对象对调用侧保持 attribute access。
- 保留旧函数名兼容。
- 测试证明当前 recipes 的 request sampling 不变。

第二 patch 再删除旧文件和旧名字：

- 删除 `vrl/rollouts/collector/configs.py`。
- 删除 `collector_config_cls` 相关字段。
- 删除或收缩 `vrl/models/registry.py`。
- 删除未使用的 `max_batch_requests`。

这个两步拆法可以避免一次性同时改 registry、YAML、collector、tests，降低回归定位成本。
