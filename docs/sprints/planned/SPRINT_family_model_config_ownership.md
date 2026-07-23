# SPRINT：Family model config ownership

状态：**planned（2026-07-22）**。

父 program：[Argument and state ownership](../SPRINT_argument_and_state_ownership_program.md)

承接：`docs/sprints/done/SPRINT_config_as_signatures.md` deferred P3/P4。

前置：[Config argument ownership and resolution](../done/SPRINT_config_argument_ownership_and_resolution.md)
的 typed parse/build contract。

## 0. 结论先行

当前 `vrl/config/schema.py` 同时知道：

```text
SD3 / Wan / Cosmos Predict2 / Predict2.5 / Anima
Janus / NextStep / LlamaGen / Echo / Flux / CausVid / Magi
```

并维护 `_model_config_classes_by_family`。registry已经是 canonical family selection source，这张
全局 mapping是第二个 family table；family runtime又在 `model.py/runtime.py` 从
`ModelBuild.model_config` 读取同一组 key。

目标不是删除 public schema层，也不是把 Pydantic与runtime dataclass合并。目标是：

- shared model keys留在 config层；
- family-specific public schema与family代码同址；
- registry用 lazy dotted path指出 schema；
- family runtime dataclass仍表达模型构造签名；
- `ModelBuild.model_config` 保留为进程/wire mapping，但只能由已经验证的 family payload产生。

## 1. 正确的类型与位置

### Shared public section

在 `vrl/config/model_schema.py` 定义：

```text
ModelSection
shared lora section
shared memory section
shared torch_compile section
shared executor section
```

它只表达所有 family共用的 YAML contract。这个薄文件是合理的 shared abstraction：family
config module与Root schema都需要它，同时避免 `schema.py ↔ family config` circular import。

### Family public section

每个 registered family提供：

```text
vrl/models/families/<family>/config.py
    <Family>ModelSection      # Pydantic public YAML shape
    <Family>Config            # existing runtime dataclass, when one exists
```

Cosmos子 family沿用现有 nested package。共享同一 contract的变体（Wan t2v/i2v、Janus/R1）可以让
多个 registry entry指向同一 dotted class。

family即使没有额外 public key，也保留轻量 `config.py` 形状。该薄文件提供：

- lazy import boundary；
- cross-family一致性；
- heavy `model.py` 之外的 config-only import；
- grepable family owner。

这符合薄文件 keep-list，不应为了少几行把 class塞回 central schema。

### Runtime config

现有 `JanusProConfig`、`NextStep1Config`、`LlamaGenConfig`、`Emu3Config`、`GlmImageConfig` 等仍是
plain dataclass，表达 model wrapper constructor需要的已解析值。它们可以移动到 family
`config.py`，public package re-export保持。

Pydantic `<Family>ModelSection` 与 dataclass `<Family>Config` **不合并**：

- 前者是用户可写 key和unknown-key/cross-field validation；
- 后者包含 model wrapper defaults、resolved dtype/device、sampling/load参数；
- generic build adapter从 `ModelBuild` 明确构造后者。

## 2. T0 — 建立完整 ownership inventory

对每个 registry entry记录：

```text
canonical family
aliases
public model keys
shared keys
family runtime readers
runtime config class / builder
conditional required keys
dotted import paths
bundled presets
```

审计必须覆盖：

- literal `.get("key")` / `["key"]`；
- `plain_mapping(build.model_config)`；
- `model_config_revision_kwargs`；
- normalizer mutations；
- registry `config_cls/config_builder` dotted strings；
- tests直接 import和public `__init__` re-export；
- eval/DPO entrypoint绕过 online registry的构造路径。

本阶段输出表进入 Sprint implementation notes，不创建永久手写 allow-list。

## 3. T1 — Registry成为 family schema选择 source

给 `ModelFamilyEntry` 增加准确命名的 dotted field：

```python
model_section_cls: str
```

不要叫 `config`、`component` 或 `handler`；它明确表示 public `model:` section class，并与
`TokenFamilyBuild.config_cls` 的 runtime constructor config区分。

修改：

- registry每个 entry声明 family section class；
- `vrl/config/schema.py` 根据 normalized family取得 entry并 lazy import；
- unknown-key walker从同一 entry得到 `ConfigBlock`；
- variant class集合从 registry entries派生；
- 删除 `_model_config_classes_by_family` / `_model_config_variant_classes` 手写表。

registry仍只存字符串，不 import Pydantic/family heavy module。`model_section_cls` 是真实 lazy
protocol boundary，应 **KEEP**。

### 构造期验证

- empty dotted path失败；
- alias normalization后仍取 canonical entry；
- registered family缺 schema path失败；
- unknown family由现有 authoritative family validation报错，不退回宽松 base而吞 key；
- non-training legacy entrypoint若确实需要 shared-only fallback，必须显式声明该边界。

## 4. T2 — 迁移 family public schema

分批迁移，每批均 load全配置：

### Token families

- Janus Pro / R1；
- NextStep；
- LlamaGen；
- Emu3；
- GLM-Image。

同时把 runtime config dataclass移出 heavy `model.py`，更新 registry dotted `config_cls` 和 public
re-export。runtime/model/runner只 import轻量 config module。

### Denoise families

- SD3.5 / Flux；
- Wan t2v/i2v；
- Cosmos Predict2 / Predict2.5 / Anima；
- Echo；
- CausVid；
- Magi。

只移动 family-specific fields/validators。共享 `path/revision/lora/memory/torch_compile/use_lora/executor`
继续继承 `ModelSection`，不能在每个 family复制。

每批删除 central class后立即证明：

- family合法 key通过；
- sibling family的 key失败；
- alias使用同一 class；
- default与迁移前一致；
- import config不加载 torch/diffusers/upstream source。

## 5. T3 — Family payload validate once，wire mapping保持

T0/T1 config build保留 typed family section。registry构造 `ModelBuild` 时：

1. 去掉 routing-only shared keys；
2. 把 validated family payload `.model_dump()` 为 plain mapping；
3. 传入 `ModelBuild.model_config`；
4. family runtime builder立即构造 `<Family>Config` 或调用明确 resolver。

`ModelBuild.model_config` **KEEP**：

- 它跨 driver/Ray worker；
- family字段异质；
- normalizer需要在 serialization前处理 path；
-强行建立一个全局 union会让 neutral interface import所有 family。

但 raw mapping不能继续作为“未验证任意字典”。tests要证明 unknown family key在 weight load前失败。

### Runtime config default owner

- runtime-only defaults留在 `<Family>Config`；
- user-facing default只在 public section；
- public `None` 表示“未覆盖 runtime default”时，builder不传该 kwarg；
- 不在 runtime.py再写同一个 literal fallback；
- checkpoint/model architecture常量可以由 runtime dataclass default引用，不能复制数字/string。

## 6. T4 — Conditional required fields与错误质量

family-specific条件放 family section validator或family resolver：

- reference-conditioned family的 reference path；
- CausVid license/pinned source/checkpoint；
- Magi source/config/component paths；
- Anima single-file artifacts；
- NextStep VAE/frozen module；
- Wan dual-stage/expert fields。

原则：

- 纯 YAML形状/跨字段条件：Pydantic family section；
- filesystem/Hub/checkpoint/backend条件：family resolver；
- model class/upstream runtime条件：family constructor。

不能把 filesystem或GPU检查塞进 public schema，也不能把可在 load期发现的 unknown key拖到模型加载。
错误必须带完整 `model.<key>` 路径。

## 7. T5 — Central schema收口

最终 `vrl/config/schema.py` 的 model部分只保留：

- Root `model` field；
- shared `ModelSection` import；
- registry-driven selector/validator facade；
- error re-anchoring到 `model.`。

删除：

- family Pydantic class definitions；
- family-to-class mapping；
-手写 variant tuple；
-与 family runtime重复的 key comments/table。

selector薄函数 **KEEP**：它连接 Root schema、registry lazy import和unknown-key walker，是 framework
adapter，不应为了少几行内联两处。

## 8. ALL_CAPS 与 prompt/constants

保持：

- checkpoint/revision/SHA/file names；
- codebook/token/patch/model architecture dimensions；
- Janus R1 byte-sensitive prompt protocol；
- Magi official-runtime env/protocol translation；
- CausVid audited source glob/digest。

这些常量可与 runtime config同在 family package，但不把 byte-sensitive prompt迁到 YAML。

本 Sprint不迁 Kling reward prompt table；它属于 reward model，不属于 family model config。

## 9. What changes / what stays

### 改变

- family schema ownership与registry mapping；
- heavy model文件中的 runtime config class移到轻量 family config module；
- public family payload validate once；
- central schema不再枚举 family vocabulary。

### 保持

- OmegaConf/Pydantic/runtime三种不同角色；
- registry neutral/lazy；
- `ModelBuild` plain wire mapping；
- family runtime builder与normalizer；
- public imports/re-exports；
- cross-family config.py一致形状，即使某个文件很薄。

## 10. Non-goals

- 不把所有 family config合成 union dataclass。
- 不让 central config import torch/diffusers/upstream packages。
- 不删除 ModelBuild或registry。
- 不统一本来不同的 family defaults。
- 不移动 sampling section中真正跨 family的 generation参数。
- 不以 LOC为目标删除薄 family config files。

## 11. Acceptance gates

- 64 bundled experiments load/build；
- 每个 family至少一个 valid key与一个 sibling-only invalid key；
- aliases；
- conditional required true/false；
- registry dotted schema/runtime class import；
- config-only import不出现 heavy model/upstream modules；
- DPO/eval/online三类 entrypoint；
- typed payload snapshot与迁移前逐 key相同；
- `ruff` touched files、`git diff --check`；
- CPU-only，无 Ray/GPU。

## 12. Definition of Done

- [ ] central schema无 family class/table。
- [ ] 每个 registry family有 lazy `model_section_cls`。
- [ ] 每个 family public/runtime config定义在 family package。
- [ ] shared keys只定义一次。
- [ ] unknown/sibling key在模型加载前失败。
- [ ] ModelBuild仍是纯数据、可序列化、family-neutral。
- [ ] config-only import不触发 heavy dependency。

## 13. References

- `vrl/config/schema.py`
- `vrl/config/unknown_keys.py`
- `vrl/config/builders.py`
- `vrl/families/registry.py`
- `vrl/families/names.py`
- `vrl/models/interfaces/runtime.py`
- `vrl/models/steps/token/build.py`
- `vrl/models/families/janus_pro/model.py`
- `vrl/models/families/nextstep_1/model.py`
- `vrl/models/families/llamagen/model.py`
- `vrl/models/families/emu3/model.py`
- `vrl/models/families/glm_image/model.py`
- `vrl/models/families/wan_2_1/model.py`
- `vrl/models/families/causvid/model.py`
- `vrl/models/families/magi_1/model.py`
- `docs/sprints/done/SPRINT_config_as_signatures.md`
