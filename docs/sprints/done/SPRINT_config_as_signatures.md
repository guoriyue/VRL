# SPRINT: Config as signatures (torch-style required args)

状态：**done（P1 + P2 已落地 main，594cba3，2026-06-12，G1-G4 全过；2026-06-17 归档至 done/）**；P3（per-family
model config）/ P4（schema 派生或退役）作为未来扩展未启动（deferred，非本次交付门）。本 sprint 触发了
`docs/sprints/done/SPRINT_config_unknown_key_warning.md` §2 预留的重启条件之二（"正在系统性
补全 schema 字段——为了类型化本身"），是该 parked follow-up 的正式继任。

> **2026-07-13 后续更正：**下文把 `reward_view` 判为活 config 消费链的结论已经失效。
> collector 现在不再接受按名字选择 scoring view；trajectory 必须恰好提供一个 view，零个或多个
> 都会失败。旧 `reward_view` config 投影和 `_reward_view_name` 已整体删除，不应恢复为局部内联。

落地摘要（实施时与设计的偏差均有审计依据）：

```text
P1  types.py：8 个实验语义字段删默认值转必填（optim/lr、n、
    rollout_batch_size、timestep_fraction、total_epochs、output_dir、
    drop_zero_advantage）；builders.py 收缩为 section_to_dataclass 通用
    构造器 + 字段→路径布局表 + 两个显式桥（rollout.n 别名、precision
    展开）；OnlineTrainer.config 转必填参数（裸 TrainerConfig() 路径删除）；
    base trainer/actor yaml 删 20+ 条与 dataclass 重复的默认值，保留
    故意分叉的 ema 块 / drop_zero_advantage / timestep_fraction（必填
    单一副本）与 ??? 标记。
P2  orchestration 组 yaml 去重（strict/continuous 共 9 条 dup，保留
    mode 判别符与 max_pending_rollouts=2 有意覆盖）；冗余
    configure_ar_rollout 删除（builders 别名桥已覆盖两处赋值）。
P2 范围修正（审计驱动）：
  - rollout/diffusion.yaml 内容去重跳过——那些键经 flat-merge 进
    request.sampling 到达 worker，G3 快照只覆盖 TrainerConfig，无 gate
    不动；待 request.sampling 有等价快照 gate 后再做。
  - "新建 typed RolloutConfig dataclass" 推迟：RolloutConfig 名字已被
    collector 投影和 pydantic lint 各占一次，且 nextstep_1 硬依赖
    sampling["noise_level"] 的键存在性——等引擎真正消费 typed 旋钮时
    再立项（届时换名，如 GenerationRolloutKnobs）。
  - 审计建议的三个"死键"删除（rollout.reward_artifact /
    rollout.train_segments 分支 / reward_view 对）复查后全部保留：
    各自有真实下游消费链（collector/core.py:164,248、
    trajectory/builders.py:426），是可选功能查找路径而非死代码。
验收  G1 load-all 19 实验过；G2 缺键报错一次列全完整 YAML 路径（pin
    测试，覆盖节级+标量级聚合）；G3 18 个实验 resolved TrainerConfig 与
    重构前逐字节相等（/tmp/vrl_config_snapshot，一次性产物）；G4 全套
    741 测试绿，e2e 收集正常，ruff 干净。

对抗验证轮（3 个 validator）抓到并已修复：
  - BLOCKER ×2：yaml 去重删除的键在 builder 之外还有裸读取方——
    online.py:117 的 actor.gradient_checkpointing（所有 diffusion online
    run 会在 bundle 构建后崩）与 train_dpo.py 的三处 require()
    （resume_strict / gradient_checkpointing / max_norm，DPO 启动即死）。
    教训：G1/G3 只覆盖 build_configs 路径，recipe 运行期裸读取是盲区。
    修法统一为"OmegaConf.select + 从 dataclass 字段对象派生默认值"
    （TrainerConfig.__dataclass_fields__[name].default），默认值仍只有
    一份。
  - 布局表消灭：审查后把 _TRAINER_SCALAR_SECTIONS 等三张手维护表 +
    完整性断言整体替换为字段级 metadata={"yaml": ...}（节名/带点路径/
    "bridged"），builder 用 fields()+get_type_hints 全派生——布局声明
    与字段本体同址，零平行结构。
  - 严格性恢复：六个 typed 节内的未知键从 warn-and-drop 改回硬报错
    （dataclass 即该节完整词汇，拼错的超参必须拒绝启动）；显式 null 节
    （actor.ema=null）报错而非静默全默认；节级/标量级缺键聚合到同一条
    报错。
  - 周边：profile preset 删 11 条与 TorchProfilerConfig 重复的默认值；
    launcher.py 的 output_dir 静默 fallback 删除（required+load 期执法）。
  - 记录在案未做：9 个实验文件里与默认值相等的重述（ppo_epochs:1 等）
    保留——实验级重述是显式选择声明，不是默认值的第二副本；
    configs/base/algorithm/grpo.yaml 的 4 条 dup 留给 P2.5/P3。
```

## 0. Core Decision

一个键是否必填，只声明在**一处**：typed config dataclass 的字段列表。
无默认值的字段 = 必填（torch 函数签名语义），有默认值 = 可选。YAML 只携带
值，不携带必填性；builder 退化为通用构造器，缺字段时报完整 YAML 路径。
机器执行，无需任何人（或 LLM）在读取处判断键的类别。

参照系（见 `docs/sprints/reading/`）：

- torch：函数签名，无默认值参数即必填，`TypeError` 报参数名。
- vLLM：整套 CLI 从 typed config dataclass 生成，无默认值字段 = required，
  help 来自字段 docstring（`reading/vllm.md` §12，`arg_utils.py:236-349`）。
- 共同原则：**必填性声明在定义处，由构造时机器执行**。

## 1. 今天的问题：同一个键的契约写在三个地方

以 `trainer.total_epochs` 为例，三层互相矛盾的声明同时存在：

```text
vrl/trainers/core/types.py:205   total_epochs: int = 10000      # 类型层：可选，默认 10000
vrl/config/builders.py:103       require(cfg, "trainer.total_epochs")  # builder 层：必填
configs/base/trainer.yaml:4      total_epochs: 10000            # YAML 层：又一份默认值
```

后果（全部已实测）：

- `TrainerConfig` 的字段默认值**全部是死代码**——builder 对每个键
  `require()`，dataclass 默认值永远不会被用到（除非测试直接构造）。
- base yaml 的默认值和 dataclass 默认值是两份手维护副本，正属于
  AGENTS.md "derive, don't duplicate" 禁止的形态（已经分叉：dataclass
  `n: int = 4` vs base yaml 无 `n`、由 rollout 组提供）。
- `build_trainer_config` 是 70 行逐键搬运（builders.py:84-110），每加一个
  字段要同步改三处。
- 必填性无处可查：想知道哪些键必填，只能读 builder 代码。

对照组：`build_algorithm_config` 走的 `_dataclass_payload`
（builders.py:26-36）已经从 dataclass 字段**派生**合法键集——证明 torch
形态在本仓库可行且已有先例。

## 2. 目标形态

### 2.1 通用构造器（builder 的归宿）

```python
def section_to_dataclass(cls, cfg, path: str):
    """Construct ``cls`` from the YAML section at ``path``.

    Required = field without default (torch signature semantics). Missing
    fields raise naming the full YAML paths; unknown keys go through the
    existing unknown-key warner (warn-and-pass preserved), then dropped.
    """
```

- 字段名集合从 `dataclasses.fields(cls)` 派生（`_dataclass_field_names`
  已存在）。
- 缺必填字段：捕获 dataclass `TypeError`（原生就会列出全部缺失参数名），
  重抛时把字段名映射成 `actor.optim.lr` 形式的完整路径。
- 未知键：复用 `vrl/config/unknown_keys.py` 的警告器（warn-and-pass 决策
  不变），然后丢弃——修正 `_dataclass_payload` 当前直接 raise 的不一致。

### 2.2 必填/可选的判定结果直接长在字段上

```python
@dataclass(slots=True)
class TrainerConfig:
    output_dir: str                  # 必填：实验语义，无 sane default
    total_epochs: int                # 必填：实验语义
    save_freq: int = 50              # 可选：基建旋钮，默认值只在这里
    log_freq: int = 1
    ...
```

- 实验语义键（output_dir、total_epochs、n、rollout_batch_size、lr ...）
  **删掉默认值** → 签名必填。
- 基建旋钮保留默认值，且 **base yaml 里与 dataclass 默认值相同的条目删除**
  ——默认值只活在一处。base yaml 收缩为：组合结构 + 各 group 真正想
  覆盖的值（如 `rollout/ar_discrete.yaml` 的 `n: 8` 是 AR 家族的有意
  覆盖，保留）。
- 跨节桥接只剩 precision policy 展开（一条 `precision:` 键派生四个字段），
  保留为 builder 里的显式代码——这是真实的适配 UX，不是搬运。

### 2.2.1 历史别名迁移（2026-06-12，clean win）

`rollout.n` / `rollout.n_samples_per_prompt` 曾是同一概念（GRPO 组大小）
的两种拼法（diffusion 家族用 `n`、AR 家族用 `n_samples_per_prompt`），靠
builder 里一段别名桥接 + schema 重复声明 + collector 特判读取维持。这是纯
历史包袱，已统一为 `n_samples_per_prompt`（自我说明的描述性名，优于隐晦的
`n`）：全部 rollout yaml 用该拼法、dataclass 字段同名改成
`n_samples_per_prompt`、collector/schema/exclude-set/读取处随之更新、别名桥
删除。收益：该字段从 `"bridged"` 降为普通 `"rollout"` 标量（字段名==yaml 键），
`"bridged"` 从此只剩 precision 一种含义。**未迁移**的"不同构"是 yaml 的正当结构而非
包袱：`rollout:` 节多消费者（`n`/`rollout_batch_size`→trainer，
`denoise_mode`/`sde`/...→生成 worker），强行镜像 dataclass 会撕碎可组合性。

### 2.3 家族异质的 model 节

全局 schema 表达不了 "reference_image 仅在 predict2 global 模式必填"。
torch 的答案是：不同函数有不同签名。对应到本仓库：**每个家族在自己的
包里声明自己的 config dataclass**（与 `families/<name>/runtime.py` 同级，
镜像 wan/state.py 的家族自治先例），条件必填放 `__post_init__`。引擎和
共享胶水不读 model 节的裸键。

### 2.4 与现有机制的关系

- **`???` + loader 执法**（2026-06-12 已落地，`loading.py` 的
  `missing_keys` 检查）：保留，作为 typed 层覆盖不到的键（如
  `trainer.entrypoint`，在任何 dataclass 构造之前就被读取）的兜底。
  两层互补：loader 抓未填的 `???`，构造器抓缺失的必填字段。
- **`vrl/config/schema.py`（pydantic lint）**：维持 lint 角色不变；
  P4 阶段评估从 dataclass 派生它（或直接退役），消除最后一份手维护副本。
- **unknown-key warn-and-pass**：决策不变，实现收敛到通用构造器一处。

## 3. 迁移阶段

```text
P1  trainer/actor 节：TrainerConfig 字段默认值审计（实验语义键删默认值），
    build_trainer_config 收缩为 section_to_dataclass + 显式桥接；
    base/trainer.yaml、base/actor.yaml 删除与 dataclass 重复的默认值。
P2  rollout 节核心旋钮（n / rollout_batch_size / sample_batch_size /
    denoise_mode / sde 块）建 RolloutConfig dataclass，同样收缩。
P3  per-family model config dataclass（predict2 先行——它有现成的条件
    必填案例 reference_mode/reference_image）。
P4  schema.py 从 dataclass 派生或退役。
```

每阶段独立可交付、可回退；P1 单独就消灭三层重复的主体。

## 4. 验收 gate

```text
G1  对全部 19 个实验：load_config + build_configs 全过（现有
    test_load_all_experiments 即 CI lint，零 GPU）。
G2  错误质量：删掉任一必填键后，报错必须给出完整 YAML 路径且一次列全
    该节所有缺失键（pin 测试）。
G3  零行为变化：迁移分支上对 19 个实验快照 resolved TrainerConfig 并
    与 main 对比逐字段相等；快照脚本是一次性验证产物，验完即删，
    不入库（遵守 no-exact-config-tests）。
G4  全量回归：tests/config + trainers + scripts + rollouts。
```

## 5. Non-Goals

- 不把运行时 config 换成 pydantic——plain dataclass + `__post_init__`
  是既有约定（consistency 优先）；pydantic 留在 lint 层。
- 不做 vLLM 式 CLI 生成——dotlist override 已覆盖需求。
- 不改 YAML 组合机制（defaults 组、merge 顺序）。
- 不在本 sprint 里动 reward / data 节（各有独立的 validation 入口，
  等 P1-P2 的模式验证后再评估）。

## 6. References

- `vrl/config/builders.py:26-36`（_dataclass_payload，目标形态的先例）
- `vrl/config/builders.py:84-110`（逐键 require 搬运，P1 主要消灭对象）
- `vrl/trainers/core/types.py:160-214`（TrainerConfig，全字段带死默认值）
- `configs/base/trainer.yaml`（第三份默认值副本 + 既有 `???` 标记）
- `vrl/config/loading.py`（loader 端 `???` 执法，2026-06-12）
- `docs/sprints/reading/vllm.md` §12（pydantic dataclass 生成 CLI 的先例）
- `docs/sprints/done/SPRINT_config_unknown_key_warning.md` §2（被触发的重启条件）
