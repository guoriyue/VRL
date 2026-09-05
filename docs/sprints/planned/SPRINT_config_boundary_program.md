# SPRINT PROGRAM: Config boundary — type it once, then delete the machinery that existed because it wasn't

状态：**in progress（2026-09-05 审计；S0 门已就位；S1–S4 done，S5a done，S5b 进行中）**

前置（全部 done，本 program 是它们的收官）：[[SPRINT_config_unknown_key_warning]]、
[[SPRINT_config_as_signatures]]、[[SPRINT_config_argument_ownership_and_resolution]]、
[[SPRINT_family_model_config_ownership]]、[[SPRINT_config_resolution_consolidation]]、
[[SPRINT_config_string_settings]]。

## 0. 结论先行

设计文档里承诺的边界只有一条：

```
YAML ──load_config──▶ DictConfig ──parse_config──▶ RootConfig ──build_configs──▶ 运行时 dataclass
      (OmegaConf：合并/插值/覆盖/???)      (Pydantic：一次定型)          (纯投影，无点路径)
```

代码没有守住这条线。**`parse_config` 之后 raw `DictConfig` 没有死**：它和 typed `root` 一起往下流，
同一个函数里一行读 `OmegaConf.select(cfg, ...)`、下一行读 `built.root.data`。下游 builder 有的接
raw、有的接 root、有的两种都接。因为没有消费者知道自己会拿到哪种形状，**所有人都用鸭子类型的
访问器**——这就是"到处都乱"的机械成因，不是风格问题。

可量化的症状（`vrl/` 内）：

| 读点路径的方式 | 调用 | 文件 |
|---|---|---|
| `OmegaConf.select` | 66 | 14 |
| `cfg_get`（.get → [] → getattr） | 50 | 8 |
| `require` / `optional_none` / `path_exists` | 45 | 9 |
| `cfg_path` | 27 | 10 |
| `OmegaConf.to_container` 裸调 | 19 | 11 |
| `precision._select`（`cfg_path` 的私有复刻） | 1 impl | — |

七种读法、三种转 plain dict（`plain_mapping`/`to_builtin`/`to_builtin_deep`）、**六处**各自判定
"unknown key"（walker / `extra="forbid"` / `extra="allow"`+`_drop_unknown_extras` /
`_dataclass_payload` / `section_payload_and_missing` / `parse_reward_inference_config`）、四种
错误文案、三种 unknown-key 政策（raise / warn / **静默丢弃**）。

第二个成因：**schema 只有一半是 schema。** `RootConfig` 有 31 个 `Any` 字段带 `# reader: X`
注释——那是 key 登记簿，不是类型。`precision: Any`，旁边一组 `PrecisionConfig` 等 dataclass
**从未被实例化**（grep 零命中），存在的唯一目的是喂 unknown-key walker。`actor`/`trainer` 是
`extra="allow"` 的开放袋子，真正的类型住在另一个 dataclass 的 `metadata={"yaml": ...}` 里，
靠一台"布局引擎"（`_online_runtime_section_shape` + `section_payload_and_missing` +
`validate_yaml_home`）把两边缝起来。结果同一个值（`optim`/`ema`/`ppo_epochs`/`output_dir`…）
同时活在两个对象、两套类型系统里。

**核心判断：unknown-key walker、lint 的 code sweep、yaml-metadata 布局引擎、幽灵 precision
dataclass、七个访问器——全部是"schema 没定型"的补偿机器。把 schema 定型（全树
`extra="forbid"`，每个 section 一个真 model），这些机器就整体可删。** 这是删代码的 program，
不是加抽象的 program。

## 1. 现状实锤（按成因归类）

### 1.1 边界穿孔：raw 与 typed 同流

- `vrl/scripts/common/online.py:811,853,855,1118`：持有 `built.root` 的同时 4 处
  `OmegaConf.select(cfg, ...)`；`:837` `load_prompt_examples_from_config(cfg.data)` 传 raw 节点，
  而 `built.root.data` 就在下一行。
- `vrl/run.py:299-320`：同一函数里 `resolve_distributed_resources(cfg)` 吃 raw，
  `RayGenerationConfig.from_cfg(built.root)` / `RolloutCollectorConfig.from_cfg(built.root)` 吃
  typed——后两者因此用 `cfg_get` 鸭子读（测试里 22+5 处又喂 dict）。
- `TrainerConfig.from_cfg(cfg)` 吃 raw，用 `require`/`path_exists` 重走一遍树。
- `resolve_precision_policy(root)` 吃 typed root，但 `root.precision` 是 `Any`=raw dict，于是内部
  用私有 `_select` 再鸭子读一遍；legacy `actor.optim.allow_tf32` 检查靠 getattr 链穿过
  pydantic model 的 extra 属性——能跑纯属偶然。
- `vrl/scripts/denoise/encode_targets.py:165-178,244`：`root = parse_config(cfg)` 之后立刻 8 处
  `OmegaConf.select(cfg, ...)` 读 root 上已有的值。`sana_aesthetic_report.py` 同款 6 处。

### 1.2 半个 schema

- `schema.py` 31 个 `Any`、`sampling_schema.py` 37 个 `Any`；`precision`/`distributed.resources`/
  `rollout.torch_profiler`/`rollout.trajectory_storage` 都是 `Annotated[Any, ConfigBlock(cls)]`
  ——类型信息只给 walker 用，pydantic 本身不校验。
- `precision.py:59-95` 六个 dataclass 零实例化；解析由 260 行手写 `_parse_role`/`_select` 完成。
- `actor`/`trainer`：`_OnlineRuntimeSection(extra="allow")` + `_drop_unknown_extras` 前置过滤 +
  `_OFFLINE_DPO_{ACTOR,TRAINER}_FIELDS` 手写集合 + `_validate_offline_dpo_surface` 用
  `model_fields_set` 反推。
- 错误文案靠字符串手术对齐：`_extract_error_message` 改写 pydantic 三种 error_type，
  `_revalidate_section` 用 `startswith("unknown ")` 再前缀一次 section 名。

### 1.3 unknown-key 只在一条路上生效

`require_no_unknown_keys(cfg)` 只被 `validate_training_config` ← `build_configs` ←
`run.resolve_online_run` 调到。**十个**只走 `load_config + parse_config` 的脚本
（encode_targets / anima_fixed_eval / cosmos_predict25_{frame_prefix_gate,kling_eval} /
anima/generate / generation_bottleneck_profile / native_denoise_probe / teacache_drift_probe /
sana_checkpoint_compare / wan_robotics_checkpoint_eval）拼错 key 静默放行。`supervise.py:934`
把这件事写成了条件分支——那是诚实版的同一个洞。

另外三处 `OmegaConf.update(..., force_add=True)` 在 load 之后造 key
（`train.py:168`、`anima/generate.py:255`、`generation_bottleneck_profile.py:104`），绕过登记簿；
后者一行 `overrides=[f"precision.rollout.dtype={...}"]` 就能走正门。

### 1.4 脚本层各自再解析一遍

- **五份** `sampling.*` 投影，默认值互相矛盾：`_sampling.py`（`guidance_scale` 1.0、
  `max_sequence_length` 512、缺 `num_frames` 静默补 93）vs 两个 Anima 脚本（4.5 / 128）vs
  `frame_prefix_gate._sampling_dimension`（缺则 raise）。`_sampling.py` 本来就是为消重而建，
  只被 5 个里的 2 个采用。五份全读 raw `DictConfig`，`sampling_schema.py` 的默认值对谁都不作数。
- `_kling_reward.resolve_kling_worker_config` 与 `wan_robotics_checkpoint_eval._reward_worker_config`
  同形不同名，都绕过 `RewardRuntimeConfig.kwargs`。
- `resolved_config.yaml`：`checkpointing.save_resolved_config` 无版本戳写出，被四个脚本用三种
  机制读回（`load_config` ×3、raw `OmegaConf.load` ×1——后者是为了读已退役的 `trainer.eval`）。
  `sana_aesthetic_report` 因此需要 sha256 协议钉 + 一台手写递归 dotted-path differ。
- `vrl/rewards/service/server.py:86` 四行手写 unknown-key（只查顶层，`worker_config` 不查）；
  `kling_video_reward._from_dataclass` **静默丢弃** unknown；danbooru taxonomy 是 `vrl/` 里唯一
  的 `yaml.safe_load`，import 时执行、零校验。
- `utils/config.import_from_path` 接受 `module.attr`，`train._import_callable` 拒绝它——同一条
  加载路径上两套 import-path 语法。

## 2. 目标形态（end state）

1. `load_config` 是唯一碰 YAML/OmegaConf 合并的地方（现状已如此，保持）。
2. `parse_config(cfg) -> RootConfig` 是**唯一**校验点：全树 `extra="forbid"`，unknown-key 由
   pydantic 原生报 `loc`，不再有第二套 walker；`require_no_unknown_keys` 折进 `parse_config`
   ——于是十个脚本自动补上守门。
3. `parse_config` 之后 **`DictConfig` 死亡**：`build_configs(root)`、
   `resolve_distributed_resources(root)`、`TrainerConfig.from_root(root)`、
   `resolve_precision_policy(root.precision)`……签名一律吃 typed。运行时不再存在点路径读取。
4. 每个 YAML section 恰有一个 pydantic model 拥有它的 key 与类型；运行时 dataclass 是它的**投影**
   （字段名对字段名的构造），不是第二份 schema。`OPEN` 只剩两处合法：`reward.kwargs.<name>`
   （per-reward 契约由 reward 类持有）与 `data.manifest` 混合键。
5. 因 (2)(3)(4) 而失去存在理由、整体删除的机器：`unknown_keys.py`（walker、`ConfigBlock`、
   `OPEN` 哨兵、`select`/`variants`）、`lint.py` 的 code sweep（无点路径可扫；yaml sweep 退化为
   "全部实验 parse 通过"，`tests/config` 已覆盖）、`_online_runtime_section_shape` /
   `section_payload_and_missing` / `validate_yaml_home` / `metadata={"yaml": ...}`、
   `precision.py` 六个幽灵 dataclass + `_select` + `_reject_legacy_keys`、
   `cfg_get`/`cfg_path`/`require`/`optional_none`/`path_exists`/`to_builtin`、
   `_extract_error_message` 的 literal/extra 分支与 `_revalidate_section` 的前缀手术。

## 3. Sprint 序列（每步独立可 merge，每步过 S0 门）

### S0 — 零行为变化门（已就位）

`scratch: config_snapshot.py` 对 78 个 bundled experiment 转储 `merged DictConfig` +
`BuiltConfigs` 全部 typed 字段为 canonical JSON；基线 `snapshot_baseline.json`（77 built，
0 error；1 个 yaml 是非实验文件）。每个 sprint 结束 diff 必须为空，或差异逐条有据。
现有门：`tests/config` 529 passed / 11s，`python -m vrl.config.lint` 双绿。

### S1 — 先堵洞，不动 schema（**done**；snapshot diff 为空，77 实验逐字节相同）

落地（决策 A 已由 owner 批准）：
- `require_no_unknown_keys` 折进 `parse_config`：十个只走 `load_config + parse_config` 的脚本
  自动获得守门；`validate_training_config` 不再单独调用。
- unknown-key 报错文案统一为一种：`unknown <a.b.c>[, <d.e>]`——walker 与 pydantic
  `extra="forbid"`（`_extract_error_message`）现在产出同一句话。之前两套文案为同一件事。
- 两处 `force_add` 后置改写 → `load_config(overrides=[...])`：`generation_bottleneck_profile`
  （`precision.rollout.dtype`，顺带删掉 `frame_count` 幽灵别名）、`anima/generate`
  （`_configure_lora_for_inference` 改 `_lora_overrides` 返回 dotlist，需要时重 load）。
- `online.py`：`run_online_recipe` 里 raw `cfg` 只剩两处合法用途——喂 `resolve_online_run` 与
  `save_resolved_config`。其余 6 处（conditioning / manifest / reference_image / sampler.type /
  algorithm.kind / production gate）+ `load_prompt_examples_from_config(cfg.data)` +
  `resolve_training_context(cfg)` 全部改读 `built.root`。`_preflight_production_video_reward`
  签名改 `RootConfig`。
- `encode_targets`：8 处 raw 读改 `root.*`（顺带修 `model.path=None` → `"None"` 字符串隐患）。
- `to_builtin`（浅）删除，7 个调用点改 `to_builtin_deep`（`OmegaConf.to_container` 本就递归，
  两者对 OmegaConf 输入等价）。
- 两个 reward worker_config 投影（kling / wan robotics）改读 `RewardRuntimeConfig.from_cfg(cfg).kwargs`。
- `load_prompt_examples_from_config`：`task_type` 缺省对 typed/raw 输入行为一致（`or` 回退）。
- 测试：3 个"未知 key 静默放行/丢弃"用例翻转为拒绝；生命周期测试的假 root 补齐
  algorithm/rollout.sde/data（它之前靠 raw cfg 偷读这些）。

**从 S1 挪出的两项**（各归其真正 owner）：`train.py` rank-local `visible_devices` 的
`force_add` 是资源派生，随 S4 进 `resolve_distributed_resources`；`sana_aesthetic_report` 6 处
raw 读的函数被测试用微型 DictConfig 直喂，随 S6 脚本层一起改 typed。
`precision._select` 不做过渡替换（`_select` 把 None 当缺省、`cfg_path` 不把 None 当缺省，
语义不同），直接在 S2 随 precision 定型删除。

### S2 — precision 定型（**done**；snapshot diff 为空）

- `precision.py`：六个零实例化的 dataclass 换成六个 pydantic section
  （`PrecisionConfig` / `TrainingPrecisionConfig` / `RolloutPrecisionConfig` /
  `QuantizationConfig` / `PromptEncodersPrecisionConfig` / `DiffusionMathPrecisionConfig`），
  全部 `extra="forbid"`（新 `ClosedConfigBase`）。dtype 词汇、float32 模式、quantization
  format/recipe 在 **parse 时**校验（validator 复用原有 normalize 函数与 `QuantizationPolicy`，
  报错文案不变）；`outer_autocast` 用 `StrictBool`，0/1/"false" 照旧拒绝。
- `resolve_precision_policy(section: PrecisionConfig | None)`：只吃 parsed section，
  40 行完成 training→rollout 继承 + 两个默认；260 行手写 `_parse_role` / `_select` /
  `_reject_legacy_keys` / `actor.optim.allow_tf32` legacy 分支全部删除。`allow_tf32` 现在就是
  一个 unknown key（`unknown actor.optim.allow_tf32`），和其它已删 key 同一条路。
- `RootConfig.precision: PrecisionConfig | None`——不再是 `Annotated[Any, ConfigBlock(...)]`，
  walker 从 pydantic 字段自动下钻。
- `_extract_error_message(exc, *, section=)`：把 bare section 的 loc 前缀交给格式化器本身，
  `_revalidate_section` 里的两段 `startswith` 字符串手术删除；类型错误（原先丢 loc）现在
  报 `<path>: <pydantic msg>`。
- 调用方：vrl 14 处 + tests 40 处 `resolve_precision_policy(root)` → `(root.precision)`；
  4 处喂 raw DictConfig 的测试改 `parse_config(cfg).precision` 或
  `PrecisionConfig.model_validate(...)`；`TrainerConfig.from_cfg` 的无 precision 回退改走
  `parse_config`（S3 整体重写该函数）。
- `tests/config/test_precision.py`：fixture 从 `RootConfig(**top)` 直构改为
  `parse_config(...).precision`（走同一道门），"两种输入形状必须一致"的 8 个用例随 `_select`
  一起消失（只剩一种形状），改为一个"resolver 拒绝非 section 输入"的负向用例。

### S3 — actor / trainer 定型（**done**；snapshot diff 为空）

- `ActorSection` / `TrainerSection` 改成真 model（`ClosedConfigBase`）：每个标量一个显式
  typed 字段（`StrictInt`/`StrictBool`/`Literal`），**嵌套块直接用消费它的运行时 dataclass 作
  类型**（`optim: OptimConfig`、`ema: EMAConfig`、`debug`、`precision_drift_guard`、
  `precision_correction`、`rollout_orchestration`、`torch_profiler`）——pydantic 原生校验
  stdlib dataclass（未知 key、缺必填、`__post_init__` 全部在 parse 时触发），所以嵌套块的
  key/默认值/范围检查只在 dataclass 上写一次，不再复制成第二个 pydantic model。
- 必填语义留在运行时 dataclass（无默认=必填，torch 签名语义）：`TrainerConfig.from_root` 按
  "字段名属于哪个 section"投影（名字由 `ActorSection`/`TrainerSection` 拥有，投影派生，
  不再有 `metadata={"yaml": ...}`），缺失一次性汇总报完整路径（`actor.optim.lr` 等粒度保留）；
  显式 `null` 仍拒绝（`model_fields_set`）。`OnlineBatchPlan.from_root` 同理。
- 删：yaml-metadata 布局引擎全部——`_online_runtime_section_shape`、`_OnlineRuntimeSection`
  （`extra="allow"` + `_drop_unknown_extras`）、`_online_runtime_section_block`、
  `section_payload_and_missing`、`validate_yaml_home`；`schema.py` 不再 import
  `vrl.trainers.online.config`（依赖方向反过来：投影 import schema）。
- `build_offline_dpo_trainer_config(root, dpo)`：读 typed `root.actor`；"adafactor 下显式设了
  AdamW-only 键"改为"值偏离默认"判定（dataclass 无 fields_set；偏离默认才是真 footgun）。
  `train_dpo.py` 的 `actor.*`/`trainer.*` 读全部改 typed。
- 顺带：`rollout.prompts_per_batch` / `n_samples_per_prompt` 改 `StrictInt`（bool 不是 batch
  维度，原来靠 `require_exact_int` 在 plan 里拒；现在 parse 即拒）；`_extract_error_message`
  把 pydantic 对 dataclass 字段的 `unexpected_keyword_argument` 也映射成 `unknown <path>`。
- **保留**：`_OFFLINE_DPO_{ACTOR,TRAINER}_FIELDS` + `_validate_offline_dpo_surface`——它是
  "离线入口消费哪些键"的刻意隔离表，与 model/sampling 的 family-select 不同（离线只有一个
  变体），做成 kind-select 变体是为形状一致而加机器，不做。

### S4 — distributed / data / rollout 定型（**done**；snapshot diff 为空；决策 B 已由 owner 批准）

- `distributed.resources: DistributedResourceConfig | None`——消费它的 dataclass 直接做 section
  类型（与 S3 同款），pydantic 在 parse 时构造/校验；`gpu_pool` / `reward.device` /
  `reward.gpu_pool` 改 `Literal`，坏词在 parse 报 `unknown distributed.resources.<role>.<key>=...`。
  删 `_distributed_resource_config_from_cfg` 与三个 `_parse_*_pool/_device` 手写解析器；
  `resolve_distributed_resources(root)` 只吃 typed root，reward inference 回退改读
  `RewardRuntimeConfig.from_cfg(root.reward)`；`reward_inference_configs_from_cfg`（raw walk）删除。
  测试 120 处调用点由脚本机械包上 `parse_config(...)`（14 个文件）；顺带清出 3 处测试 fixture
  里早已不存在的 `distributed.reward` 假 key 和一个被 builder 拒绝的 `sleep_offload` 假 kwarg
  ——它们此前能过，只因为旧路径根本不校验。
- `data.preprocessing` / `data.sampler` → `DataPreprocessingSection` / `DataSamplerSection`
  （手写 `ConfigBlock` 元组删除）；`DataConfig` 其余 `Any` 全部定型（`StrictInt`/`StrictBool`/
  `str`）；validator 的 `key in dict` 改为 `is None`。
- `rollout`：`window_size`/`window_range`/`return_prev_sample_mean`/`cache_ref_noise_pred` 定型；
  `torch_profiler: TorchProfilerConfig | None`、`trajectory_storage: TrajectoryStoragePolicy | None`
  ——collector 的 flat 投影跳过 dataclass 值（原来靠"是 dict 就跳过"），
  `trajectory_storage_policy_from_cfg` 接受已构造的 policy。
- 读方全部改 typed：`resolve_training_context(root)`、`resolve_gradient_checkpointing_mode(root)`、
  `compile_conflicts(root)`、`validate_production_*(root)`、`train.py`（`resolve_train_target` /
  `_verdict_dir` / rank-local 收窄）、`train_dpo.py` 的 data/sampling 读、`online.py` 的
  preprocessing/sampler 读、`prompts.py` 的 `image_field`/`caption_field` 缺省。
- `train.py` 的最后一处 `force_add` 消失：rank-local 收窄只改 env，物理 GPU 序号作为
  `distributed.resources.visible_devices=[N]` **loader override** 重新 load——没有 load 之后的树改写。
- `vrl/run.py` 两处 `resolve_distributed_resources(cfg)` 改传 `built.root`：`resolve_online_run` 里
  raw `cfg` 现在只剩 `build_configs(cfg)` 一处消费。

### S5a — 拆 walker（**done**；snapshot diff 为空）

- `algorithm` 是最后一个靠 walker 的 `select`/`variants` 才能判 unknown-key 的 section。改成
  `AlgorithmConfig(kind, kl_reward_coef, hyperparameters)`：一个 before-validator 按 `kind` 选运行时
  dataclass，把其余 key 交给 pydantic `TypeAdapter(dataclass)` 校验（类型、必填、`__post_init__`；
  未知 key 用 dataclass 的 init 字段集判——**独立 TypeAdapter 对 stdlib dataclass 默认忽略多余 key**，
  这是本步实测出的一个坑），构造好的实例放进 `hyperparameters`。`build_algorithm_config` /
  `_dataclass_payload`（含它的 `ignored_keys = {"kind","kl_reward_coef"}` hack）删除，
  `BuiltConfigs.algorithm = root.algorithm.hyperparameters`。
- `ConfigBase` 改 `extra="forbid"`，`ClosedConfigBase` / `_ClosedModelSection` 合并消失。
- **删 `vrl/config/unknown_keys.py` 整个文件**（walker、`ConfigBlock`、`OPEN`、`select`/`variants`）
  和 schema 里为它而生的 `_model_section_block*` / `_sampling_section_known_fields` /
  `*_variant_classes` / `Annotated[..., ConfigBlock(...)]`。unknown-key 现在只有一个机制：
  pydantic `extra_forbid` + dataclass 字段的 `unexpected_keyword_argument`，由
  `_extract_error_message` 汇总成同一句 `unknown a.b, c.d`（全部一次报出、排序——和 walker 的
  UX 一致）。
- `lint.py` 的 code sweep（AST 扫 `cfg_get`/`require`/`select` 点路径）删除——没有点路径可扫了；
  CLI 保留，只做 "全部实验 parse 通过" 一件事（Makefile / CI 入口不变）。
- 测试：`tests/config/test_unknown_keys.py` 重写为 parse 语义（walker 专属的 ConfigBlock 派生
  用例删除）；`test_schema.py` 里 48 处 `find_unknown_keys(cfg)` 通过 `tests/config/helpers.unknown_keys`
  （从 parse 报错反解 key 列表）保持断言原样；算法段测试改直接校验 `AlgorithmConfig`
  （其中两个参数值本来就不合法——`add_kl_coefficient=0.2` 给 bool 字段、`segment_weights=[1.0]`
  给 dict 字段——旧路径从不校验值）。

### S5b — 拆访问器（进行中）

删 `cfg_get` / `cfg_path` / `require` / `optional_none` / `path_exists`，剩余读方全部改 typed
attribute；`resolve_training_resume_config(root)`、`RayGenerationConfig.from_root`、
`RolloutCollectorConfig.from_root`、`load_prompt_examples_from_config(DataConfig)`、
`build_train_launch(root)`、perf 脚本 `prepare_sampling_state(model, root)`、
`torch_compile_for_role` 只认 `Mapping | TorchCompileSection`。

### S6 — 脚本层与外围

- 五份 sampling 投影 → 读 `root.sampling` typed，**默认值归零**（缺就报错，schema 是唯一默认
  源）。**决策点 C**：Anima 两脚本当前默认 `guidance_scale=4.5`/`max_sequence_length=128` 会变
  为必填——需要 owner 确认这是预期（推荐：是，脚本不该私藏训练超参默认）。
- `resolved_config.yaml` 加 `schema_version` 戳；读端统一一个 `load_resolved_run_config()`，
  `sana_curve_verdict` 的 raw-load 逃生口随之关闭。
- `RewardServiceConfig` 改 pydantic `extra="forbid"`（`worker_config` 进入校验）；
  `kling_video_reward._from_dataclass` 的静默丢弃改 raise；danbooru taxonomy 改
  `OmegaConf.load` + 一个小 model，脱离 import 时副作用。
- `train._import_callable` 与 `utils.config.import_from_path` 合一（`module:attr` 唯一语法）。

## 4. 明确不做

- 不合并 pydantic public schema 与运行时 dataclass（[[SPRINT_config_argument_ownership_and_resolution]]
  §0 已裁决：public section 可投影到多个 runtime owner）。
- 不动 YAML preset 的组织与 `defaults:` 语义。
- 不引入 Hydra / 结构化 OmegaConf（`OmegaConf.structured`）——它会把类型校验拉回 merge 阶段，
  与 (2) "一个校验点" 冲突。

## 5. 三个需要 owner 拍板的决策

| # | 决策 | 推荐 |
|---|---|---|
| A | unknown-key 守门覆盖全部 `parse_config` 入口，含读归档 `resolved_config.yaml` 的脚本 | 是；归档配置带退役 key 本就该显式处理，不该靠"恰好没检查" |
| B | `resolve_distributed_resources` 签名改吃 `RootConfig`（90 处测试调用） | 是，一次 sed；否则 raw 路径永远删不掉 |
| C | 脚本层 sampling 默认值归零（Anima 4.5/128 变必填） | 是；训练超参默认只能有一个来源 |
