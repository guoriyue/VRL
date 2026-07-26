# SPRINT：Config resolution consolidation — 归拢散落的 resolved-config

状态：**done（2026-07-25，P1–P10 均已裁决并落地或有据撤销）**。

> **执行状态（2026-07-24，全部 6 phase 已处理）**：
> - **P1 已落地** `4e6b9a47`——`collector/core.py` 直接读 `self.config.trajectory_storage` /
>   `.kl_reward_coef`，修掉 §2.2/§4 描述的会 `raise TypeError` 的 latent crash（`score_rollouts`
>   路径，CPU 快测未覆盖，实测确会崩）。
> - **P2 已落地** `57077fe3`——reward inference map 聚合到 `RewardRuntimeConfig.inference_configs`
>   （build 时解析一次），`resolve_distributed_resources` 读它；raw-cfg walk 保留为隔离 resource
>   测试的 fallback（该函数有 ~100 个调用点，全改签名不划算）。
> - **P3 已调研→撤销**（0 编辑）——三处标的经查都是**合理边界**，合并会改变行为：AR
>   paged/engine/scheduler 旋钮是执行策略而非 sampling 内容（有守卫测试
>   `test_scheduler_policy_is_not_duplicated_in_sampling_params` 断言它们**不该**进
>   `ARSamplingParams`）；`samples_per_chunk` 的另两处服务不同执行路径、默认值不同；
>   `trajectory_storage` per-chunk 正是 §5 Non-Goal 保护的 wire/进程边界重解析（executor.py:338）。
>   **本 sprint 对 P3 的乐观定性被证据推翻**——这里没有可清的散落。
> - **P4 已落地** `62d7a562`——5 份 `_resolve_device` + 2 份 `_resolve_sampling` 聚合进
>   `scripts/eval/_device.py` / `_sampling.py`（net −64 行）。
> - **P5 已落地** `21c57523`——`online.py` 深处的 `OmegaConf.select` raw 重读改为读 `BuiltConfigs`
>   typed 字段（use_lora/kl_coef/sft/model.path），并修一处 `model.path` None→"None" 隐患。
> - **P6 部分落地** `a2029924`——`_resolve_rollout_num_workers` 折进 `_resolve_role_num_workers`；
>   其余 5 项经查是**合理的 family-specific 代码**（echo 双入口、wan offload 被测试 pin 的 backstop、
>   wan topology / vae_decode_memory 需碰 ModelBuild/wire 边界、单文件 checkpoint 三家 precedence 已
>   分化），保留——印证 §5 Non-Goals。
>
> **总结**：真正可清的散落比 survey 初判的更少。P1 修了一个真 crash，P2/P4/P5 落了三处实在的聚合，
> P6 收了一处重复；P3 与 P6 大部分标的经执行验证是**合理边界而非散落**——这更强地印证了 §0 的结论：
> origin 近期已把绝大多数散落收干净，剩的多是各安其位的边界。
>
> **P7（追加，2026-07-24）：`ResolvedRun` 编排聚合座——已落地 `1ec5cbd7`。** P1–P6 之后用户
> 正确指出："random splitted" 的真身不在叶子 resolver（那些各有其主），而在**编排**：
> `run_online_recipe` 内联手串 `build_configs → OnlineRunConfig → family entry →
> resolve_distributed_resources → RayGenerationConfig.from_cfg → trainer_torch_device →
> RolloutCollectorConfig.from_cfg`，`train_dpo` 再抄共享子集——组合逻辑本身没有 owner。
> 新增 **`vrl/run.py`** 作为**唯一组合座**：
> `ResolvedRun`（built + family + resources + device，全入口共享）、
> `ResolvedOnlineRun`（+ run + generation + collector）、`resolve_run` / `resolve_online_run`
> （逐字保序复刻原脚本语句与守卫，fail-fast 次序不变）。两个入口改为一次 resolve + 解包，
> `OnlineRunConfig` 随迁。**新入口从此只调组合座、读字段，不再手串链条**——"config 如何变成
> runtime 对象"有了一个可指的地方。这是 §3 SINGLE-BOUNDARY + RESOLVED-BUNDLE 模式补上的
> 最后一层（入口层的 bundle），仍不是 god-object：它组合各层 owner，不吸收它们。

> **P8（追加，2026-07-24）：模型物化座——已落地 `2095b003`（附带修真 bug `44cfaa1d`）。** 第二条
> 被手抄 12 处的链：`resolve_model_build → checkpoint identity → build_rollout/replay → identity
> 复查`。`vrl/run.py` 增加 `ResolvedModel` + `resolve_model`（便宜投影段）与 `materialize`
> （bundle + 复查段，`context` 保住各站点逐字报错措辞）；两段拆分是因为 online 的 checkpoint
> preflight 恰在两段之间（lifecycle 测试 pin 次序）。已接座：online、wan DPO、两个 sana eval。
> **有据保留 inline**：从不算 identity 的站点（perf/generation/denoise/anima/wan_i2v——接座会新增
> 全树哈希与未 pin revision 的 raise 路径）、cosmos eval（`parameter_dtype_override` + pin 措辞）、
> `launcher.py`/`worker.py`（run 层之下，import 成环；worker 复查是进程边界防御）。顺带发现并修复
> wan_robotics eval 的潜伏 TypeError（传 raw DictConfig 且漏 `precision`，被测试 fake 遮蔽）。

> **P9（追加，2026-07-24）：收座带来的删除红利——已落地 `3f8386b7`。** 接座之后按五形态复查，
> 删掉三个因此丧失存在理由的扁平函数：`RolloutCollectorConfig.get()` 鸭子类型适配器（typed 分支
> form-2 死亡，余下读者改直读 `request_sampling`，适配器及其自测一并删除）、
> `resolve_family_model_build` 门面（生产调用者归零，TEST-ONLY=死，sana e2e 测试改走 entry 门面）、
> anima `_resolve_device`（第 6 份逐字拷贝，此前因 torch-作参签名逃过 P4 去重）。有据保留：
> `OnlineRunConfig.from_root`（独立被测的投影契约，keep-list）、`reward_inference_configs_from_cfg`
> 的 fallback 分支（约 100 个 resource 测试依赖其从 cfg 派生 reward placement 的语义）。

> **P10（追加，2026-07-25）：全仓 resolver/choreography 终扫——已落地 10 个批提交
> `60401174`..`306b7009`。** 用户指出"类似问题全仓很多处"，遂做全仓审计（10 区域猎手 × 逐条
> 对抗验证，本 session 全部已裁决的 keep 烘进规则防重报）：**55 条候选 → 40 条确认 / 15 条否决**
> （否决的是被误当散落的合理边界：AR 旋钮分离、wire 边界、swap_linears 循环、cross-rank moment
> reduction 等）。40 条中 **39 条执行、1 条按其自身判决 keep**（adjudicated eval 前缀）。分 10 批
> 落地（config-families / ray-utils / generation-core / generation-bindings / models-core /
> families-token / families-media / rollouts-trajectory / trainers-algos / rewards-scripts），
> 每批独立 ruff + config lint + fast 子集验证。**净删约 1500+ 行**：删 test-only/零调用（precision
> 模块、trainable-state facade、stack_trajectory_batches、ray facade、record_function 别名、
> GenerationRuntimeCapabilities 4 死字段等）、抽共享 owner（peel_peft、ordered_covering_chunks、
> resolve_image_token_replay、set_mu_shifted_timesteps、require_exact_int、resolve_kling_worker_config
> 等约 15 个）、扁平化 `_TrajectoryBatchBuilder` 类、折叠 causvid bundle。附带发现并修 2 个潜伏
> bug（causvid full-finetune replay 的 apply_full_finetune 签名 TypeError；wan_robotics 已在早前批修）。
> **另发现一个预存的测试隔离 flake**（`test_chunk_memory_shadow::test_probe_fits_confirms_and_truncates_steps`
> 在全量顺序里非确定性失败、单独/干净 HEAD 均正常）——非本轮引入，留作独立 issue。

父 program：[Argument and state ownership](SPRINT_argument_and_state_ownership_program.md)

前置（origin 已落地）：[Config argument ownership and resolution](SPRINT_config_argument_ownership_and_resolution.md)

## 0. 一句话

> 用户诉求：“all kinds of resolved config build xxx live everywhere in my repo; I don't
> want them scattered; maybe a single data class or something to resolve them.”

**“一个 data class 解决所有 resolve” 几乎肯定是错的方向**——那是 god-object，正是你自己
的 AGENTS.md 明令拒绝的形态（“A single god ‘resolve everything’ class is an ANTI-pattern the
repo would reject — the goal is clear OWNERSHIP and one-construction-site, not one giant class”）。
把 config-layer 的 precision、reward、distributed、per-family model build 全塞进一个类，会把今
天清晰的 ownership 边界重新糊成一坨，且立刻违反“Resolved struct 每个字段都要有 non-logging
consumer”的死字段规则（一个万能类的字段必然对多数 caller 是死的）。

**真正的问题不是“函数太多”，而是两类具体形态：**

- (a) **runtime-re-resolution**：少数下游 call site 明明已经拿到 typed / resolved 值，却又
  重新读 raw cfg 或把已 resolve 的 struct 再喂回 parser。这才是你感受到的“散落”。
- (b) **几处真实重复**：同一个 resolver 被逐字抄了 3–5 份（eval 的 `_resolve_device`、
  generation 的 `samples_per_chunk`、wan 的 topology 复算）。

而且 **origin 最近的 ~63 个 commit 已经把绝大部分散落收干净了**（见 §1）：`build_configs →
BuiltConfigs` 已是训练脚本唯一的顶层 construction site，`families/registry.py` 的
`resolve_model_build` 已是唯一的 config→ModelBuild 边界。你想要的“单一 resolve 入口”其实
**已经存在**，只是分层的（config layer 一个、model build 一个），而不是一个巨类。

**本 sprint 的正确交付**：把剩下的 ~6–7 处 runtime-re-resolution / 重复，**上提回已有的
owner**（或抽一个共享 helper），而不是新建任何抽象。这是一次收尾，不是重构。

## 1. 现状版图

八个 area 的调查共列出约 200 条 “resolver” 条目（跨 area 有大量重复列举，例如
`resolve_family_model_build`、`build_family_runtime_bundle` 在 families / generation 两处都出现）。
其中**绝大多数是 `keep`**：它们是真实的 boundary / per-family variant / typed `__post_init__`
校验器 / lazy-import dispatch。拿到 `consolidate` / `hoist` / `dedup` 判定的只有约 6–7 组
去重后的动作。分层归属如下。

### 1.1 config layer —— 顶层单一 construction site（已成型）

`vrl/config/` 已经是目标末态，不是散落源头：

- `build_configs`（`vrl/config/builders.py:427`）返回 frozen `BuiltConfigs`
  （root/algorithm/precision/trainer/reward/resume），是 online/offline 训练脚本**唯一**顶层入口。
- `validate_training_config`（`vrl/config/validation.py:328`）返回 `(RootConfig, PrecisionPolicy)`
  **对**，专为“precision 不被二次 resolve”而设计；`build_trainer_config` 用
  `precision or resolve_precision_policy(cfg)` 兜底。
- `builders.py` 把 merged tree 切成 per-layer typed config；`schema.py` 拥有 pydantic typed
  边界 + cross-field 规则；`algorithm/data/precision/reward_inference` 各拥有恰好一个投影。

### 1.2 families/registry —— 唯一 config→ModelBuild 边界（已成型）

`ModelFamilyEntry.resolve_model_build`（`vrl/families/registry.py:235`）是**唯一**把
validated `RootConfig` 变成 `ModelBuild` 的地方，type-guard 拒绝 raw OmegaConf，下游一律读
`ModelBuild` 而非 cfg。`resolve_family_model_build`（`vrl/models/steps/denoise/build.py:165`）
是它的 free-function facade。

### 1.3 model construction —— 保留 contract，一致算法共享实现

执行后的现行裁决比初稿更精确：

- denoise families 继续通过 registry 的 `model_cls` 暴露一致 `from_build` contract；
- 加载算法真正相同的 families 共用
  `DiffusersPipelineModelBase.from_build`，差异由 `_pipeline_classname`、
  `_frozen_encoder_names`、`_prompt_encoder_on_cpu` 等显式 declarations 表达；
- 有额外 scheduler、memory 或 checkpoint invariants 的 family 保留 override；
- token `*_config_from_build` 仍是 registry 的 typed per-family variants，保持分立。

因此“不拍平成 data table、保留跨 family grepability”的原则不变，但不再要求保留逐字相同的
per-family wrappers。

### 1.4 origin 已经消灭的散落（引用近期 commit）

| 已落地 | commit |
| --- | --- |
| validation/build 返回 typed `BuiltConfigs`，reward 不再 tuple 下标 | `a6a2cd2b` |
| data loader 只有一个 derivation owner | `456f7069` |
| rollout worker defaults 一次解析为 frozen settings | `520ab5a8` |
| checkpoint resume policy 回到 checkpoint owner | `4c13b2ee` |
| FSDP/DDP strategy 只消费已解析设置 | `877ef23e` |
| online trainer / batch geometry / lifecycle 按 consumer 拆分 | `217b7c32` `78e0242c` `8d65d2af` `a13cd3c6` |
| generation request 改 typed 正向投影，local knobs fail closed | `080ebb53` |
| online schema keys 从 typed owner 派生 | `cb6e2db2` |
| “give each YAML projection one owner” | `eaa681c6` |
| “derive family model schemas from registry” | `cba79bdc` |
| colocate family model configs（wan/cosmos/denoise/janus/nextstep/llamagen/emu3/glm） | `727ed032` `a2fac492` `c24d1d48` `3e6367c9` `ff136861` `966275fa` |
| build from validated config（resolve_model_build 边界成型） | `ca0dfd9f` |

**结论**：你担心的“scattered build xxx everywhere”在 config layer / families / rollouts /
ray / trajectory 五个 area 里**已经基本不存在**。本 sprint 只处理剩余尾巴。

## 2. 真正的问题（不是“函数太多”）

### 2.1 (a) 合法边界 / per-family resolver —— 必须保持分立

这些**不是**散落，删掉/合并它们才是倒退：

- **顶层 boundary resolver**：`build_configs`、`build_trainer_config`
  （`vrl/config/builders.py:223`，layout 从字段 `metadata={'yaml':...}` 派生，无平行表）、
  `resolve_model_build`、`resolve_distributed_resources`（`vrl/ray/resources.py:162`，被
  schema.py:781 显式指向为 value owner）。每个是一个真实 ownership，单一 construction site。
- **lazy-import dispatch**：`algorithm_config_class`（`vrl/config/algorithm.py:9`）、
  `new_gatherer`（`vrl/families/registry.py:412`）——保持 config-parse 路径不加载 torch/runtime。
- **per-role device 三兄弟**：`_resolve_role/rollout/reward_devices`（resources.py:626/657/829）
  语义各异（trainer=head、rollout=disjoint、reward=disjoint-from-both），且已共享
  `_slice_pool_with_overlap_fallback` / `_requested_role_gpu_count` 真正的公共逻辑。跨家族
  grepability 优先，保留三份外壳。
- **刻意的 wire-boundary backstop**：`ChunkPlacementPolicy.__post_init__`
  （`chunk_placement.py:32`）故意重复 `RolloutWorkerSection` 的 allowed-set 校验，是 defense-in-depth。
- **合法的 process-boundary re-resolution**：`apply` 侧 executor 在 Ray worker 端重新解析
  serialized wire dict（`full_sequence_denoise/executor.py:338`），跨进程边界，必须留。

### 2.2 (b) runtime-re-resolution —— 该上提回 owner（真正的散落）

这是用户真正感受到的“散落”，共四簇：

1. **reward inference map 从 raw cfg 二次 walk**。`reward_inference_configs_from_cfg`
   （`vrl/config/reward_inference.py:102`）在 GPU placement 阶段深处
   （`vrl/ray/resources.py:171`）重新读 `reward.components/kwargs` 原始 cfg，构造
   `{name: RewardInferenceConfig}`——**但** `RewardConfig._validate_reward` 在 schema 校验时
   **已经**对每个 component 调过 `parse_reward_inference_config`。同一批 typed struct resolve
   了两次。本 area 唯一的真实重复。

2. **collector 把已 typed 的 storage policy 又喂回 parser（且是 latent bug）**。
   `vrl/rollouts/collector/core.py:210`：

   ```python
   trajectory_storage_policy = trajectory_storage_policy_from_cfg(
       cfg_get(self.config, "trajectory_storage", None),
   )
   ```

   `self.config` 是 `RolloutCollectorConfig`，其 `trajectory_storage` 字段 **已经是** resolved
   的 `TrajectoryStoragePolicy`（`from_cfg` 在 config.py:83 已 resolve 过一次）。`cfg_get` 命中
   `.get()` adapter（config.py:33，`return getattr(self, name)`）→ 返回 typed policy dataclass →
   喂回 `trajectory_storage_policy_from_cfg`（storage.py:43），后者 `to_builtin` 不转 dataclass、
   非 Mapping → **直接命中 `raise TypeError`**（storage.py:57）。所以除了冗余，这是一条会崩的
   路径（当前 CPU 测试大概没走到带 in-loop KL/storage 的 scoring 分支才没炸）。正解是
   `self.config.trajectory_storage` 直接读。

3. **generation request.sampling 被 ~6 处各自 re-parse**。`request.sampling` 是个
   `dict[str, Any]` 的 per-request runtime payload，已有唯一 owner
   `DiffusionRequestLayout.parse_sampling_params`（`full_sequence_denoise/layout.py:73`）/
   `ARRequestLayout.parse_sampling_params`（`token_autoregressive/layout.py:45`），但下面这些
   site **绕过** owner 直接读 raw dict：

   - `samples_per_chunk` 在 3+ 处各自 resolve：layout 的 parser、`build_engine_plan`
     fallback（`execution/planner.py:18`）、`DistributedExecutionPlanner.plan_with_engine`
     （`execution/chunk_placement.py:218`）。
   - `trajectory_storage` 每个 chunk 重新 derive：`apply_wire_storage_policy`
     （`full_sequence_denoise/executor.py:324`），而策略是 request-constant。
   - AR 的 paged/engine/scheduler 旋钮（`ar_paged_block_size`、`ar_paged_cache_dtype`、
     `ar_engine`、`ar_scheduler_batch_size`）**不在** `ARSamplingParams` 里，被
     `_ar_runner`（executor.py:102）、`require_native_ar_engine`（executor.py:158）、
     `resolve_scheduler_batch_size`（layout.py:68）直接从 raw dict 读。

   ⚠️ 关键约束：这是 **per-request runtime payload，不是 launch YAML**，所以修法**不是**
   “hoist 到 config layer”，而是让两个已有的 `parse_sampling_params` 成为**唯一**的 request-side
   parser。

4. **families 内部把 normalizer 已 settle 的东西在 from_build 时复算**：
   - `wan_topology_from_build`（`wan_2_1/config.py:169`）被 T2V/I2V/replay 三处 from_build
     各调一次，每次重新 normalize+validate normalizer 已装好的 topology。
   - `_resolve_wan_offload_mode`（`wan_2_1/model.py:1287`）重读 rollout 并重查
     `normalize_wan_model_build` 已经 reject 过的 legacy key。
   - `resolve_echo_video_dimensions`（`echo/config.py:57`）在 model.py 和 runtime.py 两个
     build site 各跑一遍同样的 model/sampling cross-check。
   - `vae_decode_memory_from_config`（`denoise/common/vae_decode_memory.py:31`）把 raw
     `model.memory.vae_decode` 在 runtime-bundle-build 时才 parse 成 `VaeDecodeMemory`——
     这个 **parse** 可以上提到 `resolve_model_build`（apply 因需要 live VAE 仍留 runtime）。

### 2.3 (c) 真实重复 —— 该 dedup

- **`_resolve_device` 被逐字抄 5 份**：`cosmos_predict25_kling_eval.py:260`、
  `sana_aesthetic_checkpoint_eval.py:769`、`sana_checkpoint_compare.py:394`、
  `video_reward_suite.py:399`、`wan_robotics_checkpoint_eval.py:830`，只有 error 措辞不同。
  `eval/` 没有共享 util module。
- **`_resolve_sampling` 两份构造同一个 10-key dict**：`cosmos:288` / `wan_robotics:591`，
  从相同的 `sampling.*`/`rollout.*` YAML 路径读进 untyped dict。
  （注意刻意的 **非重复**：`sana_aesthetic` 的 `_resolve_sampling:780` 返回
  `OFFICIAL_SAMPLING_PROTOCOL` 常量以保复现独立于训练 SDE——保持分立。）
- **`_resolve_rollout_num_workers`（resources.py:790）≈ `_resolve_role_num_workers`
  （resources.py:1020）**：dead-code form 4，body 几乎同体，仅 zero-worker 语义不同。
- **单文件 checkpoint 解析同 precedence 抄 3 份**：anima `_resolve_artifact`
  （`cosmos/anima/model.py:578`，最干净、已用 `hf_hub_download`）、echo
  `_resolve_echo_checkpoint`、llamagen `_resolve_checkpoint_file`（`llamagen/model.py:429`）
  共享“explicit > local-root > hub + validate_checkpoint_source_member”precedence。
- **online.py 深处重读 raw cfg**：`model.use_lora` 经 `OmegaConf.select` 读了 **4 处**
  （`_default_reference_model:375`、`_export_transformer_lora:414`、
  `_export_language_model_lora:440`、gradient-checkpointing 路径）；`algorithm.kl_coef`/
  `sft_weight`、`data.sft_latents`、`model.path/revision` 同样在 run 深处从 raw cfg 重读，
  尽管 `build_configs` 已产出 typed struct。

## 3. 提案：SINGLE-BOUNDARY + RESOLVED-BUNDLE（不是 god class）

设计原则，全部是**扩展 origin 已建立的模式**，零新抽象：

1. **一个 config 投影一个 owner**。每类 resolve 已有单一 owner（config layer 的
   `build_*` / `resolve_*`；families 的 `resolve_model_build`；request 层的
   `parse_sampling_params`）。剩余散落 = 下游没走 owner。修法 = 让下游走 owner，不新建 owner。

2. **compute-once → read-everywhere via 已有的 Resolved bundle**。已 resolve 的值挂在既有的
   typed 出参上（`BuiltConfigs` / `RewardRuntimeConfig` / `ModelBuild` / `ARSamplingParams`），
   下游读字段，不重算。**遵守死字段规则**（AGENTS.md）：每个新挂上去的字段必须有 non-logging
   consumer（control-flow / 传给 runtime/Ray / 能 raise 的 validation），否则不挂。

3. **runtime-re-resolution 一律上提到最近的已有 owner**——launch-time 的上提到 config layer
   / `resolve_model_build`；per-request 的收敛到 `parse_sampling_params`。

### 3.1 Before / After 示例

**示例 A：reward inference map（launch-time hoist）**

```python
# BEFORE — vrl/ray/resources.py:171，GPU placement 深处重读 raw cfg
def resolve_distributed_resources(cfg):
    reward_inference = reward_inference_configs_from_cfg(cfg)   # 重新 walk raw reward.*
    ...

# AFTER — config layer resolve 一次，挂上 RewardRuntimeConfig，placement 读字段
# build_reward_config(...) 里已持有 typed RewardConfig，顺手产出 map：
built.reward.inference_configs  # {name: RewardInferenceConfig}，已由 schema 校验期算出
def resolve_distributed_resources(cfg, reward_inference):   # 由 caller 传入 resolved map
    ...
```

**示例 B：trajectory storage（读 typed 字段，删 re-resolution + latent bug）**

```python
# BEFORE — vrl/rollouts/collector/core.py:210（会命中 raise TypeError 分支）
trajectory_storage_policy = trajectory_storage_policy_from_cfg(
    cfg_get(self.config, "trajectory_storage", None),   # 已是 typed policy，喂回 parser 崩
)

# AFTER — 直接读已 resolve 的 typed 字段
trajectory_storage_policy = self.config.trajectory_storage
```

副产品：`self.config` 不再被当 generic cfg node 用后，`RolloutCollectorConfig.get()` adapter
（config.py:33）失去唯一 re-resolution caller，可评估随之删除（`generation_sampling` 已直接读
typed 字段）。

### 3.2 明确不合并（keep-list）

- denoise `from_build` contract + token `*_config_from_build` —— 不拍平成 data table；
  identical denoise loading algorithms 由 `DiffusersPipelineModelBase.from_build` 共享，
  family-specific invariants 继续留在 declarations 或 override。
- 顶层 boundary resolver（`build_trainer_config`、`resolve_model_build`、
  `resolve_distributed_resources`）—— 真实 ownership，**不并入任何总类**。
- lazy-import dispatch（`algorithm_config_class`、`new_gatherer`）—— 保持 torch off config-parse。
- per-role device 三兄弟 —— 语义各异，已共享真正的公共逻辑。
- `ChunkPlacementPolicy.__post_init__` wire backstop、executor.py:338 worker-side re-parse
  —— 故意的跨边界 defense / process-boundary，**保留**。
- causvid SHA256/source audit、magi 多 folder snapshot —— 真 family-specific gate，**不折进**
  共享 checkpoint helper。

## 4. 迁移增量

每个 phase 独立可交付、可回退、可单独测；按 value/risk 排序（高在前）。

```text
P1  [bug + trivial] collector/core.py:210 读 self.config.trajectory_storage；
    评估删 RolloutCollectorConfig.get() adapter。
    收益最高：同时修掉一条会 raise TypeError 的路径 + 去 re-resolution。
    Gate: rollouts/trajectory 定向测 + 一条覆盖 in-loop scoring 的 pin 测试。

P2  [config-layer hoist，clean win] reward_inference map 在 config layer resolve 一次
    （挂 BuiltConfigs 或 RewardRuntimeConfig），传给 resolve_distributed_resources；
    删 ray/resources.py 对 raw cfg 的 reward walk。
    Gate: reward 有/无、in_process|http 两 transport、placement 测试无 raw-cfg 读取。

P3  [generation request 单 owner] 让 parse_sampling_params 成为唯一 request-side parser：
    - 把 ar_paged_block_size/ar_paged_cache_dtype/ar_engine/ar_scheduler_batch_size
      并入 ARSamplingParams；
    - parsed samples_per_chunk / step-count 线程进 planner + chunk_placement 成本；
    - trajectory_storage 在 executor 里 parse 一次，不再 per-chunk。
    收敛 ~6 处 re-resolution 到两个已有 typed params struct，零新抽象。
    Gate: diffusion + AR 各族 request→params 测试；samples_per_chunk 单一来源 pin。

P4  [scripts dedup] eval/ 抽一个共享 helper（resolve_eval_device + dtype，吸收
    “auto→precision，cpu→fp32”规则）替换 5 份 _resolve_device / _resolve_dtype；
    cosmos+wan 的 _resolve_sampling 收敛到一个 typed sampling-protocol resolver。
    保持 sana OFFICIAL_SAMPLING_PROTOCOL 分立。
    Gate: 五脚本 import + device/dtype 解析 pin；sana 复现协议不变。

P5  [online.py raw-cfg gate 上提] model.use_lora（×4）/ algorithm.kl_coef / sft_weight /
    data.sft_latents / model.path/revision 从 BuiltConfigs 的 typed 字段读，删 OmegaConf.select
    重读。tensor 加载体保留，只把 gate/paths 换成 typed。
    Gate: LoRA on/off、SFT on/off、reference-model 分支的构造测试。

P6  [families local dedup，低 value]
    - wan_topology_from_build：normalizer 存 typed WanTopology，from_build 读一次；
    - _resolve_wan_offload_mode：直接读 rollout.pipeline_offload_mode；
    - resolve_echo_video_dimensions：resolve 一次；
    - vae_decode_memory parse 上提到 resolve_model_build（typed VaeDecodeMemory 下发，
      apply 留 runtime）；prompt_encoder_dtype fallback 折进 PrecisionPolicy；
    - 抽 resolve_single_file_artifact 共享 anima/echo/llamagen 单文件 checkpoint precedence，
      保留各家 repo/filename 默认与 family-specific gate；
    - _resolve_rollout_num_workers 折进 _resolve_role_num_workers（role-参数化 allow_zero/min）。
    Gate: 各族 CPU 构造 smoke + 现有 ray resources 测试。
```

P1–P2 单独就把用户能感知的散落（会崩的 re-resolution + 唯一的 config-layer 重复）清掉；
P3–P6 是收尾质量项，可按需排期。

## 5. Non-Goals

- **不新建任何 god “resolve everything” class / `*Manager` / `*Container`。** 用户的
  “single data class” 直觉在此仓库是反模式；已有的分层 owner 就是答案。
- **不把 `from_build` / `*_config_from_build` contract 拍平成 data table。**
  identical denoise algorithms 可以共用 `DiffusersPipelineModelBase.from_build`；
  token builders 和有额外 invariants 的 denoise override 保持分立。
- **不合并 boundary resolver / lazy-import dispatch / per-role device 三兄弟**。
- **不把 `_distributed_resource_config_from_cfg` 从 `vrl/ray/` 挪到 `vrl/config/builders.py`**：
  schema.py:781 显式把 reader 指向 resources.py，这是刻意的 ownership 选择，不是散落。
- **不动 process-boundary / wire-boundary 的合法 re-resolution**（executor.py:338、
  ChunkPlacementPolicy backstop）。
- **不动 `validate_production_*` gate**（validation.py:94，故意读 raw cfg 做 file/manifest I/O，
  不能进 pydantic）。
- **不改四层 config 架构**（OmegaConf → Pydantic → resolver/builder → runtime config 保持）。
- **不做无差别 YAML/格式清扫**；experiment-level 显式 pin 保留。

## References

**尾巴所在（本 sprint 目标）：**

- `vrl/config/reward_inference.py:102`（reward_inference_configs_from_cfg）/ `vrl/ray/resources.py:171`（调用点）
- `vrl/rollouts/collector/core.py:210`（trajectory_storage re-resolution + latent TypeError）/
  `vrl/rollouts/collector/config.py:33`（.get adapter）/ `vrl/trajectory/storage.py:43,57`
- `vrl/generation/bindings/full_sequence_denoise/layout.py:73` / `executor.py:324`；
  `vrl/generation/bindings/token_autoregressive/layout.py:45,68` / `executor.py:102,158`；
  `vrl/generation/execution/planner.py:18` / `chunk_placement.py:218`
- `vrl/ray/resources.py:790,1020`（rollout vs generic num_workers）
- eval `_resolve_device` ×5：`cosmos_predict25_kling_eval.py:260`、`sana_aesthetic_checkpoint_eval.py:769`、
  `sana_checkpoint_compare.py:394`、`video_reward_suite.py:399`、`wan_robotics_checkpoint_eval.py:830`；
  `_resolve_sampling`：`cosmos_predict25_kling_eval.py:288` / `wan_robotics_checkpoint_eval.py:591`
- `vrl/scripts/common/online.py:375,414,440,390`（raw-cfg 重读 use_lora / sft）
- families：`wan_2_1/config.py:169`、`wan_2_1/model.py:1287`、`echo/config.py:57`、
  `models/steps/denoise/common/vae_decode_memory.py:31`、`models/steps/denoise/base.py:499`、
  `cosmos/anima/model.py:578`、`llamagen/model.py:429`

**已成型的 owner（不要动）：**

- `vrl/config/builders.py:427`（build_configs → BuiltConfigs）、`:223`（build_trainer_config）
- `vrl/config/validation.py:328`（validate_training_config → typed pair）
- `vrl/families/registry.py:235`（resolve_model_build，唯一 config→ModelBuild 边界）
- `vrl/ray/resources.py:162`（resolve_distributed_resources，唯一 config→resource 边界）

**origin 已消灭的散落（近期 commit）：**

- `a6a2cd2b` `456f7069` `520ab5a8` `4c13b2ee` `877ef23e` `217b7c32` `78e0242c` `8d65d2af`
  `a13cd3c6` `080ebb53` `cb6e2db2` `978ee3c8`（argument-ownership program）
- `eaa681c6`（give each YAML projection one owner）、`cba79bdc`（derive family model schemas）、
  `727ed032` `a2fac492` `c24d1d48` `3e6367c9` `ff136861` `966275fa`（colocate family model config）、
  `ca0dfd9f`（build from validated config）

**相关 done sprint：**

- `docs/sprints/done/SPRINT_config_argument_ownership_and_resolution.md`（本 sprint 的直接前置——
  origin 已把顶层 resolve-once / typed-result / consumer-split 落地）
- `docs/sprints/done/SPRINT_config_as_signatures.md`（config as torch signatures）
- `docs/sprints/done/SPRINT_single_caller_inlines.md`（单调用内联 + form-4 同体上提）
- `docs/sprints/done/SPRINT_family_model_config_ownership.md`（family model config 归属）
- `docs/sprints/done/SPRINT_argument_and_state_ownership_program.md`（父 program）
