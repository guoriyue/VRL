# SPRINT: 补审被错分的生产训练入口层（scripts/diffusion/）

状态：**done（2026-07-11）**。父：`SPRINT_fbag_00_overview.md`。逐函数审计已完成：全部生产
入口判为 cohesive-keep，两个明确 finding 已落地提交，附带的 perf `fp8_math` 重复项审后判 KEEP。
config-resolve 全 preset 冒烟（`tests/config/test_load_all_experiments.py` 33/33）确认每个
entrypoint 字符串仍解析。逐入口判决见 §4，验证见 §5。

> 这是 function-bag 审计暴露出的**范围漏洞**,不是一条具体缺陷。sweep agent 判定
> `vrl/scripts/{perf,eval,data}` 是正确的一次性生命周期,但 `vrl/scripts/diffusion/` 被错分。

## 0. 一句话

`vrl/scripts/diffusion/` 下的生产训练/生成入口层不是一次性 probe,而是由 ~15 个 config preset
通过 `trainer.entrypoint` 点名字符串调度。它们因为落在 `scripts/` 下,被第一轮深度审计(只覆盖
`vrl/{trainers,config,utils,trajectory,rollouts,nn,models}`)排除。补审结论:**全部内聚,无 form-2/3/4
问题**——每家 `train.py` 都是薄 recipe wrapper,委托共享 `run_online_recipe`,只携带各自合法的家族
钩子(collector kwargs / after_bundle),没有一家手抄了共享序列。

## 1. 为什么它们是长期资产,不是 probe

AGENTS.md:一次性 probe 的价值是"它产出的答案",可以是 procedural 函数袋;长期资产"live in
canonical paths, are referenced by other code/docs, survive cleanup"。这 7 个文件全部被 config
`trainer.entrypoint` **点名字符串**引用(plain-symbol grep 会漏,正是 five-forms 里
"module:function 字符串调度"的坑):

| 入口文件 | 被调度的 entrypoint 字符串 |
|---|---|
| `scripts/diffusion/train.py` | `vrl.scripts.diffusion.train:train_diffusion_grpo`(descriptor 家族的通用入口:sd3.5、wan-t2v、dance-grpo、dppo 等全走这里,家族由 `model.family` 分派) |
| `scripts/diffusion/cosmos/train.py` | `...cosmos.train:train_cosmos_predict2_grpo` / `:train_cosmos_predict25_grpo` / `:train_cosmos_predict25_diffusion_nft` |
| `scripts/diffusion/flux/train.py` | `...flux.train:train_flux_diffusion_nft`(GRPO 走通用入口,仅 NFT 留家族 recipe) |
| `scripts/diffusion/wan_2_1/train.py` | `...wan_2_1.train:train_wan_2_1_i2v_grpo`(仅 I2V;T2V 走通用入口) |
| `scripts/diffusion/wan_2_1/train_dpo.py` | `...wan_2_1.train_dpo:train_wan_2_1_dpo`(离线 DPO,不走 `run_online_recipe`) |
| `scripts/diffusion/cosmos/anima/generate.py` | 生产采样 CLI(+ test importer) |

> 事实修正(2026-07-11 补审):第一版此表虚列了 `scripts/diffusion/sd3_5/train.py` /
> `train_sd3_5_grpo` 与 wan 的 `train_wan_2_1_grpo`——两者都不存在。sd3.5 与 wan-T2V 都是
> descriptor 家族,走通用 `train_diffusion_grpo`(证据:`recipe/online/flow_matching_grpo.yaml`
> 的 entrypoint + sd3_5 实验 include 它)。另两个顶层脚本 `scripts/diffusion/generate.py`(家族无关
> 生成 CLI)与 `encode_targets.py`(SFT-latents 生产者)是 generation/data 生命周期,正确地不在本
> recipe-entrypoint 审计范围内。

这些是每次真训练都会跑的代码,一旦有死分支/家族间抄漏/单调用者拆分,影响的是生产,不是丢弃品。

## 2. 动作:按 library-core 审这 7 个文件

复用第一轮同一套判据(五形态 + thin-function 保留清单 + 对抗性 verify),重点看**跨家族抄写**
——AR/diffusion 各家 train.py 极易出现 form-4(某家 train.py 手抄了共享 recipe 序列但步骤微妙
错位,如历史上 cosmos3 把量化放到 compile 之后)。审计要 diff 每家 train.py 的主体 vs
`vrl/scripts/common/online.py` 的共享 recipe,而不是看调用点。

具体:对每个入口的每个 `train_*` 函数,问 form-2(某分支的输入还有没有生产者)、form-4(主体是
不是共享 `run_online_recipe` 的手抄变体)、form-3(family train.py 内部的私有拆分)。

### 2.1 已落地 findings（committed）

- `cosmos/anima/generate.py::_resolve_sampling` 删除 7 个从未被生成请求读取的训练期 sampling key；
  CLI adapter 只保留实际传入 prompt encoding 与 `VideoGenerationRequest` 的 5 个值。
- `wan_2_1/train_dpo.py` 删除私有 `_trainer_precision_label`，复用
  `vrl.trainers.precision.normalize_mixed_precision` 这一 Accelerate 协议适配边界。

两者均已在提交代码中生效(复核 2026-07-11:`train_dpo.py:38,124` 用 `normalize_mixed_precision`；
`generate.py::_resolve_sampling` 只剩 5 个 key)。

## 3. 附带:一个 perf 目录的重复 helper

`vrl/scripts/perf/common/fp8_math.py` 的 `amax_scale` / `tensorwise_fp8_matmul` 手抄了
`vrl/nn/quantization/fp8.py` 的量化核心(form-4/5):`amax_scale` 重实现私有 `_amax_scale`
的 scalar 分支(`fp8.py:72`),`tensorwise_fp8_matmul` 重实现 `Fp8Linear.forward` 的
tensorwise 分支(scalar amax-scale 两操作数后 `torch._scaled_mm` bf16 累加)。
它正确地从核心 import 了 `FP8_E4M3_MAX` 常量,但抄了序列本身。

**判决(2026-07-11 补审后):KEEP,不改。** 逐项核对后,form-4/5 在这里不成立:

1. **probe 仍活着**:`amax_scale`/`tensorwise_fp8_matmul` 被 3 个 perf probe 导入
   (`fp8_recipe_accuracy.py`、`fp8_rollout_drift_probe.py`、`fp8_linear_benchmark.py`),都是用户保留
   的 perf 测量档案。
2. **它们是独立的参考/基准,不是生产代码的便利复制**:`fp8_rollout_drift_probe` 用
   `tensorwise_fp8_matmul` 当"fp8 对 RL log-prob 漂移影响"的独立参照,`fp8_linear_benchmark` 定义自己
   的最小 `_Fp8Linear` 基准类。让它们改调生产 `Fp8Linear.forward` 反而 ① 拿不到裸 tensorwise
   matmul(生产 forward 裹着 reshape/bias/blockwise/rowwise/master-drop 机制),② 把测量口径耦合到
   生产代码演进——测量与被测同源正是要避免的。
3. **唯一会静默 rot 的真魔数已单源**:`FP8_E4M3_MAX` 已从 `nn/quantization/fp8.py` import;重复的只是
   `(amax/MAX).clamp_min(1e-12).to(float32)` 这一行 e4m3 amax-scale 的教科书定义,不是可微妙错位的
   多步序列。而生产侧的 `_amax_scale` 是私有的,为一个 probe 把它公开会反向扩大生产模块 API 面。

这符合"独立性/一致性优先于减行数"与 perf-sprint 保留原则,不动。

## 4. 逐入口判决（补审闭合，2026-07-11）

| 入口文件 | 判决 | 依据 |
|---|---|---|
| `diffusion/train.py` | cohesive-keep | 薄 recipe wrapper + 两个 lazy-import 边界(`build_bundle`/`build_replay_bundle`,被 cosmos/flux/wan 复用,不是私拷) |
| `cosmos/train.py` | cohesive-keep | 3 个入口共用 `_after_bundle_built`(3 caller,合法)+ predict2 的 per-sample reference 校验钩子,无手抄共享序列 |
| `flux/train.py` | cohesive-keep | 唯一差异是 NFT 无 `reference_model_getter`(docstring 说明),其余全委托 |
| `wan_2_1/train.py` | cohesive-keep | 仅 I2V,携带 per-sample reference 校验钩子;T2V 已归通用入口 |
| `wan_2_1/train_dpo.py` | cohesive-keep | 唯一不走 `run_online_recipe` 的离线入口(docstring 说明);`metric_fields` 从 `DPOStepMetrics` 派生(源真理,非硬编码);`_build_encoders` 是长流程里的概念抽取,保留 |
| `cosmos/anima/generate.py` | cohesive-keep | 生成 CLI,各 `_resolve_*` 是独立多分支概念;finding 已落地 |

form-2(分支输入无生产者)、form-3(family 内私有拆分)、form-4(手抄共享序列)在这 6 个文件里均未命中。
未发现新的死符号或可合并的单调用者拆分。

## 5. 验证

- **config-resolve 全 preset 冒烟**:`tests/config/test_load_all_experiments.py` **33/33 通过**——每个实验
  的 `trainer.entrypoint` 字符串仍解析,确认没有入口被误删/误改。
- **入口文件回归**:`tests/scripts/test_anima_generate.py`、`test_encode_sft_targets.py`、
  `tests/trainers/test_offline_dpo_timesteps.py` **18/18 通过**。
- **全仓 lint**:`ruff check vrl/ tests/` 全绿(无未用导入/变量/可简化项)。
- 本轮为纯闭合:除已提交的两个 finding 外无新增代码改动,故无 KEEP→改的落地需要再跑对抗性 verify。

## 引用

- 调度证据:`vrl/config/presets/recipe/online/*.yaml` 的 `trainer.entrypoint`
- 共享 recipe(form-4 对照基准):`vrl/scripts/common/online.py:run_online_recipe`
- perf 重复:`vrl/scripts/perf/common/fp8_math.py` vs `vrl/nn/quantization/fp8.py:72,168`
