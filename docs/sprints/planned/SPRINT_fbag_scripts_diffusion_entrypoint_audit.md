# SPRINT: 补审被错分的生产训练入口层（scripts/diffusion/）

状态：入口 body-level 审计已完成（2026-07-11，findings 全部落地，正式 `make verify`
门禁通过：1701 passed、16 skipped、23 deselected）；仅 §3 的 perf fp8_math 重复 helper
仍待处置。父：`SPRINT_fbag_00_overview.md`。

> 这是 function-bag 审计暴露出的**范围漏洞**,不是一条具体缺陷。sweep agent 判定
> `vrl/scripts/{perf,eval,data}` 是正确的一次性生命周期,但 `vrl/scripts/diffusion/` 被错分。

## 0. 一句话

`vrl/scripts/diffusion/` 下的 7 个文件不是一次性 probe,而是**生产训练/生成入口层**,由 ~15 个
config preset 通过 `trainer.entrypoint` 点名字符串调度。它们因为落在 `scripts/` 下,被第一轮
深度审计(只覆盖 `vrl/{trainers,config,utils,trajectory,rollouts,nn,models}`)排除。需要按
library-core 同样的五形态审一遍。

## 1. 为什么它们是长期资产,不是 probe

AGENTS.md:一次性 probe 的价值是"它产出的答案",可以是 procedural 函数袋;长期资产"live in
canonical paths, are referenced by other code/docs, survive cleanup"。这 7 个文件全部被 config
`trainer.entrypoint` **点名字符串**引用(plain-symbol grep 会漏,正是 five-forms 里
"module:function 字符串调度"的坑):

| 入口文件 | 被调度的 entrypoint 字符串 |
|---|---|
| `scripts/diffusion/train.py` | `vrl.scripts.diffusion.train:train_diffusion_grpo`(4+ 个 flow_matching recipe) |
| `scripts/diffusion/cosmos/train.py` | `...cosmos.train:train_cosmos_predict2_grpo` / `:train_cosmos_predict25_grpo` / `:...nft` |
| `scripts/diffusion/sd3_5/train.py` | `...sd3_5.train:train_sd3_5_grpo` |
| `scripts/diffusion/flux/train.py` | `...flux.train:train_flux_diffusion_nft` |
| `scripts/diffusion/wan_2_1/train.py` | `...wan_2_1.train:train_wan_2_1_grpo` / `:train_wan_2_1_i2v_grpo` |
| `scripts/diffusion/wan_2_1/train_dpo.py` | `...wan_2_1.train_dpo:train_wan_2_1_dpo` |
| `scripts/diffusion/cosmos/anima/generate.py` | 生产采样 CLI(+ test importer) |

这些是每次真训练都会跑的代码,一旦有死分支/家族间抄漏/单调用者拆分,影响的是生产,不是丢弃品。

## 2. 动作:按 library-core 审这 7 个文件

复用第一轮同一套判据(五形态 + thin-function 保留清单 + 对抗性 verify),重点看**跨家族抄写**
——AR/diffusion 各家 train.py 极易出现 form-4(某家 train.py 手抄了共享 recipe 序列但步骤微妙
错位,如历史上 cosmos3 把量化放到 compile 之后)。审计要 diff 每家 train.py 的主体 vs
`vrl/scripts/common/online.py` 的共享 recipe,而不是看调用点。

具体:对每个入口的每个 `train_*` 函数,问 form-2(某分支的输入还有没有生产者)、form-4(主体是
不是共享 `run_online_recipe` 的手抄变体)、form-3(family train.py 内部的私有拆分)。

### 2.1 已落地 findings

- `cosmos/anima/generate.py::_resolve_sampling` 删除 7 个从未被生成请求读取的训练期 sampling key；
  CLI adapter 只保留实际传入 prompt encoding 与 `VideoGenerationRequest` 的 5 个值。
- `wan_2_1/train_dpo.py` 删除私有 `_trainer_precision_label`，复用
  `vrl.trainers.precision.normalize_mixed_precision` 这一 Accelerate 协议适配边界。

### 2.2 body-level 审计结论（2026-07-11 落地）

范围修正：§1 表已滞后——`sd3_5/train.py` 与 wan t2v 入口已并入通用
`diffusion.train:train_diffusion_grpo`，实存入口为 6 个文件（generic/cosmos/flux/
wan_i2v/wan_dpo/anima_generate）+ AR 侧 `ar/train.py`（同一 definition seam 的生产者，
顺带纳入）。

**FIX（wrong derivation，2 个真 bug）**

- `wan_2_1/train_dpo.py::_build_encoders`：encode 归一化方向反了——
  `(latents - mean) * latents_std` 应为 `/ latents_std`。证据：仓库权威 decode
  （`models/diffusion/wan_2_1/model.py::decode_latents`）为 `raw = z * std + mean`
  （代码取 `1.0/config.latents_std` 再除），encode 必须是其逆 `(raw - mean)/std`；
  上游 flow_grpo wan pipeline 同。原实现把 latent 按通道放大 std²（wan std≈1.5-2.8
  → 2-8 倍），transformer 吃到 OOD 输入。回归测试
  `tests/scripts/test_wan_dpo_encoders.py`（真/假两侧 + decode 往返）。
- `cosmos/anima/generate.py`：metadata 行记录 `seed = prompt_seed + sample_index`
  是虚构——`prepare_sampling`（anima/model.py）用**一个** generator seed 整批采样，
  单样本没有独立 seed。改记真实可复现的 batch seed（`prompt_seed`）。

**REMOVE（死写入 / 零生产者协议面）**

- `wan_2_1/train.py` 的 `metadata.setdefault("conditioning", "reference_image")`：
  全仓无读者（唯一 `"conditioning"` 运行时读点是 parity probe 断言的 batch_context，
  其来源是 `model.export_batch_context` 的硬编码，与 example.metadata 无关）。
  回归测试 `tests/scripts/test_wan_i2v_collector_kwargs.py`（含 metadata 不被改写）。
- `OnlineRecipeDefinition`（common/types.py）9 个死字段/死钩子，全部 grep + git -S
  双证据：`build_bundle`（生产侧 `build_replay_bundle or build_bundle` 回退从不触发，
  8 个构造点全部传了 replay；唯一消费者是 lifecycle 测试 fake）→ 删字段、
  `build_replay_bundle` 转必填；`model_getter`/`scheduler_getter`（零自定义生产者，
  历史用户在 e0f1d2a4/d14cdd55 descriptor 迁移中退役）→ 默认行为内联进 online.py；
  `configure_trainer`/`weight_dtype_getter`/`before_step`/`after_step`/`metric_row_hook`
  （零生产者；before/after_step 的最后用户随 Ray reward framework 一起删除）；
  `description`（零读者）。
- 级联清理：`RecipeDeviceContext`（单点构造单点读，纯重包装本地变量）；
  `OnlineRecipeStack` 里 hook 死后失去全部消费者的 7 个字段
  （model/collector/algorithm/evaluator/trainer_config/collector_config/output_dir）；
  `diffusion/train.py::build_bundle` 与其唯一底层
  `build_family_runtime_bundle_from_cfg`（models/diffusion/build.py）；
  `ar/train.py::_build_bundle` + 单调用者 `_resolve_family_imports` 内联。

**KEEP（对抗性核查后保留）**

- cosmos `_predict2_collector_kwargs` global 分支返回的 `reference_image` kwarg：
  与 launcher `_build_executor_kwargs` 的注入同源（cfg.model.reference_image）但不
  等价——hook 侧多 fail-fast 存在性校验 + expanduser，且被
  `tests/scripts/test_predict2_reference_kwargs.py` 钉死为 launch-contract 协议行为。
- cosmos `_after_bundle_built` 薄包装：携带 `require_method=False` 实参差异，非纯转发。
- `completed_step/completed_epoch` 进度键：`write_checkpoint_meta` 的人读
  checkpoint_meta.json 契约（provenance），有 writer 契约消费。
- DPO 手工桥接 TrainerConfig 默认值：DPO 不走 `build_configs()`，从 dataclass 字段
  派生默认属"derived, never copied"的合法形态。
- 备注（未处置）：`train_dpo.py` 的 `str(data_cfg.cache_dir) or None` 在 yaml 显式
  `null` 时会得到字符串 "None"（现有 preset 均为 `""`，不触发）；
  `_normalize_per_sample_reference_images` 名字滞后（只 validate 不 normalize），
  改名属 taste，未动。

## 3. 附带:一个 perf 目录的重复 helper

`vrl/scripts/perf/common/fp8_math.py` 的 `amax_scale` / `tensorwise_fp8_matmul` 手抄了
`vrl/nn/quantization/fp8.py` 的量化核心(form-4/5):`amax_scale` 重实现私有 `_amax_scale`
的 scalar 分支(`fp8.py:72`),`tensorwise_fp8_matmul` 重实现 `Fp8Linear.forward` 的
tensorwise 分支(scalar amax-scale 两操作数后 `torch._scaled_mm` bf16 累加)。
它正确地从核心 import 了 `FP8_E4M3_MAX` 常量,但抄了序列本身。

**优先级低**:它在一次性 perf 目录下,不进生产路径。若那个 perf probe 仍在用,让它改调
`nn/quantization/fp8.py` 的核心(prior-art 规则对依赖内部同样适用);若 probe 已答完问题,
按一次性生命周期直接连脚本删除。先确认 probe 是否还需要,再决定。

## 4. 验证

审计产出结构化 findings 后,同样过对抗性 verify(默认 KEEP,偏置控制不变)。任何"该改"落地前,
config-resolve 全 preset 冒烟(已有 `tests/config/test_load_all_experiments.py`)确认 entrypoint
字符串仍解析。

## 引用

- 调度证据:`vrl/config/presets/recipe/online/*.yaml` 的 `trainer.entrypoint`
- 共享 recipe(form-4 对照基准):`vrl/scripts/common/online.py:run_online_recipe`
- perf 重复:`vrl/scripts/perf/common/fp8_math.py` vs `vrl/nn/quantization/fp8.py:72,168`
