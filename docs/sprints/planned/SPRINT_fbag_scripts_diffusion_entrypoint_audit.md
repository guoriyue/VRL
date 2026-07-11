# SPRINT: 补审被错分的生产训练入口层（scripts/diffusion/）

状态：in progress（2026-07-10）。父：`SPRINT_fbag_00_overview.md`。已落地两个明确 finding；
7 个生产入口的完整逐函数审计仍未关闭。

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

其余入口仍须按 §2 完成 body-level 对照；这两个局部 finding 不代表全量审计已结束。

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
