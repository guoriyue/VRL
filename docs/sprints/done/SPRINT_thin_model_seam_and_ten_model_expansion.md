# SPRINT: 薄化模型 seam + 十个新模型扩张（diffusion + AR，含 GLM-Image）

状态：**done。Phase 0 收官（2026-07-07）；Phase 1 完成（2026-07-08，10/10 落地 + 真权重 rollout 验证）**。

> **Phase 1 验证记录（2026-07-08，RTX 5090 32GB + CPU 兜底）**：探针 = `vrl/scripts/diffusion/generate.py`
> （生产 `sde_step_with_logprob` 循环 + first-step replay parity）/ AR 家族用 executor 直驱。
> 10/10 全部 replay parity **0.0e+00**（AR 家族由 rollout↔teacher-forced 等值测试锁定）：
> SANA（bf16 尾数坏死线性注意力——二分定位后 from_spec 映射到 fp16，4a0c8e5e，修复后 SDE 直出杂志级）、Lumina-Image-2（摄影级输出）、PixArt-Σ（ddim 阶梯首战）、
> HunyuanImage-2.1（17B CPU 验证；1024px/20 步复跑画质干净——首跑斑块系 6 步/512px 探针省时设置）、HunyuanVideo（13B + tiled decode）、Mochi（倒 sigma 标准化实证）、
> CogVideoX（v-pred ddim + BFCHW）、Emu3（4163 受限 token 直出高质量图）、LlamaGen（vendored GPT 256 token）、
> GLM-Image（transformers 5.13 升级后落地；原生 1024px CPU 全程验证——1280 prior token 采样 +
> 20 步 DiT 解码，输出为全战役最佳画质的摄影级图像）。
> **栈变更**：transformers 4.57.6 → **5.13.0**（GLM-Image 硬依赖；两处兼容修复：cache_rows 的
> legacy-cache 适配、emu3 replay loader 的 shard-index 遍历；全套件 672 passed）。
性质：**架构瘦身重构（Phase 0）+ 模型覆盖扩张（Phase 1，10 个）**。
承接 [[SPRINT_model_family_coverage]]（覆盖度 index）与 [[SPRINT_physical_ai_model_support]]（优先级边界）。

> **Phase 0 终局对账（2026-07-07）**：11 轮薄化，净删 ~1500 行样板；descriptor 家族 5 个
> （sd3_5/qwen_image/predict2_5/wan_2_1/wan_2_1_i2v rollout）；diffusion runner 归零；AR 装配线归一；
> capability 全仓单一构造点；死键/死函数两轮全仓扫描零命中。**挂条件延期项**（非遗漏）：
> ① per-family train.py ×6（**薄化第十六轮再审+落地，2026-07-09**——纯转发的折叠，真差异的保留）
> ② executor-as-data（等 owner 决策）
> ③ AR chunk 模板化（**薄化第十五轮落地，2026-07-08**——触发条件"5 个 AR 家族"已满足）
> ④ echo compile/fp8 GPU 验证（80GB 卡）⑤ cosmos caps 契约统一（**薄化第十五轮落地，2026-07-08**）
> ⑥ SamplingState 基类（**薄化第十四轮落地，2026-07-08**）。~~下一步 = Phase 1 接 SANA~~——**已落地**
> （[[SPRINT_sana_t2i]] 已归档 done/；CPU 复验 2026-07-09：tests/models/diffusion/sana 4 passed +
> tests/config 168 passed）。Phase 1 全量收官见文首验证记录；挂条件项 ②④ 不属于本 sprint
> 的完成判据，需要时按各自触发条件另立工作。

## 0. 一句话

**问题不是"加模型难"，是"每个家族的 `runtime.py` 把同一套 build 编排抄了一遍"。**
先做 Phase 0 把这层重复折叠成一个由 registry 描述符驱动的通用 builder——之后"加一个模型"才真正逼近
vLLM 那种薄脚本（`model.py` 真代码 + registry 一条 + yaml 一份 + test 一个）。Phase 1 用这个薄 seam
接 10 个模型：4 个 T2I 扩散、3 个 T2V 扩散、3 个 AR（含用户点名的 **GLM-Image**）。

## 1. 根因盘点：重复在哪，薄层在哪（第一手证据）

### 1.1 已经薄的部分——registry（不要动）

`vrl/rollouts/families/registry.py` 已经是声明式的。加一个 diffusion 家族在 registry 侧只是一条：

```python
# vrl/rollouts/families/registry.py:130-146
register_rollout_family(
    _diffusion_entry(
        family="flux",
        task="t2i",
        aliases=("flux_1_dev",),
        executor_cls="vrl.models.diffusion.flux.runtime:FluxChunkExecutor",
        runtime_builder="vrl.models.diffusion.flux.runtime:build_flux_runtime_bundle",
        runtime_spec_extractor="vrl.models.diffusion.flux.runtime:extract_flux_runtime_spec",
        request_prefix="flux",
        default_task_type="text_to_image",
    ),
)
```

`_diffusion_entry(...)`（`registry.py:92-138`）已经把 collector / gatherer / capability / executor_kwargs 全部
默认好——这层是好的，**Phase 0 不碰**。

### 1.2 深重复的部分——每个家族的 `runtime.py` build 编排

真正的重复在这里。`vrl/models/diffusion/sd3_5/runtime.py` 里这五个函数，在**每个** diffusion 家族里
几乎逐字复制：

```python
# vrl/models/diffusion/sd3_5/runtime.py:46-175（节选骨架）
def extract_sd3_5_runtime_spec(cfg, device, weight_dtype):        # 只是 task_variant 不同
    return extract_runtime_spec(cfg, device, weight_dtype, task_variant="t2i")

def build_sd3_5_runtime_bundle(spec):                             # 通用编排：from_spec→lora/ft→
    model = SD3_5Model.from_spec(spec)                            #   quantize→compile→set_num_steps→
    if spec.use_lora: model.apply_lora(spec)                     #   组装 RuntimeBundle
    else: model.apply_full_finetune()
    apply_rollout_quantization(model, spec)
    ...  # compile / set_num_steps / RuntimeBundle(...)  ← 这 40 行每个家族都一样

def build_sd3_5_replay_runtime_bundle(spec): ...                 # 同一套编排的 replay 变体
def build_sd3_5_runtime_bundle_from_cfg(cfg, device, wd): ...    # 2 行包装
def build_sd3_5_replay_runtime_bundle_from_cfg(cfg, device, wd): ...  # 2 行包装
```

家族之间**真正不同的只有 5 个值**：`Model` 类、`ReplayModel` 类、`transformer` diffusers classname
字符串（如 `"SD3Transformer2DModel"`，见 `runtime.py:125`）、scheduler loader、`task_variant` + 两个
`runtime_caps` flag。其余全是复制。AR 侧同理：`ar/janus_pro/runtime.py:53-122` 的
`build_janus_pro_runtime_bundle` / `build_janus_pro_replay_runtime_bundle` / `extract_janus_pro_runtime_spec`
也是同一套编排换个类名。

### 1.3 结论：哪些该折叠、哪些是真薄层（保留）

| 层 | 现状 | Phase 0 处理 |
|---|---|---|
| registry `_diffusion_entry` / `RolloutFamilyEntry` | 已声明式、薄 | **保留** |
| `build_*_runtime_bundle` / `build_*_replay_runtime_bundle` | 每家族逐字复制 40+ 行 | **折叠**进共享通用 builder |
| `extract_*_runtime_spec` / 两个 `*_from_cfg` 包装 | 每家族 2~4 行样板 | **折叠**（由描述符提供 task_variant）|
| `*ChunkExecutor.build_chunk_encoded`（diffusion）| 家族特有（要 repeat 哪些 embed）| **保留**（真薄层）|
| `model.py`（`from_spec`/`encode_prompt`/`forward_step`/`decode_latents`）| 家族特有真代码 | **保留**（这才是模型本体）|
| AR `runner.py` / executor 的 `forward_plan`/`gather_chunks` | 家族特有（tokenize/decode/VQ）| **保留**（真薄层）|
| `vrl/scripts/.../<family>/train.py` | 每模型 ~59 行纯转发（如 `scripts/diffusion/sd3_5/train.py`）| **折叠**成 registry 驱动的通用 entrypoint |
| 每家族 `__init__.py` re-export 聚合器 | 样板 | **收薄**（只 re-export 真需要对外的符号）|

> 补充证据（架构盘点确认）：AR 侧**没有** `DiffusionModelBase` 的对应基类——每个 AR 模型是裸
> `nn.Module` 直接实现 Protocol，所以 AR 家族的每模型重复比 diffusion 更多。另：`vrl/models/vla/`
> 已死（只剩 `__pycache__`，无 `.py`、无 import），Phase 0 顺手删掉这些 stale 编译产物。
> 今天接一个模型的实际触碰面 ≈ **4 个 model 包文件 + registry 一条 + train.py 一份 + ≥1 yaml = 6~7 个文件**
> （对比 vLLM 单文件）。

> **判据**：一个函数若在家族之间只有类名/字符串不同 → 是样板，折叠；若含模型独有的张量语义
> （哪个 embed 要 repeat、怎么 tokenize、怎么 VQ decode）→ 是薄层，保留。

## 2. Phase 0：把 build 编排折叠成描述符驱动的通用 builder

**目标**：让 diffusion 家族的 `runtime.py` 从"5 个复制函数 + 1 个 executor"缩到"1 个 executor（只留
`build_chunk_encoded`）"，build 编排全部由 registry 描述符 + 一个共享 builder 承担。

> **进度（2026-07-01，loop tick 1）**：
> - ✅ 共享 builder 落地：`vrl/models/diffusion/build.py` 的 `build_diffusion_runtime_bundle` /
>   `build_diffusion_replay_runtime_bundle`（family-agnostic，不 import 任何家族类，无循环依赖）。
> - ✅ 参考家族 sd3_5 迁移完成：`sd3_5/runtime.py` 的 `build_*` 从 ~90 行编排缩成薄 stub，删掉 8 个只在
>   编排里用的 import。派发契约（`module:build_sd3_5_runtime_bundle(spec)` 字符串路径）不变。
> - ✅ 死代码 `vrl/models/vla/` 已删（§2.6）。
> - ✅ 验证：`tests/models` + `tests/rollouts` 309 passed；相关 wiring/precision/config/vae-memory/launcher
>   119 passed。两个 pin 测试（policy-source-scan、replay-namespace-patch）已改指向共享 `build.py` 并保留
>   全部行为断言。剩余失败（cosmos3/echo 缺 FAMILY_MODEL_CLASSES、rollout runtime AttributeError）经 stash
>   对比确认为**干净树预存**，与本次无关。
> **进度（2026-07-01，loop tick 2）**：
> - ✅ qwen_image 迁移完成（纯样板，直接套 sd3_5 stub）。
> - ✅ flux 迁移完成：flux 有两处家族特有逻辑（NFT `previous` adapter + 动态时间步），给共享 builder 加了
>   两个可选 hook `after_lora` / `after_construct`——flux 逻辑留在 flux stub 的闭包里，通用体不认识 NFT。
>   NFT 的 `attach_previous_adapter=True` 只从 `scripts/diffusion/flux/train.py` 传入，registry/GRPO 路径不变。
> - ✅ 扩测试覆盖：wiring 测试的 replay 参数化新增 flux + qwen_image（原本只有 sd3_5/wan/cosmos_predict2）。
> - ✅ 验证：wiring/vae-memory/config + flux/ + qwen_image/ **75 passed**；`tests/models`+`tests/rollouts`
>   **311 passed**（预存失败照旧 deselect）。
> - ⏭️ 待办（后续 tick）：video/anima 保留自建 replay builder（多 transformer/单文件，形状不同，可只复用
>   rollout 侧 `build_diffusion_runtime_bundle`）；§2.1 registry 描述符字段、§2.4 train.py 折叠、§2.5 `ARModelBase`。

> **进度（2026-07-01，loop tick 3）**：
> - ✅ §2.5 `ARModelBase` 落地：`vrl/models/ar/base.py`，上提 AR 两处 byte-identical 逻辑
>   （`load_trainable_state` 用 `type(self).__name__` 做 label；`disable_adapter` 走 `self.language_model`）。
>   `JanusProModel` / `NextStep1Model` 改继承 `ARModelBase`，删掉重复方法 + 各自 3 个转为未用的 import
>   （`contextlib` / `disable_adapter_on` / `load_weights_into`）。4 个类（含两个 Replay 子类）继承验证通过。
>   前瞻价值：Phase 1 的 3 个 AR 模型（GLM-Image/Emu3/LlamaGen）直接继承，不再抄这两段。
> - ✅ 验证：`tests/models/ar` + `tests/generation/ar` + config **61 passed**；`tests/models`+`tests/rollouts`
>   **311 passed**。两个预存 collection error（cosmos3/echo）确认与本次无关且未加重。
> - 📌 架构结论（本 tick 勘察确定，非待办）：**video/anima 的 rollout builder 是真发散不是重复**——wan 多变体
>   `_resolve_model_cls(t2v/i2v)` + 特有 metadata/caps；echo 不做 quantization/compile；anima 单文件 artifact
>   解析。硬套通用体要加一堆 skip flag 污染通用体，故**保留自建**，(a) 项无干净可做的部分，关闭。
> - 📌 §2.4 train.py 折叠**降级/暂缓**：train.py 经 `trainer.entrypoint` dotted path 被每个实验 yaml 引用，
>   且家族有 GRPO/NFT/DPO 多 recipe（如 flux 的 `build_flux_nft_*`）。折叠会改 config entrypoint 契约、
>   blast radius 覆盖所有实验 yaml，不是无人值守 tick 的干净单元——留给有人值守时做。
> - ⏭️ 剩余 Phase 0：§2.1 registry 描述符字段（把 model_cls/transformer_classname 等挪进 entry，让 stub
>   进一步变薄）、§2.5 的 `build_ar_runtime_bundle` 折叠（janus config 构造复杂、双 capability，需谨慎）。

> **薄化全量审计（2026-07-01，AST 扫描全部 runtime.py 的 build_/extract_ 函数）**：
>
> | 家族 | rollout / replay builder | 状态 |
> |---|---|---|
> | sd3_5 / qwen_image | 11 / 14 行 [DELEGATES] | ✅ 已薄 |
> | flux | 30 / 53 行 [DELEGATES]（多出的是 NFT/动态时间步闭包，真家族逻辑） | ✅ 已薄 |
> | wan_2_1 | 68 / 76 行自建 | ✅ 真发散（多 transformer 变体），保留 |
> | echo | 41 / 64 行自建 | ✅ 真发散（无 quant/compile、LTX wrapper replay），保留 |
> | anima | 63 / 57 行自建 | ✅ 真发散（单文件 artifact 解析），保留 |
> | cosmos3 | 32 / 30 行自建 | ✅ 真发散（`_apply_train_knobs`、无 compile），保留 |
> | **cosmos predict2 / predict2_5** | 59+48 / 48+49 行自建 | ⚠️ **~90% 同通用形状但不能盲折**：runtime_caps
> 无 `family_capability` 键（与 sd3_5/flux/qwen 不一致！）、各带家族 metadata（reference_image /
> model_revision / skip_text_encoder）。折叠须给共享 builder 加 caps/metadata 参数并决定 caps 契约
> 是否统一——**有人值守项**，且 caps 不一致本身值得单独审（消费方读不读 `family_capability`？）。 |
> | **AR janus_pro / nextstep_1** | 22+27 / 22+22 行自建 | ⚠️ **bundle 组装 ~18 行逐字重复**（差异只在
> capability 常量与 `supports_chunked_execution`；家族真逻辑是 config-from-spec 函数，那部分保留）。
> **建议在接 3 个 AR 新家族（GLM-Image/Emu3/LlamaGen）前折叠 `build_ar_runtime_bundle(spec, ...)`**，
> 否则每个新 AR 家族再抄一份。model.py 侧已由 ARModelBase 收薄（tick 3）。 |
> | `*_from_cfg` 包装（各家族 6-8 行×2） | — | ✅ 保留：它们是 train.py `trainer.entrypoint` 的派发目标，
> 折叠属于 §2.4（有人值守项）。 |

> **薄化第二轮（2026-07-01，有人值守）——AR 折叠落地 + cosmos 折叠落地 + 剩余家族逐一判定**：
>
> **已落地**（`build_ar_runtime_bundle` + 共享 builder 新增 `runtime_caps` 全量覆盖 / `extra_metadata(model, spec)` 两参）：
> - ✅ **AR janus_pro / nextstep_1**：4 个 builder → 9-15 行 stub（`vrl/models/ar/build.py`）。caps 差异
>   （R1 按 `ar_task` 选 capability）留在 janus stub。GLM-Image/Emu3/LlamaGen 落地时直接用。
> - ✅ **cosmos predict2（rollout+replay）**：历史 caps dict（无 `family_capability` 键）原样传入——
>   `capabilities.py:177` 只在键存在时覆盖、缺失回落 registry 声明，值相同 → **零行为变化**。
>   `reference_image` 走 `extra_metadata`。
> - ✅ **cosmos predict2_5（仅 rollout）**：非 LoRA 分支的 fail-loud 保留（共享 builder 调
>   `model.apply_full_finetune()`，Predict2.5 定义为 raise "NFT requires LoRA"）。
>
> **判定保留（各有具体根因，不是"懒得折"）**：
> - **predict2_5 replay（49L）**：两处真发散——UniPC 调度器（非 `load_flow_match_scheduler`）+ 非 LoRA
>   分支用 `model.apply_full_finetune()`（故意 raise）而通用 replay 用 `enable_transformer_full_finetune`
>   helper（SD3 系 replay model 的 `apply_full_finetune` 会碰 pipeline 而炸，所以通用不能改）。折叠要加
>   `scheduler_classname` + `full_finetune` 两个 knob 只服务一个家族 = knob 蔓延，保留。
> - **wan rollout（68L）**：**其实可折**（下一个有人值守单元）——`_resolve_model_cls(task_variant)` 在
>   stub 里先解析再传 `model_cls`，caps/家族名切换也在 stub 定，`boundary_ratio`/`trainable_transformers`
>   走 `extra_metadata(model, spec)`（model 依赖的 metadata 正是这个签名的设计原因）。replay（76L）
>   多 transformer 构造，保留。wan 是最高流量视频家族，折叠单独一个 commit + 全量 wan 测试。
> - **echo rollout（41L）**：机械可折且现有配置下零行为变化（echo 配置没设 compile/quant，通用 builder
>   的这两步是条件 no-op）。但**语义变化**：今天 echo 静默忽略这两个 knob（按本仓库 no-op-knob 规则这
>   本身是个 bug——用户设了没效果），折叠后会真的生效于 LTX wrapper——**需要 GPU 验证 echo+compile
>   后再折**。echo replay（64L）是 LTX wrapper 工厂，保留。
> - **anima rollout（63L）**：可折但要先把 artifact 路径解析（原地 mutate `spec.model_config`）挪进
>   `AnimaModel.from_spec`（本来就该在那），metadata 的三个 path 键走 `extra_metadata` 读已解析的
>   config。preview-grade 家族，ROI 低，可选。replay（57L）自建调度器 + `load_anima_transformer`，保留。
> - **cosmos3（32/30L）**：`_apply_train_knobs` 替换了整个 lora/full-finetune 分支且无 compile 步——
>   折叠等于用 hook 换掉 builder 心脏，不是折叠。probe-grade，保留。
>
> **薄化第三轮（2026-07-01）——wan/anima rollout 折叠落地，折叠收官**：
> - ✅ **wan rollout（68→34L）**：t2v/i2v 解析（model_cls/capability/caps flag）留在 stub 做前置，
>   规范化 task_variant 与 model 依赖的 metadata（boundary_ratio/trainable_transformers）走
>   `extra_metadata`（merge 在通用键之后，覆盖原始 task_variant）。
> - ✅ **anima rollout（63→35L）**：单文件 artifact 解析留在 stub 做前置（原地改 `spec.model_config`，
>   发生在 from_spec 之前——委托内部才调 from_spec，顺序保持）；`_resolve_artifact` 不挪
>   （replay 路径与 wiring 测试还在用它）。新增模块级 `ANIMA_FAMILY_CAPABILITY`，executor 类属性复用。
> - ❌ **echo rollout 撤出折叠名单（修正第二轮判断）**：echo 文件头 docstring 明确记录
>   "Quantization and torch.compile are intentionally not wired in Stage 1 ... add them once validated
>   on an 80GB card"——跳过是**有记录的刻意决策**，不是静默 no-op bug（且量化侧本有 worker 的
>   `assert_rollout_quantization_applied` 兜底 fail-loud）。折叠会在未验证前接线，违背该决策。
>   80GB 卡验证过 LTX+compile/fp8 后再折。
>
> **薄化第四轮（2026-07-01，用户拍板）——echo rollout + predict2_5 replay 折叠**：
> - ✅ **echo rollout（41→21L）**：用户决定接线、GPU 验证后置。共享 builder 的 quant/compile 对 echo
>   变为"配置开才生效"（现有配置全关 → 行为零变化）。⚠️ **两条路径仍未在 LTX transformer 上验证**——
>   module docstring 与 stub docstring 都标了"开 knob 前先在 80GB 卡跑 parity"。
> - ✅ **predict2_5 replay（49→35L）**：flux 模式统一。NFT 的 LoRA-only 守卫**前置到 stub**（比原来更好：
>   原来先加载完整个 transformer 才 raise，现在加载前就 fail，错误信息逐字保留；模型自己的
>   `apply_full_finetune` raise 仍在，护住其它调用面）。UniPC 用新的 `scheduler_classname` 参数
>   （与 `transformer_classname` 完全对称的数据参数）。wiring 测试的 loader patch 改指共享 build 模块。
>
> **折叠终态（22 个 build 函数，不含 from_cfg/chunk_encoded）**：**18 个委托，4 个保留**。
> 保留清单及行级根因：wan replay（76L，多 transformer 构造）、echo replay（64L，LTX wrapper 工厂 +
> 自建调度器）、anima replay（57L，自建 shift 调度器 + `load_anima_transformer`）、cosmos3 两侧
> （62L，`_apply_train_knobs` 替换 builder 核心分支）——这 4 个的"发散"都是**构造方式**不同（多件套/
> 非 diffusers 工厂/核心分支替换），不是参数差异，折叠必然把共享 builder 变成 hook 拼盘。
> 验证：`tests/models + tests/rollouts + tests/generation/ar + config` **356 passed**。
> ⏭️ echo 的 GPU 验证事项（开 compile/fp8 前）：80GB 卡真 rollout + `debug.first_step` logprob parity。

> **薄化第五轮（2026-07-01）——§2.1+§2.4 落地：registry 描述符彻底消灭 wrapper（qwen_image pilot）**：
> - ✅ 机制：`RolloutFamilyEntry` 新增 `build: DiffusionFamilyBuild`（model_cls/replay_cls/
>   transformer_classname/task_variant/memory_owner/scheduler_classname/runtime_caps，纯字符串数据）；
>   `vrl/models/diffusion/build.py` 新增 4 个 generic 函数（`extract_family_runtime_spec` 从
>   `cfg.model.family` 解析家族并盖到 spec 上；`build_family_runtime_bundle`/`_replay_` 用 `spec.family`
>   查描述符；2 个 from_cfg）；`RuntimeBuildSpec` 新增 `family` 字段（随 Ray payload 走，worker/launcher
>   **零改动**——contract 本就带 family，payload 归一化是白名单外透传）；新增通用 train entrypoint
>   `vrl/scripts/diffusion/train.py:train_diffusion_grpo`（sd3_5/qwen 的 train.py 逐字同构证明了它家族无关）。
> - ✅ pilot：qwen_image 端到端迁移——runtime.py 的 5 个 builder/extractor 函数**全删**（只剩 capability
>   常量 + executor），`scripts/diffusion/qwen_image/` 整目录删除，实验 yaml entrypoint 切到通用 recipe，
>   wiring 测试换 descriptor 专测（含 spec.family 缺失 fail-loud 断言）。
> - **新家族的落地面（当下起）**：`model.py`（真代码）+ `runner.py`（若需要）+ `runtime.py`
>   （capability + executor，~30 行）+ registry 一条带 `build=` 描述符 + yaml（entrypoint 用通用 recipe）
>   ——**零 builder 函数、零 train.py**。SANA 按此接。
> - ✅ **sd3_5 同法迁移完成（第二个 descriptor 家族）**：5 个函数删、`scripts/diffusion/sd3_5/` 目录删、
>   5 个实验 yaml entrypoint 切通用 recipe（fsdp yaml 经 defaults 继承自动跟随）、3 处测试改指 generic
>   （precision-bridge 用 `extract_family_runtime_spec`；wiring 的 descriptor 专测参数化覆盖
>   sd3_5+qwen_image；vae-memory 参数化加 spec_family 列）。全仓 `build_sd3_5_*`/`train_sd3_5_*` 零残留。
> - **descriptor 家族名册（终态）**：sd3_5、qwen_image（+ 未来所有 data-only 新家族，SANA 起）。
>   **其余家族为何不迁（都是"stub 里有代码"）**：flux（NFT hook 闭包）、wan（t2v/i2v 变体解析 +
>   model 依赖 metadata）、anima（artifact 路径解析前置）、predict2/predict2_5（extra_metadata lambda +
>   NFT 守卫）、echo（replay 是 LTX wrapper 工厂，descriptor 的 replay_cls 装不下）、cosmos3
>   （`_apply_train_knobs`）、AR janus/nextstep（config-from-spec 是真代码，且 janus 双 capability）。
>   描述符装数据不装代码——这条边界就是"能否零函数"的判据。
> - 验证：**439 passed**（迄今最宽集合：+ execution worker 测试）。

> **薄化第六轮（2026-07-02）——model.py 层：`DiffusersPipelineModelBase` 上提逐字重复成员**：
> - 方法级 AST 哈希审计发现 builder 层之下还有一层重复：7 个 pipeline-backed model 类
>   （sd3_5/flux/qwen/cosmos3/predict2/predict2_5/wan）的 `pipeline`/`device`/`scheduler`/`raw_handle`
>   逐字相同，`_set_transformer` 6/7 相同、`trainable_modules` 8/9、`apply_full_finetune` 语义等价
>   （`pipeline.transformer` ≡ `self.transformer` 同一对象）。
> - ✅ 新增 `base.py:DiffusersPipelineModelBase(DiffusionModelBase)`：`__init__(pipeline, device)` +
>   上述 8 个成员。`set_num_steps` 采用 **flux 的动态 shifting 版本**为基类实现（对静态调度器行为
>   等价、读 `self.scheduler` 兼容 replay）——全家族零 override。7 个类换基类、删重复成员：
>   **净 -172 行**（+121/-293）。
> - 保留的真 override（各 1 处）：sd3 `_set_transformer`（attention processor 重装）、wan
>   `trainable_modules`/`apply_lora`/`load_trainable_state` 等（多 transformer）、predict2_5
>   `apply_full_finetune`（NFT raise）。`_lora_dtype` 的 3 份相同 override 保留——`LoraModelMixin`
>   在 MRO 中先于新基类，挪进基类会被 mixin 默认遮住，改 mixin 会改 cosmos 系行为。
> - echo/anima 不采用（非 pipeline-backed：LTX wrapper / 单文件 checkpoint），留在 `DiffusionModelBase`。
> - 新家族增益：SANA 等 model.py 不再写这 8 个成员，只写 from_spec + 四个生成抽象方法 + replay 投影。
> - 验证：**439 passed** + 8 类继承/override 断言抽查。

> **薄化第七轮（2026-07-02）——runner/executor 层：no-op 骨架与 base 等价 override 清除**：
> - **executor 侧关键发现**：`DiffusionChunkExecutorBase.build_chunk_encoded` 的默认实现
>   （`repeat_encoded_batch`：非 tensor/None/已对齐 batch 自动 passthrough）**已经与 sd3_5 / wan-T2V /
>   wan-I2V / predict2_5 / echo 的 override 行为等价**——5 个 override 是纯冗余，全删。flux 的唯一差异
>   （`text_ids` [seq,3] 无 batch 维不能 repeat）改为**数据**：base 新增 `chunk_passthrough_keys`
>   类属性，flux 只写 `("text_ids",)` 一行。保留的 2 个 override 各有真逻辑：predict2（从
>   generation_request 兜底 reference_image）、cosmos3（batch=1 全 passthrough 语义）。
> - **runner 侧**：`common/backbone.py` 新增 `DiffusionBackboneRunnerBase`（no-op `postprocess_branch`
>   / `finalize_noise_pred`），5 个 runner 类继承并删掉逐字相同的 no-op（sd3/flux/wan-T2V/wan-I2V 各删
>   2 个、qwen 删 1 个）。runner 剩下的全是真差异：`build_branch` 的 kwargs 映射（各家 transformer
>   签名不同）+ qwen 的范数保持 CFG 数学。
> - 净 **-216 行**（+67/-283）。验证：**537 passed**（首次含全量 `tests/generation`）。
> - 新家族增益：SANA 的 executor 大概率零方法（qwen 同款，纯类属性）；runner 只写 `build_branch`。

> **薄化第八轮（2026-07-02）——predict2_5 升级 descriptor 家族（第三个）+ 死键清链**：
> - **死键判定**：`model_revision` / `skip_text_encoder` 两个 bundle-metadata 键全仓库零读取方
>   （dead-field 规则）→ 两个 `_from_spec` helper + 两个 extra_metadata lambda + cosmos3 的同款死键
>   整链删除。死 cap `supports_diffusion_nft`（此前已判死）一并删。
> - **删完发现 predict2_5 只剩一处"代码"**：NFT LoRA 守卫 → 数据化为描述符新字段
>   `requires_lora_reason: str | None`（generic builder 在加载 transformer 前 fail-loud）。
>   于是 predict2_5 完整升级 descriptor：runtime.py 只剩 capability + executor，6 个函数全删。
> - **train.py 的 recipe 保留**（真 recipe 差异：NFT 不传 `reference_model_getter`——用 previous adapter
>   替代参考模型），但其 `_build_predict25_*` 转发改指 generic from_cfg。eval 脚本
>   （`cosmos_predict25_kling_eval.py`）改指 generic extract/build。
> - **descriptor 家族名册更新**：sd3_5、qwen_image、**cosmos-predict2.5**。predict2 也已接近
>   （死键删完后只剩 reference_image 兜底 lambda——它读 generation_request，是 executor 层真逻辑）。
> - 验证：**577 passed**（+ 全量 tests/scripts）。

> **薄化第九轮（2026-07-02，用户拍板）——runner 并入 model.py，runner.py 文件归零**：
> - **无状态四家（sd3_5/qwen/flux/wan）走激进版**：模型类自己实现 backbone-runner 协议
>   （bases += `DiffusionBackboneRunnerBase`，`cfg_mode`/`cfg_base`/`build_branch` 上类，qwen 保留
>   范数 CFG `finalize_noise_pred`），`forward_step` 传 `self`——runner **类和文件都消失**。
>   "怎么调我的 transformer"回归模型知识。wan I2V 在子类覆写 `build_branch`，replay 类经继承链自动获得。
>   sd3 的 attention-processor 安装 helper 随迁 model.py。
> - **cosmos predict2/predict2_5 走保守版**：它们的 runner 是**带每步状态的真策略对象**
>   （构造参数是当步 sigma，EDM 预条件/velocity 数学在 postprocess/finalize 里）——类保留、整体搬进
>   model.py，文件消失。
> - 终态：`vrl/models/diffusion/**/runner.py` **0 个文件**；净 -109 行（442+/551-）。
>   家族文件形状收敛为：**model.py（模型全部知识）+ runtime.py（引擎参数表）**。
> - capability 常量去重同轮完成（前一 commit）：descriptor 三家的 runtime.py 常量删除，
>   worker 从 launch contract 注入（registry 是唯一构造点）。
> - 验证：**577 passed**（backbone parity 测试真实走过 model-as-runner 路径）。
> **遗留审计项**：predict2_5 写入 caps 的 `supports_diffusion_nft` **全仓库无读取方**（dead cap，
> 按 dead-field 规则应删或补上本该存在的校验）；cosmos/wan/echo/anima 系 caps 无 `family_capability`
> 键与 sd3/flux/qwen 不一致（行为无差——`capabilities.py:177` 缺键回落 registry 声明——但契约分裂）。

### 2.1 扩展 registry 描述符

给 `RolloutFamilyEntry`（或 `_diffusion_entry`）补上现在藏在各家族 `build_*` 里的 5 个值：

```python
# vrl/rollouts/families/registry.py —— 在 _diffusion_entry 增加：
#   model_cls:            "vrl.models.diffusion.<f>.model:<F>Model"
#   replay_cls:           "vrl.models.diffusion.<f>.model:<F>ReplayModel"
#   transformer_classname:"SD3Transformer2DModel"      # diffusers 类名
#   task_variant:         "t2i"                          # 供 extract_runtime_spec
#   runtime_caps_extra:   {"supports_reference_conditioning": False}
```

### 2.2 新增一个共享 builder（唯一一份编排）

在 `vrl/models/loader.py`（或新文件 `vrl/models/diffusion/build.py`）放一份通用编排，签名吃描述符：

```python
def build_diffusion_runtime_bundle(spec, entry) -> RuntimeBundle:
    model = _import(entry.model_cls).from_spec(spec)
    (model.apply_lora if spec.use_lora else model.apply_full_finetune)(spec)
    apply_rollout_quantization(model, spec)
    if (c := spec.torch_compile or {}).get("enable"):
        model.torch_compile_transformer(c["mode"])
    if spec.num_steps is not None:
        model.set_num_steps(spec.num_steps)
    return RuntimeBundle(model=model, ..., runtime_caps={
        "family_capability": entry.capability.to_dict(), **entry.runtime_caps_extra})

def build_diffusion_replay_runtime_bundle(spec, entry) -> RuntimeBundle: ...  # replay 变体，同理
```

launcher 从 registry entry 直接调这两个通用 builder，不再按家族名 import `build_<f>_runtime_bundle`。

### 2.3 迁移每个现有 diffusion 家族

对 sd3_5 / flux / qwen_image / wan_2_1 / echo / cosmos*（共 ~9 个）：删掉各自的 `build_*_runtime_bundle` /
`build_*_replay_runtime_bundle` / `extract_*_runtime_spec` / 两个 `*_from_cfg`，把那 5 个值挪进 registry
描述符。`runtime.py` 只留 `*ChunkExecutor`。逐家族迁移、逐家族跑 `tests/generation/diffusion/`。

### 2.4 折叠 per-model `train.py` entrypoint

现状：每个家族有一份 `vrl/scripts/<seam>/<family>/train.py`（如 `scripts/diffusion/sd3_5/train.py` 59 行），
只是把 family 名 + `*_from_cfg` builder 转发给 `run_online_recipe(...)`。Phase 0 后 builder 已由 registry
描述符统一，`train.py` 也应折叠成**一个** registry 驱动的通用 entrypoint（`python -m vrl.scripts.train
--family sana ...`），删掉每模型的转发文件。这样"接一个模型"不再需要写 train 脚本。

### 2.5 AR 侧：补一个 `ARModelBase` + 折叠 build 编排

架构盘点确认 AR **没有** diffusion 那样的共享基类。Phase 0 给 AR 加一个 `vrl/models/ar/base.py:ARModelBase`
（把 janus_pro / nextstep_1 里重复的 replay/adapter/versioned-slot 逻辑上提，对齐 `DiffusionModelBase`），
并把 `build_*_runtime_bundle` / `extract_*_runtime_spec` 折叠成一个 `build_ar_runtime_bundle(spec, entry)`。
AR 的 executor（`forward_plan`/`gather_chunks`）含 tokenize/decode/VQ，是真薄层，保留。

### 2.6 清理死代码

删掉 `vrl/models/vla/`（只剩 `__pycache__`，无 `.py` 源、无 import 引用）——stale 编译产物，属 Phase 0
顺手清理范围（同源同生命周期）。

### 2.7 Phase 0 完成判据

- `git grep -c 'def build_.*_runtime_bundle' vrl/models/` 从 ~11 家族×2 降到 **2 个共享函数**（+ AR 1 个）。
- `vrl/scripts/**/train.py` 每模型转发文件删除，替换为 1 个 registry 驱动 entrypoint。
- `vrl/models/vla/` 删除，`ar/base.py:ARModelBase` 落地并被 janus_pro / nextstep_1 继承。
- 全量 `pytest tests/generation tests/config tests/rollouts` 绿。
- `tests/config/test_load_all_experiments.py` 绿（所有现有 experiment 仍能 resolve）。
- 随机抽一个家族做 1-step 生成 smoke（如 `tests/generation/diffusion/test_diffusion_metrics.py`）不回归。

## 3. Phase 1：十个新模型清单（index——每个模型已拆为独立落地 sprint）

**2026-07-01 起本节降级为 index**：每个模型有自己的落地 sprint（按仓库惯例，照 [[SPRINT_flux_t2i]] /
[[SPRINT_qwen_image_t2i]] 的模式），模型事实 / 技术点 / KILL-RISK 门 / 验收全在各自 sprint 里，
本表只留 checkbox 与一句话定位。**接入建议顺序**（依赖与风险排序）：
SANA → Lumina2 → Emu3 → HunyuanVideo → Mochi-1 → GLM-Image → HunyuanImage-2.1 → LlamaGen →
（PixArt-Σ / CogVideoX 需先过 DDIM logprob 门，见各自 sprint §0）。

> ⚠️ Evidence-first：HF repo id 与 diffusers 类名是**预期**，落地第一步必须核对真实类名——核对不过就
> 记录并降级/换模型，不硬接。SANA 已核对（2026-07-01）；PixArt-Σ / CogVideoX 已确认**非 flow-matching**，
> 带 KILL-RISK 门。

### 3.1 Tier A — T2I 扩散（纯 diffusion seam，最省）

- [x] **1. SANA**（2026-07-08 GPU 验证：探针 rollout + replay parity 0 误差 + 与 SanaPipeline 同 seed 视觉一致） → [[SPRINT_sana_t2i]] — 1.6B linear-DiT + DC-AE，**已核对 seam 吻合**，
  Phase 0 之后第一个薄接样例（模板 qwen_image）。
- [x] **2. PixArt-Σ**（code-landed c06dd987；DDIM 门已开 db97675c；GPU 探针待权重） → [[SPRINT_pixart_sigma_t2i]] — ⚠️ **epsilon-prediction 非 flow-matching**，
  须先过 DDIM logprob 门（门 A 扩展 / 门 B 换模型）。
- [x] **3. Lumina-Image 2.0**（code-landed eb9e7e25；GPU 探针待权重） → [[SPRINT_lumina_image_2_t2i]] — 2.6B flow-matching + Gemma-2，
  与 SANA 同形状类，排 SANA 之后增量最小。
- [x] **4. HunyuanImage-2.1**（GPU 验证 2026-07-08：17B CPU 探针 parity 0） → [[SPRINT_hunyuan_image_2_1_t2i]] — ~17B 双编码器（MLLM+byT5），
  LoRA-only，T2I 侧最重，diffusers 支持须先验证。

### 3.2 Tier B — T2V 视频扩散（套 Wan/Cosmos 5D 潜变量 seam）

- [x] **5. HunyuanVideo**（code-landed 051b5f83；GPU 探针待权重） → [[SPRINT_hunyuan_video_t2v]] — 13B flow-matching，embedded-guidance
  单分支（runner 照 flux）× 5D 布局（照 wan）。
- [x] **6. CogVideoX**（code-landed f2616506，v-pred 走 sde_type=ddim；GPU 探针待权重） → [[SPRINT_cogvideox_t2v]] — ⚠️ **v-prediction 非 flow-matching**，与 PixArt-Σ
  共享 DDIM logprob 门（另加 v-pred 分支）。GLM 系 T2V。
- [x] **7. Mochi-1**（code-landed 70d18b3c，倒 sigma 标准化；GPU 探针待权重） → [[SPRINT_mochi_1_t2v]] — 10B AsymmDiT 真 flow-matching + 真 CFG，
  T2V 里调度器最干净，预期 rollout/replay 都能委托共享 builder。

### 3.3 Tier C — AR（自回归，套 janus_pro/nextstep_1 seam + ARModelBase）

- [x] **8. GLM-Image** ⭐ → [[SPRINT_glm_image_ar_t2i]] — **用户点名**。9B AR + 7B frozen diffusion
  decoder，只训 AR 段（LoRA）；两个 KILL-RISK 门（语义 token logprob 可达性、decoder 后处理定位）。
  建议 Emu3 之后接。
- [x] **9. Emu3** → [[SPRINT_emu3_ar_t2i]] — 8B 纯 next-token（transformers 原生类），GLM-Image 的
  前置压力测试；技术点是约束解码要挂进 RL 采样。
- [x] **10. LlamaGen** → [[SPRINT_llamagen_ar_t2i]] — 学术基线，无 HF 原生类（零-vendor 优先：
  LLaMA 架构尝试 `LlamaForCausalLM` 复现）；优先级最低，vendor 成本超标可降级关闭。

### 3.4 覆盖平衡校验

| 类型 | 数量 | 编号 |
|---|---|---|
| T2I 扩散 | 4 | 1–4 |
| T2V 扩散 | 3 | 5–7 |
| AR（含 GLM-Image） | 3 | 8–10 |
| **合计** | **10** | |

## 4. 接一个模型的通用形状（Phase 0 之后 = 真薄）

**Diffusion 家族**（例：SANA）需要新增/改动：

```text
vrl/models/diffusion/sana/model.py      # 真代码：from_spec / encode_prompt / prepare_sampling /
                                        #   forward_step / decode_latents + SanaReplayModel
vrl/models/diffusion/sana/runtime.py    # 只留 SanaChunkExecutor（build_chunk_encoded：repeat 哪些 embed）
vrl/models/diffusion/sana/__init__.py
vrl/rollouts/families/registry.py       # +1 条 _diffusion_entry(..., model_cls=..., transformer_classname=...)
configs/model/diffusion/sana/base.yaml  # 照 sd3_5/medium.yaml
tests/generation/diffusion/test_sana_*  # 1-step forward + config resolve
```

**AR 家族**（例：Emu3）额外多一个 `runner.py`（decode-step 原语），executor 的
`forward_plan`/`forward_chunk_plan`/`gather_chunks` 是家族特有——参照 `ar/janus_pro/`。

**不碰**的共享层（家族无关）：`common/*`、`flow_matching.py`、`loader.py` 的通用部分、算法/rollout 层。

## 5. 每个模型的完成判据（loop 每一轮的 DoD）

一轮（一个模型）算完成，必须全绿：
1. diffusers/transformers 真实类名已核对并写进 model.py（不是猜的）。
2. `pytest tests/generation/<seam>/test_<family>_*` 绿：至少一个 1-step forward 契约测试。
3. `tests/config/test_load_all_experiments.py` 绿：新 yaml 能 resolve。
4. registry round-trip：`FAMILY_REGISTRY["<family>"]` 能 build 出 runtime bundle（可加一个 `test_registry` 断言）。
5. 勾掉 §3 对应 checkbox，`git commit`（英文 message，如 `models: add SANA T2I diffusion family`）。

## 6. 怎么搭 hourly loop workflow（教学）

### 6.1 机制：会话内 /loop（ScheduleWakeup）

采用**会话内自节奏循环**（不是云端 trigger）。原因：本仓库在本地 `/home/mingfeiguo/...`，改动必须落到
本地代码；`ScheduleWakeup` 每小时把**同一对话**唤醒一次，上下文（本 sprint、已接模型）自然累积，改动直接
落本地。代价：**只在会话/终端存活时有效**，关掉终端循环即停（重开后说"继续 model loop"即可恢复）。

### 6.2 循环怎么跑

每次唤醒（间隔 3600s ≈ 1 小时）执行同一个 loop prompt：

> 打开 `docs/sprints/planned/SPRINT_thin_model_seam_and_ten_model_expansion.md`。若 Phase 0（§2）未完成，
> 先推进 Phase 0，不接新模型。否则从 §3 找第一个未勾选 `[ ]` 的模型，按 §4 薄接、按 §5 判据验证、勾选、
> commit。一次只接一个；完成或遇硬阻塞就停，等下一跳。十个全勾完则报告并结束循环。

### 6.3 你要会的操作

```text
暂停循环：  直接说"停掉 model loop" —— 我下一跳不再 ScheduleWakeup，循环自然结束
改频率：    说"改成每 3 小时" —— 我把 delaySeconds 调大（上限 3600s/1h，更长要外部 cron）
立刻先跑一轮：说"现在先接一个" —— 我立即执行一轮，不等下一跳
恢复：      重开终端后说"继续 model loop"
```

> 注意：`ScheduleWakeup` 上限是 1 小时（runtime clamp [60,3600]）。若要比 1 小时更稀疏的节奏、或要在
> 关终端后仍跑，改用本地 crontab（`0 */3 * * * cd <repo> && claude -p "<loop prompt>"`）——那是 §6.1
> 里被否掉的备选，机器长期开机时可切过去。

### 6.3.1 Phase 1 与 loop 的边界（2026-07-01，tick 4 记录）

**Phase 0（纯重构）适合无人值守 hourly loop；Phase 1（接真模型）不适合。** 原因：

- Phase 0 的验证是 fast 单元测试（wiring/config/registry），无人值守能自证正确——已完成 sd3_5/flux/
  qwen_image 迁移 + ARModelBase + 删 vla，全程 tests 绿。
- Phase 1 每个模型的核心是**生成/replay 的 flow-matching / AR logprob 数学**（见 sd3_5/model.py 的
  `prepare_sampling`/`forward_step`/`export_replay_tensors`/`restore_eval_state`，其中 `debug.first_step`
  断言 rollout logprob == replay logprob）。这类正确性**只能用真权重跑一次生成做 parity 验证**；wiring/config
  测试用 fake tiny transformer，只查 bundle 结构，**不验证生成数学**。
- 因此在"无真权重 + 无 GPU 生成 parity"的无人值守环境里写 model.py 让 wiring 变绿就勾 checkbox，是**假信号**
  ——模型并未被验证能生成/训练。按仓库 Evidence-First 与"验证后才报完成"，不这么做。

**结论**：loop 的自主安全工作（Phase 0）已基本穷尽。Phase 1 需要三者之一：(a) 提供 SANA 等模型的真权重 +
GPU，让某一跳能跑生成 parity；(b) 转为**有人值守**逐个接（我写 model.py、你在 GPU 上跑 parity 确认）；
(c) 把 loop 重定向到剩余的可选 Phase 0 项（§2.1 需先给 `RuntimeBuildSpec` 加 `family` 字段并过 Ray
序列化，或 `build_ar_runtime_bundle` 折叠——都是更大、非纯净的单元）。tick 4 未提交任何代码，停在此决策点。

### 6.4 一个诚实提醒

**提醒频率 ≠ 完成速度**。真正接一个模型（尤其 Tier B/C、以及整个 Phase 0 重构）通常不止一小时。
每小时唤醒的作用是**节奏器**：把我拉回 checklist 接着做，不保证每小时产出一个可用家族。Phase 0 期间一跳
可能只推进一部分——正常。

## 7. 非目标

- **不做 omni / 统一理解+生成需要新 rollout+logprob seam 的模型**（BAGEL、Qwen3-Omni、HunyuanImage-3.0）——
  见 [[SPRINT_model_family_coverage]] §5，各自独立大 sprint。GLM-Image 之所以入选，是因为它的 AR 段能套
  现有 `nextstep_1` seam，而非要求全新 seam。
- **不做 VLA / 动作策略**（OpenVLA/PI0.5）——不同 RL 范式，见 [[SPRINT_physical_ai_model_support]]。
- **Phase 0 不动 registry 的 `_diffusion_entry` 声明式结构、不动 `common/*` 与算法层**——只折叠 build 编排。
- 不追 flux / qwen_image（已实现，见 `vrl/models/diffusion/{flux,qwen_image}/`）。

> **薄化第十一轮（2026-07-02）——wan per-entry descriptor + 全模型死键清扫**：
> - **wan 变体机器蒸发**：registry 本就有 `wan_2_1`/`wan_2_1_i2v` 两条 entry 却共用一个 builder 在运行时
>   重推导变体（registry 已知的信息被二次查询——与 capability 双份构造同病）。两条 entry 各带
>   `DiffusionFamilyBuild`（model_cls/task_variant 按 entry 固定）后，`_MODEL_BY_TASK`/`_resolve_model_cls`/
>   `_task_variant_from_cfg`/extract 包装/rollout stub 全删；replay 保留自建（多 transformer），
>   `DiffusionFamilyBuild.replay_cls` 改 optional（None = 家族自管 replay，generic fail-loud）。
>   配置里冗余的 `task_variant: i2v` 死 knob 一并删（i2v 全部经 family 选择）。
> - **全模型 metadata 死键审计**（精确 -F grep 每键读者）：`ar_task`/`dtype`/`model_path`/`task_variant`/
>   `use_lora` 零读者、`bundle.metadata` 无整体消费——generic 五键按 provenance-only 规则**标注保留**
>   （build.py 单点注释）；家族特有死键**删除**：wan 的 boundary_ratio（读者仅测试断言，改断言
>   `model.boundary_ratio` 行为面）/trainable_transformers/reference_image、anima 的三个 path 键、
>   predict2 的 reference_image（读者全是 request.metadata，另一对象）+ 死 helper `_reference_image_from_spec`。
> - **死函数扫描（vrl/models 全模块级函数 × 仓外调用方）：零命中**——前几轮已清完。
> - 验证：**623 passed**。descriptor 家族名册：sd3_5、qwen_image、cosmos-predict2.5、**wan_2_1、
>   wan_2_1_i2v（rollout 侧）**。

> **薄化第十二轮（2026-07-07）——flux 升级第六个 descriptor 家族；builder hook 机制退役**：
> - **判决修正**：flux 的两个 hook 装的都是模型知识（runner 合并同款论证）——NFT `previous` adapter
>   搬进 `FluxModel.apply_lora`（配置驱动：`model.nft_previous_adapter: true`，from_spec/prepare_replay
>   双守卫拦全参）；动态时间步搬进 `FluxReplayModel.prepare_replay`（`DiffusionModelBase` 新增 no-op
>   协议，generic replay builder 构造后统一调用）。
> - **hook 参数退役**：`after_lora`/`after_construct` 随唯一用户消失，从共享 builder 签名删除。
> - flux runtime.py 173→41 行（仅 executor）；train_flux_grpo 删除（== generic recipe，4 个 GRPO yaml
>   切 `train_diffusion_grpo`）；NFT recipe 保留（真差异：无 reference model）但 build 转发走 generic，
>   其实验 yaml 增加 `nft_previous_adapter: true`。
> - descriptor 名册：sd3_5、qwen_image、predict2_5、wan×2、**flux**。验证：**623 passed**。

> **薄化第十三轮（2026-07-07，"全部可疑再审"）——四家 rollout descriptor 化 + 两处误判翻案**：
> - **cosmos3 翻案**：`_apply_train_knobs` 的函数体就是 generic 的 lora/full-finetune+compile 序列手抄本
>   （此前只看调用点就判"替换核心分支"，没读函数体——误判）。且其顺序 compile 后才 quantize，与
>   "quantize 必须在 compile 前让 inductor 看到 fp8 模块"相反，潜伏 bug 随迁移 generic 修正。
> - **echo/anima recipe 翻案**：`weight_dtype_getter` 传的就是 `run_online_recipe` 的默认分支——全仓
>   该参数皆冗余（已从 generic 与 flux NFT recipe 删除）；echo/anima recipe 因此与 generic 逐字等价，
>   echo train.py 整目录删除、train_anima_grpo 删除，3 个 yaml 切 generic entrypoint。
> - **四家升级**：predict2（全 descriptor，含 replay）、echo/cosmos3/anima（rollout descriptor，
>   replay 自建保留：LTX 工厂 / pipeline-shell 复用 / 自建调度器）。anima 的 artifact 解析搬进
>   `AnimaModel.from_spec`（模型知识回家）；其 replay extract 因 e2e 字符串契约 + 自辩注释保留，
>   body 改走 generic。
> - **janus R1 wan 病修复**：recipe 在 extract 后手动突变 `spec.ar_task` ——改为 extract 按
>   `cfg.model.family` 派生（registry 早有 `janus_pro_r1` entry；R1 配置须声明 family）。
> - **descriptor 名册终态：10/13**（sd3_5、qwen_image、flux、predict2、predict2_5、wan×2、echo、
>   cosmos3、anima——后三家 rollout 侧）。runtime.py 里剩余全部函数：4 个自建 replay（真构造发散）
>   + 2 个 AR extract/config（真家族逻辑）+ anima replay extract（命名契约）。
> - 验证：**623 passed**。净 -296 行。

> **薄化第十四轮（2026-07-08）——终局对账延期项⑥落地：`DiffusionSamplingStateBase`**：
> - 17 个 `*SamplingState` dataclass 的字段审计：`latents`/`timesteps`/`scheduler`/`guidance_scale`
>   四字段 17/17 逐字相同；前三个正是 chunk executor 唯一会碰的 engine 契约（echo docstring 早已
>   口头声明该契约，现在有了类型载体）。`prompt_embeds`(15/17)/`do_cfg`(14/17) **不上提**——
>   进基类会给 flux/echo/hunyuan_video/cosmos3 造死字段（dead-field 规则）。
> - 新基类落在既有 `base.py`（不开新文件），全 17 类挂基类、删重复声明；echo/cosmos3 的字段行内
>   布局注释迁入各自 docstring。构造点全 kwargs，字段重排零行为差。
> - 验证：**1549 passed**（15 处失败为环境缺依赖的历史红测：ltx_core/glm_image/paddle-OCR，
>   main 基线 worktree 同一集合同红）。净 -23 行（+70/-93）。

> **薄化第十五轮（2026-07-08）——终局对账延期项③⑤落地：AR chunk 模板 + caps 单一派生**：
> - **③ AR chunk 模板化**（触发条件满足：AR 家族 ×6）：`ARDiscreteChunkExecutorBase` 模板
>   （validate→seed→prefill→ARDecodeLoop→VQ decode→token mask→result 骨架收进基类）+ 共享
>   `ARDiscreteChunkResult`（四家逐字相同的 13 字段收敛为一个，另携 `prefill_forwards` 遥测）+
>   单一 `ARDiscreteChunkGatherer`（镜像 diffusion 侧全家族共用 `DiffusionChunkGatherer` 的
>   registry 形状），全部落在既有 `vrl/generation/ar/executor.py`。janus/emu3/glm_image/llamagen
>   各留一个直线式 `prepare_chunk_inputs` hook（emu3 另 override `chunk_token_mask` 处理强制
>   结构位）；**nextstep_1（连续 token 3 元组 finalized）与 R1（generate_with_refine 反转控制流）
>   按第十一轮设计裁决留在模板外**。registry 四家 gatherer 指向共享类；净 **-317 行**。
> - **⑤ cosmos caps 契约统一——裁决为"停止发布"而非"统一发布"**（owner 复核后二改：第一版
>   `diffusion_runtime_caps()` 统一派生被否——派生自 registry capability 的 caps 与 worker 经
>   launch contract 拿到的是**同一个对象**，`declared.with_runtime_caps(derive(declared)) ==
>   declared` 恒等，统一地发布同义反复仍是赘肉）。终版：diffusion 双 builder、wan/anima/cosmos3
>   手写 replay builder、AR `build_ar_runtime_bundle` **全部不再发布 runtime_caps**（AR 的
>   `supports_chunked_execution: not replay` 也是假动态——replay 侧无读者，rollout 侧恒等于
>   registry 声明 True）；`DiffusionFamilyBuild.runtime_caps` 字段（6 份手抄 dict）删除。
>   **三改（owner 拍板）：整条 runtime_caps 通道删除**——`RuntimeBundle.runtime_caps` 字段、
>   worker 的 caps 拷贝、`FamilyCapability.with_runtime_caps` 合并全拆；capability 的唯一存储 =
>   registry entry，经 launch contract 到 worker，executor 自声明常量只做 fail-loud 交叉检查
>   （`_check_executor_capability`，原 `_merge_loaded_capability` 语义收窄后改名）。"有消费者、
>   无生产者"的 seam 不值得为假想的未来动态 flag 保留——真需求（如 chunk probe 回填
>   `default_max_samples_per_chunk`）出现时再建通道。顺手清掉 wan runtime `__all__` 三个指向
>   已删函数的死条目（F822）。
> - 验证：tests/{models,generation,rollouts,config,architecture,scripts,trainers} 全绿；改动文件
>   ruff clean。

> **薄化第十六轮（2026-07-09）——终局对账延期项①落地：per-family train.py 再审**：
> - **逐份裁决（引用到行）**：janus_pro/train.py（85L）与 nextstep_1/train.py（52L）是**纯转发**——
>   janus 的 family= 闭包甚至 `del family`（死仪式），r1 的 ar_task 本就由 cfg.model.family 在
>   extractor 内派生；两份删除，折叠进新的家族无关入口 `vrl/scripts/ar/train.py:train_ar_grpo`
>   （镜像 diffusion 侧 `train_diffusion_grpo` 的形状；builder 从 registry entry 的
>   `runtime_builder`/`runtime_spec_extractor` + 新增 `replay_runtime_builder` 字段解析——
>   与 Ray worker 走同一批 import 字符串）。5 个 AR 实验 yaml 改指通用入口。
> - **顺带解锁**：emu3/glm_image/llamagen 三个 Phase 1 新家族此前**没有任何训练入口**（只落了
>   rollout 侧）；registry 声明 replay builder 后它们经 train_ar_grpo 直接可训（契约测试锁定
>   6 个 AR 家族的三条 import 字符串全部可解析）。
> - **保留的（真差异，不折叠）**：flux/train.py = NFT 专用（无 reference_model_getter，KL 参照走
>   disable_adapter）；wan_2_1/train.py = i2v 的 collector_kwargs_getter（参考图解析）+ 手写
>   replay builder（多 transformer，generic replay 路径对它 fail-loud）；cosmos/train.py =
>   predict2 V2W 参考图接线 + predict2.5 GRPO/NFT 双入口；wan train_dpo.py = 离线 DPO 另一套
>   循环。这些是 recipe 层真代码，不是样板。

### 事实基础（全部引用到行级）

1. **基类失衡**：`DiffusionChunkExecutorBase` 715 行共享机器 vs `ARChunkExecutorBase`（`vrl/generation/ar/executor.py`）
   仅 100 行（layout / `_ar_runner` 接线 / `_align_tokenizer_output` / engine 守卫）。AR 家族因此各自手搓
   ~350 行编排。
2. **决定性事实：`forward_plan` 无生产调用方**。全仓 `.forward_plan(` 只有
   `tests/models/ar/janus_pro/test_r1_model.py:389` 和 `tests/e2e/test_real_checkpoint_rl.py:474`。
   生产路径（`worker.py:391` 起）只走 `forward_chunk_plan` 逐 chunk 派发（AR 无 pipelined 路径）。
   diffusion 基类的 `forward_plan` 就是三行：`run_sample_chunks_with_oom_retry(chunks, forward_chunk_plan)
   → gather_chunks`——家族不写。
3. **双装配漂移**：AR 每家族的 `forward_plan`（janus 140L / nextstep 136L / R1 90L，合计 366L）各自
   独立组装 trajectory+metrics，而 chunk 路径由 gatherer 再组装一遍——同一产物两条装配线
   （已存在实际分叉：janus `forward_plan` 感知 `use_ar_scheduler`，`JanusProChunkGatherer` 硬编码
   `ar_scheduler_enabled: False`）。生产只走 gatherer 线，`forward_plan` 线只被测试覆盖。
4. **chunk 路径的骨架重复**：`forward_chunk_plan`（93/94/64L）共享 tokenize+align → embed →
   `ARDecodeLoop` → decode-payload 骨架，但每份是**可读的直线代码**，真差异（janus 离散 VQ /
   nextstep 连续 flow-head / R1 `generate_with_refine` 反转控制流）占每份约 1/3。
5. 琐碎项：`capability()` 2L 逐字相同未上提；janus 缺省 `plan()` 在两测试被调、nextstep 无 `plan()`；
   gatherer 的 cat-字段循环（89/97/64L 里各 ~30L）是数据形状。

### 设计裁决

**做（按序）：**

1. **`ARChunkExecutorBase` 补齐请求级三件套**（对齐 diffusion 基类）：`plan()`（build_engine_plan +
   capability）、`capability()`、`forward_plan()` = chunks → `forward_chunk_plan` → `gather_chunks`
   （复用 `run_sample_chunks_with_oom_retry`，AR 顺带获得 OOM 降级）。**删除 3 份家族 `forward_plan`
   （-366L）**，装配线归一为 gatherer 一条。两个测试调用方原样工作且从此见证"测试路径 == 生产路径"。
2. **gatherer 瘦身**：加 `cat_chunk_fields(chunks, fields) -> dict` 小助手（字段清单已是
   `layout.ordered_chunks(row_fields=...)` 的数据），gatherer 各删 ~30L 的手写 `torch.cat` 段；
   engine-counters 组装与 `_ar_engine_counters` 合并单份。
3. `capability()` 上提基类。

**明确不做（及为什么）：**

- **`forward_chunk_plan` 不做模板方法化**。骨架重复 ~40L×2（R1 形状不同不算），拆成 4 个 hook 会把
  可读的 90 行直线流碎成跨文件跳读——AGENTS.md 反对为省行数 flatten/碎片化薄函数。**触发条件**：
  GLM-Image/Emu3/LlamaGen 落地后骨架 ×5 份时再模板化（届时收益翻倍、且新家族可当 pilot）。
- **R1 不进任何模板**：其控制流反转（委托 `model.generate_with_refine` + `image_sampler` 回调），
  只继承基类新 `forward_plan`，chunk 路径保持自有。
- **`use_ar_scheduler` 的跨 prompt 批调度语义随家族 `forward_plan` 一起删除**：它只存在于无生产调用方
  的路径上（生产 chunk 路径本就是 per-chunk 调度），属于死语义；TokenScheduler 本体与其单测不动。

### 风险与验证

- 最大风险：`forward_plan` 语义换成 chunk+gather 后两个测试调用方的输出等价性——它们正是等价性见证人，
  跑 `tests/models/ar/` + `tests/generation/ar/` + `tests/e2e/test_real_checkpoint_rl.py`（收集级）验证。
- 迁移序：基类三件套 + janus 删 `forward_plan` → 全测 → nextstep → R1 → gatherer 瘦身 → 全测提交。
- 预期净变化：约 **-350 行**，AR 执行面从"每家一条完整流水线"收敛为"每家一个 chunk 步骤 + 一个装配器"。

## 参考

- 重复证据：`vrl/models/diffusion/sd3_5/runtime.py:46-175`、`vrl/models/ar/janus_pro/runtime.py:53-122`
- 薄层证据（保留）：`vrl/models/diffusion/base.py:41-96`（抽象方法）、`registry.py:92-146`（声明式 entry）
- 落地形状：`configs/model/diffusion/sd3_5/medium.yaml`、`tests/generation/{diffusion,ar}/`
- GLM-Image：https://github.com/zai-org/GLM-Image
- 上游覆盖/边界：[[SPRINT_model_family_coverage]]、[[SPRINT_physical_ai_model_support]]、[[SPRINT_janus_pro_upstream_reconcile]]
</content>
</invoke>
