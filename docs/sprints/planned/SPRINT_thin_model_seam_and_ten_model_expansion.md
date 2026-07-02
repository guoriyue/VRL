# SPRINT: 薄化模型 seam + 十个新模型扩张（diffusion + AR，含 GLM-Image）

状态：**planned（2026-06-30）**。性质：**架构瘦身重构（Phase 0）+ 模型覆盖扩张（Phase 1，10 个）**。
承接 [[SPRINT_model_family_coverage]]（覆盖度 index）与 [[SPRINT_physical_ai_model_support]]（优先级边界）。
本 sprint 是**可落地**的：Phase 0 一次性重构，Phase 1 每小时接一个模型（配 hourly loop trigger，见 §6）。

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
> - ⏭️ 待办（后续 tick）：迁移 flux / qwen_image（同 single-transformer 形状，直接套）；video/anima 保留自建
>   replay builder（多 transformer/单文件，形状不同）；§2.1 registry 描述符字段、§2.4 train.py 折叠、
>   §2.5 `ARModelBase`。

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

## 3. Phase 1：十个新模型清单（每小时接一个）

接入前提：Phase 0 已合入（否则每个模型仍要抄 build 编排）。每个模型的落地 = **真薄层**：
`model.py` +（AR 才需要）`runner.py` + registry 一条描述符 + yaml 一份 + test 一个。

> ⚠️ Evidence-first：下表 HF repo id 与 diffusers 类名是**预期**，落地第一步必须核对该模型在当前
> diffusers/transformers 里的真实 pipeline/transformer 类名——**类名决定它能否套 seam**。核对不过就
> 记录到本表并降级/换模型，不硬接。

### 3.1 Tier A — T2I 扩散（纯 diffusion seam，最省）

- [ ] **1. SANA** — `Efficient-Large-Model/Sana_1600M_1024px`（NVIDIA，linear-DiT，1024px 高效）。
  seam：diffusion T2I。训练：LoRA。为什么：cosmos-rl 已支持、极省显存，最适合当 Phase 0 之后的第一个薄接样例。
- [ ] **2. PixArt-Σ** — `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`（T2I DiT，弱 CFG，社区常用基线）。
  seam：diffusion T2I。训练：LoRA。
- [ ] **3. Lumina-Image 2.0** — `Alpha-VLLM/Lumina-Image-2.0`（flow-matching DiT，Gemma 文本编码器）。
  seam：diffusion T2I。训练：LoRA。技术点：文本编码器非 CLIP，`encode_prompt` 要对齐。
- [ ] **4. HunyuanImage-2.1** — `tencent/HunyuanImage-2.1`（大 T2I DiT，强中文/文字渲染）。
  seam：diffusion T2I。训练：**LoRA-only**（体量大）。

### 3.2 Tier B — T2V 视频扩散（套 Wan/Cosmos 5D 潜变量 seam）

- [ ] **5. HunyuanVideo** — `tencent/HunyuanVideo`（T2V DiT）。seam：diffusion 视频（5D latent）。
  训练：LoRA。为什么：[[SPRINT_model_family_coverage]] Tier-2 已点名；VAE/scheduler 需单独对齐。
- [ ] **6. CogVideoX** — `THUDM/CogVideoX-5b`（Zhipu/GLM 的 T2V，与 GLM-Image 同门）。
  seam：diffusion 视频。训练：LoRA。技术点：3D VAE + T5 文本编码器。
- [ ] **7. Mochi-1** — `genmo/mochi-1-preview`（T2V DiT，AsymmDiT）。seam：diffusion 视频。训练：LoRA。

### 3.3 Tier C — AR（自回归，套 janus_pro/nextstep_1 seam）

- [ ] **8. GLM-Image** ⭐ — `zai-org/GLM-Image`（16B = 9B AR〔GLM-4-9B〕+ 7B diffusion decoder + Glyph Encoder）。
  **用户点名**。seam：**混合 AR+decoder**，形状最贴 `nextstep_1`（AR 连续 + flow/decoder head），不是纯离散
  `janus_pro`。训练：LoRA-only（16B）。技术点：AR 段出语义 token → diffusion decoder 出像素；logprob 打在
  AR 段。落地前先决定 RL 到底训哪一段（AR 语义段 vs decoder）——建议先只训 AR 段，decoder frozen（对齐
  [[SPRINT_frozen_component_preservation]]）。
- [ ] **9. Emu3** — `BAAI/Emu3-Gen`（纯 next-token AR T2I，单一 transformer）。seam：`ar_discrete`（最贴 janus_pro）。
  训练：LoRA。为什么：最干净的离散 AR，验证 GLM-Image 之前的 AR seam 压力测试。
- [ ] **10. LlamaGen** — `FoundationVision/LlamaGen`（AR 离散 T2I，class/text 条件 + VQ decode）。
  seam：`ar_discrete`。训练：LoRA。技术点：VQ tokenizer 与 janus 的 VQ decode 路径可复用。

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

## 参考

- 重复证据：`vrl/models/diffusion/sd3_5/runtime.py:46-175`、`vrl/models/ar/janus_pro/runtime.py:53-122`
- 薄层证据（保留）：`vrl/models/diffusion/base.py:41-96`（抽象方法）、`registry.py:92-146`（声明式 entry）
- 落地形状：`configs/model/diffusion/sd3_5/medium.yaml`、`tests/generation/{diffusion,ar}/`
- GLM-Image：https://github.com/zai-org/GLM-Image
- 上游覆盖/边界：[[SPRINT_model_family_coverage]]、[[SPRINT_physical_ai_model_support]]、[[SPRINT_janus_pro_upstream_reconcile]]
</content>
</invoke>
