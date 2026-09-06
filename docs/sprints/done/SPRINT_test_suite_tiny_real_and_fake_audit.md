# SPRINT: 测试套件 tiny-real 化、假替身审计与配置断言整改（planned）

> **Historical correction (2026-07-13).** The physical-stage contract and its
> `RayPipelineStageWorker` test discussed below were subsequently deleted because
> the whole seam had zero production consumers. The timing investigation remains
> a historical finding, not a reason to restore that test-only adapter.

状态：已落地 / done。全套 T1–T6 已在 commit 84584d2（"test: tiny-real fixtures, dedup fakes, prune declaration-only config asserts"，main 祖先）一次性实现：AR fixtures.py（build_stub_janus_mmgpt/build_stub_janus_model）、anima test_backbone_parity.py + build_tiny_anima_transformer、_ONLINE_RECIPES/_RECIPES 从 _experiment_names() glob 派生、interfaces 契约参数化注册家族、replay-export 对齐 predict2+predict2.5+anima、cosmos/wan-i2v from_build loading test、conftest marker 注释、_wait_until 沉入 continuous/_helpers.py。190 测试全绿。仅 T3.4/T6.4（doc 明标可选、nightly-only 提速观察）未做，不计入 scope。
保留、哪些测试什么都没证明该删、哪些配置断言违反 no-exact-config、哪些 parity/infra 缺口要补。
findings + 路径 + 整改逻辑都在本文。按 T1→T6 分轨道做，轨道间基本无依赖（T1 的 AR fixtures 是 T5
AR parity 的前置）；每个 item 独立可 PR。

> 方法：按 7 个维度（fakes-audit / real-load-audit / tiny-fixture-coverage / parity-contract-design /
> removable-redundant / config-assertion-audit / infra-markers-speed）逐个真实读源码 + 跑测试验证，
> 每条 finding 都过了一轮对抗式复核（verifier_correction）。本文是把复核修正折叠进去后的定稿——
> 凡复核推翻或收窄了自动结论的地方，下文都按修正后的版本写，并在 §校验记录 标出最关键的 6 处反转。

---

## 1. 核心结论 (TL;DR)

**diffusion 侧已经有一套"tiny-real config-init"的黄金标准（`tests/models/diffusion/fixtures.py` +
`registry.py`）：从 config 直接构造真实 diffusers transformer（~6.7K 参数、CPU、不 from_pretrained、
不下载、seed 可复现），再用 `record_forward_calls` 钉住 wrapper 传给真实签名的 kwargs。本 sprint 把
这套标准推到全套——重点是还没迁移的 AR 侧——同时删掉证明不了任何东西的测试、重写复制 YAML 字面值的
配置断言、补齐 parity 矩阵、收紧 infra。**

用户的核心诉求贯穿全文：**要的是"小的、同/近架构的真实模型，而不是同一个大模型"**——
`build_tiny_cosmos_transformer()` 是真实的 `CosmosTransformer3DModel`，只是小；它无法像手写 fake 那样
随上游签名漂移而静默腐烂。手写 fake 重声明签名才是腐烂源。

**但"tiny-real > fake"不是无脑替换。** 大量 fake 是合法的协议/边界替身（fake Ray ref、fake VAE 只需
`.decode` 形状、scripted KV 探针、外部 OCR 引擎），换成真实模型零收益甚至会摧毁断言。判据是
AGENTS.md 的"一致性优先于行数削减"：只动**重声明了神经网络 forward 签名、会随上游漂移而静默腐烂的
复制品**，其余合法边界 fake 一律保留。

按动作拆分（去重后约 50 条 finding）：

| 动作 | 数量 | 含义 |
|---|---|---|
| **FAKE→TINY**（手写 fake 迁 tiny-real） | 2 | anima forward_step 唯一未迁移的 diffusion 签名 fake；理由真实可行 |
| **STUB-CONSOLIDATE**（AR stub 去重到共享 fixtures） | 1（含 4 文件） | janus mmgpt 表面被 4 个文件各写一遍且定义漂移 |
| **REMOVE**（删除证明不了东西的测试） | 1（含子项） | `test_prompt_dataset_configs.py` 整文件复制 YAML 字面值 |
| **CONFIG-REWRITE**（配置断言改测行为/结构） | 5 | 钉死 YAML 字面值 / 手维护实验名集合 / 手维护 allowlist |
| **PARITY/COVERAGE**（补 parity 与契约矩阵） | 5 | anima 缺真实 backbone parity；RuntimeModel 契约从不跑真实家族；replay-export 对齐只有 predict2 |
| **INFRA**（marker / 速度 / flaky 卫生） | 6 | dead marker、gpu CI lane 文档过度承诺、_wait_until 重复、Ray per-test spin-up |
| **KEEP**（合法边界 fake / 已是黄金标准） | ~30 | fake Ray、fake VAE、scripted KV 探针、外部引擎、行为/不变量配置测试——真实边界，不要碰 |

**真正的靶子集中在四处**（其余碰了反而引入风险）：

- **(a) AR 侧的 mmgpt stub 在 4 个文件里各写一遍（最大单点）**——`_StubMMGPT`/`_StubVQ`/各自的
  `_StubLM`/`_RecordingLM`/`_LM` 把 Janus `MultiModalityCausalLM` 表面手写了四份，定义还互相漂移。
  但 AR **无法**迁 tiny-real（见 §非目标），所以是 stub 去重到共享 fixtures，不是 fake→tiny。
- **(b) anima 是 diffusion 唯一未迁 tiny-real 的 forward_step parity**——sd3/wan/predict2/predict2.5
  都用 `build_tiny_*_transformer` + `record_forward_calls`，只有 anima 还手写 `_ConstantAnimaTransformer`
  重声明 Cosmos 签名。可干净迁移（它的真实 backbone 就是 `CosmosTransformer3DModel`）。
- **(c) `test_prompt_dataset_configs.py` 整文件复制 YAML 声明**——逐行把 manifest 路径、resolution、
  num_workers 抄进 Python 断言，正是 no-exact-config 规则禁止的反模式；结构/行为覆盖已在别处。
- **(d) 一批手维护的常量列表会随源漂移而静默漏测**——`_ONLINE_RECIPES`（16 条，盘上 17 条 online
  recipe，漏了 `online_nft_motion_physics`）、`expected` 实验名全集（18 条字面）、reward kwargs
  allowlist（6 条路径）。应从 `_experiment_names()` glob 派生，符合 AGENTS.md "从真相源派生校验集合"。

诚实说：fake Ray runtime、fake VAE（只需 `.enable_*`/`.decode` 形状）、scripted KV/OCR 探针、
gated e2e 真权重套件、`test_schema.py` 的验证行为测试、per-test seeding——**全部合理，不要碰**。

---

## 2. 判据 (the rubric we applied)

**FAKE→TINY（该迁）的唯一信号：** fake 重声明了某个真实神经网络模型的 **forward 签名**，且该真实模型
有 config-init 构造路径（diffusers transformer 都有 `__init__(config)`），所以一旦上游改 kwarg，
手写 fake 会静默继续通过——这正是 tiny-real 要消灭的腐烂。

**KEEP（合法 fake）的类别：**
- `protocol/boundary fake` — fake Ray ref/actor、fake HF `from_pretrained` 返回、fake 远程 transport、
  fake VAE 只需 `.enable_tiling/.enable_slicing` 或 `.decode` 形状、fake torch `GradScaler` 脚本化
  backoff、外部 OCR 引擎。这些是 AGENTS.md 明确白名单的"a fake VAE that only needs .decode shape"。
- `behavior/equivalence probe` — 输出是刻意 scripted 的确定值，用来钉一个真实模型给不出的不变量
  （batched-step == per-row-step 位等价、CFG combine == 2.0、`calls == []` 证明某分支被绕过、
  per-row KV 顺序）。真实模型会摧毁这些断言。
- `red-line guard` — `forward` 故意 raise 以证明某路径永不被走（`_PolicyWrapperGuard`、`_FakeHead.__call__`）。
- `ABC subclass fake` — 继承真实抽象基类只实现扩展点（`_FakeTorchReward(TorchRewardModel)`），随源增删
  方法会显式报错，不会静默腐烂。
- `minimal-protocol conformance probe` — 故意缺方法以触发 `require_*_model` 守卫（`_IncompleteReplayModel`）。

**配置断言的判据：** configs 是声明。测 **loader/validation 行为**（哪些输入 raise、哪些 warn、哪些
load）与**结构约定**（loader 是合法 discriminator、section 存在、单一 component），**绝不**断言字面
YAML 值（manifest 路径、resolution、lr、entrypoint 字符串）。手维护的"该有哪些 recipe/哪些允许内联"
集合应从 glob 真相源派生，不要手抄。

**删除测试的判据：** 见 §"何时可以删除测试"。核心：断言纯属声明回显 **且** 真实行为/结构覆盖已在别处。

---

## 3. T1 — AR tiny-fixtures 与 stub 去重（最高杠杆，T5 AR parity 的前置）

AR 侧完全没有 `fixtures.py`（`tests/models/ar/` 下只有 `__init__.py` + 各家族测试），而 janus mmgpt 表面
在 4 个文件各写一遍。这是全套最大的单点重复。**关键约束：AR 不迁 tiny-real**（见 §非目标 D）。

### T1.1 — 建 `tests/models/ar/fixtures.py`，去重 janus stub 的**结构脚手架**（risk: low, effort: M）

- [ ] 新建 `tests/models/ar/fixtures.py`（AR 类比 `tests/models/diffusion/fixtures.py`），导出：
  - `build_stub_janus_mmgpt(*, ...)` —— canonical `_StubMMGPT` 容器，含验证过的属性表面
    `language_model / gen_head / gen_vision_model / gen_aligner / gen_embed / prepare_gen_img_embeds`，
    以及 `_StubVQ.decode_code(ids, shape)`。
  - `build_stub_janus_model(**config)` —— 用上面的 mmgpt 包成 `JanusProModel`，替换 4 个文件里各自的
    `_model()` helper。
  - 复用（不复制）diffusion 的 `record_forward_calls`（`tests/models/diffusion/fixtures.py:167`，
    family-agnostic，只注册 forward_pre_hook）。
- [ ] **docstring 必须写明 AR 不能 tiny-real 的真实理由**（按 §非目标 D 的精确措辞，逐路径写，
  不要写"impossible for all AR"的笼统假话）。

证据（4 份各写一遍、定义漂移）：
```python
# test_kv_decode.py:86 / test_replay.py:60 / test_janus_paged_attention_one_step.py:126 各自:
class _StubMMGPT(nn.Module):
    self.language_model = _StubLM() / _RecordingLM()
    self.gen_vision_model = _StubVQ()
    self.gen_head = nn.Linear(HIDDEN, JANUS_IMAGE_VOCAB_SIZE) / _RecordingHead()
    self.gen_aligner = nn.Identity()      # ← test_r1_model.py 的 _MMGPT 完全省略了这个
    self.gen_embed = nn.Embedding(JANUS_IMAGE_VOCAB_SIZE, HIDDEN)
```
生产真相源：`runtime.py:108` 的 `replay_modules=("language_model","gen_embed","gen_aligner","gen_head")`
就是这个契约；`JanusProModel.__init__`（`model.py:193-199`）已对 `gen_head/gen_vision_model/language_model`
做 `hasattr` 守卫并 raise。

> **关键修正（复核推翻了"单一 builder 吃下所有"的提案）：** 不要做"一个 `_StubMMGPT` + `recording`/
> `with_lm_head` 两个布尔"的 god-builder。**4 个文件的 `language_model.forward` 行为各不相同，且都是各自
> 测试的承重 instrumentation，不是可吸收的"重复签名"：**
> - `test_kv_decode._RecordingLM`：制造特定 KV cache（`key=full(call_index)`, `value=key+100`）+ 写
>   `hidden[:, -1, 0] = 10+call_index`，测试断言精确值 `[10.,10.,11.,11.]` / `[12.,...]` 和 KV-prefill
>   调用 shape 序列 `[(2,3,H),(2,3,H),(4,1,H)]`。
> - `test_janus_paged_attention._RecordingLM`：只记录 `**kwargs` 返回 zeros，测试断言
>   `language_model.calls == []`（即它**必须不被调用**，paged backend 接管）。
> - `test_replay._StubLM`：纯 identity trunk（`last_hidden_state == inputs_embeds`），无 `.logits`。
> - `test_r1_model._LM`：经 `lm_head` 返回 `.logits`（`forward_text_logits`/`_sample_selfcheck_text` 需要），
>   `_VQ` 带 `quantize.embedding`（`_resolve_vq_latent_channels` 需要），且**无** `gen_aligner`。
>
> **结论：fixtures 只 canonical 化结构脚手架（mmgpt 容器 + `_StubVQ` + `gen_embed`/`prepare_gen_img_embeds`
> + 一个可选的 recording head/记录列表），每个测试的 `language_model.forward` 行为 instrumentation 留在本地
> 注入。** nextstep 暂不纳入（它当前只有 request-parsing 测试，无 model-level fixture 需求）。

什么覆盖会丢失：无。这是去重 + 集中化，不删任何断言；4 个文件本地的行为 instrumentation 全部保留。

非目标：不把 diffusion 的 tiny-real 与 AR 的 stub-builder 合并成一个模块——两族真实模型约束不同
（diffusion=config-init 真实 transformer；AR=外部 trust-remote-code，无 config-init），应保持 sibling 关系。

---

## 4. T2 — real-load → tiny 替换与 gate 修复

### T2.1 — anima `test_forward_step.py` 迁 tiny-real，补真实 backbone parity（risk: low, effort: M）

这是 diffusion 侧唯一未迁的签名 fake，同时出现在 `fakes-audit` / `real-load-audit` /
`tiny-fixture-coverage` / `parity-contract-design` 四个维度——是确凿的高优先级靶子。

证据：
```python
# tests/models/diffusion/cosmos/anima/test_forward_step.py:11-35
class _ConstantAnimaTransformer(torch.nn.Module):
    config = SimpleNamespace(in_channels=1)
    def forward(self, *, hidden_states, timestep, encoder_hidden_states, padding_mask, return_dict):
        value = encoder_hidden_states[:, :1, :1].reshape(-1, 1, 1, 1, 1)
        return (torch.ones_like(hidden_states) * value,)   # 重声明 Cosmos 签名 + 闭式输出
```
生产真相源：`vrl/models/diffusion/cosmos/anima/model.py:496-502` 是
`CosmosTransformer3DModel.from_config(...)`——和 `build_tiny_cosmos_transformer()` 同一个类。

- [ ] **保留** `_ConstantAnimaTransformer` 这个 const-flow contract 测试**不动**——它故意需要一个真实模型
  给不出的可控常量输出，用来钉 `noise_pred_cond==ones` / `noise_pred_uncond==zeros` / raw-latent
  passthrough（`transformer.calls[0]["hidden_states"]==latents`）/ sigma 直传（`timestep==0.7`）。
- [ ] **新增** `tests/models/diffusion/cosmos/anima/test_backbone_parity.py`，镜像 predict2.5 exemplar：
  `build_tiny_cosmos_transformer()` + `record_forward_calls()`，跑 `AnimaModel.forward_step`，断言两次
  forward、断言 `cond != uncond`（真实 backbone 响应 prompt）、断言 CFG combine。
- [ ] **三处承重修正（写测试时必看，照抄 exemplar 会引 bug）：**
  1. **anima 的 CFG 公式是 `combined = uncond + guidance*(cond-uncond)`**（`model.py:326`），**不是**
     predict2/predict2.5 的 `cond + guidance*(cond-uncond)`。直接复制 exemplar 的断言行就是真 bug。
  2. **`timestep==0.7` 钉的是 sigma 原值直传，不是 EDM→timestep 转换**——`forward_step`（`model.py:301-302`）
     把 `current_sigma` 原样传给 transformer，`state.timesteps=700.0` 是无用 decoy。自动结论里的
     "EDM-sigma→timestep conversion" 措辞是错的。
  3. **通道几何**：anima 把 latents **直接**喂 transformer（无 runner 侧通道扩展，不像 predict2.5 的
     `DiffusionBackboneCaller`），所以 `AnimaSamplingState.latents` 必须携带
     `transformer.config.in_channels` 个通道（tiny fixture 下 = 5，因 `build_tiny_cosmos_transformer`
     设 `in_channels=4+1` 且 `concat_padding_mask` 内部再加 1）。const fake 用
     `torch.ones_like(hidden_states)`（1 通道进=出）掩盖了真实模型 `in_channels(5)!=out_channels(4)` 的
     不对称——这正是 tiny-real 多覆盖的一点。考虑给 `build_tiny_cosmos_transformer` 加
     `concat_padding_mask`/`in_channels` 参数，或新增 `build_tiny_anima_transformer`
     （`in_channels==out_channels`, `concat_padding_mask=True`），符合 anima 不 concat condition-mask 的路径。

什么覆盖会丢失：无（const 测试保留，parity 是新增）。这关上 diffusion parity 矩阵唯一的 fake→tiny 洞。

### T2.2 — scheduler logprob parity 在 clean CI 静默 no-op（risk: low, effort: M）

证据：`tests/models/diffusion/test_scheduler_logprob_parity.py:41-44` 用 `local_files_only=True` +
`pytest.skip` on cache miss。clean CI runner（无缓存 checkpoint）上，4 个 cache-loaded 家族 + 显式的
`test_predict2_scheduler_exercises_edm_conversion` 全部 skip，只有 in-code 构造的 anima case 跑。

> **关键修正（复核收窄了 severity）：** 自动结论的 headline "EDM sigma_max=80 regression 在 clean CI 零
> 覆盖" 是**夸大**的。EDM→flow 转换分支（`vrl/math/diffusion/flow_matching.py:91-114`）在 clean CI 里被
> `tests/math/test_diffusion_flow_matching.py` 用 cache-free `_FakeScheduler` + 合成 `_EDM_SIGMAS`
> 表无条件覆盖（同样钉 `ratio==1` 不变量 + O(1) 量级）。真正在 clean CI 退化为 e2e-only 的是 **predict2
> 真实 checkpoint 的实际 sigma 表**——而那已被 e2e cosmos_predict2 case 钉住。所以这是"真实 sigma 表覆盖
> 缺口 + 一条会静默 no-op 的误导性测试 comment"，不是"回归零覆盖"。

- [ ] 选 **option (a)**：发布/缓存一个每家族的 tiny scheduler-only HF repo（scheduler_config.json 几 KB，
  非权重），加入 `_FAMILY_SCHEDULERS` 用 pinned revision 经 `from_config` 加载，去掉 `local_files_only`。
  这保留本文件独有的"真实 config 表"价值。
- [ ] **不要**断言字面 sigma 值（config-as-declaration）；保留 `ratio==1` 不变量 + `sigma.max()>1` 量级 pin。
- [ ] 在 docstring 注明完整真实表 parity 由 e2e cached case 冗余覆盖。

什么覆盖会丢失：option (b)（in-code 构造 EDM scheduler）与已在 clean CI 跑的合成 math 测试大量重复，
故选 (a)。

---

## 5. T3 — 可删除 / 冗余测试清理（逐条覆盖丢失分析）

### T3.1 — 删除 `test_prompt_dataset_configs.py` 中复制 YAML 字面值的测试（risk: low, removal_safe: 部分）

整文件 5 个测试里，**3 个删、2 个保留**——复核修正了"删整文件"的过宽提案。

证据：
```python
# tests/config/test_prompt_dataset_configs.py:16-78
assert cfg.data.manifest == "datasets/ocr/train.txt"      # 逐字复制 configs/dataset/ocr.yaml
assert cfg.data.preprocessing.resolution == 0
assert cfg.data.sampler.dataloader_num_workers == 4
```

- [ ] **删 / 重写** `test_*dataset*`（lines 14-78，含 ocr/geneval/pickscore_sfw/videophy_i2v/pickapic_v2）：
  这是逐行复制 YAML 声明。`_validate_data`（`schema.py:148-206`）只校验这些 key **存在**、从不校验值，
  所以那些 bool/int 值断言纯属声明回显。
  - 什么覆盖会丢失：无独有合法覆盖。**别处已覆盖**：(1) loader/preprocessing/sampler **结构验证**由
    `test_schema.py::test_valid_data_loaders_are_accepted`（参数化全 3 个 loader）+
    `test_prompt_image_manifest_requires_image_caption_fields` + `test_unknown_sampler_type_raises` 钉住；
    (2) 每个 dataset group 都被某 experiment 消费，`test_load_all_experiments.py::test_all_experiments_load_and_validate`
    对每个 experiment 跑 `require_training_config`，真实 merged `DataConfig` 已被验证。
  - **执行注意**：不要塌缩成"每个 YAML parse 一下"的裸 smoke（会与 all-experiments 循环重复）；该文件唯一
    边际价值是"每个 dataset **group** 文件单独 parse 成 valid `DataConfig`"（group 是可独立复用的 building
    block）。保留一个 per-group 结构 smoke（构造 `DataConfig` + 断言 loader 是合法 discriminator +
    `_validate_data` 不 raise），不做字面等值。覆盖 `pickscore_sfw`（自动 finding 的 files 列表漏了它）。
- [ ] **删** `test_prompt_rewards_stay_in_reward_configs`（lines 81-89）——见 T3.2，改为通用化。
- [ ] **删** `test_sd35_prompt_dataset_experiments_load_and_validate`（lines 92-101）——见 T3.3。

> **关键修正：** 自动结论说"删整文件"，但复核确认它有 dataset-group 内部字段的结构覆盖（all-experiments
> 循环不覆盖 group 文件本身的独立可加载性），所以是"删 3 个 + 把 dataset 测试重写成 per-group 结构 smoke"。

### T3.2 — `test_prompt_rewards_stay_in_reward_configs`：2 条字面表，通用化（risk: low）

证据：`assert list(cfg.reward.components.keys()) == [reward_name]`（geneval→['geneval'], pickscore→['pickscore']）。

- [ ] **删**这个 2 条字面表，但**通用化保留**它唯一不冗余的部分。
  - "每个 reward 配置恰好一个 component" 已被 `test_load_all_experiments.py::test_reward_configs_are_single_reward_building_blocks`
    对全部 7 个 reward YAML 用 `len(components)==1` 值无关地覆盖。
  - **唯一独有的是"component KEY 名 == 注册 reward 名"**——`RewardConfig.components` 在 `schema.py:52-53`
    是 `Annotated[dict, OPEN]`（注释明写 reward 名 user-chosen，open by design），`require_training_config`
    不校验 key 是否注册；真正消费 key 的是 `MultiReward.from_dict→get_reward(name)`（运行时 KeyError）。
  - **执行**：改成通用测试——对每个 reward YAML 断言 component key 解析到注册名（或由文件名派生），而非
    2 条字面表。这样新增第三个 prompt reward 不会漏守。
  - 什么覆盖会丢失：裸删会丢"component key 能解析到注册名"这条守卫（运行期 KeyError 风险）；故通用化而非裸删。

### T3.3 — `test_sd35_prompt_dataset_experiments_load_and_validate`：load+validate + 字面 entrypoint（risk: low, removal_safe: true）

证据：`assert cfg.trainer.entrypoint == "vrl.scripts.diffusion.sd3_5.train:train_sd3_5_grpo"`。

- [ ] **删**这一条（lines 92-101）。load+validate 已被 `test_all_experiments_load_and_validate` 完整覆盖
  （`expected` 集合含这两个 sd3_5 experiment）；entrypoint **解析为真实 callable** 的行为由
  `test_unified_train_entrypoint_reads_yaml_entrypoint` 拥有（geneval/pickscore/ocr 三个 yaml entrypoint
  完全相同且 `train.py:20 def train_sd3_5_grpo` 真实存在）。字面字符串断言只多保证"YAML 还写着这串字符"。
  - 什么覆盖会丢失：无。

### T3.4 — Ray per-test 集群 spin-up，无 session fixture（risk: low, effort: M, 低优先级 nightly-only）

证据：`tests/rewards/ray/test_runtime.py`（3 个 `ray.init/shutdown` 周期）等 4 个文件，共 ~6 次独立集群
spin-up，无 `tests/ray/conftest.py`、无 session-scoped `ray_cluster` fixture。全部 `pytestmark=slow_test`
（只命中 nightly lane，不影响 PR 延迟）。

> **关键修正：** 真实 per-test 成本是 finally 里的显式 `ray.shutdown()`，**不是**缺复用机制——生产
> `vrl/ray/runtime.py:84` 已有 `if self.init_ray and not ray.is_initialized(): ray.init(...)` 守卫，
> runtime 已经按复用语义工作。isolation 担忧（`test_resource_lifecycle.py` 的 release 断言）在共享集群下
> **安全**——`shutdown()` 只拆 actor group/placement group，从不拆 Ray 集群。真正的 blocker 是 GPU 测试
> （`test_runtime.py:161-205`）传 `num_gpus` 而 CPU 测试不传——单个 session fixture 服务不了两种 init shape。

- [ ] （可选，nightly 提速）加 session-scoped `ray_cluster` fixture（init 一次，去掉 per-test `shutdown()`），
  scope 为 CPU-only 或始终预留 GPU。**先量 nightly wall-time 再投入。**
  - 什么覆盖会丢失：无（纯提速，不删测试）。category 标 redundant 略偏——这是速度观察。

---

## 6. T4 — 配置断言重写（no-exact-config 整改 + 手维护常量派生）

这一轨的共同病根：**断言复制声明值** 或 **手维护一个会随 glob 真相源漂移的常量列表**——后者正是
AGENTS.md "不要手维护重复类型化结构的常量、应从真相源派生" 的硬规则。

### T4.1 — `_ONLINE_RECIPES` 手维护列表已漂移，从 glob 派生（risk: low）

证据：`tests/config/test_precision.py:135` 的 `_ONLINE_RECIPES` 有 16 条；盘上有 17 个 online recipe，
**漏了 `diffusion/cosmos_predict2_5/online_nft_motion_physics`**（实测它当时的等价解析为
`training.dtype=bf16`, `rollout.dtype=bf16`, and `diffusion_math.dtype=fp32`，
无 legacy key，本会 PASS，是被静默漏测的真覆盖）。

- [ ] **保留**测试（`policy.diffusion_math == 'fp32'` 是 resolver 强制的受保护轴不变量，`"mixed_precision" not in cfg.actor`
  守旧 split 配置面不被重引入——都有 teeth）；**删手列表**，从 `_experiment_names()` 过滤
  `not Path(name).name.startswith("offline_")` 派生（实测得正好 17 条 online recipe）。
- [ ] **同步修第二处相同病灶**：`tests/scripts/test_online_precision_bridge.py:20` 的 `_RECIPES`（4 条手列表，
  同样漏 `online_nft_motion_physics`）也从 glob 派生。
  - 什么覆盖会丢失：无；反而**找回**被漏测的 recipe 覆盖。removal_safe=false（不删测试，只改 list 来源）。

> Current note: `policy.training == policy.rollout` is no longer a resolver
> tautology. The two role blocks are independently explicit, so this assertion
> now protects the online-recipe alignment contract. The protected-math and
> legacy-key assertions remain independently useful.

### T4.2 — `test_experiments_are_grouped_by_model_family` 钉死 18 条实验名全集，改结构断言（risk: low）

证据：`tests/config/test_load_all_experiments.py:114-138` 的 `expected` 是 18 个名字的字面集 +
`assert set(_experiment_names()) == expected`。git history 确认 add/remove/rename churn 频繁，每次都会
无故 break 这个测试。

- [ ] 把字面集等值换成它真正想守的结构约定：每个路径首段 ∈ `{'ar','diffusion'}`（line 138 已断言，保留）
  + `all(len(Path(n).parts)==3 for n in names)`（family/model/recipe 三段分组）。
  - 什么覆盖会丢失：无。"all load+validate" 已由 `test_all_experiments_load_and_validate` 用同一 rglob 覆盖；
    字面集隐含的 3 段深度由新增的 `len(parts)==3` 断言守住（**必须加，否则深度分组覆盖静默回归**）。

### T4.3 — `test_experiments_use_dataset_groups` 的 reward kwargs allowlist：6 条路径硬编码（risk: low, effort: M）

证据：`tests/config/test_load_all_experiments.py:145-152` 硬编码 6 个 YAML 路径的 allowlist，
`if rel not in allowed_reward_kwargs: inline_reward_kwargs.append(rel)`（白名单放行）。

- [ ] **保留** `inline_data==[]`（干净结构规则）；reward kwargs 改用 allowlist 实际编码的底层规则：
  内联 `reward.kwargs` 只允许覆盖已由 reward group 提供的 leaf 标量，**不能**声明新 component——
  即"内联的 component keys ⊆ 其 reward group default 提供的 keys"（实测 6 个文件内联的键都是
  `kling_video_reward.yaml`/`videocon_physics.yaml` 声明键集的严格子集，可推导、零误报）。

> **关键修正：** **不要**用自动结论的兜底方案"派生 allowlist = defaults 含 video-reward group 的实验"——
> 该集合是 7 个（`cosmos_predict2/online_grpo_kling_video_reward` 引入了 group 但不内联 kwargs），≠ 字面
> allowlist 的 6 个，替换会**放松**断言（让那个文件将来内联任意 kwargs 也不被检出）。用上面的子集校验。
  - 什么覆盖会丢失：无（子集规则更严，覆盖 allowlist 的全部意图且不漂移）。

### T4.4 — `test_algorithm_config_dispatches` 钉 kl_estimator / train_segments / entrypoint 字面值（risk: low）

证据：`tests/config/test_load_all_experiments.py:241-251`，`assert algo_cfg.kl_estimator == "k2"`（复制
`online_grpo_ocr.yaml:12`）、`train_segments == {...}`（复制 `token_grpo_multisegment.yaml:13-16`）、
entrypoint 字符串。

- [ ] **保留** dispatch 断言（`isinstance(algo_cfg, expected_type)` + `EXPECTED_ALGO_TYPE` 交叉校验——
  真实 wiring 行为，对应 `builders.py:188-222`）。**删 3 个声明 pin：**
  - `kl_estimator=='k2'`：删。覆盖丢失分析——它弱代理了"非默认覆盖经 `_dataclass_payload` 存活"，但该
    覆盖流转已被 `test_cli_overrides_reach_typed_trainer_config`（`_dataclass_payload` 是 family/section 无关）
    稳健覆盖。**不要**改成"in 合法集合"（若覆盖失效回落到默认 'k3'，'k3' 仍在集合内，抓不到）。
  - `train_segments == {...}`：改成 `set(algo_cfg.train_segments) == {'initial_image','selfcheck_text','final_image'}`
    ——段名是 `compute_loss`（`multisegment.py:65,75-83`）真实读取的结构契约，True/False 是调参声明。
  - entrypoint 字符串：删（纯声明重复）。若想补强，改成对 r1 配置调 `_import_callable` 验证可导入
    （`train.py` 已有 `train_janus_pro_r1_ocr_grpo`），而非字面比对。
  - 什么覆盖会丢失：无独有合法覆盖；段名结构契约保留。

### T4.5 — `test_prompt_dataset_configs.py` dataset 断言整体重写（risk: low）

与 T3.1 合并执行——把 lines 16-78 的逐行 YAML 复制改成 per-group 结构 smoke（构造 `DataConfig` +
loader 是合法 discriminator + `_validate_data` 不 raise），覆盖含 pickscore_sfw 的全部 5 个 group。

---

## 7. T5 — parity / 契约矩阵补全

### T5.1 — RuntimeModel/ReplayModel 契约从不跑真实家族类，新家族可静默跳过（risk: med, effort: M）

证据：`test_runtime_model_contract.py:126-134` / `test_replay_model_contract.py:108-116` 只对手写
`_MinimalRuntimeModel`/`_MinimalReplayModel`/`_DiffusionModelBaseStub` 断言 Protocol shape，**从不**断言
注册家族（JanusProModel / NextStep1Model / SD3_5Model / Wan* / Cosmos* / AnimaModel）满足协议。

> 复核确认这是真缺口：CPU CI 上没有任何真实家族流经 `require_runtime_model`/`require_replay_model`
> （`test_minimal_replay_runtime_wiring.py` 只调 `require_minimal_replay_bundle`，只校验 metadata role
> 字符串；AR builder 被 `_TinyRuntimeModel` monkeypatch 掉）。唯一真实家族过 gate 的是 gated e2e
> （`WM_RUN_REAL_MODEL_TESTS`+CUDA）。当前 3 个真实 ReplayModel 都满足协议——所以这是**回归预防 gate**，
> 不是现存 bug。

- [ ] 加参数化契约测试，对每个注册家族断言其 replay-model 类满足 `ReplayModel`/`RuntimeModel`，矩阵 keyed
  off family registry（`tests/rollouts/runtime/test_family_registry.py:18-30` 已枚举 9 个家族）。
- [ ] 实现注意：runtime-checkable Protocol 的 `isinstance()` 需要实例；类级 gate 用
  `callable(getattr(cls, m))`（`_missing_callables` 已是此法）。复用 `test_minimal_replay_runtime_wiring`
  的 monkeypatched builder 路径廉价构造，或直接对类检查必需方法不实例化。
  - 什么覆盖会丢失：无（纯新增）。removal_safe=false。

### T5.2 — 无 AR 侧 architecture parity（Janus/NextStep generate+replay 数值全靠 fake）（risk: low, effort: L, 文档+track 项）

证据：diffusion 有 `not torch.allclose(cond, uncond)`（真实模型响应 prompt）；AR 的 KV-decode/replay 断言
全建立在 scripted-constant fake 上（`hidden[:, -1, 0] = float(10 + call_index)`），只钉 wiring/shape，从不
证明真实 AR trunk 产出 prompt-dependent logits。NextStep 在 CPU 侧 forward/replay 数值零覆盖。

- [ ] **标为结构性 coverage-gap，不是立即修**。真实 tiny-real AR parity 要么靠 gated e2e（已有，
  `test_real_checkpoint_rl.py` CASES 含 janus_pro + nextstep_1），要么 vendor 一个上游 trunk 的 config-init
  最小构建。
- [ ] **(a)** 在 T1.1 新建的 `tests/models/ar/fixtures.py` docstring 显式记录这个洞（类比 `registry.py`
  的 "Cosmos/Wan-I2V have no tiny repo" 注释），让与 diffusion 的不对称可见。
- [ ] **(b)** 若上游 janus 暴露 from_config 路径（`JanusProReplayCore.__init__(config)` 用
  `LlamaForCausalLM(config.language_config)` + `model_name_to_cls`，但仍 import `janus` 包），在
  `importorskip` 后加 `build_tiny_janus_mmgpt` 让真实 tiny trunk 替换 `test_kv_decode` 的 logits-flow 断言。
  - 什么覆盖会丢失：无（现有 fake 是唯一 CPU-runnable KV/segment 覆盖，**不删**）。

### T5.3 — replay-export 对齐契约只有 predict2，钉成跨家族不变量（risk: low, effort: M）

证据：`tests/models/diffusion/cosmos/predict2/test_replay_export_alignment.py:44-59` 钉每个导出 replay
tensor 的 dim-0 == sample batch（生于真实 GPU-gate bug：`init_latents` leading-1 dim 在 sample_batch_size>1
时被丢，restore KeyError）。

> **关键修正（复核收窄了候选 + 改了表述）：** **Wan-I2V 应剔除**——其 `condition` 来自
> `pipe.prepare_latents(..., batch_size, ...)` 一开始就是 batch_size，`image_embeds` 经
> `_align_optional_batch` 强制对齐且无法对齐就 raise，到 export 时已 sample-batched，不存在 predict2 的
> leading-1 静默丢弃风险；且其 roundtrip 已被 `wan_2_1/test_backbone_parity.py:133-162` 覆盖。
> **predict2.5 和 anima 的 export 已调用 `align_replay_tensor`/`_align_replay_tensor`（生产已防住），
> 但无测试钉住**——真正的窄缺口是"把已正确的生产对齐行为钉成跨家族回归守卫"，不是"这些家有同一个 bug"。

- [ ] 参数化对齐契约，候选限定 **predict2 + predict2.5 + anima**（剔除 Wan-I2V）：对每个 model 的
  `export_replay_tensors(state, batch_size=4)` 断言每个 tensor dim-0 == 4。predict2 测试折叠进参数化或保留。
  - 什么覆盖会丢失：无（纯 state plumbing，无 forward）。

### T5.4 — Cosmos / Wan-I2V 无 tiny HF pipe，from_pretrained 装配未在 PR CI 测（risk: low, effort: S）

证据：`registry.py:7` docstring "Cosmos / Wan-I2V have no such repo on the Hub, so they are absent"；
`_TINY_PIPELINES` 只有 wan-t2v + sd3。生产 `predict2/model.py:131` 的 `Cosmos2VideoToWorldPipeline.from_pretrained`
/ `wan_2_1/model.py:402` 的 `WanImageToVideoPipeline.from_pretrained` 只被 gated e2e 覆盖（Wan-I2V 连 e2e 都没有）。

> **关键修正（复核改了首选方案 + 丢弃同义反复提案）：** **不要**做"自审计测试断言 archs_with_tiny_pipe()
> 等于预期集合"——那是把 `_TINY_PIPELINES.keys()` 再抄一份的同义反复，且根本无法实现"Hub 发布 tiny repo
> 时标记"（测试不查 Hub）。真正没便宜覆盖的是 **`from_build` 的冻结/dtype/safety-checker 分支**（cosmos
> `_PassthroughSafetyChecker` 替换 + `torch.set_grad_enabled(True)` + dtype staging, `model.py:118-144`；
> wan-i2v `enable_sequential`/`model_cpu_offload` 分支, `model.py:430-438`）。注意 predict2 的 wrapper
> 接受真实 tiny transformer + forward_step 已被 `predict2/test_backbone_parity.py` + `common/test_decode_layout_parity.py`
> 用 `SimpleNamespace(pipeline)` 便宜覆盖了，所以"never wiring-tested cheaply" 过宽。

- [ ] **首选**：按 sd3 的 monkeypatch-`from_pretrained` 模式（`sd3_5/test_model_loading.py` 用 `_FakePipeline`
  断言冻结/dtype/device staging，零下载），给 cosmos predict2 和 wan-i2v 各加一个 `from_build` 加载测试，
  覆盖 safety-checker passthrough + grad 重开 + offload 分支。
- [ ] **补充（可选）**：若发布 pinned tiny Cosmos/Wan-I2V pipe repo，扩 `_TINY_PIPELINES` 覆盖真实
  diffusers 装配；在 docstring 写明 regeneration 路径。
  - 什么覆盖会丢失：无（纯新增）。

---

## 8. T6 — infra / marker / 速度 / flaky 卫生

### T6.1 — `optional` / `distributed` marker 注册了但零使用（risk: low）

证据：`pyproject.toml:111-112` 注册两个 marker；`conftest.py:55-67` 有两条永不触发的 gating 分支；
`grep pytest.mark.distributed/optional` 全仓零命中。这是 commit `23fec3c` "vLLM-style marker gating" 落地的
vLLM-parity 脚手架（`ci_envs.py` docstring 写明 "structure identical to vLLM's"），受一致性规则保护。

> **关键修正：** **不要**用自动结论的 option (a)（给 `tests/ray/test_ray_actor_pool.py` 加
> `@pytest.mark.distributed`）——那些 `cross_node=True` 测试是纯 in-process 单元测试（调
> `require_actor_gpu_ids()` 配手搭 dict，不起 Ray、不需第二节点/GPU），已带 file-level
> `pytestmark = slow_test`；加 distributed marker 会误标 + 双 marker 隐藏真实覆盖。全仓**没有**真正的
> distributed 测试（`grep mp.spawn/torchrun/init_process_group` 零命中），没有诚实的 lane member。

- [ ] **只做 option (b)**：在 `conftest.py` 加一行注释，说明这两个 marker 是预留的 vLLM-parity lane、
  当前无成员，防止未来 reviewer 当 dead code 删掉。**不删**（一致性规则保护）。

### T6.2 — `gpu` marker docstring 过度承诺不存在的 "GPU CI lane"（risk: low）

证据：`pyproject.toml:110` 写 "gpu: ...selectable for a GPU CI lane"，但 `.github/workflows/ci.yml` 三个
job 全是 `runs-on: ubuntu-latest`（CPU），无任何 GPU/self-hosted runner。

> **关键修正：** (1) 唯一含 "GPU CI lane" 措辞的是 `pyproject.toml:110` 一处——`conftest.py:5-10` 只说
> marker "both selects and skips"，从不承诺 lane，应从证据剔除。(2) "real-CUDA paths get zero CI coverage"
> 被夸大——CPU 可跑的 `test_vllm_paged_attention_import_gate.py` 用 fake vLLM 内部模块覆盖了
> `VllmPagedAttentionKernels` 整套 wiring/契约；真正零 CI 覆盖的只是"真实 CUDA kernel 的数值执行"，
> 那本就无法在 CPU runner 跑（是预期约束）。(3) `ci.yml:49,77` 已诚实陈述现实，唯一不一致的就是 pyproject 这句。

- [ ] **软化 `pyproject.toml:110` 措辞**，明说 gpu 测试只在本地/未来 GPU runner 跑（对齐 `ci.yml` 已有的
  诚实表述）。auto-skip 本身保留。（"加 GPU CI lane" 是可选项，不是必须。）

### T6.3 — `_wait_until` deadline-loop helper 跨文件重复（risk: low）

证据：`test_contracts.py:106-112` 定义 `async def _wait_until(condition, timeout_s=5.0)`；
`test_schedule.py:307-310` 把同一模式内联重写了一次（0.001 poll + 5.0s deadline 一致）。

> **关键修正（复核把目标文件改对了）：** 自动结论说"沉到 `conftest.py`"并引 "no new lean files" 反对建
> helpers 模块——**用反了规则**。全仓只有一个 `tests/conftest.py`（只含 hook + autouse fixture），无"放
> 纯共享 helper"的先例；本仓既定约定恰恰是**可 import 的 `_helpers.py`**（精确先例：
> `tests/trainers/online/_helpers.py` 被 6+ 个同级测试 import）。一个只装 `_wait_until` 的 `conftest.py`
> 本身才是 "new lean file"。

- [ ] 沉到 `tests/rollouts/orchestration/continuous/_helpers.py`（对齐 `_helpers.py` 既有约定），两个文件
  都 import；保留 0.001 poll + 5.0s deadline。注意 `test_schedule.py` 顶部需补 `import asyncio`。
  - 什么覆盖会丢失：无（纯去重，行为等价）。低优先级 test polish。

### T6.4 — `test_pipeline_contracts.py` 最慢 fast-lane 测试（2.24s）实为启了本地 Ray 实例（risk: low）

证据：`--durations` 排第一的 `test_ray_stage_worker_loads_handler_from_stage_config` 2.24s。

> **关键修正（复核推翻了自动诊断）：** 自动结论说这是"一次性 ray import 成本，无真实集群"——**实测全错**。
> `import ray` 仅 0.164s 且 `from vrl.generation.ray import RayPipelineStageWorker` 根本不加载 ray（全 lazy）。
> 2.24s 全在 `worker.worker_metadata()`（`stage_worker.py:48-61`）里的 `current_gpu_ids() → ray.get_gpu_ids()`
> （`vrl/ray/dependencies.py:42-44`），它**隐式 `ray.init()` 起了一个本地 Ray 实例**（日志 "Started a local
> Ray instance"）。`stage_worker.py:52` 不是廉价 lazy 延迟，正是起 Ray 的那行。

- [ ] **保留测试**（它是唯一经 `import_from_path` 跑 `RayPipelineStageWorker` 真实 handler-factory 的测试，
  companion 用 `_LocalStageActor` fake）。**真正廉价的优化在本测试逻辑**：它只断言
  `worker_metadata()["worker_id"]/["stage"]`，从不断言 node_ip/gpu_ids，且 `worker_metadata` 用
  `contextlib.suppress` 包住 ray 调用——改成断言 stage_name/worker_id 而不触发 ray-backed gpu/node 查询，
  即可去掉这个纯副作用的 Ray spin-up，零覆盖损失。
  - 什么覆盖会丢失：无（删除不安全，handler-factory 路径独有；优化是不调 live-ray metadata）。

---

## 9. 何时可以删除测试 (when to remove)

本 sprint 用的删除判据，蒸馏如下。**两个条件必须同时满足才删：**

1. **断言纯属声明回显 / 同义反复**：断言复制了 YAML 字面值（manifest 路径、resolution、lr、entrypoint
   字符串）、复制了 dataclass 默认值、或在结构上恒真（`compute==rollout` 因 resolver 从同一源派生）。
2. **真实行为 / 结构覆盖已在别处**：该测试想守的 loader/validation 行为或结构约定，已被一个 value-agnostic
   的测试覆盖（如 `test_schema.py` 的验证分支、`test_all_experiments_load_and_validate` 的全实验循环、
   `test_reward_configs_are_single_reward_building_blocks` 的全 reward 单 component）。

**反向（不可删，即使看起来冗余/像 fake）：**
- 测试驱动**真实生产代码路径**，fake 只替换 import-graph 之外的边界（Ray、HF 网络、GPU kernel、外部引擎）——
  断言的是生产逻辑，不是"mock 返回它被告知的值"。
- fake 输出是 scripted 确定值，钉一个真实模型给不出的不变量（位等价、CFG combine、`calls==[]`、KV 顺序）。
- 这是某分支/契约的唯一覆盖（grep 确认别处没有），删了会丢真实回归守卫。
- parity/契约测试钉 refactor 不变量（batch-independence、序列化 round-trip 的 derived field）。

**删除前流程：** 枚举将删的断言 → grep 确认覆盖在别处且 value-agnostic → 删 → 跑 pytest 确认零回归。
当删除会让某结构约定失守时，**重写成 value-agnostic 版本**（如 T3.2 通用化 key 解析、T4.2 结构断言），
不要裸删丢覆盖。

---

## 10. 非目标 / 保留项 (KEEP) — 一致性优先于清理

以下 fake 与模式经逐个读源 + 跑测试验证为**合法边界 / 已是黄金标准 / 行为测试**，本 sprint **一律不动**。
按类别说明为什么留：

**A. 协议 / 边界 fake（换 tiny-real 零收益甚至破坏，项目规则明确白名单）**
- `_FakeRef/_FakeRay/_FakeWorker/_FakeActor`（`test_chunk_dispatch.py:33-63`）——canonical "fake Ray ref"，
  `completion_rank` 确定性控制 `ray.wait` 顺序，跑真实 `run_actor_jobs`/executor 的 LPT 派发逻辑。真实集群
  慢且完成顺序非确定。
- `_FakeVAE`（`test_vae_decode_memory.py:17`、`test_decode_layout_parity.py`）——只记录
  `enable_tiling/enable_slicing`（policy 的全部契约）；`_IdentityDecodeVAE` 是 identity 探针（真实 VAE 上采样
  会破坏 shape-equality 断言）。已配 `build_tiny_*` transformer，是正确的 split 模板。
- ~~`_FakeModule/_FakeModel`（`test_frozen_module.py:17`）~~——已随 trainer-side frozen_offload 撤销删除
  （2026-06-13，见 `SPRINT_memory_plan_full` Phase 0）：ReplayModel 不加载这些模块，无 park/offload 可测。
- `_FakeModule/_FakePipeline`（`sd3_5/test_model_loading.py:11`）——fake diffusers pipeline assembly，是唯一
  钉 per-component dtype-routing map `{transformer:fp32,vae:fp32,default:fp16}` 的测试；真实 SD3.5 是多 GB 下载。
  断言的是 loader 构建的 dict，不是 YAML 字面值。
- `_FakeScaler/_FakeOptimizer` + `_Algorithm/_Collector/_Evaluator/_SpyEMA`（`test_grad_scaler.py`）——
  scripted GradScaler backoff（真实 scaler 无法被强制 skip）+ 协作者协议 fake；模型是真实 `nn.Linear`。
- `_FakeParameter/_FakeModule/_FakeGatherer/_FakeCapability`（`generation/ray/test_runtime_config.py`）——
  driver-CUDA-ownership 守卫 + gatherer/capability 协议 fake。（注：自动结论里"cuda-ownership 守卫被触发"
  的某个测试实因上游 device-overlap 先 raise 而"碰巧通过"，是接线缺陷，但不改 keep 结论。）
- `_StubBackend`（`test_ar_attention_backends.py:10`）——只捕获 `resolve_attention_backend` 转发的 kwargs；
  真实 backend 需 vLLM/GPU 且 dispatcher 从不 inspect backend 对象，只 inspect builder 签名。
- `_FakeFlashAttention*`（`test_vllm_paged_attention_import_gate.py`）——vLLM kernel 是 GPU-only 外部依赖
  behind import gate，fake 镜像 vLLM 0.21.0 的 18 个 metadata 字段（有 rot 风险但无廉价真实替代）；
  真实数值路径在 `test_vllm_paged_attention_real_ops.py` 的 `@gpu`。
- reward transport fake（`_FakeTorchReward(TorchRewardModel)` ABC 子类、`_FakeActorRuntime`/`_EmptyActorRuntime`
  Ray actor 边界、`_FakeRewardModel` factory）+ kling `_Fake*`（多 GB Qwen2-VL，无 tiny Hub repo）+
  `_FakePaddleOCR`（heavy 外部 OCR 引擎）——全是外部 transport/网络/引擎边界。

**B. 行为 / 等价探针（真实模型会摧毁断言）**
- `_RecordingTransformer + _Adapter`（`test_backbone_contract.py`）——`**kwargs`（不重声明签名），钉
  `transformer_calls` metric 与 batched-vs-separate-CFG 路由契约（唯一覆盖 `DiffusionBackboneOutput.metrics`）。
- `_StubKVModel/_StubHFTrunk/_StubHFModel`（`test_torch_attention_backend.py`）——per-row 确定性算术钉
  batched-step == per-row-step 位等价（NextStep 仍跑 per-branch step）；`_StubHFTrunk` 是 call-contract recorder。
- janus `_StubVQ`（`decode_code` 返回零张量）——frozen VQ decoder 边界，wrapper 只读 shape；在三个主测试里
  连 `decode_code` 都没被调，只满足 `__init__` attr 守卫。
- `_PolicyWrapperGuard`（`test_offline_dpo_timesteps.py`）——red-line guard 叠在**真实** `build_tiny_wan_transformer`
  上，是 AR 该收敛的参照模式（本身已是黄金标准）。
- `_FakeScheduler`（`test_diffusion_flow_matching.py`）+ `_FakeHead`（`test_ar_flow_matching.py`，`__call__`
  故意 raise 守 `.net`-vs-forward 契约）——纯 math 契约探针。
- continuous-orchestration 的 `asyncio.sleep(0.05)` "progress pauses"——event-gated（`asyncio.Event`），
  非 wall-clock flaky。

**C. 协议一致性测试（钉 refactor 不变量 / 序列化边界）**
- `test_replay_model_contract` vs `test_runtime_model_contract`——两个不同协议 + 两个 `require_*` 守卫，
  错误信息与结构不变量互不重叠。
- `test_capabilities` round-trip——hand-written 自定义序列化器的跨进程 wire format，钉 derived `profiler_label`
  fallback（唯一覆盖）。
- `test_logging`——handler 去重 + propagate + `.3f` 格式行为（唯一覆盖，47 处生产使用）。

**D. AR 不迁 tiny-real（保留 stub，只去重）——逐路径真实理由**
- **NextStep**：`NextStep.from_pretrained` / `NextStepPipeline` 来自 `nextstep_model`/`gen_pipeline`，
  全仓不可导入（`ModuleNotFoundError`）、未声明依赖、无 in-repo config-init。
- **Janus replay 路径**：`MultiModalityCausalLM` **有** config-init（标准 HF `PreTrainedModel`，
  `JanusProReplayCore` 已用 `LlamaForCausalLM(config.language_config)` 构造），但 `janus` 包**未在
  `pyproject.toml` 声明**（仅本机 editable install），CI clean-install 缺失——所以 trunk 迁移虽可行但会破
  CI。（自动结论"import janus fails / config-init impossible"是错的，真实理由是"未声明依赖"。）
- **Janus decode/paged 路径**：`_RecordingLM`/`_RecordingPagedBackend` 是**行为注入** recording double
  （写 `hidden[:,-1,0]=10+call_index`，测试断言精确张量），真实模型会**移除**这层覆盖。且其针对的完整
  `MultiModalityCausalLM` 表面需要 vision tower，非 config-init-trivial。

**E. 已是黄金标准 / 已正确的行为测试（不动，作为模板）**
- `tests/models/diffusion/fixtures.py` + `registry.py`——可 import 的 builder（比 conftest fixture 更灵活、
  可参数化 seed/geometry），8 个文件跨家族 import，offline-skip 干净。**这是全套该收敛的参照。**
- `tests/config/test_schema.py`——每个测试驱动 `parse_config`/`DataConfig` 验证行为（test-injected 输入触发
  code path），零 fake，是配置测试该长的样子。
- `test_all_experiments_load_and_validate` / required-field / `???` mandatory-marker——value-agnostic 安全网
  （presence + validate），git-modified YAML 必须过它，不会随 lr/save_freq 编辑腐烂。
- ~~`test_vae_decode_memory` 的 `frozen_offload modules=['vae']`~~——该用例已随 frozen_offload 撤销删除
  （2026-06-13）：`model.memory` 现仅 `vae_decode` 一个 section，无跨 policy ownership 可测。
- gated e2e 真权重套件（`test_real_checkpoint_rl.py`）——三重 gate（`WM_RUN_REAL_MODEL_TESTS` + CUDA+memory +
  network-free cached snapshot），是真数值层，**不要试图 tiny 化**。
- `test_tiny_pipeline_wiring`——~1MB tiny-pipe 真实 `from_pretrained` 装配，config-init fixture 覆盖不了的
  廉价真实加载层。
- `test_scheduler_logprob_parity`——真实 per-family scheduler parity 矩阵（AR 排除正确，无 SDE replay 路径），
  是 parity 维度该收敛的模板（除 T2.2 的 clean-CI no-op 问题）。
- `test_replay_export_alignment`（predict2）——纯算术 export 测试（`object.__new__`，无 transformer），
  不该硬塞 fixture。
- gpu auto-skip + e2e/distributed/optional gating 接线（已验证正确）；per-test seeding（局部、可 grep、
  无跨测试耦合，**不要**引入全局 autouse seed fixture）；heavy deps lazy import（collection 1.9s 内无错）。

**F. 通用非目标**
- 不为清理而清理：只动有明确 rot 风险（手维护漂移列表、重声明签名 fake）或 no-exact-config 违规的测试。
- 不 flatten thin 抽象（registry 懒加载、cross-family 统一形状）。
- 不把 diffusion tiny-real fixtures 与 AR stub-builder 合并成一个模块。

---

## 11. 执行顺序

按杠杆排（轨道内 item 已标 risk/effort）：

1. **T1** — 建 `tests/models/ar/fixtures.py`，去重 janus stub 结构脚手架（T5.2/T5.1 AR 部分的前置）。
2. **T4** — 配置断言重写：先 `_ONLINE_RECIPES`/`_RECIPES` 从 glob 派生（T4.1，零风险、找回漏测）、
   实验名集合结构化（T4.2）、reward kwargs 子集校验（T4.3）、dispatch 字面 pin 删除（T4.4）。
3. **T3** — 删 `test_prompt_dataset_configs.py` 的 3 个字面值测试 + dataset 测试重写成 per-group smoke
   （T3.1/T3.5 合并、T3.2 通用化、T3.3 删）。
4. **T2** — anima fake→tiny + 补 backbone parity（T2.1）；scheduler parity tiny-repo（T2.2）。
5. **T5** — RuntimeModel 契约参数化跑真实家族（T5.1）；replay-export 对齐跨家族（T5.3）；cosmos/wan-i2v
   `from_build` 加载测试（T5.4）；AR parity gap 文档化（T5.2，track 项）。
6. **T6** — infra 卫生：marker 注释（T6.1）、gpu docstring 软化（T6.2）、`_wait_until` 去重（T6.3）、
   pipeline-contract Ray spin-up 优化（T6.4）。

每个 item 改完跑对应 `pytest` 子集 + 必要时 config-resolve，确认零回归再 PR。

---

## 12. 校验记录（对自动审计结果的独立复核与关键反转）

逐个真实读源 + 跑测试后，本文已把以下 6 处最关键的反转折叠进对应 item，记录于此以便复核：

1. **AR fixtures 单 builder 提案 → 只 canonical 化结构脚手架**（T1.1）：4 个文件的 `language_model.forward`
   行为各不相同且都是承重 instrumentation（scripted KV 值 / `calls==[]` / identity trunk / `.logits`），
   两个布尔无法在不变 god-builder 的前提下吸收，本地注入必须保留。
2. **anima parity CFG 公式**（T2.1）：是 `uncond + g*(cond-uncond)`，**不是** predict2 的
   `cond + g*(cond-uncond)`；`timestep==0.7` 钉 sigma 原值直传，**非** EDM→timestep 转换。照抄 exemplar 会引 bug。
3. **scheduler parity severity**（T2.2）：EDM 转换分支在 clean CI 已被合成 math 测试无条件覆盖；真正缺的是
   真实 sigma 表（已 e2e 冗余）——是"误导性 no-op comment + 真实表缺口"，非"回归零覆盖"。
4. **reward kwargs allowlist 兜底方案错误**（T4.3）：派生"defaults 含 video-reward group"集合是 7 个 ≠ 字面
   6 个，会**放松**断言；应用"内联 keys ⊆ group default keys"子集校验。
5. **`_wait_until` 目标文件**（T6.3）：应沉到 `_helpers.py`（既有约定，`tests/trainers/online/_helpers.py`
   为先例），**不是** `conftest.py`——只装一个 helper 的 conftest 本身才是 "new lean file"。
6. **pipeline-contract 2.24s 诊断**（T6.4）：实测**不是**一次性 ray import，而是 `worker_metadata()` 里
   `ray.get_gpu_ids()` **起了本地 Ray 实例**；优化杠杆在本测试逻辑（不调 live-ray metadata），非"ray import"。

其余被复核确认无误的关键判断：anima 是 diffusion 唯一未迁的签名 fake（可干净迁）；AR stub 4 文件重复且
定义漂移属实；`test_prompt_dataset_configs` 是逐行 YAML 复制；`_ONLINE_RECIPES` 已漂移漏 1 条；
RuntimeModel 契约从不跑真实家族；A/B/C/D/E 类 fake 全部为合法边界 / 行为探针 / 黄金标准，保留。

---

## 关键文件引用

**黄金标准（参照，不动）**
- `tests/models/diffusion/fixtures.py`（`build_tiny_*` + `add_lora_adapters` + `record_forward_calls:167`）
- `tests/models/diffusion/registry.py:7,48-68`（`load_tiny_pipeline` + Cosmos/Wan-I2V 无 tiny-repo 注释）
- `tests/trainers/test_offline_dpo_timesteps.py:103-164`（`_PolicyWrapperGuard` 叠真实 tiny transformer）
- `tests/models/diffusion/cosmos/predict2_5/test_backbone_parity.py`（parity exemplar）

**T1/T5 AR**
- `tests/models/ar/janus_pro/test_kv_decode.py:25-103`、`test_replay.py:31-77`、`test_r1_model.py:64-115`、
  `tests/generation/ar/test_janus_paged_attention_one_step.py:89-143`（4 份 mmgpt stub）
- `vrl/models/ar/janus_pro/model.py:193-199`（`__init__` attr 守卫）、`:1070-1102`（`JanusProReplayCore` config-init）、
  `:1168`（`AutoModelForCausalLM.from_pretrained trust_remote_code`）；`runtime.py:108`（`replay_modules`）
- `tests/models/interfaces/test_runtime_model_contract.py:126-134`、`test_replay_model_contract.py:108-116`；
  `tests/rollouts/runtime/test_family_registry.py:18-30`（家族枚举）

**T2 anima / scheduler**
- `tests/models/diffusion/cosmos/anima/test_forward_step.py:11-35`（`_ConstantAnimaTransformer`）
- `vrl/models/diffusion/cosmos/anima/model.py:294-332`（forward_step + CFG `uncond+g*(cond-uncond)` @ :326）、
  `:470-502`（`CosmosTransformer3DModel.from_config`）
- `tests/models/diffusion/test_scheduler_logprob_parity.py:16-18,41-44,56-65`；
  `tests/math/test_diffusion_flow_matching.py`（合成 EDM 表，clean CI 覆盖）；`vrl/math/diffusion/flow_matching.py:91-114`

**T3/T4 config**
- `tests/config/test_prompt_dataset_configs.py:14-101`（删/重写）
- `tests/config/test_load_all_experiments.py:114-138,141-164,167-176,179-194,241-251`（实验名集 / allowlist /
  单 component / 全实验循环 / dispatch pin）
- `tests/config/test_precision.py:135,155-167`、`tests/scripts/test_online_precision_bridge.py:20`（`_ONLINE_RECIPES`/`_RECIPES`）
- `tests/config/test_schema.py`（行为测试参照）；`vrl/config/schema.py:52-53,87-89,118,148-206`、`vrl/config/builders.py:188-222`

**T5 parity / coverage**
- `tests/models/diffusion/cosmos/predict2/test_replay_export_alignment.py:44-59`；predict2.5/anima export 对齐
  （`model.py` 的 `align_replay_tensor`/`_align_replay_tensor`）
- `vrl/models/diffusion/cosmos/predict2/model.py:131`、`vrl/models/diffusion/wan_2_1/model.py:402,430-438`
  （`from_pretrained` 装配 / offload 分支）；`sd3_5/test_model_loading.py:11`（monkeypatch-from_pretrained 模板）

**T6 infra**
- `pyproject.toml:110-112`（marker）、`tests/conftest.py:5-10,46-67`、`.github/workflows/ci.yml:24-58,49,77`
- `tests/rollouts/orchestration/continuous/test_contracts.py:106-112`、`test_schedule.py:307-310`（`_wait_until`）；
  `tests/trainers/online/_helpers.py`（`_helpers.py` 先例）
- `tests/generation/pipeline/test_pipeline_contracts.py:196-216`；`vrl/generation/ray/stage_worker.py:48-61`、
  `vrl/ray/dependencies.py:42-44`（`ray.get_gpu_ids` 起本地 Ray）
- `tests/rewards/ray/test_runtime.py:60-205` 等（Ray per-test spin-up）；`vrl/ray/runtime.py:84`（已有复用守卫）
