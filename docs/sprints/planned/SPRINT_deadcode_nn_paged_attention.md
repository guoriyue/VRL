# SPRINT: `nn` paged-attention 死字段级联清理（planned）

状态：**RECONCILED（2026-07-24）**，对齐 main @ `7c748532`（= `origin/main` tip，自审计基线 `88ed756e` 起落地约 63 个 cleanup/refactor commit）。本次复核结论：**10 条全部仍需做（STILL_VALID）**，其中 **3 条零变动**（`free`、`debug_info` 三兄弟、`Fp4Linear.recipe`）、**7 条 RELOCATED**（字段定义未变，仅 caller/test 行号因 token 重构下移）；**0 条已由 origin 落地**、**0 条情况已变**。风险分布不变：**8 low + 2 medium**（medium：`ARAttentionStepInput.position`、`VllmDecoderPagedSequenceState.branch`）。全部来自 `vrl/nn/` paged-attention 路径及其两个 caller（`paged_attention_helpers.py` / `nextstep_1/runner.py`）与 `vrl/nn/quantization/fp4.py`。

> **执行状态（2026-07-24）**：全部 10 条已落地 `c5046266`（分页注意力死字段级联 + `_ar_config` model 参数收敛）。

**复核要点（一句话）**：审计跑在旧树 `88ed756e`，但本簇触及的 4 个生产文件（`vrl/nn/layers/attention/paged.py`、`vrl/nn/modules/ar_decoder.py`、`vrl/nn/kernels/attention/vllm_paged.py`、`vrl/nn/quantization/fp4.py`）**自基线起字节未变**（`git log 88ed756e..HEAD` 对这几个文件为空），所有死判在同一行号仍成立；唯一的漂移是两个 caller 文件（`paged_attention_helpers.py`、`nextstep_1/runner.py`）与 `test_janus_paged_attention_one_step.py` 的 token 重构使删除动作里的构造 kwarg / 断言行号整体下移——本文档已逐条更新为 `7c748532` 上的当前行号。

来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）.
关联：[[SPRINT_deadcode_00_overview]]；本 sprint 推翻 [[SPRINT_small_function_consolidation]]（A5 对 `ARAttentionBackend.free` 的 "protocol boundary KEEP" 判定，见 §1.1）；承接 [[SPRINT_grab_bag_file_audit]]（`Sequence` import 仅因 `free` 保留，见 §1.1）；判例参照 [[SPRINT_diffusion_request_branch_dead_fields_cleanup]]（删除同形 write-only `DiffusionBranch.name`，见 §1.8）；格式范本 [[SPRINT_trajectory_views_types_dead_fields_cleanup]].

## 0. 一句话

这是一条 paged-attention 数据结构上的**紧耦合死字段级联**：`ARAttentionConfig` / `ARAttentionPrefillInput` / `ARAttentionStepInput` / `VllmDecoderPagedSequenceState` 上有一整串"只被日志字典读、或只被自校验读、或根本零 reader"的字段，删一个会连带删掉喂养下一个的 writer。主导 kind 是 **dead-field**（8/10），最锋利的一条是 `VllmDecoderPagedSequenceState.branch`：它的唯一去向是 `_pack_step` 把 `request.branch_names` zip 进 `next_states.branch`，而 `next_states.branch` 从不被任何控制流读取——整条 `branch_names → state.branch` 消费图终结于一个无人读的字段，删掉源头字段即可拆掉整条链。误删风险集中在两处**同名字段碰撞**：`ARAttentionPrefillInput.branch`（真活，`ar_decoder.py:157` 用来生成出现在 KV 耗尽 `RuntimeError` 里的 sequence id）必须与死的 `state.branch` 区分；`ARAttentionStepInput.position/row_indices` 必须与各家族**自有** `TokenStepBatch.position/row_indices`（大量活 reader）区分——本 sprint 已逐 receiver 消歧，删除动作只碰 paged-attention 类型。

**整簇按 ONE PR 落地**：这些字段互为 writer/reader，分批提交会在中间态留下 `TypeError`（frozen+slots dataclass 收到已删字段的 kwarg）或 F401（孤儿 import）。下方 §1 给出依赖序删除顺序。

**与在飞 sprint 的关系（本次复核重核）**：in-flight 的 [[SPRINT_native_generation_engine_program]] 未提交改动集中在 `vrl/generation/ray/`、`vrl/ray/`、`vrl/models/steps/denoise/base.py`；本簇触及的文件**无一在该 sprint 的 worktree 修改集内**，该 sprint 文档也不 stage 任何 `free`/`debug_info`/`branch_names`/`sequence_ids`/paged-state 字段的未来用法，故本簇可独立落地，无需 "sequence after the sprint" 排序。

### 依赖序删除顺序（reviewer 友好，实际一次提交）

1. **`debug_info` 三兄弟（§1.2）先删** —— 它们是 `model_key` / `cache_layout_version` 的唯一 reader。
2. **`ARAttentionConfig.model_key` + `cache_layout_version`（§1.3）** —— debug_info 一走即零 reader，本 PR 走 model_key **全删** 分支（选项 A）。
3. **`ARAttentionStepInput` 上的 `metadata`/`row_indices`/`position`（§1.5–1.7）** —— 三个 write-only / test-only 字段，删字段 + 删各构造 kwarg。
4. **`VllmDecoderPagedSequenceState.branch`（§1.8）先于 `branch_names`** —— 删 `state.branch` 使 `_pack_step` 里 `zip(states, request.branch_names)` 的 `branch` 产物失去去向，随即删 zip，`branch_names` 遂零 reader，再删 `ARAttentionStepInput.branch_names` 及其 writer。
5. **`VllmDecoderPagedSequenceState.row`（§1.9）** —— 与 branch 同类 self-copy，一并删。
6. **`ARAttentionPrefillInput.sequence_ids`（§1.4）** 与 **`ARAttentionBackend.free`（§1.1）** 独立，任意顺序。
7. **`Fp4Linear.__init__ recipe`（§1.10）** 完全独立文件，可最后单独审。

## 1. 待删清单（仍有效）

> 本节保留全部 10 条 STILL_VALID/RELOCATED finding。medium-risk 两条（§1.7 `position`、§1.8 `branch`）重点标注；其余按上面的依赖序编排。RELOCATED 条目的「位置」行给出 `7c748532` 上的当前行号，并标注行号相对基线 `88ed756e` 的移动（字段定义本身未变）。

### 1.1 `ARAttentionBackend.free` — dead-function（form 1，risk=low）· **STILL_VALID（零变动）**
- 位置：`vrl/nn/layers/attention/paged.py:119-120`（文件字节未变，同基线）
- 判死证据：
  - `grep -rn '\.free(' vrl/ tests/ --include='*.py'` → **ZERO**（无任何调用点）。
  - `grep -rn 'def free' vrl/ tests/ --include='*.py'` → 仅 `paged.py:119`；另一命中是前缀误配 `vrl/models/steps/denoise/common/lora.py:171 freeze_checkpoint_owned_adapter_params`（`def free` 是 `freeze` 的前缀，不同符号）。无 override、无 `getattr`/字符串 dispatch。
  - 函数体：`def free(self, sequence_states): del sequence_states` —— 纯 no-op 默认，两个具体 backend（`TorchNativeDecoderAttentionBackend`、`VllmDecoderPagedAttentionBackend`）与测试 fake `_RecordingPagedBackend` 均不 override，整个栈无任何 `free/release/cleanup/__del__/close` teardown 路径。
  - `grep -n 'Sequence' vrl/nn/layers/attention/paged.py` → 仅 `:9` import 与 `:119` free 签名——删 `free` 后 `Sequence` 成孤儿 import。
- 动作：删 `paged.py:119-120` 的 `free`；把 `paged.py:9` 的 `from collections.abc import Mapping, Sequence` 收缩为 `from collections.abc import Mapping`（`Mapping` 仍被 `ARAttentionConfig.extra`、`debug_info` 返回类型、metadata 字段使用，保留）。无测试引用 `free`，无测试清理。删后 `ruff check vrl/nn/layers/attention/paged.py` 须无 F401。
- 注意（推翻先例）：[[SPRINT_small_function_consolidation]] 记有 "`ARAttentionBackend.free` 按 protocol boundary KEEP"，理由是"具体 backend 可覆盖"的**假想 override**论证——与同一 sprint 已删除的假想 `set_epoch` 钩子同形。在当前 consumer-is-source-of-truth 规则下，一个既无 implementer 又无 caller 的契约方法不是 boundary。commit / sprint note 须显式记录本删除**推翻**该 KEEP 判定；同时记 [[SPRINT_grab_bag_file_audit]] 中 "`Sequence` 仅因 `free` 保留" 的前提随之失效。

### 1.2 `ARAttentionBackend.debug_info`（+ `VllmDecoderPagedAttentionBackend.debug_info` + `VllmPagedAttentionKernels.debug_info`）— dead-function（form 1，kernels 变体 TEST-ONLY，risk=low）· **STILL_VALID（零变动）**
- 位置：`vrl/nn/layers/attention/paged.py:122-127`、`vrl/nn/modules/ar_decoder.py:121-128`、`vrl/nn/kernels/attention/vllm_paged.py:233-242`（三文件均字节未变）
- 判死证据：
  - `grep -rn 'debug_info' vrl/ tests/ --include='*.py'` → 恰 6 命中：`paged.py:122`（base def）、`vllm_paged.py:233`（def）、`ar_decoder.py:121`（override def）/`:123`（`super().debug_info()` 内部自调）/`:127`（`self.kernels.debug_info()` 内部自调）、`tests/nn/kernels/test_vllm_paged_attention_import_gate.py:54`（唯一外部 caller）。
  - `ar_decoder.py:121` 的 override **零 caller**；其内部两处调用只喂自己。唯一外部 caller 是 import-gate 测试的一条断言 `assert kernels.debug_info()["attention_kernels"] == "vllm_paged_attention_kernels"`（`:54`）—— TEST-ONLY = 死。第三个 subclass `TorchNativeDecoderAttentionBackend` 既不 override 也不调用。三个函数体都是纯 dict builder（log/display，无副作用）。
  - `grep -n 'Mapping' vrl/nn/kernels/attention/vllm_paged.py` → 仅 `:6` import 与 `:233` debug_info 返回类型——删 debug_info 后 `Mapping` 成孤儿 import。
- 动作：删三个 `debug_info`；删 `tests/nn/kernels/test_vllm_paged_attention_import_gate.py:54` 断言（相邻 `:53 assert "vllm.v1.attention.backends.flash_attn" in kernels.modules` 已行为性覆盖 import gate，无需替代断言）；把 `vllm_paged.py:6` 的 `from collections.abc import Callable, Mapping` 收缩为 `from collections.abc import Callable`。`paged.py` 与 `ar_decoder.py` 的 `Mapping`/`Any` import 有其它活用途，保留。删后对四个文件跑 ruff check/format。
- 注意：这是级联头——`debug_info` 走后，`model_key`（§1.3）与 `cache_layout_version`（§1.3）失去唯一 reader，故 §1.3 走全删分支。

### 1.3 `ARAttentionConfig.model_key` + `ARAttentionConfig.cache_layout_version` — dead-field（dead-field-rule，risk=low）· **RELOCATED（仅 1 处 test setter 行号 +1）**
- 位置：`vrl/nn/layers/attention/paged.py:25`（`model_key`）、`:27`（`cache_layout_version`）（`paged.py`/`vllm_paged.py` 字节未变）
- 判死证据：
  - `grep -rn 'cache_layout_version' vrl/ tests/` → 仅定义 `paged.py:27` + 两个 reader `paged.py:126`、`vllm_paged.py:239`，二者都在 `debug_info` dict 内；**无任何构造** 传 `cache_layout_version=`（生产构造 `_ar_config` 与全部测试构造都省略，取默认 `"vllm.v1"`）。
  - `grep -rn 'model_key' vrl/nn/ vrl/models/ tests/`：生产唯一构造点 `ar_attention_backends.py:129 model_key=str(getattr(getattr(model, "config", None), "model_path", family))`（未变）；reader 仅 `paged.py:33-34`（自校验 `if not self.model_key: raise`）、`paged.py:125` 与 `vllm_paged.py:237`（两处 debug_info）；测试 reader 仅 `tests/nn/modules/test_torch_attention_backend.py:227 assert backend.config.model_key == "stub-model"`。`debug_info` 一删（§1.2），`model_key` 除自校验外零 reader。
  - `grep -rnE 'asdict|dataclasses\.fields|vars\(' vrl/nn/` → 无（无反射/序列化路径能读写这两字段）；`grep -rn 'cache_layout' vrl/config/ docs/sprints/{planned,parked}/` → 无 YAML knob、无 planned 用法。
- 动作（本 PR 因 §1.2 删 `debug_info`，走 **model_key 全删** = 选项 A）：
  - `cache_layout_version`：删字段 `paged.py:27` + 两 reader 行 `paged.py:126`、`vllm_paged.py:239`（无论 debug_info 是否删都安全，无测试引用）。
  - `model_key`：删字段 `paged.py:25`、自校验 `paged.py:33-34`、reader 行 `paged.py:125` 与 `vllm_paged.py:237`、生产 kwarg `ar_attention_backends.py:129`；删全部测试 setter/reader：`tests/nn/modules/test_torch_attention_backend.py:34` 与断言 `:227`、`tests/nn/kernels/test_vllm_paged_attention_import_gate.py:28` 与 `:49`、`tests/nn/kernels/test_vllm_paged_attention_real_ops.py:22`、`tests/nn/layers/test_paged_attention_contract.py:20` 与 `:22`、`tests/nn/modules/test_ar_decoder_module_contract.py:79`、`tests/generation/bindings/token_autoregressive/test_janus_paged_attention_one_step.py:52`（**行号已从基线 `:51` 移到 `:52`**，该测试文件被 token 重构轻微上/下移；其余 8 处 model_key test 行号在 `7c748532` 上均不变）。
- 注意：**不要**对 `model_key` 用 "annotate display/provenance-only" 选项——debug_info 一删，`model_key` 连 display reader 都没有（自校验不算 consumer），标注只会保住一个纯死字段。`model_key` 是无默认值的 required 字段，删字段却不删这 9 处构造会在每个构造点抛 `TypeError`，故测试清理是硬性的。选项 B（保留 model_key）仅在 debug_info 删除被推翻时才适用，本 PR 不走。

### 1.4 `ARAttentionPrefillInput.sequence_ids` — dead-field（form 2：live reader，dead semantics，risk=low）· **RELOCATED（构造点行号下移，动作不涉及 test 清理）**
- 位置：`vrl/nn/layers/attention/paged.py:46, 53-56`（`paged.py`/`ar_decoder.py` 字节未变）
- 判死证据：
  - `grep -rn 'sequence_ids' vrl/ tests/ --include='*.py'` → 仅定义 `paged.py:46` + 自校验 `paged.py:53-55` + 唯一 reader `ar_decoder.py:156 sequence_ids = request.sequence_ids or tuple(f"{request.branch}:{row}:{...}")`（`:174 sequence_id=str(sequence_ids[row])` 使用）。测试 / YAML / pyproject **零命中**。
  - 全部 `ARAttentionPrefillInput` 构造**都用 keyword 且无一传 `sequence_ids=`**；两个生产构造点**行号已下移**：`paged_attention_helpers.py:240`（基线 `:249`）、`nextstep_1/runner.py:206`（基线 `:222`）——默认 `()` 恒 falsy，故 `request.sequence_ids or ...` 的真值分支不可达，reader 退化为生成 id 兜底，正是 form 2 的失败形态。parked [[SPRINT_ar_shared_prompt_prefill]] 规划 KV fan-out/共享前缀页，从不引入 caller 供给的 sequence id。
- 动作：删 `sequence_ids` 字段（`paged.py:46`）、删 `__post_init__` 长度校验（`paged.py:53-56`）、把 `ar_decoder.py:156` 的 `request.sequence_ids or ...` 无条件内联为生成的 `f"{branch}:{row}:{n}"` tuple（`:174` 的 `str(...)` 包裹变冗余，因生成 id 已是 str）。无测试清理。
- 注意（同名碰撞）：done 的 [[SPRINT_ar_step_result_dead_fields_cleanup]] 里 `ARStepResult/ARStepBatch.sequence_ids` 是**另一个已删类型**（commits 2aa0254/f5ff639），与本字段无关，勿混。

### 1.5 `ARAttentionStepInput.metadata`（+ prefill metadata 的 `"family"` 键）— dead-field（form 1 applied to data + dead metadata key，risk=low）· **RELOCATED（两 caller 行号下移）**
- 位置：`vrl/nn/layers/attention/paged.py:80`（`paged.py`/`ar_decoder.py` 字节未变）
- 判死证据：
  - `grep -rn 'request\.metadata|step\.metadata|prefill\.metadata' vrl/nn/` → 唯一命中 `ar_decoder.py:155 self._max_new_tokens_from_metadata(request.metadata)`，是 **PREFILL** 路径读 `"image_token_num"`；两个 backend 的 step 体都不读 `request.metadata`。
  - `grep -rnF 'metadata["family"]' vrl/ tests/` → **ZERO**；`.get("family"` 在 `vrl/nn/` 无命中（其余 `.get("family"` 命中都是 checkpoint/config payload，非 attention metadata）。
  - step metadata 的两个 writer（**行号已下移**）：`paged_attention_helpers.py:300 metadata={"family": self.family}`（基线 `:317`）、`nextstep_1/runner.py:255 metadata={"family": "nextstep_1"}`（基线 `:271`）——写一次、无人读。prefill metadata 里 `"family"` 键同样零 reader（现已与 `"image_token_num"` 合并到单行 dict）：`paged_attention_helpers.py:244 metadata={"family": self.family, "image_token_num": image_token_num}`（基线 `:253`）、`nextstep_1/runner.py:210 metadata={"family": "nextstep_1", "image_token_num": image_token_num}`（基线 `:226`），`"image_token_num"` 才是活键。
- 动作：删 `ARAttentionStepInput.metadata` 字段（`paged.py:80`）+ **两个** step writer（`paged_attention_helpers.py:300`、`nextstep_1/runner.py:255`）；从两处 prefill metadata dict（`helpers:244`、`runner:210`）删死键 `"family"`，保留 `"image_token_num"`（现为同行 dict，删键即可）。**保留** `ARAttentionPrefillInput.metadata` 本身——它真被 `ar_decoder.py:155` 读。无测试清理（无测试传 step `metadata=` 或断言 step metadata）。
- 注意（构造点易漏）：`ARAttentionStepInput` 是 `@dataclass(frozen=True, slots=True)`，删字段却漏删 `nextstep_1/runner.py:255` 的 writer 会在 nextstep_1 首次 paged decode step 抛 `TypeError: unexpected keyword argument 'metadata'`。两个 step writer 都要删。

### 1.6 `ARAttentionStepInput.row_indices` — dead-field（form 1 applied to data：zero readers，risk=low）· **RELOCATED（两 producer 行号下移）**
- 位置：`vrl/nn/layers/attention/paged.py:79`（`paged.py` 字节未变）
- 判死证据：
  - `grep -rn 'row_indices' vrl/nn/ ...`：`ARAttentionStepInput.row_indices` 只有定义 `paged.py:79` + 两个 producer（**行号已下移**）——`paged_attention_helpers.py:299 row_indices=tuple(batch.row_indices + batch.row_indices)`（基线 `:316`）、`nextstep_1/runner.py:254 row_indices=tuple(batch.row_indices * (2 if has_uncond else 1))`（基线 `:270`）。
  - `grep -rn 'row_indices' vrl/nn/modules vrl/nn/kernels` → **无**：两个 backend 的 step 体都不读 `request.row_indices`；`__post_init__` 也不校验它。
  - 其余全部 `row_indices` 命中都是**各家族自有** `TokenStepBatch.row_indices` / lane 类型（`nextstep_1/runner.py`、`paged_attention_helpers.py`、`cache_rows.py` 局部变量）——真活的 select/scatter 逻辑，勿动。
- 动作：删 `ARAttentionStepInput.row_indices` 字段（`paged.py:79`）+ **两个** producer kwarg（`paged_attention_helpers.py:299`、`nextstep_1/runner.py:254`）。无测试清理（无测试对 `ARAttentionStepInput` 传 `row_indices=`）。runner 自有 `batch.row_indices` 逻辑保留。
- 注意（两 producer 且语义不一致）：两个 producer 语义竟不同（helpers 无条件 double，nextstep 仅 `has_uncond` 时 double）——这个无人察觉的分歧本身就是"值从不被读"的旁证。同 §1.5，frozen+slots，漏删任一 kwarg 即 `TypeError`。

### 1.7 `ARAttentionStepInput.position` — dead-field（dead-field-rule，TEST-ONLY reader，**risk=medium**）· **RELOCATED（caller + janus 断言行号下移）**
- 位置：`vrl/nn/layers/attention/paged.py:78, 89-90`（`paged.py`/`torch_attention.py`/`ar_decoder.py` 字节未变）
- 判死证据：
  - `grep -rn 'request\.position|step\.position' vrl/ tests/` → 唯一命中 `tests/generation/bindings/token_autoregressive/test_janus_paged_attention_one_step.py:136 assert step.position == 0`（**基线 `:140`，因该测试文件重构上移到 `:136`**）—— TEST-ONLY = 死。
  - `grep -n 'position' vrl/nn/modules/torch_attention.py` → **无**；`VllmDecoderPagedAttentionBackend._pack_step` 从 `state.length` 推 `cache_positions`、从 `state.next_position_id` 推 `position_ids`，从不读 `request.position`。
  - 全部 `position=` 构造点：`paged_attention_helpers.py:298`（基线 `:315`）、`nextstep_1/runner.py:253`（基线 `:269`）、`tests/nn/modules/test_torch_attention_backend.py:78,141,152,237`（不变）、`test_janus_vllm_paged_attention_backend.py:56`（不变）、`test_nextstep_vllm_paged_attention_backend.py:68`（不变）、`tests/nn/layers/test_paged_attention_contract.py:43`（不变）。
- 动作：删 `position: int` 字段（`paged.py:78`）+ `__post_init__` 校验（`paged.py:89-90`）；删上述全部 8 个 `position=` kwarg（含易漏的 `tests/nn/layers/test_paged_attention_contract.py:43`——该字段无默认值，漏改会把测试变成 `TypeError` 崩溃而非其本意的 sequence_states 校验）；删 test-only 断言 `test_janus_paged_attention_one_step.py:136`。
- 注意（medium：同名碰撞，勿误删）：**不要**碰 `batch.position` / `TokenStepBatch.position` 的任何读点——`vrl/models/steps/token/base.py:160`、`paged_attention_helpers.py:271`、家族 runner、`test_token_loop.py`、`test_token_scheduler.py` 都是**另一个活类型**。原始 auditor action 曾漏掉 `test_paged_attention_contract.py:43`（不在 `test_torch_attention_backend.py` / binding 目录内），本条已补入。

### 1.8 `VllmDecoderPagedSequenceState.branch`（级联至 `ARAttentionStepInput.branch_names`）— dead-field（dead-field-rule + cascade，writer-only chain，**risk=medium**）· **RELOCATED（caller + janus 断言行号下移）**
- 位置：`VllmDecoderPagedSequenceState.branch` 定义 `vrl/nn/modules/ar_decoder.py:29`，writer `ar_decoder.py:175`（prefill）与 `:229`（self-copy relabel）；`ARAttentionStepInput.branch_names` 定义 `vrl/nn/layers/attention/paged.py:77` + 校验 `:87-88`（`ar_decoder.py`/`paged.py`/`torch_attention.py` 字节未变）
- 判死证据：
  - `grep -rn '\.branch\b' vrl/nn/ ...`：`state.branch` 的读点只有 `ar_decoder.py:175`（writer）与 `_pack_step` self-copy（`:229`）；**唯一活的 `.branch` 读** 是 `ar_decoder.py:157 f"{request.branch}:..."`（`ARAttentionPrefillInput.branch`，用于生成出现在 KV 耗尽 `RuntimeError` 的 sequence id）与 `paged.py:51-52`（prefill 输入自校验）——**这两个是 prefill 输入字段，保留**，与死的 `state.branch` 不是一回事。
  - `grep -rn 'branch_names' vrl/ tests/`：reader 仅 `ar_decoder.py:235 for state, branch in zip(states, request.branch_names, strict=True)`（把 branch zip 进 `next_states.branch`）与 `paged.py:87-88`（自校验）。`TorchNativeDecoderAttentionBackend.step` **完全忽略** `branch_names`。整条 `branch_names → zip → state.branch` 消费图终结于无人读的 `state.branch`。
  - caller 机械构造 `["cond"]*B + ["uncond"]*B`（**行号已下移**：`helpers:297`、`nextstep_1/runner.py:226,240,252`），结果按**位置切片** 回收（`helpers` 的 `updated_states[:B]/[B:]`、runner 同形），从不按 branch 名路由——label 不携带路由信息。
  - `grep -rnF '"branch"' / getattr(.*branch / asdict` → 无字符串键/反射路径保活。判例：[[SPRINT_diffusion_request_branch_dead_fields_cleanup]] 删过同形 write-only `DiffusionBranch.name`。
- 动作：
  - 删 `VllmDecoderPagedSequenceState.branch`（`ar_decoder.py:29`）及其 writer（`:175 branch=request.branch`；`_pack_step` 里 `next_states` 的 `branch=branch`（`:229`）及 `zip(states, request.branch_names)`（`:235`）—— `next_states` 变为对每个 state 仅 `length+1`/`next_position_id+1` 的 copy）。
  - 删 `ARAttentionStepInput.branch_names`（`paged.py:77`）+ `__post_init__` 校验（`paged.py:87-88`）。
  - 删构造点（**行号已下移**）：`paged_attention_helpers.py:297` kwarg，`nextstep_1/runner.py:226,240`（局部 list 构建）与 `:252`（kwarg）。
  - 测试清理：删 `branch_names=` kwarg —— `tests/nn/layers/test_paged_attention_contract.py:42`、`tests/nn/modules/test_torch_attention_backend.py:77,140,151,236`、`test_janus_vllm_paged_attention_backend.py:55`、`test_nextstep_vllm_paged_attention_backend.py:67`（这几处**行号不变**）；删断言 reader `test_janus_paged_attention_one_step.py:135 assert step.branch_names == ...`（**基线 `:139`，重构后 `:135`**）；从该文件 `_PagedState` fake 删已变 write-only 的 `branch` 字段（定义 `:45`、prefill 写 `:66`、self-copy 写 `:84`；基线 `:44/:65/:83`）。**保留** `test_janus_paged_attention_one_step.py:131` 的 `[request.branch for request in backend.prefill_requests]` 顺序检查（**基线 `:135`，重构后 `:131`**，那是活的 `ARAttentionPrefillInput.branch`）。
- 注意（medium：级联 + 同名碰撞 + 断言 reader 易漏）：(1) 删除有严格序——先删 `state.branch` 使 zip 产物失去去向，再删 `branch_names`；(2) `test_janus_paged_attention_one_step.py:135` 是**断言 reader**（非构造），漏删会在删字段后 `AttributeError`；(3) `ARAttentionPrefillInput.branch` 与 `state.branch` 同名但一活一死，务必 receiver 消歧。
- ⚠ 复核偏差（沿用，已再次确认）：finding 的 `lines` 元字段记 `branch` 定义在 `ar_decoder.py:31`、self-copy writer 在 `:231`；实测 `class VllmDecoderPagedSequenceState`（`:25`）下字段序为 `sequence_id:28 / branch:29 / row:30 / length:31 / next_position_id:32 / block_ids:33`，self-copy 的 `branch` writer 实为 `:229`。即 `branch` 定义为 `:29`（action 正文的 `:29` 正确，仅 `lines` 头字段的 `31` 有偏差）。核心死判（`state.branch` 零生产 reader）不受影响。

### 1.9 `VllmDecoderPagedSequenceState.row` — dead-field（dead-field-rule，only reader is a self-copy，risk=low）· **RELOCATED（仅 janus fake 行号 +1）**
- 位置：`VllmDecoderPagedSequenceState.row` 定义 `vrl/nn/modules/ar_decoder.py:30`，writer `:176`（prefill）与 `:230`（self-copy）（`ar_decoder.py` 字节未变）
- 判死证据：
  - `grep -rn '\.row\b|\brow=' vrl/ tests/`（排除 `row_indices`/`row_idx`/log 串）→ `state.row` 的唯一非写读点是 `_pack_step` self-copy `ar_decoder.py:230 row=state.row`，外加 test fake 的同形 copy（`test_janus_paged_attention_one_step.py:85`，**基线 `:84`**）。
  - row 身份是**位置性**的、从不从 state 查：`_build_paged_forward_inputs` 用 `for row_idx, state in enumerate(states)`（`ar_decoder.py:400`），`select_paged_states` 用 `return [states[index] for index in row_indices]`（`paged_attention_helpers.py:71`，按 runner 自有 `row_indices` 索引）。protocol 把 `sequence_states` 类型标为 opaque `tuple[Any, ...]`，无 boundary 命名该字段。
  - 无 `asdict/fields()/vars`、无 `"row"`/`getattr` dispatch、无 YAML（runtime state 非 config）、无 display/provenance 标注。
- 动作：删 `row` 字段（`ar_decoder.py:30`）及其两 writer（`:176 row=row`、`:230 row=state.row`）；更新 `test_janus_paged_attention_one_step.py` 的 `_PagedState` fake（删镜像的 `row` 字段：定义 `:46`、prefill 写 `:67`、self-copy 写 `:85`；基线 `:45/:66/:84`）。两个生产构造点都是 keyword 式且在 `ar_decoder.py` 内，删除安全。
- ⚠ 复核偏差（沿用，已再次确认）：finding 的 `lines` 元字段记 `row` 定义在 `ar_decoder.py:32`；实测为 `:30`（writer `:176`、`:230` 与 finding 一致）。核心死判（`state.row` 唯一非写读点是 self-copy）不受影响。

### 1.10 `Fp4Linear.__init__` 的 `recipe` 参数 — dead-arg（form 2：validation-only，唯一合法值即默认值，risk=low）· **STILL_VALID（零变动）**
- 位置：`vrl/nn/quantization/fp4.py:151-154, 167`（文件字节未变，同基线）
- 判死证据：
  - `grep -rn 'Fp4Linear(' vrl/ tests/`：全部生产 caller（`fp4.py:269 swap 内部`、`quantized_rollout_drift_probe.py:197`、`quantized_linear_benchmark.py:124`）都是 `Fp4Linear(x)` 无 recipe；唯一传非默认值的是 rejection 测试 `tests/nn/quantization/test_fp4.py:195 Fp4Linear(nn.Linear(64,32), recipe="rowwise")`（在 `:193 def test_invalid_recipe_rejected`）。
  - `grep -n 'recipe' vrl/nn/quantization/fp4.py`：`:153-154 if recipe != "nvfp4": raise`、`:167 self.recipe = recipe`、`:237 extra_repr` 读 `self.recipe`。函数体从不 branch on recipe（对比 `Fp8Linear` 在 `fp8.py:105,112,149,155` 真按 `self.recipe` 分支的三 recipe）。
  - config 层上游已拒绝 nvfp4 带 recipe：`vrl/config/precision.py:51 "nvfp4": _QuantizationFormatRules(allowed_recipes=(), default_recipe=None)`；`swap_linears_to_nvfp4` 本身无 recipe kwarg。in-progress `SPRINT_nvfp4_rollout.md` 明确 "use `format: nvfp4` without a recipe"，无第二 recipe 计划。
- 动作：删 `recipe` 参数与 `if recipe != "nvfp4": raise` 守卫；`self.recipe = "nvfp4"` 改为普通属性（仍被 `extra_repr`（`fp4.py:237`）与 `quantized_rollout_drift_probe.py:211 print(f"recipe={rollout_head.recipe} ...")` 读，故保留属性）。删 rejection 测试 `tests/nn/quantization/test_fp4.py:193-195`（`test_invalid_recipe_rejected`）。
- 注意：`tests/models/steps/token/test_rollout_quantization.py` 参数化构造 `linear_type(nn.Linear(...))` 不传 recipe，跨方案构造仍有效；`recipe` 不在 `state_dict`（测试只断言 `{"weight","bias"}`）——这两处**无需改**，删后重跑确认绿即可。config 层 `test_fp4.py:283 test_nvfp4_policy_rejects_fp8_recipes` 是 `QuantizationPolicy` 的独立守卫（`:284-285`），与 `Fp4Linear.recipe` 无关，**保留**。

## 2. 已由 origin 落地（本次复核确认，无需再做）

（无）—— 自审计基线 `88ed756e` 起，`origin/main` 的约 63 个 cleanup commit **无一触及本簇的 4 个生产文件**（`paged.py`、`ar_decoder.py`、`vllm_paged.py`、`fp4.py` 对 `git log 88ed756e..HEAD` 均为空 diff），故 10 条死判在同一行号全部保留，零条被提前落地。

## 3. 情况已变（需重新评估）

（无）—— 所有 finding 的死判前提（零 reader / dead semantics / validation-only）在 `7c748532` 上逐条重验通过；唯一变化是 §1.3–§1.9 中 caller 文件（`paged_attention_helpers.py`、`nextstep_1/runner.py`）与 `test_janus_paged_attention_one_step.py` 的**行号漂移**，已在 §1 各条按当前行号更新，并标注基线行号供 diff 对照。**行号漂移不改变任何动作语义**，故不计入"情况已变"。

## 4. 验证协议

- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅跑本任务触及的 Python 文件，禁止仓库级 `ruff format .` / `ruff check --fix .`）。
- **全簇完成后**：
  - `pytest tests/nn/ tests/generation/bindings/token_autoregressive/ tests/models/steps/`（覆盖 paged-attention 契约、两个 backend、janus/nextstep binding、fp4 量化、rollout 量化）不新增失败。
  - `pytest -m "not e2e and not slow_test"` 子集不新增失败。
  - `vrl.config.lint` 与 `ruff check .` 保持全绿。
- **基线（清理前，2026-07-24，main @ `7c748532`）**：删除前须先跑一遍 fast subset + `vrl.config.lint` + `ruff check .` 记录当前 green/pre-existing-failure 数，作为回归基线（审计基线 `88ed756e` 上曾记 2620 passed / 7 pre-existing failures，`7c748532` 已含 63 个后续 commit，须重新取数为准）。删除后这三项须相对**当前基线**不新增失败。
- **本簇触及的测试文件（逐条动作提取，行号为 `7c748532`）**：
  - §1.1 `free`：无。
  - §1.2 `debug_info`：`tests/nn/kernels/test_vllm_paged_attention_import_gate.py`（删 `:54` 断言）。
  - §1.3 `model_key`/`cache_layout_version`：`tests/nn/modules/test_torch_attention_backend.py`（`:34,:227`）、`tests/nn/kernels/test_vllm_paged_attention_import_gate.py`（`:28,:49`）、`tests/nn/kernels/test_vllm_paged_attention_real_ops.py`（`:22`）、`tests/nn/layers/test_paged_attention_contract.py`（`:20,:22`）、`tests/nn/modules/test_ar_decoder_module_contract.py`（`:79`）、`tests/generation/bindings/token_autoregressive/test_janus_paged_attention_one_step.py`（`:52`，基线 `:51`）。`cache_layout_version` 本身无测试。
  - §1.4 `sequence_ids`：无。
  - §1.5 `metadata`：无。
  - §1.6 `row_indices`：无。
  - §1.7 `position`：`tests/nn/modules/test_torch_attention_backend.py`（`:78,141,152,237`）、`test_janus_vllm_paged_attention_backend.py`（`:56`）、`test_nextstep_vllm_paged_attention_backend.py`（`:68`）、`tests/nn/layers/test_paged_attention_contract.py`（`:43`）、`test_janus_paged_attention_one_step.py`（删 `:136` 断言，基线 `:140`）。
  - §1.8 `branch`/`branch_names`：`tests/nn/layers/test_paged_attention_contract.py`（`:42`）、`test_torch_attention_backend.py`（`:77,140,151,236`）、`test_janus_vllm_paged_attention_backend.py`（`:55`）、`test_nextstep_vllm_paged_attention_backend.py`（`:67`）、`test_janus_paged_attention_one_step.py`（删 `:135` 断言 + fake `branch` 字段 `:45,66,84`；基线 `:139` 与 `:44,65,83`）。
  - §1.9 `row`：`test_janus_paged_attention_one_step.py`（fake `row` 字段 `:46,67,85`；基线 `:45,66,84`）。
  - §1.10 `recipe`：`tests/nn/quantization/test_fp4.py`（删 `:193-195` `test_invalid_recipe_rejected`）；`tests/models/steps/token/test_rollout_quantization.py`、`tests/models/steps/denoise/common/test_lora_fp8_build.py` 无需改，重跑确认绿。

## 5. Non-Goals

- 不删被"能 raise 的校验"、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）——本簇明确**保留**：`ARAttentionPrefillInput.branch`（`ar_decoder.py:157` 生成 KV 耗尽错误里的 sequence id + `paged.py:51-52` 校验）、`ARAttentionPrefillInput.metadata`（`ar_decoder.py:155` 读 `image_token_num`）、`ARAttentionConfig.family`/`block_size`/`extra`（backend 行为读）、各家族 `TokenStepBatch.position`/`row_indices`（真活 select/scatter）。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`（显存残留配额）、`ensure_loaded`、`process_gpu_used_bytes`（NVML）、sana/hunyuan `prepare_latents` 修复。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function——`ARAttentionBackend.prefill`/`step` 契约方法、两个 backend 的统一形状、`select_paged_states`/`scatter_paged_states` 保持不动。
- cluster-specific non-goals：
  - **不碰 `Fp8Linear`**——它是三 recipe 真分支（`fp8.py:105,112,149,155`），与 `Fp4Linear` 单 recipe 非对称，`Fp4Linear.recipe` 的删除**不**推广到 fp8。
  - **不改 runner/helpers 自有 `batch.row_indices`/`batch.position`/`branch_names` list 构建之外的路由逻辑**——位置切片回收（`updated_states[:B]/[B:]`）是活逻辑，仅删喂给已删 paged-attention 字段的那几行。
  - **不合并 §1.8 的删除步序**——必须先 `state.branch` 后 `branch_names`，否则中间态留下无去向的 zip。

## References

改动文件与行（file:line，`7c748532`）：
- `vrl/nn/layers/attention/paged.py`：`:9`（import 收缩）、`:25`（model_key）、`:27`（cache_layout_version）、`:33-34`（model_key 校验）、`:46,53-56`（sequence_ids）、`:51-52`（prefill branch 校验，保留）、`:77,87-88`（branch_names）、`:78,89-90`（position）、`:79`（row_indices）、`:80`（step metadata）、`:119-120`（free）、`:122-127`（debug_info）、`:125,126`（debug_info reader 行）
- `vrl/nn/modules/ar_decoder.py`：`:29`（state.branch 定义）、`:30`（state.row 定义）、`:121-128`（debug_info override）、`:155`（prefill metadata reader，保留）、`:156,174`（sequence_ids 内联）、`:157`（prefill branch reader，保留）、`:175,176`（prefill state writer）、`:229`（self-copy branch writer）、`:230`（self-copy row writer）、`:235`（zip）、`:400`（enumerate 位置索引，保留）
- `vrl/nn/kernels/attention/vllm_paged.py`：`:6`（import 收缩）、`:233-242`（debug_info）、`:237,239`（model_key/cache_layout_version reader 行）
- `vrl/nn/modules/ar_attention_backends.py`：`:129`（`_ar_config` 内 `model_key=`）
- `vrl/nn/quantization/fp4.py`：`:151-154`（recipe 参数 + 守卫）、`:167`（self.recipe）、`:237`（extra_repr，保留读）
- `vrl/models/steps/token/paged_attention_helpers.py`：`:240`（prefill 构造）、`:244`（prefill "family" 键）、`:293`（step 构造）、`:297`（branch_names）、`:298`（position）、`:299`（row_indices）、`:300`（step metadata）
- `vrl/models/families/nextstep_1/runner.py`：`:206`（prefill 构造）、`:210`（prefill "family" 键）、`:226,240`（branch_names 局部构建）、`:245`（step 构造）、`:252`（branch_names kwarg）、`:253`（position）、`:254`（row_indices）、`:255`（step metadata）
- `vrl/config/precision.py`：`:51`（nvfp4 无 recipe 上游校验，证据引用）
- `vrl/scripts/perf/quantized_rollout_drift_probe.py`：`:211`（`rollout_head.recipe` 读，保留属性）
- 测试：`tests/nn/kernels/test_vllm_paged_attention_import_gate.py:28,49,54`、`tests/nn/kernels/test_vllm_paged_attention_real_ops.py:22`、`tests/nn/layers/test_paged_attention_contract.py:20,22,42,43`、`tests/nn/modules/test_torch_attention_backend.py:34,77,78,140,141,151,152,227,236,237`、`tests/nn/modules/test_ar_decoder_module_contract.py:79`、`tests/generation/bindings/token_autoregressive/test_janus_vllm_paged_attention_backend.py:55,56`、`tests/generation/bindings/token_autoregressive/test_nextstep_vllm_paged_attention_backend.py:67,68`、`tests/generation/bindings/token_autoregressive/test_janus_paged_attention_one_step.py:45,46,52,66,67,84,85,131,135,136`、`tests/nn/quantization/test_fp4.py:193-195`

关联 sprint：
- [[SPRINT_deadcode_00_overview]]
- [[SPRINT_small_function_consolidation]]（§1.1 推翻其 `free` KEEP 判定）
- [[SPRINT_grab_bag_file_audit]]（§1.1 `Sequence` import 保留前提失效）
- [[SPRINT_diffusion_request_branch_dead_fields_cleanup]]（§1.8 同形 write-only branch 删除判例）
- [[SPRINT_ar_step_result_dead_fields_cleanup]]（§1.4 `sequence_ids` 同名不同类型辨析）
- [[SPRINT_ar_shared_prompt_prefill]]（parked，§1.4/§1.8 确认无 planned reader）
- [[SPRINT_native_generation_engine_program]]（in-flight，已核本簇文件不在其 worktree 修改集，无需排序等待）
- [[SPRINT_trajectory_views_types_dead_fields_cleanup]]（格式范本）
