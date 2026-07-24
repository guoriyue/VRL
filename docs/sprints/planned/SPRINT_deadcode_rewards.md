# SPRINT: `rewards` 死代码清理（planned）

状态：**RECONCILED（2026-07-24）**，对齐 main @ `7c748532`（= `origin/main` tip，自审计基线 `88ed756e` 起累计约 63 个 cleanup/refactor commit）。原始 13 条死代码审计发现，本次复核结果：**10 条仍需处理**（8 条 STILL_VALID + 2 条 RELOCATED，行号已随代码位移刷新）、**3 条已由 origin 落地**（3 个零消费者包 facade `__getattr__`，同一 commit `94761143` 一并删除）、**0 条情况已变**。仍待做的 10 条：1 条 medium（`cosmos3_reasoner` 不可达的 snapshot-download 回退分支）+ 9 条 low（5 个死参数、2 个 test-only 死字段、1 处 form-4 重复实现）。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查），本次对当前 checked-out 树逐条 re-grep 复核。
关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_reward_identity_and_score_keys]]（§1.8 的 `default_revision` 是该 sprint 删掉最后一个非默认 producer 后残留的死参数）、[[SPRINT_reward_service]]（原 §1.13 facade 现已由 origin 收敛，见 §2）、[[SPRINT_cosmos3_full_support]]（parked；§1.1 的 reasoner judge 已 shipped、不计划放开 `model_path` 守卫）。格式范本：[[SPRINT_trajectory_views_types_dead_fields_cleanup]]。

> **执行状态（2026-07-24）**：全部仍有效项已落地 `b7714bdc`（含 cosmos3 死回退改写为 fail-closed）。

## 0. 一句话

本簇清理 `vrl/rewards/` 下的死代码。三个包 `__init__.py` 的零消费者 lazy `__getattr__` re-export **已由 origin 的 `94761143`（"refactor(families): drop package export"）删除**——本次复核确认三个 `__init__.py` 现均只剩 docstring + 空 `__all__`，无需再做（见 §2）。剩余 10 条仍有效：最锋利的一条是 `cosmos3_reasoner.py` 的 `_DEFAULT_REWARD_MODEL` + `resolve_model_root` snapshot-download 回退——`__init__` 里的 `model_path` 守卫会先 raise、`resolve_model_root` 又在 `model_path` 非空时提前 return，两头夹击使这段「下载原始 unified repo」的回退永远不可达（form 2，live caller / dead semantics）；其余是 5 个所有调用点都走默认值的死参数、2 个只有测试在读的死字段、1 处 form-4 重复的 lazy-prepare 序列。误删风险主要在同名兄弟符号（`videoscore2`/`kling` 等各自的 `_DEFAULT_REWARD_MODEL`、`_init_disk_artifact_reward` 的 `media_type`、`_validate_lower_bounded` 的 `inclusive`）对各自家族是活的，一律不动。

## 1. 待删清单（仍有效）

> 本节仅保留本次复核判定 **STILL_VALID** 与 **RELOCATED** 的发现。RELOCATED 两条（§1.9 / §1.10）文件行号已随 origin 的 Kling 协议隔离重排刷新，动作语义不变。

### 1.1 `cosmos3_reasoner._DEFAULT_REWARD_MODEL` / `resolve_model_root` 回退 — dead-branch（risk=medium｜STILL_VALID）
- 位置：`vrl/rewards/models/cosmos3_reasoner.py:50`（常量）、`:45`（import）、`:129-135`（前置 `model_path` 空守卫）、`:140-144`（`resolve_model_root` 调用）；连带 `vrl/rewards/models/hub.py` docstring 的 "Shared by ... cosmos3_reasoner"。
- 复核：行号与原审计一致（50, 129-144）。`_DEFAULT_REWARD_MODEL` 仍在 `:50`；`from vrl.rewards.models.hub import resolve_model_root` 仍在 `:45`；`resolve_model_root(default_model=_DEFAULT_REWARD_MODEL)` 仍在 `:140-144`。空 `model_path` 的 `ValueError` 守卫仍先于 `:129-135` 执行，`hub.py` 在 `model_path` 非空时提前 return，故 default_model/snapshot 回退对本家族仍不可达。
- 判死证据：
  - 读 body：`cosmos3_reasoner.py:129-135` 的 `if not str(self.worker_config.get("model_path", "")).strip(): raise ValueError(...)` 在 `resolve_model_root` 调用之前执行；`hub.py` 的 `model_path` 分支在其非空时提前 `return root`（含 `if not root.exists(): raise FileNotFoundError`）。两头夹击后 `hub.py` 的 `reward_model_name`/`default_model`/`snapshot_download` 回退对本家族永不可达。
  - `grep -rn '_DEFAULT_REWARD_MODEL' vrl/rewards/models/cosmos3_reasoner.py` → 仅 `:50`（定义）与作为 `default_model=` 传入的那个永不被读的实参。
  - 预设 `vrl/config/presets/reward/cosmos3_reasoner.yaml` 设 `model_path: ""`（空 → 命中 `__init__` raise），`reward_model_name: nvidia/Cosmos3-Nano` 仅被通用 `vrl/rewards/base.py:352-368` 读作 `reward_model_version` provenance。
  - `hub.py` docstring 的 "Shared by ... cosmos3_reasoner ..." 明确把本家族列入共享者——留着即复现同一处 false-liveness 文档。
- 动作：`cosmos3_reasoner.py` 删 `_DEFAULT_REWARD_MODEL`；删过期注释；把 `resolve_model_root` 调用替换为对已被守卫校验过的 `model_path` 的直接解析，保留 `hub` 的 model_path-branch 语义含存在性检查（`root = Path(model_path).expanduser().resolve()`，不存在则以 `"Cosmos3 reasoner"` 家族标签 `raise FileNotFoundError`）；删现已无用的 `from vrl.rewards.models.hub import resolve_model_root`。`hub.py` 更新 `resolve_model_root` docstring 从 "Shared by" 列表移除 `cosmos3_reasoner`。**不动**：预设的 `reward_model_name` key、兄弟家族各自的 `_DEFAULT_REWARD_MODEL`、任何测试（无测试引用该常量或构造该模型）。触及测试：`tests/rewards/cosmos3_reasoner/`（仅回归跑，不改）。
- 注意：本条为 medium，reviewer 重点在「回退真不可达」这一 form-2 判断——删除后必须保持 `model_path` 空时 fail-fast（`__init__` raise）与非空但目录缺失时 `FileNotFoundError` 两条行为不变；同名兄弟常量（`videoscore2.py`/`unified_reward_video.py`/`kling_video_reward.py`/`videocon_physics.py` 各自的 `_DEFAULT_REWARD_MODEL`）对各自家族是活的，严禁连坐删除。

### 1.2 `VideoRewardArtifactStore.__init__(manifest_name=...)` — dead-arg（risk=low｜STILL_VALID）
- 位置：`vrl/rewards/artifacts.py:42`（参数定义）、`:57`（唯一内部读）。
- 复核：行号与原审计一致（42, 57）。`grep -rn 'manifest_name' vrl/ tests/` → 仅 `artifacts.py:42`（`manifest_name: str = "manifest.jsonl"`）与 `:57`（`self.manifest_path = self.root / manifest_name`），无任何 caller 传入。
- 判死证据：
  - `grep -rn 'VideoRewardArtifactStore(' vrl tests` → 唯一生产构造点 `vrl/rewards/base.py:334`，加 6 个测试构造（`tests/rewards/inference/test_artifact_store.py`），全部只传 `root`/`media_type`/`artifact_format`。
  - 读 `base.py` `_init_disk_artifact_reward`：keyword-only 签名、无 `**kwargs` 转发，从不转发 `manifest_name`，故无 YAML kwargs 路径可设它。测试都读字面量 `'manifest.jsonl'` 路径。
- 动作：删 `manifest_name: str = "manifest.jsonl"` 参数，在 `:57` 内联字面量（`self.manifest_path = self.root / "manifest.jsonl"`）。无测试改动。

### 1.3 `RewardFunction._init_reward_model(media_type=...)` — dead-arg（risk=low｜STILL_VALID）
- 位置：`vrl/rewards/base.py:266-290`（参数在 `:273`，唯一使用在 `:288`）。
- 复核：行号与原审计一致（266-290）。def 在 `base.py:266`，`media_type: MediaType = "image"` 参数在 `:273`，唯一使用在 `:288` 的 `build_inmemory_artifacts` lambda。两个 caller（`pickscore.py:28`、`aesthetic.py:26`）仍只传 `reward_name`/`score_key`/`model_factory`/`worker_config`，`**kwargs` 折进 `worker_config`。
- 判死证据：
  - `grep -rn '_init_reward_model' vrl tests`（排除 def）→ 仅 `vrl/rewards/functions/pickscore.py:28`、`vrl/rewards/functions/aesthetic.py:26`。两处均只传 `reward_name`/`score_key`/`model_factory`/`worker_config`，`**kwargs` 折进 `worker_config` dict，故非默认值不可达。
  - `media_type` 唯一使用在 `build_inmemory_artifacts(..., media_type=media_type)`；后者本身默认 `"image"`，故硬编码 `"image"` 对两个 live caller 行为等价。
  - in-memory video rewards（`target_dino_similarity.py`、`motion_dynamics.py`、`nsfw_safety.py`、`ocr.py`）绕开本 helper、直接调 `build_inmemory_artifacts(media_type="video")`。
- 动作：删 `media_type: MediaType = "image"` 参数，在其构造的 `build_inmemory_artifacts` lambda 里硬编码 `media_type="image"`。**保留**兄弟 `_init_disk_artifact_reward` 的 `media_type` 参数（那个经 concrete rewards 的 `**kwargs` 转发从 YAML 可达）。无测试改动。

### 1.4 `NSFWSafetyReward._model` — dead-field（test-only reader）（risk=low｜STILL_VALID）
- 位置：`vrl/rewards/functions/nsfw_safety.py:24`。
- 复核：行号与原审计一致（24）。`grep '\._model\b'` → `nsfw_safety.py:24`（`self._model = model`，赋值）与唯一 reader `tests/rewards/functions/test_nsfw_safety.py:66`（`reward._model._probability_from_classifier_result(`）。`vrl/` 无任何生产读取——打分走 `self.runtime = InProcessRewardRuntime(model=model)`。
- 判死证据：
  - 对照 `vrl/rewards/functions/ocr.py`：其 `_model` 被有文档的 `_engine` property 注入 seam 消费（由 `test_multi.py:333` 驱动，属 keep-list）；`nsfw_safety.py` 无 proxy/注释，是裸别名 + test-only reader，按死字段规则判死。
- 动作：删 `self._model = model`（`nsfw_safety.py:24`）——`model` 局部变量仍供 `InProcessRewardRuntime(model=model)` 使用；改写唯一测试 reader `tests/rewards/functions/test_nsfw_safety.py:62-66`（`test_nsfw_classifier_result_parsing_prefers_nsfw_labels`）为直接构造 `NSFWSafetyRewardModel({...})` 并调 `_probability_from_classifier_result`（`vrl/rewards/models/nsfw_safety.py:108`，构造仅需 config）。触及测试：`tests/rewards/functions/test_nsfw_safety.py`。

### 1.5 `parse_hf_repo_revision(default_revision)` + `DEFAULT_HF_REVISION` — dead-arg（非默认路径 test-only）（risk=low｜STILL_VALID）
- 位置：`vrl/rewards/models/hub.py:21`（def）、`:24`（参数）、`:10`（常量）、`:79`（`__all__`）。
- 复核：行号与原审计一致（21-25，常量 10）。`grep 'default_revision'` → 两个生产调用都省略 kwarg（走默认）；唯一非默认调用是 test-only 的 `tests/rewards/test_model_hub.py:31`（`parse_hf_repo_revision("org/model@", default_revision="stable")`）。
- 判死证据：
  - 两个生产调用点都省略 `default_revision` kwarg（走 `"main"` 默认）；唯一非默认调用 `tests/rewards/test_model_hub.py:31`——test-only，按审计口径仍判死。
  - `grep -rn 'DEFAULT_HF_REVISION' vrl tests` → 仅 `hub.py:10`（定义）、`:24`（默认值）、body 两分支、`:79`（`__all__`）；无外部 importer。
  - 承接 [[SPRINT_reward_identity_and_score_keys]]（删掉了最后的非默认 producer `kling`/`videocon` `_DEFAULT_REVISION` 别名），本条是该已完成清理的残留、非 staged future use。
- 动作：删 `default_revision: str = DEFAULT_HF_REVISION` 参数，body 两分支内联 `"main"` 默认；把 `DEFAULT_HF_REVISION` inline 或至少从 `__all__` 去掉（无外部 reader）。更新测试——**建议改写**（非删）`test_parse_hf_repo_revision_defaults_empty_revision` 为 `parse_hf_repo_revision("org/model@")` 并断言 `revision == "main"`，以继续覆盖仍活的 empty-revision 分支。触及测试：`tests/rewards/test_model_hub.py`。

### 1.6 `KlingVideoRewardModel._reward(fps, num_frames)` / `_prepare_batch(fps, num_frames)` — dead-arg（risk=low｜RELOCATED）
- 位置（**已随 origin 位移刷新**，原审计为 `295-313 / 361-373`）：`_reward` 现在 `vrl/rewards/models/kling_video_reward.py:258-270`（`fps`/`num_frames` 参数在 `:262-263`，互斥 raise 在 `:268-269`）；`_prepare_batch` 现在 `:192-230`（参数 `:196-197`，`nframes`/`fps` 分支 `:222`）。注意 origin 重排后 `_prepare_batch` 现**位于** `_reward` 之前。
- 复核：文件因 `ef0a04db`/`e62c4e90`（隔离 Kling prompt 协议）整体上移。仍死：`__call__` 唯一 caller 在 `:178` 只传 `min_pixels`/`max_pixels`/`use_norm`；probe `kling_reward_diagnosis_probe.py:69` 与 `kling_480p_discrimination_probe.py:76` 只传 `use_norm`；`_prepare_batch` 唯一调用（`:270`）转发恒为 None 的两值；`:268-269` 互斥 raise 不可达。
- 判死证据：
  - 读 body：两处对 `fps`/`num_frames` 都回落 `self.data_config`，故删参行为不变；互斥 raise（`if fps is not None and num_frames is not None: raise ValueError`）永不触发（错误串 `"cannot be set at the same time"` 全仓仅定义处，无测试断言）。
- 动作：删 `_reward`、`_prepare_batch` 的 `fps`/`num_frames` 参数，`_prepare_batch` 内部直接读 `self.data_config.fps`/`self.data_config.num_frames`；`nframes`/`fps` 分支（`:222`）**保留**（仍由 checkpoint 的 `model_config.json` `data_config.num_frames` 选中，`_load_configs` 建 `_DataConfig`）；删不可达的互斥 raise（`:268-269`）；保留 `max_pixels`/`min_pixels`/`use_norm`（均被调用点传入）。无测试改动。

### 1.7 `load_kling_video_reward_checkpoint(checkpoint_step)` / `_resolve_checkpoint_path(checkpoint_step)` — dead-arg（live caller, dead semantics）（risk=low｜RELOCATED）
- 位置（**已随 origin 位移刷新**，原审计为 `551-605`，caller `249-253`）：`load_kling_video_reward_checkpoint` 现在 `vrl/rewards/models/kling_video_reward.py:417`（`checkpoint_step` 参数 `:420`）；`_resolve_checkpoint_path` 现在 `:455-471`（参数 `:457`，always-latest 分支 `:466-467`，死的请求-步分支 `:469-470`）；唯一 caller 在 `:146-150`（传字面量 `-1`）；resolved step 作为 provenance 在 `:153` 消费。
- 复核：文件整体上移。仍死：`grep 'checkpoint_step'` 在 `vrl/` 与 `configs/` 无任何非-`(-1)` producer；其余引用仅为 `test_model_loading.py:124,170` 的 monkeypatch fake。`_resolve_checkpoint_path:466` 恒走 latest 分支，请求-步分支 `:469-470` 零 producer。
- 判死证据：
  - 唯一生产调用传字面量 `-1`（永远-latest 哨兵）；无其他 producer。测试仅 monkeypatch 假实现回显该参数并断言 `"checkpoint_step": -1`。
  - `test_real_checkpoint_rl.py` 里的 `_resolve_checkpoint_path` 是签名 `(case, field)` 的另一个本地函数，非本符号（已排除）。
  - 无 YAML/`worker_config`/reward_service 存在名为 `checkpoint_step` 的 knob。
- 动作：删 `load_kling_video_reward_checkpoint` 与 `_resolve_checkpoint_path` 的 `checkpoint_step` 参数；`_resolve_checkpoint_path` 收敛为「取最高 checkpoint-N 目录」（删请求-步分支 `:469-470`）；**保留**返回解析出的 step 名（`:153` load 日志消费的真 provenance，glob 解析、调用点不可自推）。更新 test fake（`tests/rewards/kling_video_reward/test_model_loading.py`，镜像 3-arg 签名并断言 `"checkpoint_step": -1`）。触及测试：`tests/rewards/kling_video_reward/test_model_loading.py`。

### 1.8 `_validate_probability(upper_open)`（`nsfw_safety` model）— dead-arg（single caller, default path unreachable）（risk=low｜STILL_VALID）
- 位置：`vrl/rewards/models/nsfw_safety.py:281`（def，`upper_open` 参数）、`:283`/`:285`（分支）、caller `:35-36`。
- 复核：行号与原审计一致（281-287，caller 35-37）。def 在 `:281`（`*, upper_open: bool = False`）；分支 `:283`（`out < 1.0 if upper_open else out <= 1.0`）、`:285`（`"<" if upper_open else "<="`）；唯一 caller `:35-36` 恒传 `upper_open=True`，故 `upper_open=False` 分支零 producer。无测试 import 该 helper。
- 判死证据：
  - helper 为 module-private（下划线名，不在 `__all__`），无 string/registry/YAML 可达；`upper_open` 是内部 Python 关键字非 config key（YAML 面向的是 `threshold` 值本身）。
  - 无测试断言概率错误串（`test_nsfw_safety.py` 唯一 `pytest.raises(ValueError, match="penalty_scale")` 走 `_validate_lower_bounded`；所有测试阈值 0.25/0.35/0.50 两种界下都合法）。兄弟 `_validate_lower_bounded(inclusive=...)` 被 `True`/`False` 双值调用，是活的、保留。
- 动作：删 `upper_open: bool = False` 关键字，硬编码开区间上界（`out < 1.0`）与错误串里的 `"<"` 关系。无测试改动。

### 1.9 `OCRRewardModel._device` — dead-field（test-only reader）（risk=low｜STILL_VALID）
- 位置：`vrl/rewards/models/ocr.py:47`。
- 复核：行号与原审计一致（47）。`grep '_device' vrl/rewards/models/ocr.py` → 仅 `:47`（`self._device = str(cfg.get("device", "cuda"))`，赋值、无读）。唯一 reader 仍是 test-only 的 `tests/rewards/functions/test_multi.py:333`（`functions["ocr"]._model._device == "cpu"`）。
- 判死证据：
  - 读 body：`_build_paddle_ocr()` 两条 PaddleOCR API 路径都硬编码 CPU（3.x `device="cpu", enable_mkldnn=False`；2.x `use_gpu=False`），从不接收 `self._device`。`OCRReward.resolve_execution_device` 恒返回 `"cpu"`，故喂进模型的 device 是 no-op、`_device` 只可能是 `"cpu"`，不携带信息。
  - 对照兄弟 `nsfw_safety.py`：其 `_device` 经 `_pipeline_device` 被行为消费——故本条非跨家族一致性 shape。
- 动作：删 `self._device = ...`（`ocr.py:47`）；更新 `tests/rewards/functions/test_multi.py:333` 改为经 wrapper 契约断言 CPU pin（如 `OCRReward.resolve_execution_device(device="cuda:0", kwargs={}) == "cpu"`），或直接删该断言（相邻 `requires_memory_parking is False` 已覆盖行为）。可选：停止从 `vrl/rewards/functions/ocr.py` 向 `OCRRewardModel` 的 worker_config 传 `"device"`（模型忽略它）；但 `OCRReward.__init__` 的 `device` 形参保留（`registry.py:221` 始终传入）。触及测试：`tests/rewards/functions/test_multi.py`。

### 1.10 `_build_prepared_model_in_pool` — duplicate-impl（form 4，body 与共享实现相同）（risk=low｜STILL_VALID）
- 位置：`vrl/rewards/runtime.py:29`（`_build_prepared_model_in_pool`）、`:52`（`_build_prepared_model`）；caller 分别 `:205`（pooled）与 `:231`（CPU-fallback）。
- 复核：行号与原审计一致（29-59）。两个 helper 均在、均 private，各恰一个 caller，无测试 import 任一。lazy-prepare 序列的重复完整保留。
- 判死证据：
  - 读 body：`_build_prepared_model_in_pool` 与 `_build_prepared_model` 逐字节相同的 `model = factory(worker_config); prepare = getattr(model, "prepare_for_inference", None); if callable(prepare): prepare(); return model` 序列，仅多一层 `with pool.building():` 包裹。
  - `traceback.clear_frames(load_error.__traceback__)`（`:214`）遍历并清空整条 traceback 全部 frame（含嵌套 delegate frame），故 delegation 后 failure-ownership 语义依旧成立。
- 动作：折叠重复：令 `_build_prepared_model_in_pool` 委托——`with pool.building(): return _build_prepared_model(factory, worker_config)`——或在其唯一调用点（`:205`）内联 `with pool.building():` 包裹一个 `_build_prepared_model` 调用。无测试改动（无测试引用任一 helper；`tests/rewards/inference/test_in_process_runtime.py` 在 `_cumem_allocator` 层 mock、断言 `load_scopes.append(bool(allocator.building))`，delegation 保持该观测）。
- 注意：本文件的 DO-NOT-FLAG 豁免项 `ensure_loaded` 在此文件并不存在（仅有 `_ensure_model`），本条动作与之无关、不触碰。

## 2. 已由 origin 落地（本次复核确认，无需再做）

以下三条在审计基线 `88ed756e` 上是 STILL_VALID 的零消费者包 facade，现已由 `origin/main` 删除。三个 `__init__.py` 本次复核确认均只剩 docstring + 空 `__all__`，无 `__getattr__`、无 lazy re-export、无 `from typing import Any`。**无需再做。**

- `vrl/rewards/__init__.py::__getattr__`（lazy `KlingVideoReward` re-export）+ `__all__` 的 `"KlingVideoReward"` + `from typing import Any` — 零消费者的包根 facade shim — 落地于 `94761143`（"refactor(families): drop package export surfaces nothing imports"）。复核：全文件 10 行，仅 docstring + `__all__: list[str] = []`。
- `vrl/rewards/functions/__init__.py::__getattr__` + 包门面 re-export（`MultiReward`/`get_reward`）+ `from typing import Any` + `__all__` — 零 importer 的 functions 包 facade — 落地于 `94761143`。复核：全文件 7 行，仅 docstring + `__all__: list[str] = []`。
- `vrl/rewards/service/__init__.py::__getattr__`（lazy `HttpRewardRuntime`/`RewardService`/`RemoteRewardServiceError`/`RewardServiceInfo` re-export）+ `__all__` — 零 importer 的 service 包 facade（消费者全绑定 `.client`/`.server` 具体子模块）— 落地于 `94761143`。复核：全文件 8 行，仅 docstring + `__all__: list[str] = []`。

## 3. 情况已变（需重新评估）

（无）——本次复核无 CHANGED/INDETERMINATE 发现。所有仍有效的 10 条判死语义与原审计一致，仅 §1.6/§1.7 两条因 origin 的 Kling 协议隔离重排导致文件行号位移（动作不变，位置已在 §1 刷新）。

## 4. 验证协议

- 基线已切到 **main @ `7c748532`**（= `origin/main` tip）。以下命令均对该 checked-out 树执行。
- 每条删除后：`ruff check <touched files>` + `ruff format --check <touched files>`（仅跑本任务触及的 Python 文件，先 `ruff check --fix` 再 `ruff format`）。
- 全簇完成后：`pytest tests/rewards/`（覆盖 `tests/rewards/inference/`、`tests/rewards/functions/`、`tests/rewards/kling_video_reward/`、`tests/rewards/cosmos3_reasoner/`、`tests/rewards/service/`、`tests/rewards/test_model_hub.py`、`tests/rewards/test_robotics_video_reward.py`）+ `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- 逐条动作触及的测试文件：
  - §1.1 → `tests/rewards/cosmos3_reasoner/`（仅回归，不改）
  - §1.2 → 无测试改动（`tests/rewards/inference/test_artifact_store.py` 等读字面量，回归跑）
  - §1.3 → 无测试改动
  - §1.4 → `tests/rewards/functions/test_nsfw_safety.py`（改写唯一 reader）
  - §1.5 → `tests/rewards/test_model_hub.py`（改写 empty-revision 用例）
  - §1.6 → 无测试改动
  - §1.7 → `tests/rewards/kling_video_reward/test_model_loading.py`（更新 3-arg fake 与断言）
  - §1.8 → 无测试改动
  - §1.9 → `tests/rewards/functions/test_multi.py`（改断言或删断言）
  - §1.10 → 无测试改动

## 5. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）——如 `_init_disk_artifact_reward` 的 `media_type`（YAML 经 `**kwargs` 可达）、`_validate_lower_bounded` 的 `inclusive`（双值调用）、`_resolve_checkpoint_path` 返回的 resolved step（日志 provenance）、`cosmos3` 的 `model_path` 守卫、`base.py:352-368` 读的 `reward_model_name` provenance key。
- 不动 DO-NOT-FLAG 豁免项：`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`（`rewards/base.py` 显存残留配额）、`ensure_loaded`（`rewards/runtime.py`，实际不存在于该文件）、`process_gpu_used_bytes`（NVML）、`sana`/`hunyuan` 的 `prepare_latents` 修复。§1.10 的动作与 `ensure_loaded` 无关。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function——三个包 facade 已由 origin 收敛（§2），非本簇再做项；`ocr.py` 的 `_engine` property 注入 seam（活）与 `nsfw_safety` model 的 `_device`（活）均保留。
- 兄弟家族的同名符号一律不动：`videoscore2`/`unified_reward_video`/`kling_video_reward`/`videocon_physics` 各自的 `_DEFAULT_REWARD_MODEL`；`hub.resolve_model_root` 对其余判官（videoscore2/unified_reward_video/videocon_physics）仍活，仅从 docstring "Shared by" 移除 cosmos3。
- 本簇文件均**不在** in-flight sprint（`SPRINT_native_generation_engine_program`，改动 `generation/ray/`、`vrl/ray/`、`models/steps/denoise/base.py`）的 worktree 变更集内，故无 sequencing 冲突，可独立落地。

## References

- `vrl/rewards/models/cosmos3_reasoner.py:45,50,129-144`、`vrl/rewards/models/hub.py:10,21-25,79`（`parse_hf_repo_revision`/`resolve_model_root`/`DEFAULT_HF_REVISION`）
- `vrl/rewards/artifacts.py:42,57`、`vrl/rewards/base.py:266-290,313,334,352-368`
- `vrl/rewards/functions/nsfw_safety.py:24`、`vrl/rewards/functions/ocr.py`、`vrl/rewards/functions/registry.py:221`
- `vrl/rewards/models/kling_video_reward.py:146-153,178,192-230,258-270,417,455-471`（RELOCATED 后的当前行号）、`vrl/rewards/models/nsfw_safety.py:35-36,108,281-287`、`vrl/rewards/models/ocr.py:47`
- `vrl/rewards/runtime.py:29-59,205,214,231`
- 已落地 facade（§2，`94761143`）：`vrl/rewards/__init__.py`、`vrl/rewards/functions/__init__.py`、`vrl/rewards/service/__init__.py`（均已缩为 docstring + 空 `__all__`）
- 测试：`tests/rewards/functions/test_nsfw_safety.py:62-66`、`tests/rewards/functions/test_multi.py:332-333`、`tests/rewards/test_model_hub.py:28-34`、`tests/rewards/kling_video_reward/test_model_loading.py:124,170`
- 配置：`vrl/config/reward_service/kling_overlap_gate.yaml:43`、`vrl/config/presets/reward/cosmos3_reasoner.yaml`、`pyproject.toml:140`
- 关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_reward_identity_and_score_keys]]、[[SPRINT_reward_service]]、[[SPRINT_cosmos3_full_support]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]
