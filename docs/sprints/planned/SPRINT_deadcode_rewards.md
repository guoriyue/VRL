# SPRINT: `rewards` 死代码清理（planned）

状态：**planned（2026-07-23）**。共 **13 条**确认死代码：1 条 medium（`cosmos3_reasoner` 不可达的 snapshot-download 回退分支）+ 12 条 low（3 个零消费者的包 facade `__getattr__`、5 个死参数、2 个 test-only 死字段、1 处 form-4 重复实现），全部来自 dead-code-audit workflow 的对抗验证结果，本 sprint 已逐条 re-grep 复核（无偏差）。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）。
关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_reward_identity_and_score_keys]]（§1.8 的 `default_revision` 是该 sprint 删掉最后一个非默认 producer 后残留的死参数）、[[SPRINT_reward_service]]（§1.13 facade 即该 sprint 描述的 "optional-dependency lazy public facade"，现已零消费者）、[[SPRINT_cosmos3_full_support]]（parked；§1.1 的 reasoner judge 已 shipped、不计划放开 `model_path` 守卫）。格式范本：[[SPRINT_trajectory_views_types_dead_fields_cleanup]]。

## 0. 一句话

本簇清理 `vrl/rewards/` 下的死代码，主形态是「零消费者的门面/参数」——3 个包 `__init__.py` 的 lazy `__getattr__` re-export 无任何 importer、5 个函数参数所有调用点都走默认值、2 个字段只有测试在读。最锋利的一条是 `cosmos3_reasoner.py` 的 `_DEFAULT_REWARD_MODEL` + `resolve_model_root` snapshot-download 回退：`__init__` 里的 `model_path` 守卫会先 raise、`resolve_model_root` 又在 `model_path` 非空时提前 return，两头夹击使这段「下载原始 unified repo」的回退永远不可达（form 2，live caller / dead semantics）。误删风险主要在两处：(1) 三个 facade 必须确认是真死 re-export 而非 CI 打包依赖的 lazy-import 边界——已逐个验证 `pyproject.toml` 入口、entry-point 字符串、`python -m` 路径都绑定具体子模块，非包根；(2) 同名兄弟常量（`videoscore2`/`kling` 等各自的 `_DEFAULT_REWARD_MODEL`、`_init_disk_artifact_reward` 的 `media_type`、`_validate_lower_bounded` 的 `inclusive`）对各自家族是活的，一律不动。

## 1. 待删清单（逐条，带证据与动作）

### 1.1 `cosmos3_reasoner._DEFAULT_REWARD_MODEL` / `resolve_model_root` 回退 — dead-branch（risk=medium）
- 位置：`vrl/rewards/models/cosmos3_reasoner.py:50, 129-144`；连带 `vrl/rewards/models/hub.py:41-60` docstring。
- 判死证据：
  - 读 body：`cosmos3_reasoner.py:129-135` 的 `if not str(self.worker_config.get("model_path", "")).strip(): raise ValueError(...)` 在 `resolve_model_root` 调用（140-144）之前执行；`hub.py:55-60` 的 `model_path` 分支在其非空时提前 `return root`（含 `if not root.exists(): raise FileNotFoundError`）。两头夹击后 `hub.py` 的 `reward_model_name`/`default_model`/`snapshot_download` 回退对本家族永不可达。
  - `grep -rn '_DEFAULT_REWARD_MODEL' vrl/rewards/models/cosmos3_reasoner.py` → 仅 `:50`（定义）与 `:142`（作为 `default_model=` 传入，即那个永不被读的实参）。
  - `grep -rn 'resolve_model_root' vrl/rewards/models/cosmos3_reasoner.py vrl/rewards/models/hub.py` → `cosmos3_reasoner.py:45`（import）、`:140`（唯一调用）；`hub.py:41`（定义）、`:82`（`__all__`）。
  - 预设 `vrl/config/presets/reward/cosmos3_reasoner.yaml:28` 设 `model_path: ""`（空 → 命中 `__init__` raise），`:29` `reward_model_name: nvidia/Cosmos3-Nano` 仅被通用 `vrl/rewards/base.py:352-368` 读作 `reward_model_version` provenance（已核对该 reader 存在）。
  - `hub.py:49-51` docstring 的 "Shared by ... cosmos3_reasoner ..." 明确把本家族列入共享者——留着即复现同一处 false-liveness 文档。
- 动作：`cosmos3_reasoner.py` 删 `_DEFAULT_REWARD_MODEL`（`:50`）；删 `:137-139` 过期注释；把 `resolve_model_root` 调用（`:140-144`）替换为对已被守卫校验过的 `model_path` 的直接解析，保留 `hub` 的 model_path-branch 语义含存在性检查（`root = Path(model_path).expanduser().resolve()`，不存在则以 `"Cosmos3 reasoner"` 家族标签 `raise FileNotFoundError`）；删 `:45` 现已无用的 `from vrl.rewards.models.hub import resolve_model_root`。`hub.py` 更新 `resolve_model_root` docstring（`:49-51`）从 "Shared by" 列表移除 `cosmos3_reasoner`。**不动**：预设的 `reward_model_name` key、兄弟家族各自的 `_DEFAULT_REWARD_MODEL`、任何测试（无测试引用该常量或构造该模型）。触及测试：`tests/rewards/cosmos3_reasoner/`（仅回归跑，不改）。
- 注意：本条为 medium，reviewer 重点在「回退真不可达」这一 form-2 判断——删除后必须保持 `model_path` 空时 fail-fast（`__init__` raise）与非空但目录缺失时 `FileNotFoundError` 两条行为不变；同名兄弟常量（`videoscore2.py`/`unified_reward_video.py`/`kling_video_reward.py`/`videocon_physics.py` 各自的 `_DEFAULT_REWARD_MODEL`）对各自家族是活的，严禁连坐删除。

### 1.2 `vrl/rewards/__init__.py::__getattr__`（lazy `KlingVideoReward` re-export）— dead-function（risk=low）
- 位置：`vrl/rewards/__init__.py:10-15`（+ `__all__` 的 `"KlingVideoReward"` 于 `:18`，+ `from typing import Any` 于 `:3`）。
- 判死证据：
  - `grep -rnE 'from vrl\.rewards import|import vrl\.rewards([^.]|$)|vrl\.rewards\.KlingVideoReward|vrl\.rewards:' vrl tests docs` → 零命中（包根 facade 无任何消费者）。
  - `grep -rn 'KlingVideoReward' vrl tests vrl/config` → 全部消费者都直接 import 具体子模块：`vrl/rewards/functions/registry.py:32`、`tests/rewards/kling_video_reward/*`、`tests/rewards/inference/test_runtime_factory.py:9`；模型工厂字符串 `vrl/config/reward_service/kling_overlap_gate.yaml:43` 指向 `vrl.rewards.models.kling_video_reward:KlingVideoRewardModel`（具体子模块，非包根）。
  - `pyproject.toml:140` 唯一入口 `vrl-reward-service = "vrl.rewards.service.server:main"` 也是具体子模块。
  - 读 body：`vrl/rewards/__init__.py:3` 的 `Any` 仅被 `__getattr__` 签名使用；`:5-7` 的 eager re-export（`RewardFunction`/`MultiReward`/`get_reward`/`RewardRollout`）是惯例包门面，保留。
- 动作：删 `__getattr__`（`:10-15`）、从 `__all__` 去掉 `"KlingVideoReward"`（`:18`）、删现已无用的 `from typing import Any`（`:3`）；保留 eager re-export。无测试改动（无测试触碰该 facade hook）。

### 1.3 `vrl/rewards/functions/__init__.py::__getattr__` + 包门面 re-export — dead-function（risk=low）
- 位置：`vrl/rewards/functions/__init__.py:1-22`。
- 判死证据：
  - `grep -rnE 'from vrl\.rewards\.functions import|import vrl\.rewards\.functions([^.]|$)' vrl tests docs` → 零命中（包 facade 无任何 importer）。
  - `grep -rn 'get_reward|MultiReward' ...`（排除定义与 `registry.py`）→ 所有消费者直接 import `vrl.rewards.functions.registry`：`vrl/scripts/common/factory.py:80`、`vrl/rewards/__init__.py:6`、`tests/rewards/functions/test_multi.py:8` 等。
  - 无 import-time 副作用丢失：`registry.py:28-65` 的 `_register_builtins` 是 lazy（由 `MultiReward.from_dict` 与测试显式调用），非包 import 时注册。
  - 读 body：`__getattr__`（lazy `KlingVideoReward`）+ `from vrl.rewards.functions.registry import MultiReward, get_reward` re-export + `__all__` + `from typing import Any`（仅 `__getattr__` 用）。
- 动作：将 `vrl/rewards/functions/__init__.py` 缩为「仅 docstring 的包标记」：删 `__getattr__`、删 registry re-export、删 `from typing import Any`、删整个 `__all__`。无测试改动（无测试从该包 `__init__` import）。
- 注意：与 §1.2 同一 commit 处理（两者都是同一 `KlingVideoReward` 零消费者 shim）。此形态承接仓库先例 commit `7ee3e36d`（"Remove dead fields, exports, and aliases"，曾把门面收敛到真实公共面并从本文件删掉 `register_reward`）。

### 1.4 `vrl/rewards/service/__init__.py::__getattr__`（lazy `HttpRewardRuntime`/`RewardService`/`RemoteRewardServiceError`/`RewardServiceInfo` re-export）— dead-function（risk=low）
- 位置：`vrl/rewards/service/__init__.py:7-37`。
- 判死证据：
  - `grep -rnE 'from vrl\.rewards\.service import|vrl\.rewards\.service:' vrl tests docs`（排除 `service/__init__.py` 自身）→ 零命中。
  - 四个符号全部经具体子模块消费：`vrl/rewards/runtime.py:336`（`from vrl.rewards.service.client import HttpRewardRuntime`）、`vrl/scripts/eval/unified_reward_robotics_discrimination_probe.py:469`（同）、`tests/rewards/service/test_service.py:29,39`（`.server`）；`pyproject.toml:140` 入口指向 `vrl.rewards.service.server:main`（具体子模块）。
  - 读 body：`__getattr__` 是纯无副作用的 re-export dispatch。注释里的 runpy 双导入顾虑（`python -m vrl.rewards.service.server`）由「仅 docstring 的空 `__init__`」同样满足——lazy 只在门面会 eager import `server` 时才有意义，空文件也不 eager import。
- 动作：将 `vrl/rewards/service/__init__.py` 缩为其 docstring（保留文件作为包标记），删 `__getattr__` 与 `__all__`。无测试改动（无测试 import 该 facade）。
- 注意：这三个 facade（§1.2/1.3/1.4）是「lazy-import public facade」候选，keep-list 只保护**有消费者**的门面；此三者消费者为零，仓库惯例（含各自 entry-point 与测试）一律绑定具体子模块，故非受保护边界。

### 1.5 `VideoRewardArtifactStore.__init__(manifest_name=...)` — dead-arg（risk=low）
- 位置：`vrl/rewards/artifacts.py:42, 57`。
- 判死证据：
  - `grep -rn 'manifest_name' vrl tests vrl/config` → 仅 `artifacts.py:42`（定义）与 `:57`（唯一内部读 `self.manifest_path = self.root / manifest_name`）。
  - `grep -rn 'VideoRewardArtifactStore(' vrl tests` → 唯一生产构造点 `vrl/rewards/base.py:334`，加 6 个测试构造（`tests/rewards/inference/test_artifact_store.py`），全部只传 `root`/`media_type`/`artifact_format`。
  - 读 `base.py:293-338`：`_init_disk_artifact_reward` 为 keyword-only 签名、无 `**kwargs` 转发，从不转发 `manifest_name`，故无 YAML kwargs 路径可设它。测试（`test_artifact_store.py`、`tests/rewards/kling_video_reward/test_function.py:118`、`tests/e2e/test_real_checkpoint_rl.py:706`）都读字面量 `'manifest.jsonl'` 路径。
- 动作：删 `manifest_name: str = "manifest.jsonl"` 参数，在 `:57` 内联字面量（`self.manifest_path = self.root / "manifest.jsonl"`）。无测试改动。

### 1.6 `RewardFunction._init_reward_model(media_type=...)` — dead-arg（risk=low）
- 位置：`vrl/rewards/base.py:266-290`（参数在 `:273`）。
- 判死证据：
  - `grep -rn '_init_reward_model' vrl tests` → 仅两个调用点 `vrl/rewards/functions/pickscore.py:28`、`vrl/rewards/functions/aesthetic.py:26`（另有 `base.py:313` docstring 交叉引用）。读两处 body：均只传 `reward_name`/`score_key`/`model_factory`/`worker_config`，`**kwargs` 折进 `worker_config` dict，不进入本调用，故非默认值不可达。
  - `media_type` 唯一使用在 `base.py:288` 的 `build_inmemory_artifacts(..., media_type=media_type)`；`build_inmemory_artifacts` 本身默认 `"image"`（`base.py:135-140`），故硬编码 `"image"` 对两个 live caller 行为等价。
  - in-memory video rewards（`target_dino_similarity.py`、`motion_dynamics.py`、`nsfw_safety.py`、`ocr.py`）绕开本 helper、直接调 `build_inmemory_artifacts(media_type="video")`。
- 动作：删 `media_type: MediaType = "image"` 参数，在其构造的 `build_inmemory_artifacts` lambda 里硬编码 `media_type="image"`。**保留**兄弟 `_init_disk_artifact_reward` 的 `media_type` 参数（那个经 concrete rewards 的 `**kwargs` 转发从 YAML 可达）。无测试改动。

### 1.7 `NSFWSafetyReward._model` — dead-field（test-only reader）（risk=low）
- 位置：`vrl/rewards/functions/nsfw_safety.py:24`。
- 判死证据：
  - `grep -rn '\._model\b' vrl/rewards/functions/nsfw_safety.py tests/rewards/functions/test_nsfw_safety.py` → 定义 `nsfw_safety.py:24`（`self._model = model`）+ 唯一 reader `tests/rewards/functions/test_nsfw_safety.py:66`（`reward._model._probability_from_classifier_result(...)`）。`vrl/` 无任何生产读取——打分走 `self.runtime`（`InProcessRewardRuntime(model=model)`，`runtime.py:114` 自持 `_model`）。
  - 对照 `vrl/rewards/functions/ocr.py`：其 `_model` 被有文档的 `_engine` property 注入 seam 消费（`ocr.py:57-63`，由 `test_multi.py:333` 驱动，属 keep-list）；`nsfw_safety.py` 无 proxy/注释，是裸别名 + test-only reader，按死字段规则判死。
- 动作：删 `self._model = model`（`nsfw_safety.py:24`）——`model` 局部变量仍供 `InProcessRewardRuntime(model=model)` 使用；改写唯一测试 reader `tests/rewards/functions/test_nsfw_safety.py:62-66`（`test_nsfw_classifier_result_parsing_prefers_nsfw_labels`）为直接构造 `NSFWSafetyRewardModel({...})` 并调 `_probability_from_classifier_result`（`vrl/rewards/models/nsfw_safety.py:108`，构造仅需 config）。触及测试：`tests/rewards/functions/test_nsfw_safety.py`。

### 1.8 `parse_hf_repo_revision(default_revision)` + `DEFAULT_HF_REVISION` — dead-arg（非默认路径 test-only）（risk=low）
- 位置：`vrl/rewards/models/hub.py:21-25`（常量 `:10`，`__all__` 于 `:79`）。
- 判死证据：
  - `grep -rn 'parse_hf_repo_revision' vrl tests` → 两个生产调用 `hub.py:65`、`kling_video_reward.py:639` 都省略 kwarg（走默认）；唯一非默认调用 `tests/rewards/test_model_hub.py:31`（`parse_hf_repo_revision("org/model@", default_revision="stable")`）——test-only，按审计口径仍判死。
  - `grep -rn 'DEFAULT_HF_REVISION' vrl tests` → 仅 `hub.py:10`（定义）、`:24`（默认值）、`:32,:35`（body，两分支都用它）、`:79`（`__all__`）；无外部 importer。
  - 承接 [[SPRINT_reward_identity_and_score_keys]]（`:161,198` 删掉了最后的非默认 producer `kling`/`videocon` `_DEFAULT_REVISION` 别名），本条是该已完成清理的残留、非 staged future use。
- 动作：删 `default_revision: str = DEFAULT_HF_REVISION` 参数，`:32/:35` 两分支内联 `"main"` 默认；把 `DEFAULT_HF_REVISION` inline 或至少从 `__all__` 去掉（无外部 reader）。更新测试 `tests/rewards/test_model_hub.py:28-34`——**建议改写**（非删）`test_parse_hf_repo_revision_defaults_empty_revision` 为 `parse_hf_repo_revision("org/model@")` 并断言 `revision == "main"`，以继续覆盖仍活的 empty-revision 分支。触及测试：`tests/rewards/test_model_hub.py`。

### 1.9 `KlingVideoRewardModel._reward(fps, num_frames)` / `_prepare_batch(fps, num_frames)` — dead-arg（risk=low）
- 位置：`vrl/rewards/models/kling_video_reward.py:295-313, 361-373`。
- 判死证据：
  - `grep -rn '_reward(\|_prepare_batch(' ...`（排除 def）→ `_reward` 调用点：`kling_video_reward.py:281`（`self._reward([...], [...], min_pixels=..., max_pixels=..., use_norm=...)`，无 `fps`/`num_frames`）、`vrl/scripts/eval/kling_480p_discrimination_probe.py:76` 与 `kling_reward_diagnosis_probe.py:69`（均 `model._reward([path], [prompt], use_norm=True)`）；`_prepare_batch` 唯一调用 `:373` 转发恒为 None 的两值。
  - 读 body：`:311-312` 对两者回落 `self.data_config`，故删参行为不变；`:371-372` 的互斥 raise（`if fps is not None and num_frames is not None: raise ValueError`）永不触发（错误串 `"cannot be set at the same time"` 全仓仅定义处，无测试断言）。
- 动作：删 `_reward`、`_prepare_batch` 的 `fps`/`num_frames` 参数，`_prepare_batch` 内部直接读 `self.data_config.fps`/`self.data_config.num_frames`；`:325` 的 `{"nframes": num_frames} if num_frames is not None else {"fps": fps}` 分支**保留**（仍由 checkpoint 的 `model_config.json` `data_config.num_frames` 选中，`_load_configs` 于 `:779` 建 `_DataConfig`）；删 `:371-372` 不可达的互斥 raise；保留 `max_pixels`/`min_pixels`/`use_norm`（均被调用点传入）。无测试改动。

### 1.10 `load_kling_video_reward_checkpoint(checkpoint_step)` / `_resolve_checkpoint_path(checkpoint_step)` — dead-arg（live caller, dead semantics）（risk=low）
- 位置：`vrl/rewards/models/kling_video_reward.py:551-605`（caller `:249-253`）。
- 判死证据：
  - `grep -rn 'load_kling_video_reward_checkpoint\|checkpoint_step' vrl tests` → 唯一生产调用 `kling_video_reward.py:249` 传字面量 `-1`（永远-latest 哨兵）；无其他 producer。测试仅 monkeypatch 假实现（`tests/rewards/kling_video_reward/test_model_loading.py:113,117,124,141,170,171`）回显该参数并断言 `"checkpoint_step": -1`。
  - `test_real_checkpoint_rl.py` 里的 `_resolve_checkpoint_path` 是签名 `(case, field)` 的另一个本地函数，非本符号（已排除）。
  - 读 body：`:600` `if checkpoint_step is None or int(checkpoint_step) == -1` 恒走 latest 分支；`:603-604` 的 `checkpoint_dir / f"checkpoint-{int(checkpoint_step)}"` 请求-步分支零 producer。
  - 无 YAML/`worker_config`/reward_service 存在名为 `checkpoint_step` 的 knob。
- 动作：删 `load_kling_video_reward_checkpoint` 与 `_resolve_checkpoint_path` 的 `checkpoint_step` 参数；`_resolve_checkpoint_path` 收敛为「取最高 checkpoint-N 目录」（删 `:600-604` 的请求-步分支）；**保留**返回解析出的 step 名（`:256` load 日志消费的真 provenance，glob 解析、调用点不可自推）。更新 test fake（`tests/rewards/kling_video_reward/test_model_loading.py:113-117,141,170-171`，镜像 3-arg 签名并断言 `"checkpoint_step": -1`）。触及测试：`tests/rewards/kling_video_reward/test_model_loading.py`。

### 1.11 `_validate_probability(upper_open)`（`nsfw_safety` model）— dead-arg（single caller, default path unreachable）（risk=low）
- 位置：`vrl/rewards/models/nsfw_safety.py:281-287`（caller `:35-37`）。
- 判死证据：
  - `grep -rn '_validate_probability\|upper_open' vrl tests` → 唯一 caller `nsfw_safety.py:35-36`（`_validate_probability("threshold", cfg.get("threshold", 0.35), upper_open=True)`）；定义 `:281`，分支 `:283`（`out < 1.0 if upper_open else out <= 1.0`）、`:285`（`"<" if upper_open else "<="`）。`upper_open=False`（`out <= 1.0`/`"<="`）零 producer。
  - helper 为 module-private（下划线名，不在 `__all__`），无 string/registry/YAML 可达；`upper_open` 是内部 Python 关键字非 config key（YAML 面向的是 `threshold` 值本身）。
  - 无测试断言概率错误串（`test_nsfw_safety.py` 唯一 `pytest.raises(ValueError, match="penalty_scale")` 走 `_validate_lower_bounded`；所有测试阈值 0.25/0.35/0.50 两种界下都合法）。兄弟 `_validate_lower_bounded(inclusive=...)` 被 `True`/`False` 双值调用，是活的、保留。
- 动作：删 `upper_open: bool = False` 关键字，硬编码开区间上界（`out < 1.0`）与错误串里的 `"<"` 关系。无测试改动。

### 1.12 `OCRRewardModel._device` — dead-field（test-only reader）（risk=low）
- 位置：`vrl/rewards/models/ocr.py:47`。
- 判死证据：
  - `grep -rn '_device' vrl/rewards/models/ocr.py` → 仅 `:47`（`self._device = str(cfg.get("device", "cuda"))`，赋值、无读）。唯一 reader `tests/rewards/functions/test_multi.py:333`（`functions["ocr"]._model._device == "cpu"`）——test-only。
  - 读 body：`_build_paddle_ocr()` 两条 PaddleOCR API 路径都硬编码 CPU（3.x `device="cpu", enable_mkldnn=False`；2.x `use_gpu=False`），从不接收 `self._device`。`OCRReward.resolve_execution_device`（`vrl/rewards/functions/ocr.py:38-40`）恒返回 `"cpu"`，故喂进模型的 device 是 no-op、`_device` 只可能是 `"cpu"`，不携带信息。
  - 对照兄弟 `nsfw_safety.py:33,105`：其 `_device` 经 `_pipeline_device` 被行为消费——故本条非跨家族一致性 shape。
- 动作：删 `self._device = ...`（`ocr.py:47`）；更新 `tests/rewards/functions/test_multi.py:333` 改为经 wrapper 契约断言 CPU pin（如 `OCRReward.resolve_execution_device(device="cuda:0", kwargs={}) == "cpu"`），或直接删该断言（相邻 `:332` 的 `requires_memory_parking is False` 已覆盖行为）。可选：停止从 `vrl/rewards/functions/ocr.py` 向 `OCRRewardModel` 的 worker_config 传 `"device"`（模型忽略它）；但 `OCRReward.__init__` 的 `device` 形参保留（`registry.py:221` 始终传入）。触及测试：`tests/rewards/functions/test_multi.py`。

### 1.13 `_build_prepared_model_in_pool` — duplicate-impl（form 4，body 与共享实现相同）（risk=low）
- 位置：`vrl/rewards/runtime.py:29-59`。
- 判死证据：
  - 读 body：`_build_prepared_model_in_pool`（`:42-49`）与 `_build_prepared_model`（`:52-59`）逐字节相同的 `model = factory(worker_config); prepare = getattr(model, "prepare_for_inference", None); if callable(prepare): prepare(); return model` 序列，仅多一层 `with pool.building():` 包裹。
  - `grep -rn '_build_prepared_model' vrl tests` → 两者均 private，各一个 caller（`InProcessRewardRuntime._ensure_model`：`:205` pooled 路径、`:231` CPU-fallback 路径）；无测试 import 任一 helper。
  - `traceback.clear_frames(load_error.__traceback__)`（`:214`）遍历并清空整条 traceback 全部 frame（含嵌套 delegate frame），故 delegation 后 failure-ownership 语义（在 `pool.close` 前从 traceback 清掉 helper frame 的 partial `model` 局部）依旧成立。
- 动作：折叠重复：令 `_build_prepared_model_in_pool` 委托——`with pool.building(): return _build_prepared_model(factory, worker_config)`——或在其唯一调用点（`:205`）内联 `with pool.building():` 包裹一个 `_build_prepared_model` 调用。无测试改动（无测试引用任一 helper；`tests/rewards/inference/test_in_process_runtime.py` 在 `_cumem_allocator` 层 mock、断言 `load_scopes.append(bool(allocator.building))`，delegation 保持该观测）。
- 注意：本文件的 DO-NOT-FLAG 豁免项 `ensure_loaded` 在此文件并不存在（仅有 `_ensure_model`），本条动作与之无关、不触碰。

## 2. 验证协议

- 每条删除后：`ruff check <touched files>` + `ruff format --check <touched files>`（仅跑本任务触及的 Python 文件，先 `ruff check --fix` 再 `ruff format`）。
- 全簇完成后：`pytest tests/rewards/`（覆盖 `tests/rewards/inference/`、`tests/rewards/functions/`、`tests/rewards/kling_video_reward/`、`tests/rewards/cosmos3_reasoner/`、`tests/rewards/service/`、`tests/rewards/test_model_hub.py`、`tests/rewards/test_robotics_video_reward.py`）+ `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- 基线（清理前，2026-07-23）：fast subset 2620 passed / 7 pre-existing failures（架构边界 + causvid/magi_1 打包摘要，与本清理无关）；`vrl.config.lint` 与 `ruff check .` 全绿。删除后这三项须保持。
- 逐条动作触及的测试文件：
  - §1.1 → `tests/rewards/cosmos3_reasoner/`（仅回归，不改）
  - §1.2 / §1.3 / §1.4 → 无测试改动
  - §1.5 → 无测试改动（`tests/rewards/inference/test_artifact_store.py` 等读字面量，回归跑）
  - §1.6 → 无测试改动
  - §1.7 → `tests/rewards/functions/test_nsfw_safety.py`（改写唯一 reader）
  - §1.8 → `tests/rewards/test_model_hub.py`（改写 empty-revision 用例）
  - §1.9 → 无测试改动
  - §1.10 → `tests/rewards/kling_video_reward/test_model_loading.py`（更新 3-arg fake 与断言）
  - §1.11 → 无测试改动
  - §1.12 → `tests/rewards/functions/test_multi.py`（改断言或删断言）
  - §1.13 → 无测试改动

## 3. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）——如 `_init_disk_artifact_reward` 的 `media_type`（YAML 经 `**kwargs` 可达）、`_validate_lower_bounded` 的 `inclusive`（双值调用）、`_resolve_checkpoint_path` 返回的 resolved step（日志 provenance）、`cosmos3` 的 `model_path` 守卫、`base.py:352-368` 读的 `reward_model_name` provenance key。
- 不动 DO-NOT-FLAG 豁免项：`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`（`rewards/base.py` 显存残留配额）、`ensure_loaded`（`rewards/runtime.py`，实际不存在于该文件）、`process_gpu_used_bytes`（NVML）、`sana`/`hunyuan` 的 `prepare_latents` 修复。§1.13 的动作与 `ensure_loaded` 无关。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function——三个 facade（§1.2/1.3/1.4）删的是**零消费者**的 re-export，不是在用的 lazy-import 边界；`ocr.py` 的 `_engine` property 注入 seam（活）与 `nsfw_safety` model 的 `_device`（活）均保留。
- 兄弟家族的同名符号一律不动：`videoscore2`/`unified_reward_video`/`kling_video_reward`/`videocon_physics` 各自的 `_DEFAULT_REWARD_MODEL`；`hub.resolve_model_root` 对其余判官（videoscore2/unified_reward_video/videocon_physics）仍活，仅从 docstring "Shared by" 移除 cosmos3。
- 本簇文件均**不在** in-flight sprint（`SPRINT_native_generation_engine_program`，改动 `generation/ray/`、`vrl/ray/`、`models/steps/denoise/base.py`）的 worktree 变更集内（`git status` 无任何 `rewards/` 文件），故无 sequencing 冲突，可独立落地。

## References

- `vrl/rewards/models/cosmos3_reasoner.py:45,50,129-144`、`vrl/rewards/models/hub.py:10,21-25,41-60,65,79,81-82`
- `vrl/rewards/__init__.py:3,10-15,18`、`vrl/rewards/functions/__init__.py:1-22`、`vrl/rewards/service/__init__.py:7-37`
- `vrl/rewards/artifacts.py:42,57`、`vrl/rewards/base.py:266-290,313,334,352-368`
- `vrl/rewards/functions/nsfw_safety.py:24`、`vrl/rewards/functions/ocr.py:38-40,57-63`、`vrl/rewards/functions/registry.py:32,58,221`
- `vrl/rewards/models/kling_video_reward.py:249-256,281,295-313,361-373,551-605`、`vrl/rewards/models/nsfw_safety.py:35-37,108,281-287`、`vrl/rewards/models/ocr.py:47,164-186`
- `vrl/rewards/runtime.py:29-59,205,214,231`
- 测试：`tests/rewards/functions/test_nsfw_safety.py:62-66`、`tests/rewards/functions/test_multi.py:332-333`、`tests/rewards/test_model_hub.py:28-34`、`tests/rewards/kling_video_reward/test_model_loading.py:113-117,141,170-171`
- 配置：`vrl/config/reward_service/kling_overlap_gate.yaml:43`、`vrl/config/presets/reward/cosmos3_reasoner.yaml:28-29`、`pyproject.toml:140`
- 关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_reward_identity_and_score_keys]]、[[SPRINT_reward_service]]、[[SPRINT_cosmos3_full_support]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]
