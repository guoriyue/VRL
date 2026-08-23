# SPRINT：Full-repo bloat audit（全库肿胀与死代码审计）

状态：**planned（2026-08-22）**。审计本身已完成（7 条并行审计道覆盖全部 vrl/ + tests/ +
configs/，每条 finding 附 path:line 与 grep 证据，函数体已读后才下判定；6 条最高风险判定
由主线复核确认）。本文档是结论与执行清单；执行为后续批次。

审计方法：AGENTS.md 死代码五形式 + 参数级审计（每个参数问"有无非默认调用者/函数体是否
消费"）+ placement 四规则 + 测试审计（mock 自证 / 重复定理 / 无牙墓碑）。

## 0. 结论先行

全库 ~19 万行（生产 9.6 万 + 测试 9.4 万），**没有结构性腐烂**：注册表可达性完好
（25 个 model family 全部有 preset 可达，无整族死码）、schema known-key 全部派生、
display/provenance-only 标注纪律在多数包执行良好。肿胀是真实的但集中：

- **3 个已确认 bug + 2 个行为缺口**（§1，立即修）。
- **五种跨仓肿胀模式**（§2）解释了 90% 的 finding——逐条修 finding 不如按模式修根因。
- 可安全删除/合并约 **2.5–3k 行**（生产 ~1.6k + 测试 ~1k，约 1.5%），另有 ~1k 行
  是**搬家不是删除**（utils 错置）。行数不是重点；§1 的 bug 和 §2-C 的"会撒谎的
  防护栏"比行数重要。
- 各包健康度：config/utils ≈ models > math/nn > rollouts > generation（生产干净、
  测试双路径拖累）> rewards > trainers（横向重复最重）> scripts（死旋钮最多）。

## 1. P0 — 已确认 bug 与行为缺口（复核过函数体，立即修）

| # | 位置 | 问题 | 修复 | 
|---|---|---|---|
| B1 | `vrl/generation/ray/launcher.py:290` | 清理路径调用 `session.kill_workers()`，真实 `RayGenerationSession` 只有 `kill_engines()`（session.py:182）。测试绿是因为 fake session 自己定义了假方法——mock 自证。生产 resident 启动失败时会 `AttributeError` 吞掉根因 | 改调 `kill_engines()`；`_FactorySession` 同步；测试替身换成有真实方法面的 |
| B2 | `vrl/trainers/online/trainer.py:1451` | 流式路径"所有 microbatch 被过滤"分支置 `initial_replay = None`，下游 `metrics_io.py` 直接 `.clip_fraction` → `AttributeError`。全批路径孪生代码（:1782）写的是正确的 `InitialReplayStats()`——孪生漂移的直接恶果 | `initial_replay = InitialReplayStats()`；补该分支测试 |
| B3 | `vrl/scripts/data/pickapic.py:35` | bootstrap 生成给用户照抄的前门命令硬编码已 404 的 `yuvalkirstain/pickapic_v2`；`presets/dataset/pickapic_v1.yaml:3-10` 早已记录下架并指向镜像 `pickapic-anonymous/pickapic_v1`。dataset identity 的真源在 config，脚本留了旧值副本 | 从 cfg 下传 `data.dataset_name`（bootstrap 已持有 cfg）；至少改默认为镜像 |
| B4 | `vrl/models/families/cosmos/{predict2,predict2_5}/model.py` | 两族 `from_pretrained` 用 `**build.revision_kwargs` 而非 `build.pretrained_kwargs`，**静默忽略 `model.local_files_only`**（该 key 在 wan 实验 preset 中是活的） | 换 `pretrained_kwargs` |
| B5 | `vrl/rollouts/orchestration/continuous/types.py:59` | `estimate_batch_bytes` 的 `tensor_extras` 过滤只收顶层 Tensor，而 extras 唯一真实内容是 `reward_components`（dict of Tensor，core.py:277）→ 队列字节预算系统性低估。`trajectory_tensor_bytes` 本身就递归 Mapping，过滤反而是破坏 | 删过滤，直接传 `batch.extras` |
| A1 | `vrl/models/families/wan_2_1/model.py:272,296` vs `vrl/models/peft_adapter.py:134` | **待裁决**（非直接修）：wan 手抄的 `apply_lora` 单边传 `autocast_adapter_dtype=False`，共享路径默认 `True`。注释称这是 FSDP/rollout 字节兼容硬需求——若成立则对所有 LoRA family 成立（共享侧是 bug）；若不成立则 wan 注释在撒谎。二者必有一错，需要一次 weight-sync 数值裁决 | 跑一次双侧对比后统一，赢家进共享 mixin |

## 2. 五种跨仓肿胀模式（根因层——修模式，不逐条摸鱼）

### 2-A mock 伺服双路径（重灾区：generation）

生产代码为测试替身留的分支：`executor.py` 4 处 + `weight_sync.py` 1 处
`if hasattr(x, "remote") else 本地直调`、`worker.py:550` `completion_callback=None`、
`worker.py:316` `margin`/`knee_threshold` 纯测试旋钮、`rollouts/collector/config.py:144`
三分支只有第一支有生产 producer、`prompt_collection.py:47` `stats=None`（两个生产调用者
都传值）、`batch_builder.py:37` `device=None`（唯一生产构造点恒传 "cpu"）。
**B1 就是这个模式的必然恶果**：替身有真对象没有的方法面。
修法统一：分支删掉、参数改必填、替身补真实方法面（`executor.py:686`
`_remote_engine_methods` 的 raise 写法已是范式；`test_weight_sync.py:70` 的
`_RemoteMethod` 已是现成的合格替身）。

### 2-B 孪生漂移（同一逻辑 ≥2 份拷贝，已经或正在分叉）

| 位置 | 拷贝数 | 已发生的漂移 |
|---|---|---|
| `trainers/online/trainer.py` replay/backward 循环（:1371 vs :1792）+ 收尾（:1410 vs :1837） | 2 | **B2 崩溃** |
| reward 模型 tensor→PIL（pickscore/animereward/aesthetic/hpsv3/nsfw 各一份，`decode_artifact_frames` 已存在只有 2 个消费者） | 5 | pickscore 取中间帧 vs animereward 取 batch 0 全帧——**打分对象已不同**，合并前需 reward 数值回归 |
| `cosmos3_reasoner.py:164` vs `videoscore2.py:129` Qwen-VL judge 前奏 | 2 | `.to(self.model.device)` vs `.to(self.device)`（device_map 下前者才对） |
| trainers 三份 all-reduce 统计体（advantages.py:30 / trainer.py:82 / grpo/continuous.py:362）+ 三份 dtype-only 集合原语（trainer.py:507-543） | 6 | 尚未分叉 |
| wan `apply_lora`（:241）/ `WanI2VReplayModel`（:1194）手抄共享 mixin/兄弟类 | 2 | **A1 分歧** |
| cosmos 安全检查器 stub ×2、`set_num_steps` 恒等 override ×3、p2.5 `finalize_noise_pred` 与基类逐字节同 | 6 | 未分叉 |
| sana `from_build` 手抄共享 loader 序列（flux 已是 `super().from_build` 正确形态） | 1 | 未分叉 |
| collector/core 与 prompt_collection 双秒表（一套 `VRL_PROFILE` 门控）；三个 `write_jsonl` 语义相反（append vs overwrite） | — | 语义已相反 |
| 三份 lazy-export facade 机器（trainers/data、trainers/online、trajectory）；`_PUBLIC_EXPORTS` 值重复自己的键 | 3 | — |

### 2-C 会撒谎的防护栏（比没有更糟：读者以为有保护）

- `kling_video_reward.py:47` "生产锁定键集"锁的 6 个 key 全仓无人读（只 `model_factory` 活）。
- `nn/optimization/passes.py` 四个 `conflicts()` 恒返回空元组 + 不可达校验循环（:390）；
  `OptimizationReport`/`introduces_replay_drift` 整条链零生产消费者（活的 drift 门禁走
  `REQUEST_SCOPED_DRIFT_SOURCES`，与此链无关）。
- `rollouts/collector/config.py:7` docstring 说"never from a hand-maintained list"，55 行后
  就是手写 sde key 映射（`SdeConfig` 加字段 → 用户设置静默失效，最坏一类）。
- `_OFFLINE_DPO_*_FIELDS`（schema.py:620）与真实读取集合无机械联系；
  `model.lora.init`/`init_lora_weights` 双别名 knob（preset 全用前者，denoise 读后者）。
- 无牙墓碑测试：`test_removed_inline_eval.py` 等 2 文件断言"派生机制会拒绝不在 schema 的
  key"（`test_unknown_keys.py` 已系统性证明）；`test_memory_policy_boundaries.py:52` 断言
  两个已不存在符号的字符串缺席。

### 2-D 死旋钮（零 producer 的可配置面）

- **CLI**：约 28 个 flag 全语料（.md/.yaml/.sh/tests）零出现——scripts lane 清单见其报告
  §14；最有害的是 `--use-config-lora`（开启后刻意保留一个日志自称要禁用的坏状态）与
  encode/merge 分片链（`merge_target_latents.py` 整文件零调用者）。
- **config/worker_config**：`flow_kl_use_dt`（两个 preset 都设 false=默认）、offline DPO
  `v_prediction` 分支（零 producer）、`use_adafactor`（仅测试）、reward 侧 9 个
  `.get(key, default)` 无任何 yaml 设置的 key（其中 `allow_absolute_paths` 是卡在
  permissive 位的安全旋钮，须显式裁决）。
- **参数**：`copy_ema_to(store_temp=)`、`inspect_cluster(driver_node_ip=)`、
  `actor_scheduling_strategy(capture_child_tasks=)`、`timeout_s`（reward client）、
  `_validate_rank_gpu_ids(expected_gpu_ids=None)` 等——全部"唯一非默认调用者是测试或不存在"。

### 2-E 幽灵与错置（删除量最大、风险最低）

- `vrl/families/`：纯 `__pycache__` 幽灵目录（git 追踪数 0，源码 097c60df 已迁走）。`rm -rf`。
- 零引用资产：`kling_video_reward_http.yaml` 等 5 个 preset、`kling_overlap_gate.yaml`
  （其启动说明指向不存在的文件）、`wan_i2v_logprob_parity_probe.py`（答案已逐字记录于
  parked sprint）、`wan_phys_ab_sample.py`、`anime_anatomy_report` 集群（608 行，依赖
  `rtmlib` 从未进 pyproject——装不上的僵尸，删或补依赖二选一，不许保持第三态）、
  fp8 两个 1 行兼容 facade（所有文档都已用新名——历史 KEEP 判决的前提未经 grep 验证）。
- 零导入 facade：`math/denoise`、`math/token`、`nn/modules`、`nn/layers/attention`、
  `steps/denoise/common` 5 个 re-export、`trainers/core/__init__`（唯一引用是一个负向测试）、
  danbooru `__init__` 33 个 re-export 中 29 个零外部消费者。
- utils 错置（搬家不删）：`nsys_report.py`（756 行）单消费者 → `scripts/perf/`；
  `model_diagnostics.py`（227 行）单消费者 → `trainers/`；`precision.py:_select` 重写了
  `cfg_path`；`ema.py:to()` 零调用者（且调了会破坏 parking 一致性）。
- reward inference 映射 4 个构造点（builders/reward_inference/registry-fallback/schema 重复
  parse）——数据版 form-4，收敛到 `RewardRuntimeConfig.from_cfg` 单点。

## 3. 测试侧清单（~1k 行可回收）

1. 逐字复制的测试文件/助手：hpsv3 vs videoscore2 `test_function.py` 四份同构（参 registry
   parametrize 合并）；`_FakeRuntime` ×6；`_engine`/`_ResolvedRef`/`_parking_snapshot` 各 2-3
   份（conftest 已存在可承接）；fp4/fp8 对基类定理的双份断言（真正的 per-scheme 数值保留）。
2. mock 自证：fp8 facade 测试（monkeypatch 掉唯一一行再断言它被调用）；
   `test_export_rollout_state_matches_helper`（断言实现等于实现）；三份
   `test_*_exposes_trainer_replay_methods`（注册表遍历版严格更强）。
3. 重复定理：`test_resources.py` 三对同配置同断言 + 一处三重死断言；
   `test_precision_drift_guard.py` 三对；`test_schema.py` vs `test_load_all_experiments.py`
   三对（留真 preset 版）。
4. 断言 stdlib 行为：`test_iteration_types.py`、`test_validation_cache.py`。
5. 错层测试：`test_wan_dpo_config.py:28-131` 八个测试测的是 `config/builders.py` 的定理，
   搬 `tests/config/`。
6. 手写表 → 注册表派生：`_CUSTOM_REPLAY_MODEL_CLASSES` 与 `_UNREGISTERED_REPLAY_CLASSES`
   互为重复；`test_replay_model_contract.py` 4 张写死 family 名单可由 `policy_semantics` 派生。

## 4. 本 sprint 自产代码的回审结论（诚实记录）

continuous 遥测刚落地即被审出两条成立的 finding：`continuous.active_batches` 与
queue `ready_batches` 在当前"单批不变量"下是结构性常量（≤1）——按我们自己给
`groups_discarded` 定的先例，应当缓做而没缓。处置：**删除 gauge/CSV 列与 stats key，
Sprint 2 lookahead 使两批并存时再加**（那时它才携带信息）；`queue.stats()` 的 6-key dict
唯一生产消费者是一条超时 f-string，收敛为 3 字段。双秒表问题（`VRL_PROFILE` 门控的
collector 内层计时 vs prompt_collection 恒开计时）归入 2-B 一并处置。

## 5. KEEP 汇总（已审过、下轮不再重审）

各审计道的完整 KEEP 清单在本次审计记录中，重点公示防误删：

- **generation/ray**：`RayLifecyclePlan` 全部字段有非日志消费者；`actor_pool` 公平准入拆分；
  `rank_group` 多卡链路是"已实现未启用能力"（registry 有 installer）——保留。
- **models**：25 族全可达；跨族统一薄方法是刻意的 grepability 资产；llamagen `vendor/` 是
  upstream-verbatim 快照不许改；能力方法（`apply_generation_offload` 等）走 getattr 字符串
  派发，符号 grep 零命中≠死。
- **trainers**：`_all_ranks_have_work` 的 docstring 是 NCCL 死锁证明本体；
  `_UnshardedStateStrategy` 一行委托是 lazy-import 环边界；selective checkpointing 是有实测
  收益的用户旋钮（probe-only 现状已在此记录，不算静默漂移）。
- **config**：`unknown_keys.py` 纯派生机制是全仓范本；41 个 schema 字段逐一查过 reader，
  零空旋钮。
- **scripts**：`*_probe` 生命周期制度执行良好（b0a27a8d 一次清了 10 个）；
  `wan_i2v_base_sample.py` 是刻意保留的上游归因 adapter（有 sprint 判决）；
  `reward_overlap_benchmark` 的三个"断点续跑"flag 保留但须写进 docs/perf/README。
- **rollouts**：continuous 的 Schedule/Owner/Runtime 三层各是真边界；
  `_interval_overlap_seconds` 是命名的非平凡算法。

## 6. 制度补丁（比单次清理更值钱的两条）

1. **兼容 facade 保留判决必须附 grep 证据**：本次两处历史 KEEP（fp8 facade、danbooru
   门面）都建立在"旧调用者仍存在"这个未验证前提上——这是"判 caller 不判 body"失误在
   兼容层上的变体。规则：sprint doc 里无"谁还在用旧名"的 grep 输出即默认删除。
2. **dead-flag lint 进 `make verify`**：argparse flag × 全语料 grep 的机械检查（本次审计
   脚本可直接沉淀），让旋钮腐烂在当天而非三个 sprint 后被发现。`vrl/config/lint.py` 的
   AST sweep 机制已现成，可扩展承接 `_OFFLINE_DPO_*_FIELDS` 的交叉校验。

## 7. Non-goals

- 不折叠跨族统一形状的薄方法（models）；不动 vendor 快照；不删"已实现未启用"能力
  （sequence_parallel/rank_group）。
- 不把 `tests/config` 的独立 capability matrix fixture 改成派生（注释已声明刻意独立）。
- 不为省行数合并断言不同定理的相似测试（fp4 对齐拒绝 vs fp8 blockwise 回退各自保留）。
- 2-B 的 reward tensor→PIL 合并**不得**顺手统一帧采样语义——采样策略变更是行为变更，
  须先出 reward 数值回归再定。

## 8. 执行批次与验收门

- **批次 0（P0）**：§1 六条。每条独立 commit；B1/B2 补失败路径测试；A1 出数值裁决记录。
- **批次 1**：2-E 幽灵删除 + 零导入 facade + 死旋钮（机械、低风险、量大）。
- **批次 2**：2-A 双路径拆除（generation 测试替身升级先行）。
- **批次 3**：2-B 孪生合并（trainers replay 循环、rewards judge/PIL——后者带 reward 回归门）。
- **批次 4**：2-C 防护栏修真 + §3 测试清理 + §6 制度补丁。
- 每批验收：`pytest`（受影响包全量）、`ruff check/format --check`（仅触碰文件）、
  config resolve 冒烟、`git diff --check`；删除项按"同源同生命周期"扩展清理并 grep import
  graph 确认无长期资产引用。

## 9. References

- 审计道报告（7 份，本次会话产出，finding 坐标以本文档为准）
- `AGENTS.md`（死代码五形式 / placement 四规则 / one-shot vs long-term）
- 先例：`docs/sprints/done/SPRINT_deadcode_00_overview.md`、
  `SPRINT_deadcode_rollouts_trainers_ray.md`、`SPRINT_allcaps_constants_audit.md`、
  `SPRINT_homeless_function_placement.md`、`SPRINT_docstring-truth-and-double-dedup.md`
